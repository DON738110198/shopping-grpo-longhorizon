"""Plan-bound raw-token collection for controlled active suffix rollouts.

The collector deliberately does not train.  It proves that one frozen actor can
be sampled from an exactly captured decision prompt after replaying the public
environment prefix that produced that prompt.  Every behavior-changing input is
bound by the active-branch plan before an environment lease or model request.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from urllib.request import Request, urlopen

from shopping_grpo.environment.actions import action_reject_reason
from shopping_grpo.environment.observation import render_structured_observation
from shopping_grpo.environment.projection import project_observation
from shopping_grpo.environment.tools import SHOP_TOOL_SCHEMAS, tool_call_to_action
from shopping_grpo.training.grpo.active_branch import (
    ACTIVE_BRANCH_PLAN_VERSION,
    ACTIVE_BRANCH_STRATEGY_VERSION,
    build_active_branch_plan,
    validate_actor_prompt_tokens,
    validate_decoding_config,
)
from shopping_grpo.training.grpo.adapter.runtime import (
    make_runtime_state,
    record_action_attempt,
    record_action_outcome,
    record_non_environment_decision,
    record_observation_projection,
    record_replay_transition,
    reward_breakdown,
    validate_policy_reward_config,
    validate_reward,
)
from shopping_grpo.training.grpo.pivotal_states import (
    canonical_replay_action,
    observation_sha256,
    replay_action_sha256,
    token_ids_sha256,
)
from shopping_grpo.training.grpo.replay import (
    DEFAULT_REQUIRED_ENVIRONMENT_VERSION,
    verify_replay_trajectory,
)

ACTIVE_SUFFIX_RESULT_VERSION = "shopping-active-suffix-result-v1"
ACTIVE_SUFFIX_COLLECTION_VERSION = "shopping-active-suffix-collection-v1"
SAMPLING_BACKEND_CONTRACT_VERSION = "shopping-vllm-token-completion-backend-v1"

_SAMPLING_FIELDS = (
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "repetition_penalty",
    "presence_penalty",
    "frequency_penalty",
    "min_tokens",
    "stop",
    "stop_token_ids",
    "ignore_eos",
)
_BACKEND_FIELDS = {
    "schema_version",
    "backend",
    "api",
    "server_version",
    "served_model",
    "served_model_root_sha256",
    "actor_checkpoint_sha256",
    "max_model_len",
    "effective_request",
}
_EFFECTIVE_REQUEST_FIELDS = {
    "prompt_format",
    "add_special_tokens",
    "echo",
    "n",
    "logprobs",
    "max_tokens_mode",
    "seed_schedule",
    "return_token_ids",
    "skip_special_tokens",
    *_SAMPLING_FIELDS,
}


class ActiveSuffixInfrastructureError(RuntimeError):
    """A replay, server, parser, or tensor contract failed closed."""

    def __init__(self, message: str, *, code: str | None = None):
        super().__init__(message)
        self.code = code or _infrastructure_error_code(message)


def _infrastructure_error_code(message: object) -> str:
    """Map internal detail to a finite, payload-free diagnostic code."""
    text = str(message)
    rules = (
        (("trajectory alignment", "completion token/logprob", "final suffix tensors"), "trajectory_alignment_invalid"),
        (("prefix replay", "replayed ", "replay produced", "replayed state"), "prefix_replay_invalid"),
        (("decision trace", "selected ", "turn span", "replay prefix", "prefix "), "prefix_runtime_state_invalid"),
        (("projected branch observation", "observation projection"), "projection_contract_invalid"),
        (("vLLM", "served model", "live sampling backend", "sampling backend"), "sampling_backend_invalid"),
        (("completion prompt", "echoed prompt", "plan binding"), "completion_contract_invalid"),
        (("actor checkpoint",), "actor_attestation_invalid"),
        (("captured prompt", "prompt hash", "actor_prompt"), "prompt_capture_invalid"),
        (("tool schema", "qwen3_coder parser"), "tool_contract_invalid"),
        (("encoder tokenizer", "captured tokenizer", "tool observation"), "tokenizer_contract_invalid"),
        (("environment step", "environment release", "terminal result", "terminal reward"), "environment_contract_invalid"),
        (("plan ", "active branch plan", "decoding config", "suffix_uid", "derived suffix seed"), "plan_contract_invalid"),
    )
    for prefixes, code in rules:
        if any(prefix in text for prefix in prefixes):
            return code
    return "active_suffix_contract_invalid"


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _is_sha256(value: object) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def sha256_actor_checkpoint(path: str | Path) -> str:
    """Hash every actor file and relative path to attest the served weights."""
    requested_root = Path(path).expanduser()
    if requested_root.is_symlink():
        raise ValueError("actor checkpoint must not be a symbolic link")
    root = requested_root.resolve()
    if not root.is_dir():
        raise ValueError("actor checkpoint must be a directory")
    entries = sorted(root.rglob("*"))
    if any(item.is_symlink() for item in entries):
        raise ValueError("actor checkpoint must not contain symbolic links")
    files = [item for item in entries if item.is_file()]
    if not files:
        raise ValueError("actor checkpoint directory contains no files")
    tree = hashlib.sha256()
    for item in files:
        file_hash = hashlib.sha256()
        size = 0
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                size += len(chunk)
                file_hash.update(chunk)
        relative = item.relative_to(root).as_posix()
        tree.update(f"{relative}\0{size}\0{file_hash.hexdigest()}\n".encode())
    return tree.hexdigest()


def attest_actor_checkpoint(path: str | Path, expected_sha256: str) -> str:
    """Fail before rollout when the plan's actor claim differs from disk."""
    actual = sha256_actor_checkpoint(path)
    if actual != str(expected_sha256):
        raise ActiveSuffixInfrastructureError("actor checkpoint SHA256 mismatch")
    return actual


def tool_schema_sha256(tool_schemas: Sequence[Mapping[str, object]]) -> str:
    """Hash the public OpenAI schemas in the order exposed to the actor."""
    if isinstance(tool_schemas, (str, bytes)) or not isinstance(tool_schemas, Sequence):
        raise TypeError("tool_schemas must be a sequence")
    normalized = []
    for index, schema in enumerate(tool_schemas):
        if not isinstance(schema, Mapping):
            raise TypeError(f"tool schema {index} must be an object")
        normalized.append(json.loads(_canonical_json(dict(schema))))
    if not normalized:
        raise ValueError("tool_schemas must not be empty")
    return _sha256_json(normalized)


def _effective_request(decoding_config: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(decoding_config, Mapping):
        raise TypeError("decoding_config must be an object")
    missing = set(_SAMPLING_FIELDS) - set(decoding_config)
    if missing:
        raise ValueError("decoding_config is missing sampling fields")
    return {
        "prompt_format": "token_ids",
        "add_special_tokens": False,
        "echo": False,
        "n": 1,
        "logprobs": 1,
        "max_tokens_mode": decoding_config["max_tokens_mode"],
        "seed_schedule": decoding_config["seed_schedule"],
        "return_token_ids": True,
        "skip_special_tokens": False,
        **{name: deepcopy(decoding_config[name]) for name in _SAMPLING_FIELDS},
    }


def validate_sampling_backend_contract(contract: Mapping[str, object]) -> dict[str, object]:
    """Validate the expected server and exact token-completion request contract."""
    if not isinstance(contract, Mapping):
        raise TypeError("sampling backend contract must be an object")
    if set(contract) != _BACKEND_FIELDS:
        raise ValueError("sampling backend contract fields do not match the contract")
    if contract.get("schema_version") != SAMPLING_BACKEND_CONTRACT_VERSION:
        raise ValueError("sampling backend contract version mismatch")
    if contract.get("backend") != "vllm" or contract.get("api") != "openai-completions":
        raise ValueError("sampling backend must be vLLM OpenAI completions")
    for name in ("server_version", "served_model"):
        if not isinstance(contract.get(name), str) or not contract[name]:
            raise ValueError(f"sampling backend {name} must be a non-empty string")
    for name in ("served_model_root_sha256", "actor_checkpoint_sha256"):
        if not _is_sha256(contract.get(name)):
            raise ValueError(f"sampling backend {name} must be a SHA256 digest")
    if contract["served_model_root_sha256"] != contract["actor_checkpoint_sha256"]:
        raise ValueError("served model root does not attest the actor checkpoint")
    max_model_len = contract.get("max_model_len")
    if not isinstance(max_model_len, int) or isinstance(max_model_len, bool) or max_model_len < 2:
        raise ValueError("sampling backend max_model_len must be at least two")
    request = contract.get("effective_request")
    if not isinstance(request, Mapping) or set(request) != _EFFECTIVE_REQUEST_FIELDS:
        raise ValueError("sampling backend effective_request fields do not match")
    if request.get("prompt_format") != "token_ids":
        raise ValueError("sampling backend prompt_format must equal token_ids")
    fixed = {
        "add_special_tokens": False,
        "echo": False,
        "n": 1,
        "logprobs": 1,
        "max_tokens_mode": "verl-v0.8-remaining-context",
        "seed_schedule": "first-suffix-then-sha256-turn-v1",
        "return_token_ids": True,
        "skip_special_tokens": False,
    }
    if any(request.get(name) != value for name, value in fixed.items()):
        raise ValueError("sampling backend token response capabilities do not match")
    sampling = {name: deepcopy(request[name]) for name in _SAMPLING_FIELDS}
    # Reuse the branch validator for all sampling ranges without weakening it.
    reference = {
        **sampling,
        "max_steps": 1,
        "prompt_length": 1,
        "response_length": int(max_model_len) - 1,
        "context_window": int(max_model_len),
        "context_generation_reserve": 1,
        "context_safety_margin": 0,
        "context_input_budget": max(
            1,
            int(max_model_len) - 1,
        ),
        "context_preserve_recent_groups": 1,
        "context_compaction_enable": False,
        "max_user_turns": 1,
        "max_assistant_turns": 1,
        "max_parallel_calls": 1,
        "max_tool_response_length": 1,
        "tool_response_truncate_side": "middle",
        "tokenization_sanity_check_mode": "ignore_strippable",
        "apply_chat_template_kwargs": {},
        "mm_processor_kwargs": {},
        "max_tokens_mode": "verl-v0.8-remaining-context",
        "seed_schedule": "first-suffix-then-sha256-turn-v1",
        "tool_parser": "contract-only",
        "tool_schema_sha256": "0" * 64,
        "generation_config_source": "vllm",
        "sampling_backend_contract_sha256": "0" * 64,
        "observation_token_budget": 64,
        "observation_detail_token_budget": 64,
        "observation_generic_token_budget": 64,
        "observation_search_top_k": 1,
        "observation_policy_sha256": "0" * 64,
    }
    # validate_decoding_config also checks the observation hash, so validate the
    # sampling values through a temporary real observation hash.
    from shopping_grpo.training.grpo.active_branch import observation_policy_sha256

    reference["observation_policy_sha256"] = observation_policy_sha256(reference)
    validate_decoding_config(reference)
    return json.loads(_canonical_json(dict(contract)))


def sampling_backend_contract_sha256(contract: Mapping[str, object]) -> str:
    return _sha256_json(validate_sampling_backend_contract(contract))


def build_sampling_backend_contract(
    *,
    server_version: str,
    served_model: str,
    served_model_root_sha256: str,
    actor_checkpoint_sha256: str,
    max_model_len: int,
    decoding_config: Mapping[str, object],
) -> dict[str, object]:
    """Materialize the digestable contract from actual server metadata."""
    return validate_sampling_backend_contract(
        {
            "schema_version": SAMPLING_BACKEND_CONTRACT_VERSION,
            "backend": "vllm",
            "api": "openai-completions",
            "server_version": str(server_version),
            "served_model": str(served_model),
            "served_model_root_sha256": str(served_model_root_sha256),
            "actor_checkpoint_sha256": str(actor_checkpoint_sha256),
            "max_model_len": int(max_model_len),
            "effective_request": _effective_request(decoding_config),
        }
    )


def _token_ids(values: object, name: str, *, allow_empty: bool = False) -> list[int]:
    if not isinstance(values, list):
        raise TypeError(f"{name} must be a list")
    normalized = []
    for value in values:
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{name} must contain non-negative integer token ids")
        normalized.append(value)
    if not normalized and not allow_empty:
        raise ValueError(f"{name} must not be empty")
    return normalized


def _finite_logprobs(values: object, expected_count: int) -> list[float]:
    if not isinstance(values, list) or len(values) != expected_count:
        raise ActiveSuffixInfrastructureError("completion token/logprob length mismatch")
    normalized = []
    for value in values:
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
        ):
            raise ActiveSuffixInfrastructureError("completion contains an invalid token logprob")
        normalized.append(float(value))
    return normalized


def generation_max_tokens(config: Mapping[str, object], prompt_token_count: int) -> int:
    """Mirror veRL 0.8's remaining sequence/context budget for one generation."""
    decoding = validate_decoding_config(config)
    if not isinstance(prompt_token_count, int) or isinstance(prompt_token_count, bool):
        raise TypeError("prompt_token_count must be an integer")
    remaining = min(
        decoding["response_length"],
        decoding["context_window"] - prompt_token_count,
        decoding["prompt_length"]
        + decoding["response_length"]
        - prompt_token_count,
    )
    if remaining < 1 or remaining < decoding["min_tokens"]:
        raise ActiveSuffixInfrastructureError("generation token budget is exhausted")
    return remaining


class _PlanBoundCompletionClient:
    def __init__(self, client, binding_sha256, decoding_config):
        self._client = client
        self._binding_sha256 = str(binding_sha256)
        self._decoding_config = validate_decoding_config(decoding_config)

    def complete(self, prompt_token_ids: Sequence[int], *, seed: int) -> dict[str, object]:
        prompt = list(prompt_token_ids)
        return self._client._complete_bound(
            self._binding_sha256,
            prompt,
            seed=seed,
            max_tokens=generation_max_tokens(self._decoding_config, len(prompt)),
            **{name: self._decoding_config[name] for name in _SAMPLING_FIELDS},
        )


class VllmTokenCompletionClient:
    """vLLM client that is unusable until a runner binds and attests a plan."""

    def __init__(
        self,
        model,
        base_url,
        api_key,
        timeout=180,
        transport=None,
        metadata_transport=None,
    ):
        self.model = str(model)
        base_url = str(base_url).rstrip("/")
        if not base_url.endswith("/v1"):
            base_url = f"{base_url}/v1"
        self.base_url = base_url
        self.url = f"{base_url}/completions"
        self.models_url = f"{base_url}/models"
        self.version_url = f"{base_url[:-3]}/version"
        self.api_key = str(api_key)
        self.timeout = int(timeout)
        self.transport = transport
        self.metadata_transport = metadata_transport
        self._bindings: set[str] = set()

    def complete(self, *args, **kwargs):
        del args, kwargs
        raise ActiveSuffixInfrastructureError(
            "completion client is not plan-bound; use ActiveSuffixRunner"
        )

    def _headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "User-Agent": "shopping-grpo-longhorizon/0.1",
        }

    def _get_json(self, url: str) -> object:
        if self.metadata_transport is not None:
            return self.metadata_transport(url, self._headers(), self.timeout)
        request = Request(url, headers=self._headers(), method="GET")
        with urlopen(request, timeout=self.timeout) as raw:
            return json.loads(raw.read().decode("utf-8"))

    def attest_and_bind(
        self,
        *,
        plan_sha256: str,
        expected_contract: Mapping[str, object],
        expected_contract_sha256: str,
        decoding_config: Mapping[str, object],
        actor_checkpoint: str | Path,
        actor_checkpoint_sha256: str,
    ) -> _PlanBoundCompletionClient:
        """Query the live server, hash its loaded root, and bind one exact plan."""
        expected = validate_sampling_backend_contract(expected_contract)
        if sampling_backend_contract_sha256(expected) != expected_contract_sha256:
            raise ActiveSuffixInfrastructureError("sampling backend contract SHA256 mismatch")
        live = self.inspect_backend(
            actor_checkpoint=actor_checkpoint,
            actor_checkpoint_sha256=actor_checkpoint_sha256,
        )
        actual = build_sampling_backend_contract(
            server_version=live["server_version"],
            served_model=live["served_model"],
            served_model_root_sha256=live["served_model_root_sha256"],
            actor_checkpoint_sha256=actor_checkpoint_sha256,
            max_model_len=live["max_model_len"],
            decoding_config=decoding_config,
        )
        if actual != expected:
            raise ActiveSuffixInfrastructureError("live sampling backend contract mismatch")
        if live["max_model_len"] != int(decoding_config["context_window"]):
            raise ActiveSuffixInfrastructureError("vLLM max_model_len differs from plan")
        binding_sha256 = _sha256_json(
            {
                "plan_sha256": plan_sha256,
                "backend_contract_sha256": expected_contract_sha256,
                "actor_checkpoint_sha256": actor_checkpoint_sha256,
            }
        )
        self._bindings.add(binding_sha256)
        return _PlanBoundCompletionClient(self, binding_sha256, decoding_config)

    def inspect_backend(
        self,
        *,
        actor_checkpoint: str | Path,
        actor_checkpoint_sha256: str,
    ) -> dict[str, object]:
        """Read live vLLM metadata and cryptographically tie its root to the actor."""
        version_response = self._get_json(self.version_url)
        models_response = self._get_json(self.models_url)
        if not isinstance(version_response, Mapping) or not isinstance(
            version_response.get("version"), str
        ):
            raise ActiveSuffixInfrastructureError("vLLM /version response is invalid")
        if not isinstance(models_response, Mapping) or not isinstance(
            models_response.get("data"), list
        ):
            raise ActiveSuffixInfrastructureError("vLLM /models response is invalid")
        matches = [
            item
            for item in models_response["data"]
            if isinstance(item, Mapping) and item.get("id") == self.model
        ]
        if len(matches) != 1:
            raise ActiveSuffixInfrastructureError("served model identity is ambiguous or missing")
        model_info = matches[0]
        root_value = model_info.get("root")
        if not isinstance(root_value, str) or not root_value:
            raise ActiveSuffixInfrastructureError("served model metadata is missing root")
        served_root_path = Path(root_value).expanduser()
        if served_root_path.is_symlink():
            raise ActiveSuffixInfrastructureError("vLLM served root is a symbolic link")
        actor_path = Path(actor_checkpoint).expanduser()
        actor_sha256 = sha256_actor_checkpoint(actor_path)
        if actor_sha256 != actor_checkpoint_sha256:
            raise ActiveSuffixInfrastructureError("actor checkpoint SHA256 mismatch")
        served_root = served_root_path.resolve()
        actor_root = actor_path.resolve()
        if served_root != actor_root:
            raise ActiveSuffixInfrastructureError("vLLM served root is not the requested actor")
        max_model_len = model_info.get("max_model_len")
        if not isinstance(max_model_len, int) or isinstance(max_model_len, bool):
            raise ActiveSuffixInfrastructureError("served model max_model_len is invalid")
        return {
            "server_version": version_response["version"],
            "served_model": self.model,
            "served_model_root_sha256": actor_sha256,
            "max_model_len": max_model_len,
        }

    def materialize_backend_contract(
        self,
        *,
        actor_checkpoint: str | Path,
        actor_checkpoint_sha256: str,
        decoding_config: Mapping[str, object],
    ) -> dict[str, object]:
        """Build a contract from the live server; no caller-provided digest is trusted."""
        live = self.inspect_backend(
            actor_checkpoint=actor_checkpoint,
            actor_checkpoint_sha256=actor_checkpoint_sha256,
        )
        return build_sampling_backend_contract(
            server_version=live["server_version"],
            served_model=live["served_model"],
            served_model_root_sha256=live["served_model_root_sha256"],
            actor_checkpoint_sha256=actor_checkpoint_sha256,
            max_model_len=live["max_model_len"],
            decoding_config=decoding_config,
        )

    def _complete_bound(
        self,
        binding_sha256: str,
        prompt_token_ids: Sequence[int],
        *,
        seed: int,
        temperature: float,
        top_p: float,
        top_k: int,
        min_p: float,
        repetition_penalty: float,
        presence_penalty: float,
        frequency_penalty: float,
        max_tokens: int,
        min_tokens: int,
        stop: Sequence[str],
        stop_token_ids: Sequence[int],
        ignore_eos: bool,
    ) -> dict[str, object]:
        if binding_sha256 not in self._bindings:
            raise ActiveSuffixInfrastructureError("unknown or stale plan binding")
        prompt = _token_ids(list(prompt_token_ids), "prompt_token_ids")
        stops = _token_ids(list(stop_token_ids), "stop_token_ids", allow_empty=True)
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise TypeError("seed must be an integer")
        payload = {
            "model": self.model,
            "prompt": prompt,
            "add_special_tokens": False,
            "echo": False,
            "n": 1,
            "seed": seed,
            "temperature": float(temperature),
            "top_p": float(top_p),
            "top_k": int(top_k),
            "min_p": float(min_p),
            "repetition_penalty": float(repetition_penalty),
            "presence_penalty": float(presence_penalty),
            "frequency_penalty": float(frequency_penalty),
            "max_tokens": int(max_tokens),
            "min_tokens": int(min_tokens),
            "stop": list(stop),
            "stop_token_ids": stops,
            "ignore_eos": bool(ignore_eos),
            "logprobs": 1,
            "return_token_ids": True,
            "skip_special_tokens": False,
        }
        if self.transport is not None:
            response = self.transport(self.url, payload, self._headers(), self.timeout)
        else:
            request = Request(
                self.url,
                data=json.dumps(payload).encode("utf-8"),
                headers=self._headers(),
                method="POST",
            )
            with urlopen(request, timeout=self.timeout) as raw:
                response = json.loads(raw.read().decode("utf-8"))
        if not isinstance(response, Mapping):
            raise ActiveSuffixInfrastructureError("vLLM completion response is not an object")
        if response.get("model") != self.model:
            raise ActiveSuffixInfrastructureError("vLLM completion model identity changed")
        choices = response.get("choices")
        if not isinstance(choices, list) or len(choices) != 1:
            raise ActiveSuffixInfrastructureError("vLLM completion must return exactly one choice")
        choice = choices[0]
        if not isinstance(choice, Mapping):
            raise ActiveSuffixInfrastructureError("vLLM completion choice is not an object")
        echoed_prompt = _token_ids(choice.get("prompt_token_ids"), "echoed prompt_token_ids")
        if echoed_prompt != prompt:
            raise ActiveSuffixInfrastructureError("vLLM echoed prompt token ids mismatch")
        token_ids = _token_ids(
            choice.get("token_ids"),
            "completion token_ids",
            allow_empty=True,
        )
        raw_logprobs = choice.get("logprobs")
        if not isinstance(raw_logprobs, Mapping):
            raise ActiveSuffixInfrastructureError("vLLM completion is missing logprobs")
        logprobs = _finite_logprobs(raw_logprobs.get("token_logprobs"), len(token_ids))
        finish_reason = choice.get("finish_reason")
        if finish_reason not in {"stop", "length"}:
            raise ActiveSuffixInfrastructureError("vLLM completion has an invalid finish_reason")
        if not token_ids and finish_reason != "stop":
            raise ActiveSuffixInfrastructureError("empty completion must be immediate EOS/stop")
        stop_reason = choice.get("stop_reason")
        if stop_reason is not None and not isinstance(stop_reason, (int, str)):
            raise ActiveSuffixInfrastructureError("vLLM completion has an invalid stop_reason")
        return {
            "prompt_token_ids": echoed_prompt,
            "token_ids": token_ids,
            "old_logprobs": logprobs,
            "finish_reason": finish_reason,
            "stop_reason": stop_reason,
        }


def append_assistant_turn(
    response_ids: list[int],
    response_mask: list[int],
    old_logprobs: list[float],
    completion: Mapping[str, object],
) -> tuple[int, int]:
    """Append one generated Assistant span and assert tensor alignment."""
    token_ids = _token_ids(completion.get("token_ids"), "completion token_ids")
    logprobs = _finite_logprobs(completion.get("old_logprobs"), len(token_ids))
    start = len(response_ids)
    response_ids.extend(token_ids)
    response_mask.extend([1] * len(token_ids))
    old_logprobs.extend(logprobs)
    if not len(response_ids) == len(response_mask) == len(old_logprobs):
        raise ActiveSuffixInfrastructureError("active suffix tensors lost alignment")
    return start, len(response_ids)


def append_tool_observation(
    response_ids: list[int],
    response_mask: list[int],
    old_logprobs: list[float],
    observation_token_ids: Sequence[int],
) -> tuple[int, int]:
    """Append actor-visible tool tokens without assigning policy loss to them."""
    token_ids = _token_ids(
        list(observation_token_ids),
        "observation_token_ids",
        allow_empty=True,
    )
    start = len(response_ids)
    response_ids.extend(token_ids)
    response_mask.extend([0] * len(token_ids))
    old_logprobs.extend([0.0] * len(token_ids))
    if not len(response_ids) == len(response_mask) == len(old_logprobs):
        raise ActiveSuffixInfrastructureError("active suffix tensors lost alignment")
    return start, len(response_ids)


def _required_sha(value: object, name: str) -> str:
    if not _is_sha256(value):
        raise ValueError(f"{name} must be a lowercase SHA256 digest")
    return str(value)


def summarize_active_suffix_group(
    suffixes: Sequence[Mapping[str, object]],
    *,
    expected_group_uid: str | None = None,
    expected_k: int | None = None,
) -> dict[str, object]:
    """Strictly validate one K-suffix group before reporting contrasts."""
    if isinstance(suffixes, (str, bytes)) or not isinstance(suffixes, Sequence) or not suffixes:
        raise ValueError("active suffix group must not be empty")
    if expected_k is not None and len(suffixes) != int(expected_k):
        raise ValueError("active suffix group K mismatch")
    group_ids = set()
    parent_branch_ids = set()
    replay_state_ids = set()
    task_ids = set()
    actor_checkpoint_ids = set()
    decoding_config_ids = set()
    backend_contract_ids = set()
    suffix_ids = set()
    suffix_indices = set()
    seeds = set()
    learning_valid = []
    credit_valid = []
    infrastructure_invalid = []
    for index, item in enumerate(suffixes):
        if not isinstance(item, Mapping):
            raise TypeError(f"active suffix {index} must be an object")
        group_ids.add(_required_sha(item.get("active_group_uid"), "active_group_uid"))
        parent_branch_ids.add(
            _required_sha(item.get("parent_branch_uid"), "parent_branch_uid")
        )
        replay_state_ids.add(_required_sha(item.get("replay_state_id"), "replay_state_id"))
        task_id = item.get("task_id")
        if not isinstance(task_id, int) or isinstance(task_id, bool) or task_id < 0:
            raise ValueError("task_id must be a non-negative integer")
        task_ids.add(task_id)
        actor_checkpoint_ids.add(
            _required_sha(item.get("actor_checkpoint_sha256"), "actor_checkpoint_sha256")
        )
        decoding_config_ids.add(
            _required_sha(item.get("decoding_config_sha256"), "decoding_config_sha256")
        )
        backend_contract_ids.add(
            _required_sha(
                item.get("sampling_backend_contract_sha256"),
                "sampling_backend_contract_sha256",
            )
        )
        suffix_uid = _required_sha(item.get("suffix_uid"), "suffix_uid")
        if suffix_uid in suffix_ids:
            raise ValueError("active suffix group repeats suffix_uid")
        suffix_ids.add(suffix_uid)
        suffix_index = item.get("suffix_index")
        if not isinstance(suffix_index, int) or isinstance(suffix_index, bool):
            raise TypeError("suffix_index must be an integer")
        suffix_indices.add(suffix_index)
        seed = item.get("seed")
        if not isinstance(seed, int) or isinstance(seed, bool) or seed in seeds:
            raise ValueError("suffix seeds must be unique integers")
        seeds.add(seed)
        if not isinstance(item.get("strict"), bool):
            raise TypeError("strict must be boolean")
        if not isinstance(item.get("valid_for_learning"), bool):
            raise TypeError("valid_for_learning must be boolean")
        if not isinstance(item.get("first_action_credit_eligible"), bool):
            raise TypeError("first_action_credit_eligible must be boolean")
        if not isinstance(item.get("infrastructure_invalid"), bool):
            raise TypeError("infrastructure_invalid must be boolean")
        if not isinstance(item.get("model_failure"), bool):
            raise TypeError("model_failure must be boolean")
        error_code = item.get("infrastructure_error_code")
        if item["infrastructure_invalid"]:
            if not isinstance(error_code, str) or not error_code:
                raise ValueError("infrastructure-invalid suffix lacks a bounded error code")
            infrastructure_invalid.append(item)
        elif error_code is not None:
            raise ValueError("non-infrastructure suffix carries an infrastructure error code")
        request_prompts = item.get("request_prompt_sha256")
        request_seeds = item.get("request_seeds")
        if not isinstance(request_prompts, list) or not isinstance(request_seeds, list):
            raise TypeError("request prompt hashes and seeds must be lists")
        if len(request_prompts) != len(request_seeds):
            raise ValueError("request prompt hash and seed schedules are misaligned")
        if any(not _is_sha256(value) for value in request_prompts):
            raise ValueError("request prompt hashes must be SHA256 digests")
        if any(
            not isinstance(value, int) or isinstance(value, bool)
            for value in request_seeds
        ) or len(request_seeds) != len(set(request_seeds)):
            raise ValueError("request seeds must be unique integers")
        for turn_index, request_seed in enumerate(request_seeds):
            if request_seed != _request_seed(item, turn_index):
                raise ValueError("request seed schedule does not match the suffix contract")
        first_sha = item.get("first_action_sha256")
        if first_sha is not None:
            _required_sha(first_sha, "first_action_sha256")
            first_action = item.get("first_action")
            if not isinstance(first_action, Mapping):
                raise TypeError("first_action must be an object when its SHA256 is present")
            if replay_action_sha256(
                first_action.get("tool"),
                first_action.get("parameters"),
            ) != first_sha:
                raise ValueError("first_action_sha256 does not match first_action")
        first_span = item.get("first_action_span")
        if first_span is not None and (
            not isinstance(first_span, list)
            or len(first_span) != 2
            or any(
                not isinstance(value, int) or isinstance(value, bool)
                for value in first_span
            )
            or not 0 <= first_span[0] < first_span[1]
        ):
            raise ValueError("first_action_span must be a non-empty integer span")
        if item.get("valid_for_learning") is True:
            if first_sha is None:
                raise ValueError("learning-valid suffix is missing first_action_sha256")
            if not request_seeds:
                raise ValueError("learning-valid suffix has no model request")
            response_ids = item.get("response_ids")
            response_mask = item.get("response_mask")
            old_logprobs = item.get("old_logprobs")
            if not all(isinstance(value, list) for value in (response_ids, response_mask, old_logprobs)):
                raise TypeError("learning-valid suffix tensors must be lists")
            if not len(response_ids) == len(response_mask) == len(old_logprobs):
                raise ValueError("learning-valid suffix tensors are misaligned")
            if any(value not in {0, 1} for value in response_mask):
                raise ValueError("learning-valid suffix response_mask must be binary")
            if not response_ids or not any(value == 1 for value in response_mask):
                raise ValueError("learning-valid suffix has no Assistant loss tokens")
            learning_valid.append(item)
            if item["first_action_credit_eligible"]:
                if first_span is None or first_span[1] > len(response_ids):
                    raise ValueError("credit-valid suffix has an invalid first action span")
                if any(value != 1 for value in response_mask[first_span[0] : first_span[1]]):
                    raise ValueError("first action span includes non-Assistant tokens")
                credit_valid.append(item)
    if len(group_ids) != 1:
        raise ValueError("active suffix records span multiple groups")
    for name, identities in (
        ("parent branch", parent_branch_ids),
        ("replay state", replay_state_ids),
        ("task", task_ids),
        ("actor checkpoint", actor_checkpoint_ids),
        ("decoding config", decoding_config_ids),
        ("sampling backend", backend_contract_ids),
    ):
        if len(identities) != 1:
            raise ValueError(f"active suffix records span multiple {name} identities")
    group_uid = next(iter(group_ids))
    if expected_group_uid is not None and group_uid != expected_group_uid:
        raise ValueError("active suffix group identity mismatch")
    if suffix_indices != set(range(len(suffixes))):
        raise ValueError("active suffix indices must be contiguous from zero")
    strict_values = {item["strict"] for item in credit_valid}
    first_actions = {str(item["first_action_sha256"]) for item in credit_valid}
    mixed_strict = strict_values == {False, True}
    action_divergent = len(first_actions) > 1
    return {
        "active_group_uid": group_uid,
        "suffixes": len(suffixes),
        "valid_suffixes": len(learning_valid),
        "invalid_suffixes": len(suffixes) - len(learning_valid),
        "infrastructure_invalid_suffixes": len(infrastructure_invalid),
        "credit_valid_suffixes": len(credit_valid),
        "unique_first_actions": len(first_actions),
        "mixed_strict": mixed_strict,
        "action_divergent": action_divergent,
        "eligible_for_action_credit": (
            len(credit_valid) >= 2 and mixed_strict and action_divergent
        ),
    }


def _suffix_seed(base_seed: int, group_uid: str, suffix_index: int) -> int:
    payload = f"{base_seed}:{group_uid}:{suffix_index}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


def _request_seed(suffix: Mapping[str, object], turn_index: int) -> int:
    if not isinstance(turn_index, int) or isinstance(turn_index, bool) or turn_index < 0:
        raise ValueError("suffix turn index must be a non-negative integer")
    suffix_seed = suffix.get("seed")
    suffix_uid = suffix.get("suffix_uid")
    if not isinstance(suffix_seed, int) or isinstance(suffix_seed, bool):
        raise TypeError("suffix seed must be an integer")
    _required_sha(suffix_uid, "suffix_uid")
    if turn_index == 0:
        return suffix_seed
    payload = _canonical_json(
        {
            "version": "first-suffix-then-sha256-turn-v1",
            "suffix_uid": suffix_uid,
            "suffix_seed": suffix_seed,
            "turn_index": turn_index,
        }
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


class _ReplayCaptureEnv:
    def __init__(self, env):
        self.env = env
        self.last_result = None

    def reset(self, task_id):
        self.last_result = self.env.reset(task_id)
        return self.last_result

    def step(self, action):
        self.last_result = self.env.step(action)
        return self.last_result


def _public_observation(result: Mapping[str, object]) -> str:
    state = result.get("observation_state")
    if not isinstance(state, Mapping):
        raise ActiveSuffixInfrastructureError("environment result lacks observation_state")
    return render_structured_observation(dict(state))


async def _maybe_await(value):
    return await value if inspect.isawaitable(value) else value


def _normalize_tool_calls(raw_calls: object) -> list[dict[str, object]]:
    if raw_calls is None:
        return []
    if isinstance(raw_calls, (str, bytes)) or not isinstance(raw_calls, Sequence):
        raise ActiveSuffixInfrastructureError("qwen3_coder parser returned a non-list")
    calls = []
    for raw in raw_calls:
        if isinstance(raw, Mapping):
            name = raw.get("name")
            arguments = raw.get("arguments", {})
        else:
            name = getattr(raw, "name", None)
            arguments = getattr(raw, "arguments", {})
        if not isinstance(name, str) or not name:
            raise ActiveSuffixInfrastructureError("parsed tool call is missing a name")
        arguments_error = None
        if not isinstance(arguments, Mapping):
            try:
                decoded_arguments = json.loads(arguments)
            except (json.JSONDecodeError, TypeError) as exc:
                arguments_error = str(exc)
                arguments = None
            else:
                # ShopSimulatorTool.execute normalizes every successfully decoded
                # non-object JSON value to an empty parameter object.
                arguments = (
                    dict(decoded_arguments)
                    if isinstance(decoded_arguments, Mapping)
                    else {}
                )
        calls.append(
            {
                "name": name,
                "arguments": dict(arguments) if arguments is not None else None,
                "arguments_error": arguments_error,
            }
        )
    return calls


def _bounded_tool_text(
    text: object,
    config: Mapping[str, object],
    *,
    projected_observation: bool,
) -> str:
    value = str(text or "")
    limit = int(config["max_tool_response_length"])
    if len(value) <= limit:
        return value
    if projected_observation:
        raise ActiveSuffixInfrastructureError(
            "projected observation exceeds max_tool_response_length"
        )
    side = config["tool_response_truncate_side"]
    if side == "left":
        return "(truncated)..." + value[-limit:]
    if side == "right":
        return value[:limit] + "...(truncated)"
    length = limit // 2
    return value[:length] + "...(truncated)..." + value[-length:]


def _restore_prefix_runtime_state(
    state: dict,
    resolved: Mapping[str, object],
    group: Mapping[str, object],
    config: Mapping[str, object],
) -> dict[str, int]:
    """Rebuild public AgentLoop counters at the captured decision boundary."""
    trajectory = resolved.get("trajectory")
    selected_event = resolved.get("event")
    source = group.get("source")
    if not all(isinstance(value, Mapping) for value in (trajectory, selected_event, source)):
        raise ActiveSuffixInfrastructureError("resolved branch runtime contract is incomplete")
    events = trajectory.get("decision_trace")
    spans = trajectory.get("turn_spans")
    if not isinstance(events, list) or not isinstance(spans, list):
        raise ActiveSuffixInfrastructureError("resolved branch lacks decision trace or turn spans")
    event_index = source.get("event_index")
    if (
        not isinstance(event_index, int)
        or isinstance(event_index, bool)
        or not 0 <= event_index < len(events)
    ):
        raise ActiveSuffixInfrastructureError("selected event index is invalid")
    if not isinstance(events[event_index], Mapping) or _canonical_json(
        dict(events[event_index])
    ) != _canonical_json(dict(selected_event)):
        raise ActiveSuffixInfrastructureError("selected decision event changed after resolution")

    for decision_index, raw_event in enumerate(events[: event_index + 1]):
        if not isinstance(raw_event, Mapping):
            raise ActiveSuffixInfrastructureError("decision trace contains a non-object")
        if raw_event.get("decision_index") != decision_index:
            raise ActiveSuffixInfrastructureError("decision trace indices are not contiguous")

    selected_turn_id = selected_event.get("assistant_turn_id")
    if (
        not isinstance(selected_turn_id, int)
        or isinstance(selected_turn_id, bool)
        or selected_turn_id < 0
    ):
        raise ActiveSuffixInfrastructureError("selected assistant_turn_id is invalid")
    matching_spans = [
        span
        for span in spans
        if isinstance(span, Mapping) and span.get("turn_id") == selected_turn_id
    ]
    if len(matching_spans) != 1:
        raise ActiveSuffixInfrastructureError("selected turn span is missing or ambiguous")
    selected_span = matching_spans[0]
    assistant_span = selected_span.get("assistant_span")
    if (
        selected_span.get("credit_eligible") is not True
        or not isinstance(assistant_span, list)
        or len(assistant_span) != 2
        or not all(
            isinstance(value, int) and not isinstance(value, bool)
            for value in assistant_span
        )
        or not 0 <= assistant_span[0] < assistant_span[1]
    ):
        raise ActiveSuffixInfrastructureError("selected Assistant span is invalid")
    response_tokens_before = assistant_span[0]
    if response_tokens_before >= int(config["response_length"]):
        raise ActiveSuffixInfrastructureError("selected branch exhausted response_length")
    if selected_turn_id >= int(config["max_assistant_turns"]):
        raise ActiveSuffixInfrastructureError("selected branch exhausted max_assistant_turns")
    if selected_turn_id >= int(config["max_user_turns"]):
        raise ActiveSuffixInfrastructureError("selected branch exhausted max_user_turns")

    ledger = trajectory.get("replay_ledger")
    prefix_count = group.get("prefix_action_count")
    if not isinstance(ledger, list) or not isinstance(prefix_count, int):
        raise ActiveSuffixInfrastructureError("selected replay prefix is invalid")
    prefix_ledger = ledger[:prefix_count]
    accepted_count = 0
    action_attempt_count = 0
    repeat_action_count = 0
    guard_rejection_count = 0
    consecutive_guard_rejections = 0
    recent_signatures: list[tuple[str, str, str]] = []
    previous_turn_id = -1
    restored_steps = []
    restored_action_events = []

    for raw_event in events[:event_index]:
        turn_id = raw_event.get("assistant_turn_id")
        if (
            not isinstance(turn_id, int)
            or isinstance(turn_id, bool)
            or not previous_turn_id < turn_id < selected_turn_id
        ):
            raise ActiveSuffixInfrastructureError("prefix assistant turn ids are invalid")
        previous_turn_id = turn_id
        tool = str(raw_event.get("tool") or "")
        raw_parameters = raw_event.get("replay_parameters")
        if not isinstance(raw_parameters, Mapping):
            raise ActiveSuffixInfrastructureError("prefix action lacks exact parameters")
        action = canonical_replay_action(tool, raw_parameters)
        if replay_action_sha256(tool, action["parameters"]) != raw_event.get(
            "action_sha256"
        ):
            raise ActiveSuffixInfrastructureError("prefix action SHA256 mismatch")
        if raw_event.get("prefix_action_count") != accepted_count:
            raise ActiveSuffixInfrastructureError("prefix action count drifted")
        decision_kind = raw_event.get("decision_kind")
        if decision_kind == "think":
            if (
                tool != "think"
                or raw_event.get("accepted") is not None
                or raw_event.get("repeated") is not False
            ):
                raise ActiveSuffixInfrastructureError("prefix think event is inconsistent")
            restored_steps.append(
                {
                    "index": len(restored_steps),
                    "tool": "think",
                    "parameters": action["parameters"],
                    "done": False,
                    "reward": 0.0,
                }
            )
            continue
        if decision_kind != "environment_tool" or tool == "think":
            raise ActiveSuffixInfrastructureError("prefix contains a terminal decision")
        observation_fingerprint = raw_event.get("observation_sha256")
        if not _is_sha256(observation_fingerprint):
            raise ActiveSuffixInfrastructureError("prefix action observation hash is invalid")
        signature = (
            tool,
            _canonical_json(action["parameters"]),
            str(observation_fingerprint),
        )
        repeated = signature in recent_signatures
        if raw_event.get("repeated") is not repeated:
            raise ActiveSuffixInfrastructureError("prefix repeat marker is inconsistent")
        action_attempt_count += 1
        repeat_action_count += int(repeated)
        recent_signatures.append(signature)
        del recent_signatures[:-3]
        accepted = raw_event.get("accepted")
        guard_reason = raw_event.get("guard_reason")
        error = raw_event.get("error")
        if accepted is True:
            if guard_reason is not None or error is not None or accepted_count >= len(prefix_ledger):
                raise ActiveSuffixInfrastructureError("accepted prefix action is inconsistent")
            transition = prefix_ledger[accepted_count]
            if canonical_replay_action(
                transition.get("tool"), transition.get("parameters")
            ) != action:
                raise ActiveSuffixInfrastructureError("accepted action differs from replay ledger")
            if transition.get("done") is not False:
                raise ActiveSuffixInfrastructureError("selected prefix contains a terminal action")
            restored_steps.append(
                {
                    "index": len(restored_steps),
                    "tool": action["tool"],
                    "parameters": action["parameters"],
                    "done": False,
                    "reward": 0.0,
                }
            )
            accepted_count += 1
            consecutive_guard_rejections = 0
        elif accepted is False:
            if not isinstance(guard_reason, str) or not guard_reason or error is not None:
                raise ActiveSuffixInfrastructureError("rejected prefix action is inconsistent")
            guard_rejection_count += 1
            consecutive_guard_rejections += 1
        else:
            raise ActiveSuffixInfrastructureError("prefix action has no accepted outcome")
        restored_action_events.append(
            {
                "index": action_attempt_count - 1,
                "decision_index": raw_event["decision_index"],
                "tool": tool,
                "parameters": deepcopy(raw_event.get("parameters") or {}),
                "action_sha256": raw_event["action_sha256"],
                "observation_sha256": observation_fingerprint,
                "repeated": repeated,
                "accepted": accepted,
                "guard_reason": guard_reason,
                "error": error,
            }
        )

    if accepted_count != prefix_count or accepted_count != len(prefix_ledger):
        raise ActiveSuffixInfrastructureError("decision trace and replay prefix diverge")
    if len(restored_steps) >= int(config["max_steps"]):
        raise ActiveSuffixInfrastructureError("selected branch exhausted max_steps")
    if consecutive_guard_rejections >= 3:
        raise ActiveSuffixInfrastructureError("selected branch follows terminal guard state")

    selected_action = canonical_replay_action(
        selected_event.get("tool"), selected_event.get("replay_parameters")
    )
    if replay_action_sha256(
        selected_action["tool"], selected_action["parameters"]
    ) != selected_event.get("action_sha256"):
        raise ActiveSuffixInfrastructureError("selected action SHA256 mismatch")
    if selected_event.get("prefix_action_count") != prefix_count:
        raise ActiveSuffixInfrastructureError("selected event prefix count mismatch")
    if selected_event.get("decision_index") != event_index:
        raise ActiveSuffixInfrastructureError("selected decision index mismatch")
    selected_kind = selected_event.get("decision_kind")
    selected_accepted = selected_event.get("accepted")
    if selected_kind == "environment_tool" and selected_accepted is True:
        if prefix_count >= len(ledger) or canonical_replay_action(
            ledger[prefix_count].get("tool"), ledger[prefix_count].get("parameters")
        ) != selected_action:
            raise ActiveSuffixInfrastructureError("selected accepted action lacks ledger entry")
    elif selected_kind == "environment_tool" and selected_accepted is False:
        if not selected_event.get("guard_reason"):
            raise ActiveSuffixInfrastructureError("selected rejected action lacks guard reason")
    elif selected_kind == "think":
        if selected_action["tool"] != "think" or selected_accepted is not None:
            raise ActiveSuffixInfrastructureError("selected think event is inconsistent")
    elif selected_kind == "assistant_final":
        if selected_action["tool"] != "assistant_final" or selected_accepted is not None:
            raise ActiveSuffixInfrastructureError("selected final event is inconsistent")
    else:
        raise ActiveSuffixInfrastructureError("selected decision outcome is inconsistent")

    state["steps"] = restored_steps
    state["action_attempt_count"] = action_attempt_count
    state["repeat_action_count"] = repeat_action_count
    state["recent_action_signatures"] = recent_signatures
    state["guard_rejection_count"] = guard_rejection_count
    state["consecutive_guard_rejections"] = consecutive_guard_rejections
    state["action_events"] = restored_action_events
    state["decision_events"] = [deepcopy(dict(event)) for event in events[:event_index]]
    state["decision_count"] = event_index
    state["next_assistant_turn_id"] = selected_turn_id
    return {
        "assistant_turns": selected_turn_id,
        "user_turns": selected_turn_id,
        "response_tokens_before": response_tokens_before,
    }


class ActiveSuffixRunner:
    """Single plan-bound runner for an observational, no-optimizer suffix probe."""

    def __init__(
        self,
        *,
        plan: Mapping[str, object],
        resolved_selections: Sequence[Mapping[str, object]],
        actor_checkpoint: str | Path,
        sampling_backend_contract: Mapping[str, object],
        completion_client,
        parser,
        encoder,
        env_factory,
        tool_schemas: Sequence[Mapping[str, object]] = SHOP_TOOL_SCHEMAS,
        required_environment_version: str = DEFAULT_REQUIRED_ENVIRONMENT_VERSION,
        policy_reward: object = None,
        vllm_timeout_seconds: int = 180,
        environment_timeout_seconds: int = 60,
        expected_groups: int = 2,
        expected_suffixes_per_state: int = 4,
    ):
        self.plan = json.loads(_canonical_json(dict(plan)))
        self.resolved = json.loads(_canonical_json(list(resolved_selections)))
        self.plan_sha256 = _sha256_json(self.plan)
        self.resolved_sha256 = _sha256_json(self.resolved)
        self.actor_checkpoint = Path(actor_checkpoint).expanduser().absolute()
        self.backend_contract = validate_sampling_backend_contract(
            sampling_backend_contract
        )
        self.parser = parser
        self.encoder = encoder
        self.env_factory = env_factory
        self.completion_client = completion_client
        self.tool_schemas = [json.loads(_canonical_json(dict(item))) for item in tool_schemas]
        self.tool_name_order = [
            str((schema.get("function") or {}).get("name") or "")
            for schema in self.tool_schemas
        ]
        self.tool_names = set(self.tool_name_order)
        if "" in self.tool_names or len(self.tool_names) != len(self.tool_schemas):
            raise ValueError("tool schemas must have unique function names")
        self.required_environment_version = str(required_environment_version)
        self.policy_reward = validate_policy_reward_config(policy_reward)
        self.policy_reward_sha256 = _sha256_json(self.policy_reward)
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 1
            for value in (vllm_timeout_seconds, environment_timeout_seconds)
        ):
            raise ValueError("active suffix timeouts must be positive")
        self.vllm_timeout_seconds = vllm_timeout_seconds
        self.environment_timeout_seconds = environment_timeout_seconds
        if getattr(completion_client, "timeout", None) != self.vllm_timeout_seconds:
            raise ValueError("completion client timeout differs from the runner contract")
        self.expected_groups = int(expected_groups)
        self.expected_suffixes_per_state = int(expected_suffixes_per_state)
        if self.expected_groups < 1 or self.expected_suffixes_per_state < 2:
            raise ValueError("active suffix expected group/K counts are invalid")
        self._validate_plan_fresh()
        decoding = self.plan["decoding_config"]
        if decoding["context_compaction_enable"] is not False:
            raise ValueError("active suffix collection currently requires compaction=false")
        if decoding["tool_parser"] != "qwen3_coder":
            raise ValueError("active suffix collection requires qwen3_coder parsing")
        if len(self.plan["groups"]) != self.expected_groups:
            raise ValueError("active suffix plan group count mismatch")
        if self.plan["suffixes_per_state"] != self.expected_suffixes_per_state:
            raise ValueError("active suffix plan K mismatch")
        for group, resolved in zip(self.plan["groups"], self.resolved, strict=True):
            trajectory = resolved.get("trajectory") or {}
            if trajectory.get("environment_version") != self.required_environment_version:
                raise ActiveSuffixInfrastructureError(
                    "resolved environment version differs from the required version"
                )
            if (
                trajectory.get("environment_manifest_sha256")
                != group["environment_manifest_sha256"]
            ):
                raise ActiveSuffixInfrastructureError(
                    "resolved environment manifest differs from the plan"
                )
        self.environment_manifest_sha256s = sorted(
            {group["environment_manifest_sha256"] for group in self.plan["groups"]}
        )
        if tool_schema_sha256(self.tool_schemas) != decoding["tool_schema_sha256"]:
            raise ActiveSuffixInfrastructureError("tool schema SHA256 mismatch")
        parser_stop_token_ids = getattr(parser, "stop_token_ids", None)
        if parser_stop_token_ids is None:
            raise ActiveSuffixInfrastructureError("qwen3_coder parser stop tokens are missing")
        normalized_parser_stops = _token_ids(
            list(parser_stop_token_ids),
            "parser stop_token_ids",
            allow_empty=True,
        )
        if normalized_parser_stops != decoding["stop_token_ids"]:
            raise ActiveSuffixInfrastructureError("qwen3_coder parser stop tokens changed")
        tokenizer_digest = getattr(encoder, "tokenizer_contract_sha256", None)
        if not _is_sha256(tokenizer_digest):
            raise ActiveSuffixInfrastructureError("encoder tokenizer contract is missing")
        if any(group["tokenizer_contract_sha256"] != tokenizer_digest for group in self.plan["groups"]):
            raise ActiveSuffixInfrastructureError("captured tokenizer contract mismatch")
        self.backend_sha256 = sampling_backend_contract_sha256(self.backend_contract)
        if self.backend_sha256 != decoding["sampling_backend_contract_sha256"]:
            raise ActiveSuffixInfrastructureError("plan sampling backend digest mismatch")
        self._bound_client = self._attest_backend()

    def _attest_backend(self):
        """Re-attest the actor and live server before each independent suffix."""
        self._validate_plan_fresh()
        return self.completion_client.attest_and_bind(
            plan_sha256=self.plan_sha256,
            expected_contract=self.backend_contract,
            expected_contract_sha256=self.backend_sha256,
            decoding_config=self.plan["decoding_config"],
            actor_checkpoint=self.actor_checkpoint,
            actor_checkpoint_sha256=self.plan["actor_checkpoint_sha256"],
        )

    def _validate_plan_fresh(self) -> None:
        if _sha256_json(self.plan) != self.plan_sha256:
            raise ActiveSuffixInfrastructureError("plan changed after runner initialization")
        if _sha256_json(self.resolved) != self.resolved_sha256:
            raise ActiveSuffixInfrastructureError(
                "resolved selection changed after runner initialization"
            )
        if self.plan.get("schema_version") != ACTIVE_BRANCH_PLAN_VERSION:
            raise ValueError("active branch plan version mismatch")
        if self.plan.get("strategy_version") != ACTIVE_BRANCH_STRATEGY_VERSION:
            raise ValueError("active branch strategy version mismatch")
        expected = build_active_branch_plan(
            self.resolved,
            actor_checkpoint_sha256=str(self.plan.get("actor_checkpoint_sha256") or ""),
            decoding_config=self.plan.get("decoding_config"),
            seed=self.plan.get("seed"),
            suffixes_per_state=self.plan.get("suffixes_per_state"),
        )
        if expected != self.plan:
            raise ActiveSuffixInfrastructureError(
                "plan does not exactly match the resolved selection"
            )
        safety = self.plan.get("safety") or {}
        required_safety = {
            "active_branch": True,
            "outcome_blind": True,
            "outcome_fields_read": [],
            "uses_hidden_goal": False,
            "prompt_token_hash_verified": True,
            "optimizer_enabled": False,
            "training_ready": False,
        }
        if safety != required_safety:
            raise ActiveSuffixInfrastructureError("active suffix plan safety contract mismatch")

    def _validate_suffix_request(self, group_index: int, suffix_index: int) -> None:
        # Rebuild immediately before every generation request.  This rechecks the
        # prompt capture, resolved branch, group uid, decoding hash, suffix uid,
        # and deterministic suffix seed rather than trusting mutable dictionaries.
        self._validate_plan_fresh()
        group = self.plan["groups"][group_index]
        suffix = group["suffixes"][suffix_index]
        capture = validate_actor_prompt_tokens(
            group["actor_prompt_tokens"],
            expected_sha256=group["actor_prompt_tokens"]["sha256"],
        )
        if token_ids_sha256(capture["tokens"]) != capture["sha256"]:
            raise ActiveSuffixInfrastructureError("captured prompt hash changed")
        expected_suffix_uid = _sha256_json(
            {"active_group_uid": group["active_group_uid"], "suffix_index": suffix_index}
        )
        if suffix["suffix_uid"] != expected_suffix_uid:
            raise ActiveSuffixInfrastructureError("suffix_uid changed before request")
        expected_seed = _suffix_seed(self.plan["seed"], group["active_group_uid"], suffix_index)
        if suffix["seed"] != expected_seed:
            raise ActiveSuffixInfrastructureError("derived suffix seed changed before request")
        decoding = validate_decoding_config(self.plan["decoding_config"])
        if _sha256_json(decoding) != self.plan["decoding_config_sha256"]:
            raise ActiveSuffixInfrastructureError("decoding config changed before request")

    def _replay_branch(self, env, group, resolved):
        capture_env = _ReplayCaptureEnv(env)
        report = verify_replay_trajectory(
            capture_env,
            resolved["trajectory"],
            environment_manifest_sha256=group["environment_manifest_sha256"],
            required_environment_version=self.required_environment_version,
            required_max_steps=self.plan["decoding_config"]["max_steps"],
            prefix_action_count=group["prefix_action_count"],
        )
        if report.get("verified") is not True:
            reason = str(report.get("reason") or report.get("stage") or "unknown")
            raise ActiveSuffixInfrastructureError(f"prefix replay failed:{reason}")
        if report.get("recomputed_replay_state_id") != group["replay_state_id"]:
            raise ActiveSuffixInfrastructureError("replayed state id differs from plan")
        if not isinstance(capture_env.last_result, Mapping):
            raise ActiveSuffixInfrastructureError("replay produced no current observation")
        raw_observation = _public_observation(capture_env.last_result)
        event = resolved["event"]
        if observation_sha256(raw_observation) != event.get("raw_observation_sha256"):
            raise ActiveSuffixInfrastructureError("replayed raw observation hash mismatch")
        ledger = resolved["trajectory"].get("replay_ledger") or []
        prefix = ledger[: group["prefix_action_count"]]
        config = self.plan["decoding_config"]
        if prefix:
            visible, projection = project_observation(
                tool_name=prefix[-1]["tool"],
                observation=raw_observation,
                parameters=prefix[-1]["parameters"],
                count_tokens=self.encoder.count_tokens,
                token_budget=config["observation_token_budget"],
                detail_token_budget=config["observation_detail_token_budget"],
                generic_token_budget=config["observation_generic_token_budget"],
                search_top_k=config["observation_search_top_k"],
            )
            projection_meta = projection.to_dict()
        else:
            # Session.start exposes the reset observation directly; projection
            # begins only after an environment tool returns a pending raw page.
            visible = raw_observation
            projection_meta = None
        if observation_sha256(visible) != event.get("observation_sha256"):
            raise ActiveSuffixInfrastructureError("projected branch observation hash mismatch")
        return raw_observation, visible, projection_meta, report

    async def _parse(self, token_ids: Sequence[int]) -> list[dict[str, object]]:
        parsed = await _maybe_await(self.parser.parse(list(token_ids), self.tool_schemas))
        return _normalize_tool_calls(parsed)

    def _encode_tool_observation(
        self,
        text: str,
        config: Mapping[str, object],
    ) -> list[int]:
        token_ids = _token_ids(
            list(self.encoder.encode_tool_observation(text)),
            "encoded tool observation",
            allow_empty=True,
        )
        prompt_length = int(config["prompt_length"])
        return token_ids[-prompt_length:]

    def _terminalize(self, state: dict, result: Mapping[str, object]) -> None:
        reward = result.get("reward")
        if not isinstance(reward, (int, float)) or isinstance(reward, bool) or not math.isfinite(float(reward)):
            raise ActiveSuffixInfrastructureError("terminal reward is not finite")
        if result.get("done") is not True or result.get("over") is not True:
            raise ActiveSuffixInfrastructureError("terminal result is not complete")
        public_detail = validate_reward(result.get("reward_detail"))
        if float(public_detail["terminal_utility"]) != float(reward):
            raise ActiveSuffixInfrastructureError("terminal utility differs from reward")
        state["done"] = True
        state["terminate"] = True
        state["terminal_result"] = {"done": True, "over": True}
        state["final_reward"] = float(reward)
        state["reward_version"] = public_detail["reward_version"]
        state["reward_type"] = public_detail["reward_type"]
        state["reward_valid"] = public_detail["reward_valid"]
        state["reward_unverifiable"] = not public_detail["reward_valid"]
        state["reward_detail"] = public_detail
        state["termination_reason"] = public_detail["termination_reason"]

    @staticmethod
    def _model_failure(state: dict, reason: str) -> None:
        state["terminate"] = True
        state["termination_reason"] = str(reason)
        state["error"] = str(reason)

    async def _collect_one(self, group_index: int, suffix_index: int) -> dict[str, object]:
        self._validate_suffix_request(group_index, suffix_index)
        self._bound_client = self._attest_backend()
        group = self.plan["groups"][group_index]
        suffix = group["suffixes"][suffix_index]
        resolved = self.resolved[group_index]
        env = self.env_factory()
        release_error = None
        try:
            env_timeout = getattr(env, "timeout", None)
            if (
                not isinstance(env_timeout, int)
                or isinstance(env_timeout, bool)
                or env_timeout != self.environment_timeout_seconds
            ):
                raise ActiveSuffixInfrastructureError(
                    "environment timeout differs from the runner contract"
                )
            raw_observation, visible_observation, projection_meta, replay_report = (
                self._replay_branch(env, group, resolved)
            )
            branch_visible_observation = visible_observation
            config = self.plan["decoding_config"]
            prefix_ledger = list(resolved["trajectory"]["replay_ledger"])[
                : group["prefix_action_count"]
            ]
            state = make_runtime_state(group["task_id"], config["max_steps"])
            state["replay_ledger"] = deepcopy(prefix_ledger)
            state["latest_observation_raw"] = raw_observation
            state["latest_observation"] = visible_observation
            state["latest_observation_truncated"] = bool(
                projection_meta and projection_meta["truncated"]
            )
            state["environment_manifest_sha256"] = group["environment_manifest_sha256"]
            state["public_query_sha256"] = resolved["trajectory"]["public_query_sha256"]
            state["initial_public_observation_sha256"] = resolved["trajectory"][
                "initial_public_observation_sha256"
            ]
            state["replay_observation_v2_complete"] = True
            if projection_meta is not None:
                record_observation_projection(state, projection_meta)
            restored = _restore_prefix_runtime_state(state, resolved, group, config)

            prompt_ids = list(group["actor_prompt_tokens"]["tokens"])
            response_ids: list[int] = []
            response_mask: list[int] = []
            old_logprobs: list[float] = []
            assistant_spans: list[list[int]] = []
            request_prompt_sha256: list[str] = []
            request_seeds: list[int] = []
            first_action = None
            first_action_sha = None
            first_action_span = None
            first_action_credit_eligible = False
            empty_completion_without_new_tokens = False
            response_truncated = False
            harness_limit_reason = None
            response_tokens_before = restored["response_tokens_before"]
            suffix_response_budget = config["response_length"] - response_tokens_before
            assistant_turns = restored["assistant_turns"]
            user_turns = restored["user_turns"]

            while not state["terminate"]:
                if len(state["steps"]) >= config["max_steps"]:
                    self._model_failure(state, "max_steps")
                    break
                if (
                    len(prompt_ids)
                    > config["context_window"]
                    - config["context_generation_reserve"]
                    - config["context_safety_margin"]
                ):
                    self._model_failure(state, "context_hard_limit_exceeded")
                    break
                self._validate_suffix_request(group_index, suffix_index)
                request_prompt_sha256.append(token_ids_sha256(prompt_ids))
                request_seed = _request_seed(suffix, len(request_seeds))
                if request_seed in request_seeds:
                    raise ActiveSuffixInfrastructureError(
                        "derived request seed collision",
                        code="request_seed_schedule_invalid",
                    )
                request_seeds.append(request_seed)
                current_turn_id = assistant_turns
                current_turn_record = {
                    "turn_id": current_turn_id,
                    "actor_prompt_sha256": token_ids_sha256(prompt_ids),
                    "tokenizer_contract_sha256": group["tokenizer_contract_sha256"],
                }
                state["assistant_turn_records"].append(current_turn_record)
                state["current_assistant_turn_id"] = current_turn_id
                state["next_assistant_turn_id"] = current_turn_id + 1
                completion = self._bound_client.complete(prompt_ids, seed=request_seed)
                if completion.get("prompt_token_ids") != prompt_ids:
                    raise ActiveSuffixInfrastructureError("completion prompt echo changed in runner")
                completion_ids = _token_ids(
                    completion.get("token_ids"),
                    "completion token_ids",
                    allow_empty=True,
                )
                if not completion_ids:
                    if response_tokens_before + len(response_ids) == 0:
                        raise ActiveSuffixInfrastructureError(
                            "trajectory alignment invalid: full response has no tokens",
                            code="trajectory_alignment_invalid",
                        )
                    empty_completion_without_new_tokens = not response_ids
                    assistant_turns += 1
                    current_turn_record.update(
                        {
                            "kind": "assistant_termination",
                            "credit_eligible": False,
                            "generation_termination_reason": "immediate_eos",
                        }
                    )
                    self._model_failure(
                        state,
                        "assistant_finished_without_environment_done",
                    )
                    break
                span = append_assistant_turn(
                    response_ids,
                    response_mask,
                    old_logprobs,
                    completion,
                )
                assistant_spans.append([span[0], span[1]])
                prompt_ids.extend(completion_ids)
                assistant_turns += 1
                if (
                    response_tokens_before + len(response_mask)
                    >= config["response_length"]
                ):
                    harness_limit_reason = "response_length"
                elif assistant_turns >= config["max_assistant_turns"]:
                    harness_limit_reason = "max_assistant_turns"
                elif user_turns >= config["max_user_turns"]:
                    harness_limit_reason = "max_user_turns"
                if harness_limit_reason is not None:
                    current_turn_record.update(
                        {
                            "kind": "assistant_termination",
                            "credit_eligible": False,
                            "generation_termination_reason": harness_limit_reason,
                        }
                    )
                    action = canonical_replay_action("harness_termination", {})
                    if first_action is None:
                        first_action = action
                        first_action_sha = replay_action_sha256(
                            action["tool"], action["parameters"]
                        )
                        first_action_span = [
                            span[0],
                            min(span[1], suffix_response_budget),
                        ]
                        first_action_credit_eligible = False
                    self._model_failure(
                        state,
                        "assistant_finished_without_environment_done",
                    )
                    break
                calls = await self._parse(completion_ids)
                if not calls:
                    current_turn_record.update(
                        {
                            "kind": "assistant_termination",
                            "credit_eligible": True,
                            "generation_termination_reason": "assistant_final",
                        }
                    )
                    action = canonical_replay_action("assistant_final", {})
                    if first_action is None:
                        first_action = action
                        first_action_sha = replay_action_sha256("assistant_final", {})
                        first_action_span = [span[0], span[1]]
                        first_action_credit_eligible = True
                    record_non_environment_decision(
                        state,
                        "assistant_final",
                        {},
                        state["latest_observation"],
                    )
                    self._model_failure(state, "assistant_finished_without_environment_done")
                    break
                if len(calls) > 1:
                    names = [str(call["name"]) for call in calls]
                    current_turn_record.update(
                        {
                            "kind": "tool_call",
                            "tool_names": names,
                            "tool_call_count": len(names),
                            "credit_eligible": False,
                        }
                    )
                    action = canonical_replay_action("parallel_tool_calls", {"tools": names})
                    if first_action is None:
                        first_action = action
                        first_action_sha = replay_action_sha256(
                            action["tool"], action["parameters"]
                        )
                        first_action_span = [span[0], span[1]]
                        first_action_credit_eligible = False
                    self._model_failure(state, "parallel_tool_calls")
                    break
                call = calls[0]
                name = str(call["name"])
                parameters = call["arguments"]
                current_turn_record.update(
                    {
                        "kind": "tool_call",
                        "tool_names": [name],
                        "tool_call_count": 1,
                        "credit_eligible": True,
                    }
                )
                if name not in self.tool_names:
                    unknown_parameters = parameters if isinstance(parameters, Mapping) else {}
                    action = canonical_replay_action(name, unknown_parameters)
                    if first_action is None:
                        first_action = action
                        first_action_sha = replay_action_sha256(
                            action["tool"], action["parameters"]
                        )
                        first_action_span = [span[0], span[1]]
                        first_action_credit_eligible = True
                    tool_text = _bounded_tool_text(
                        f"Unknown function '{name}'. Available tools: "
                        f"{self.tool_name_order}",
                        config,
                        projected_observation=False,
                    )
                    observation_ids = self._encode_tool_observation(tool_text, config)
                    if (
                        response_tokens_before
                        + len(response_ids)
                        + len(observation_ids)
                        >= config["response_length"]
                    ):
                        harness_limit_reason = "response_length"
                        self._model_failure(
                            state,
                            "assistant_finished_without_environment_done",
                        )
                        break
                    append_tool_observation(
                        response_ids,
                        response_mask,
                        old_logprobs,
                        observation_ids,
                    )
                    prompt_ids.extend(observation_ids)
                    user_turns += 1
                    continue
                if parameters is None:
                    action = canonical_replay_action("malformed_tool_arguments", {"tool": name})
                    if first_action is None:
                        first_action = action
                        first_action_sha = replay_action_sha256(
                            action["tool"], action["parameters"]
                        )
                        first_action_span = [span[0], span[1]]
                        first_action_credit_eligible = True
                    tool_text = _bounded_tool_text(
                        f"Invalid JSON in arguments for '{name}': "
                        f"{call['arguments_error']}",
                        config,
                        projected_observation=False,
                    )
                    observation_ids = self._encode_tool_observation(tool_text, config)
                    if (
                        response_tokens_before
                        + len(response_ids)
                        + len(observation_ids)
                        >= config["response_length"]
                    ):
                        harness_limit_reason = "response_length"
                        self._model_failure(
                            state,
                            "assistant_finished_without_environment_done",
                        )
                        break
                    append_tool_observation(
                        response_ids,
                        response_mask,
                        old_logprobs,
                        observation_ids,
                    )
                    prompt_ids.extend(observation_ids)
                    user_turns += 1
                    continue
                parameters = dict(parameters)
                action = canonical_replay_action(name, parameters)
                if first_action is None:
                    first_action = action
                    first_action_sha = replay_action_sha256(name, parameters)
                    first_action_span = [span[0], span[1]]
                    first_action_credit_eligible = True

                tool_text = None
                projected_tool_text = False
                if name == "think":
                    record_non_environment_decision(
                        state,
                        "think",
                        parameters,
                        state["latest_observation"],
                    )
                    state["steps"].append(
                        {
                            "index": len(state["steps"]),
                            "tool": name,
                            "parameters": parameters,
                            "done": False,
                            "reward": 0.0,
                        }
                    )
                    if len(state["steps"]) >= config["max_steps"]:
                        self._model_failure(state, "max_steps")
                        tool_text = "Error: maximum executed tool steps reached."
                    else:
                        tool_text = "Reasoning recorded. Continue with one environment tool call."
                else:
                    action_event = record_action_attempt(
                        state,
                        name,
                        parameters,
                        state["latest_observation"],
                    )
                    guard_reason = action_reject_reason(
                        name,
                        parameters,
                        state["latest_observation"],
                    )
                    if guard_reason is not None:
                        record_action_outcome(
                            state,
                            action_event,
                            accepted=False,
                            guard_reason=guard_reason,
                        )
                        state["guard_rejection_count"] += 1
                        state["consecutive_guard_rejections"] += 1
                        if state["consecutive_guard_rejections"] >= 3:
                            self._model_failure(state, "too_many_guard_rejections")
                            tool_text = (
                                "Error: maximum consecutive action guard rejections reached."
                            )
                        else:
                            tool_text = (
                                "Error: action guard rejected this call "
                                f"({guard_reason}); read the latest observation."
                            )
                    else:
                        try:
                            env_action = tool_call_to_action(name, parameters)
                            result = env.step(env_action)
                        except (KeyError, TypeError, ValueError) as exc:
                            raise ActiveSuffixInfrastructureError(
                                f"tool action conversion failed:{exc.__class__.__name__}"
                            ) from exc
                        if not isinstance(result, Mapping):
                            raise ActiveSuffixInfrastructureError(
                                "environment step result is not an object"
                            )
                        done = result.get("done") is True
                        if done:
                            after_observation = None
                        else:
                            after_observation = _public_observation(result)
                        state["steps"].append(
                            {
                                "index": len(state["steps"]),
                                "tool": name,
                                "parameters": parameters,
                                "done": done,
                                "reward": float(result.get("reward", 0.0)),
                            }
                        )
                        record_action_outcome(state, action_event, accepted=True)
                        record_replay_transition(
                            state,
                            action_event,
                            parameters=parameters,
                            after_observation=after_observation,
                            done=done,
                        )
                        state["consecutive_guard_rejections"] = 0
                        if done:
                            self._terminalize(state, result)
                            tool_text = "Environment terminated."
                        else:
                            visible_observation, projection = project_observation(
                                tool_name=name,
                                observation=after_observation,
                                parameters=parameters,
                                count_tokens=self.encoder.count_tokens,
                                token_budget=config["observation_token_budget"],
                                detail_token_budget=config[
                                    "observation_detail_token_budget"
                                ],
                                generic_token_budget=config[
                                    "observation_generic_token_budget"
                                ],
                                search_top_k=config["observation_search_top_k"],
                            )
                            state["latest_observation_raw"] = after_observation
                            state["latest_observation"] = visible_observation
                            record_observation_projection(state, projection.to_dict())
                            tool_text = visible_observation
                            projected_tool_text = True
                            if len(state["steps"]) >= config["max_steps"]:
                                self._model_failure(state, "max_steps")
                                # ShoppingToolAgentLoop projects the pending raw page
                                # after the tool sets max_steps, replacing its error text.

                tool_text = _bounded_tool_text(
                    tool_text,
                    config,
                    projected_observation=projected_tool_text,
                )
                observation_ids = self._encode_tool_observation(tool_text, config)
                if (
                    response_tokens_before
                    + len(response_ids)
                    + len(observation_ids)
                    >= config["response_length"]
                ):
                    harness_limit_reason = "response_length"
                    if not state["done"] and not state["terminate"]:
                        self._model_failure(
                            state,
                            "assistant_finished_without_environment_done",
                        )
                    break
                append_tool_observation(
                    response_ids,
                    response_mask,
                    old_logprobs,
                    observation_ids,
                )
                prompt_ids.extend(observation_ids)
                user_turns += 1

            if len(response_ids) > suffix_response_budget:
                response_truncated = True
                if (
                    first_action_span is not None
                    and first_action_span[1] > suffix_response_budget
                ):
                    first_action_credit_eligible = False
                response_ids = response_ids[:suffix_response_budget]
                response_mask = response_mask[:suffix_response_budget]
                old_logprobs = old_logprobs[:suffix_response_budget]
                assistant_spans = [
                    [start, min(end, len(response_ids))]
                    for start, end in assistant_spans
                    if start < len(response_ids)
                ]
                if first_action_span is not None:
                    if first_action_span[0] >= len(response_ids):
                        first_action_span = None
                        first_action_credit_eligible = False
                    else:
                        first_action_span[1] = min(
                            first_action_span[1], len(response_ids)
                        )
                        if first_action_span[0] >= first_action_span[1]:
                            first_action_credit_eligible = False
            if not len(response_ids) == len(response_mask) == len(old_logprobs):
                raise ActiveSuffixInfrastructureError("final suffix tensors are misaligned")
            breakdown = reward_breakdown(state, self.policy_reward)
            valid_for_learning = bool(breakdown["valid_for_learning"])
            invalid_reason = breakdown["invalid_reason"]
            model_failure = bool(breakdown["model_failure"])
            if empty_completion_without_new_tokens:
                valid_for_learning = False
                invalid_reason = "empty_suffix_no_trainable_tokens"
                model_failure = True
            if valid_for_learning and (
                not response_ids or first_action_sha is None or not any(response_mask)
            ):
                raise ActiveSuffixInfrastructureError(
                    "learning-valid suffix lacks an Assistant action span"
                )
            return {
                "schema_version": ACTIVE_SUFFIX_RESULT_VERSION,
                "active_group_uid": group["active_group_uid"],
                "parent_branch_uid": group["parent_branch_uid"],
                "replay_state_id": group["replay_state_id"],
                "task_id": group["task_id"],
                "suffix_index": suffix_index,
                "suffix_uid": suffix["suffix_uid"],
                "seed": suffix["seed"],
                "actor_checkpoint_sha256": self.plan["actor_checkpoint_sha256"],
                "decoding_config_sha256": self.plan["decoding_config_sha256"],
                "sampling_backend_contract_sha256": self.plan["decoding_config"][
                    "sampling_backend_contract_sha256"
                ],
                "prompt_token_ids": list(group["actor_prompt_tokens"]["tokens"]),
                "prompt_token_count": group["actor_prompt_tokens"]["count"],
                "prompt_token_sha256": group["actor_prompt_tokens"]["sha256"],
                "request_prompt_sha256": request_prompt_sha256,
                "request_seeds": request_seeds,
                "response_ids": response_ids,
                "response_mask": response_mask,
                "old_logprobs": old_logprobs,
                "assistant_spans": assistant_spans,
                "first_action": first_action,
                "first_action_sha256": first_action_sha,
                "first_action_span": first_action_span,
                "first_action_credit_eligible": first_action_credit_eligible,
                "termination_reason": state.get("termination_reason"),
                "harness_limit_reason": harness_limit_reason,
                "strict": bool(breakdown["strict"]),
                "valid_for_learning": valid_for_learning,
                "invalid_reason": invalid_reason,
                "infrastructure_error_code": None,
                "infrastructure_error_class": None,
                "model_failure": model_failure,
                "infrastructure_invalid": bool(breakdown["infrastructure_invalid"]),
                "policy_reward": float(breakdown["total"]),
                "terminal_utility": float(breakdown["terminal_utility"]),
                "reward_type": state.get("reward_type"),
                "steps": len(state["steps"]),
                "guard_rejections": int(state["guard_rejection_count"]),
                "repeat_actions": int(state["repeat_action_count"]),
                "assistant_turns": assistant_turns,
                "user_turns": user_turns,
                "response_tokens_before": response_tokens_before,
                "response_tokens_after": response_tokens_before + len(response_ids),
                "response_truncated": response_truncated,
                "replay": {
                    "verified": True,
                    "verified_transitions": replay_report["verified_transitions"],
                    "raw_observation_sha256": observation_sha256(raw_observation),
                    "projected_observation_sha256": observation_sha256(
                        branch_visible_observation
                    ),
                },
                "optimizer_enabled": False,
                "uses_hidden_goal": False,
            }
        finally:
            try:
                env.release()
            except Exception as exc:  # noqa: BLE001 - release uncertainty invalidates the suffix.
                release_error = exc
            if release_error is not None:
                raise ActiveSuffixInfrastructureError(
                    f"environment release failed:{release_error.__class__.__name__}"
                ) from release_error

    def _invalid_record(self, group_index: int, suffix_index: int, exc: Exception):
        group = self.plan["groups"][group_index]
        suffix = group["suffixes"][suffix_index]
        if isinstance(exc, ActiveSuffixInfrastructureError):
            invalid_reason = "active_suffix_infrastructure_error"
        else:
            invalid_reason = "external_failure"
        return {
            "schema_version": ACTIVE_SUFFIX_RESULT_VERSION,
            "active_group_uid": group["active_group_uid"],
            "parent_branch_uid": group["parent_branch_uid"],
            "replay_state_id": group["replay_state_id"],
            "task_id": group["task_id"],
            "suffix_index": suffix_index,
            "suffix_uid": suffix["suffix_uid"],
            "seed": suffix["seed"],
            "actor_checkpoint_sha256": self.plan["actor_checkpoint_sha256"],
            "decoding_config_sha256": self.plan["decoding_config_sha256"],
            "sampling_backend_contract_sha256": self.plan["decoding_config"][
                "sampling_backend_contract_sha256"
            ],
            "prompt_token_ids": list(group["actor_prompt_tokens"]["tokens"]),
            "prompt_token_count": group["actor_prompt_tokens"]["count"],
            "prompt_token_sha256": group["actor_prompt_tokens"]["sha256"],
            "request_prompt_sha256": [],
            "request_seeds": [],
            "response_ids": [],
            "response_mask": [],
            "old_logprobs": [],
            "assistant_spans": [],
            "first_action": None,
            "first_action_sha256": None,
            "first_action_span": None,
            "first_action_credit_eligible": False,
            "termination_reason": "active_suffix_infrastructure_invalid",
            "harness_limit_reason": None,
            "strict": False,
            "valid_for_learning": False,
            "invalid_reason": invalid_reason,
            "infrastructure_error_code": (
                exc.code
                if isinstance(exc, ActiveSuffixInfrastructureError)
                else "external_failure"
            ),
            "infrastructure_error_class": exc.__class__.__name__[:128],
            "model_failure": False,
            "infrastructure_invalid": True,
            "policy_reward": 0.0,
            "terminal_utility": 0.0,
            "reward_type": None,
            "steps": 0,
            "guard_rejections": 0,
            "repeat_actions": 0,
            "assistant_turns": 0,
            "user_turns": 0,
            "response_tokens_before": 0,
            "response_tokens_after": 0,
            "response_truncated": False,
            "replay": {"verified": False},
            "optimizer_enabled": False,
            "uses_hidden_goal": False,
        }

    async def collect(self) -> dict[str, object]:
        """Collect all planned suffixes serially with one fresh lease per suffix."""
        records = []
        group_summaries = []
        for group_index, group in enumerate(self.plan["groups"]):
            group_records = []
            for suffix_index in range(self.plan["suffixes_per_state"]):
                try:
                    record = await self._collect_one(group_index, suffix_index)
                except Exception as exc:  # noqa: BLE001 - persist bounded invalid diagnostics.
                    record = self._invalid_record(group_index, suffix_index, exc)
                group_records.append(record)
                records.append(record)
            group_summaries.append(
                summarize_active_suffix_group(
                    group_records,
                    expected_group_uid=group["active_group_uid"],
                    expected_k=self.expected_suffixes_per_state,
                )
            )
        if len(group_summaries) != self.expected_groups:
            raise ActiveSuffixInfrastructureError("active suffix summary group count mismatch")
        if len(records) != self.expected_groups * self.expected_suffixes_per_state:
            raise ActiveSuffixInfrastructureError("active suffix summary record count mismatch")
        expected_suffixes = self.expected_groups * self.expected_suffixes_per_state
        infrastructure_invalid = sum(
            int(item["infrastructure_invalid"]) for item in records
        )
        error_code_counts = Counter(
            str(item["infrastructure_error_code"])
            for item in records
            if item["infrastructure_invalid"]
        )
        cardinality_complete = len(records) == expected_suffixes and all(
            item["suffixes"] == self.expected_suffixes_per_state
            for item in group_summaries
        )
        mechanical_smoke_passed = cardinality_complete and infrastructure_invalid == 0
        return {
            "schema_version": ACTIVE_SUFFIX_COLLECTION_VERSION,
            "plan_sha256": self.plan_sha256,
            "records": records,
            "groups": group_summaries,
            "provenance": {
                "required_environment_version": self.required_environment_version,
                "environment_manifest_sha256s": self.environment_manifest_sha256s,
                "policy_reward_sha256": self.policy_reward_sha256,
                "vllm_timeout_seconds": self.vllm_timeout_seconds,
                "environment_timeout_seconds": self.environment_timeout_seconds,
            },
            "aggregate": {
                "groups": len(group_summaries),
                "suffixes_per_state": self.expected_suffixes_per_state,
                "suffixes": len(records),
                "valid_suffixes": sum(item["valid_suffixes"] for item in group_summaries),
                "infrastructure_invalid_suffixes": infrastructure_invalid,
                "infrastructure_error_code_counts": dict(sorted(error_code_counts.items())),
                "model_failure_suffixes": sum(
                    int(item["model_failure"]) for item in records
                ),
                "eligible_groups": sum(
                    int(item["eligible_for_action_credit"]) for item in group_summaries
                ),
                "cardinality_complete": cardinality_complete,
                "mechanical_smoke_passed": mechanical_smoke_passed,
            },
            "safety": {
                "active_branch": True,
                "optimizer_enabled": False,
                "training_ready": False,
                "uses_hidden_goal": False,
                "compaction_enabled": False,
                "server_attested": True,
                "generation_config_api_attested": False,
                "server_launch_command_evidence_required": True,
            },
        }
