"""Fail-closed replay verification for public ShopSimulator trajectory prefixes."""

from __future__ import annotations

from collections.abc import Mapping

from shopping_grpo.environment.actions import action_reject_reason
from shopping_grpo.environment.observation import render_structured_observation
from shopping_grpo.environment.tools import tool_call_to_action
from shopping_grpo.training.grpo.pivotal_states import (
    DEFAULT_REPLAY_MAX_STEPS,
    canonical_replay_action,
    observation_sha256,
    replay_state_id,
    validate_replay_ledger,
)

REPLAY_VERIFICATION_VERSION = "shopping-public-replay-verification-v1"
DEFAULT_REQUIRED_ENVIRONMENT_VERSION = "shopsimulator-environment-v2.1"


def _public_observation(result: Mapping[str, object]) -> str:
    observation_state = result.get("observation_state")
    if not isinstance(observation_state, Mapping):
        raise TypeError("observation_state must be an object")
    return render_structured_observation(dict(observation_state))


def _public_query(result: Mapping[str, object]) -> str:
    return str(result.get("instruction", result.get("observation", "")))


def _result(
    *,
    task_id: int,
    verified: bool,
    stage: str,
    reason: str | None,
    verified_transitions: int,
    transition_index: int | None = None,
    expected: object = None,
    actual: object = None,
) -> dict[str, object]:
    result = {
        "schema_version": REPLAY_VERIFICATION_VERSION,
        "task_id": int(task_id),
        "verified": bool(verified),
        "stage": str(stage),
        "reason": reason,
        "verified_transitions": int(verified_transitions),
        "first_divergence_index": transition_index,
    }
    if expected is not None:
        result["expected"] = expected
    if actual is not None:
        result["actual"] = actual
    return result


def _preflight_replay_contract(
    trajectory: Mapping[str, object],
    *,
    environment_manifest_sha256: str,
    required_environment_version: str,
    required_max_steps: int,
    prefix_action_count: int | None,
) -> tuple[int, list[Mapping[str, object]] | None, dict[str, object] | None]:
    """Validate every pure replay invariant before leasing or resetting a slot."""
    raw_task_id = trajectory.get("task_id")
    if not isinstance(raw_task_id, int) or isinstance(raw_task_id, bool):
        return 0, None, _result(
            task_id=0,
            verified=False,
            stage="contract",
            reason="invalid_task_id",
            verified_transitions=0,
        )
    task_id = raw_task_id
    expected_manifest = str(trajectory.get("environment_manifest_sha256") or "")
    if expected_manifest != str(environment_manifest_sha256):
        return task_id, None, _result(
            task_id=task_id,
            verified=False,
            stage="manifest",
            reason="environment_manifest_mismatch",
            verified_transitions=0,
            expected=expected_manifest,
            actual=str(environment_manifest_sha256),
        )
    if not isinstance(required_environment_version, str) or not required_environment_version:
        return task_id, None, _result(
            task_id=task_id,
            verified=False,
            stage="contract",
            reason="invalid_required_environment_version",
            verified_transitions=0,
        )
    trajectory_version = trajectory.get("environment_version")
    if trajectory_version != required_environment_version:
        return task_id, None, _result(
            task_id=task_id,
            verified=False,
            stage="contract",
            reason="trajectory_environment_version_mismatch",
            verified_transitions=0,
            expected=required_environment_version,
            actual=str(trajectory_version or ""),
        )
    ledger, ledger_error = validate_replay_ledger(
        trajectory,
        max_steps=required_max_steps,
    )
    if ledger_error is not None or ledger is None:
        return task_id, None, _result(
            task_id=task_id,
            verified=False,
            stage="contract",
            reason=ledger_error,
            verified_transitions=0,
        )
    if prefix_action_count is not None:
        prefix_count = (
            prefix_action_count
            if isinstance(prefix_action_count, int)
            and not isinstance(prefix_action_count, bool)
            else -1
        )
        if not 0 <= prefix_count <= len(ledger):
            return task_id, None, _result(
                task_id=task_id,
                verified=False,
                stage="contract",
                reason="prefix_action_count_out_of_range",
                verified_transitions=0,
            )
        ledger = ledger[:prefix_count]
        if any(bool(transition.get("done")) for transition in ledger):
            return task_id, None, _result(
                task_id=task_id,
                verified=False,
                stage="contract",
                reason="branch_prefix_contains_terminal_action",
                verified_transitions=0,
            )
    return task_id, ledger, None


def verify_replay_trajectory(
    env: object,
    trajectory: Mapping[str, object],
    *,
    environment_manifest_sha256: str,
    required_environment_version: str = DEFAULT_REQUIRED_ENVIRONMENT_VERSION,
    required_max_steps: int = DEFAULT_REPLAY_MAX_STEPS,
    prefix_action_count: int | None = None,
) -> dict[str, object]:
    """Reset and replay exact accepted actions while comparing every public-state hash.

    The returned report intentionally contains only hashes, booleans, indices, and
    error classes. Raw ShopSimulator responses may contain hidden goals and are
    never copied into the report.
    """
    task_id, ledger, preflight_error = _preflight_replay_contract(
        trajectory,
        environment_manifest_sha256=environment_manifest_sha256,
        required_environment_version=required_environment_version,
        required_max_steps=required_max_steps,
        prefix_action_count=prefix_action_count,
    )
    if preflight_error is not None:
        return preflight_error
    if ledger is None:  # pragma: no cover - the preflight return contract is exhaustive.
        raise AssertionError("replay preflight returned no ledger without an error")
    try:
        initial = env.reset(task_id)
    except Exception as exc:  # noqa: BLE001 - report only the class, never server text.
        return _result(
            task_id=task_id,
            verified=False,
            stage="reset",
            reason=f"reset_error:{exc.__class__.__name__}",
            verified_transitions=0,
        )
    if not isinstance(initial, Mapping):
        return _result(
            task_id=task_id,
            verified=False,
            stage="reset",
            reason="reset_result_not_object",
            verified_transitions=0,
        )
    actual_manifest_sha256 = initial.get("environment_manifest_sha256")
    if actual_manifest_sha256 != environment_manifest_sha256:
        return _result(
            task_id=task_id,
            verified=False,
            stage="reset",
            reason="server_environment_manifest_mismatch",
            verified_transitions=0,
            expected=environment_manifest_sha256,
            actual=str(actual_manifest_sha256 or ""),
        )
    actual_version = initial.get("environment_version")
    if actual_version != required_environment_version:
        return _result(
            task_id=task_id,
            verified=False,
            stage="reset",
            reason="environment_version_mismatch",
            verified_transitions=0,
            expected=required_environment_version,
            actual=str(actual_version or ""),
        )
    query_hash = observation_sha256(_public_query(initial))
    if query_hash != trajectory.get("public_query_sha256"):
        return _result(
            task_id=task_id,
            verified=False,
            stage="reset",
            reason="public_query_hash_mismatch",
            verified_transitions=0,
            expected=trajectory.get("public_query_sha256"),
            actual=query_hash,
        )
    try:
        current_observation = _public_observation(initial)
    except (TypeError, ValueError) as exc:
        return _result(
            task_id=task_id,
            verified=False,
            stage="reset",
            reason=f"public_observation_error:{exc.__class__.__name__}",
            verified_transitions=0,
        )
    current_hash = observation_sha256(current_observation)
    expected_initial_hash = trajectory.get("initial_public_observation_sha256")
    if current_hash != expected_initial_hash:
        return _result(
            task_id=task_id,
            verified=False,
            stage="reset",
            reason="initial_public_observation_hash_mismatch",
            verified_transitions=0,
            expected=expected_initial_hash,
            actual=current_hash,
        )

    for index, transition in enumerate(ledger):
        if current_hash != transition["before_public_observation_sha256"]:
            return _result(
                task_id=task_id,
                verified=False,
                stage="transition",
                reason="before_public_observation_hash_mismatch",
                verified_transitions=index,
                transition_index=index,
                expected=transition["before_public_observation_sha256"],
                actual=current_hash,
            )
        try:
            guard_reason = action_reject_reason(
                str(transition["tool"]),
                dict(transition["parameters"]),
                current_observation,
            )
            if guard_reason is not None:
                return _result(
                    task_id=task_id,
                    verified=False,
                    stage="transition",
                    reason=f"guard_rejection:{guard_reason}",
                    verified_transitions=index,
                    transition_index=index,
                )
            action = tool_call_to_action(
                str(transition["tool"]),
                dict(transition["parameters"]),
            )
            step = env.step(action)
        except Exception as exc:  # noqa: BLE001 - fail closed on uncertain side effects.
            return _result(
                task_id=task_id,
                verified=False,
                stage="transition",
                reason=f"step_error:{exc.__class__.__name__}",
                verified_transitions=index,
                transition_index=index,
            )
        if not isinstance(step, Mapping):
            return _result(
                task_id=task_id,
                verified=False,
                stage="transition",
                reason="step_result_not_object",
                verified_transitions=index,
                transition_index=index,
            )
        actual_done = bool(step.get("done", False))
        expected_done = bool(transition["done"])
        if actual_done != expected_done:
            return _result(
                task_id=task_id,
                verified=False,
                stage="transition",
                reason="done_mismatch",
                verified_transitions=index,
                transition_index=index,
                expected=expected_done,
                actual=actual_done,
            )
        if actual_done:
            current_hash = ""
        else:
            try:
                current_observation = _public_observation(step)
                current_hash = observation_sha256(current_observation)
            except (TypeError, ValueError) as exc:
                return _result(
                    task_id=task_id,
                    verified=False,
                    stage="transition",
                    reason=f"public_observation_error:{exc.__class__.__name__}",
                    verified_transitions=index,
                    transition_index=index,
                )
        expected_after = transition["after_public_observation_sha256"]
        actual_after = None if actual_done else current_hash
        if actual_after != expected_after:
            return _result(
                task_id=task_id,
                verified=False,
                stage="transition",
                reason="after_public_observation_hash_mismatch",
                verified_transitions=index,
                transition_index=index,
                expected=expected_after,
                actual=actual_after,
            )
    result = _result(
        task_id=task_id,
        verified=True,
        stage="complete",
        reason=None,
        verified_transitions=len(ledger),
    )
    if current_hash:
        accepted_actions = [
            canonical_replay_action(item["tool"], item["parameters"]) for item in ledger
        ]
        result["final_public_observation_sha256"] = current_hash
        result["recomputed_replay_state_id"] = replay_state_id(
            task_id,
            accepted_actions,
            current_hash,
            observation_kind="raw_public_observation",
            environment_manifest_sha256=str(environment_manifest_sha256),
            public_query_sha256=str(trajectory["public_query_sha256"]),
        )
    return result


def verify_replay_with_factory(
    trajectory: Mapping[str, object],
    *,
    environment_manifest_sha256: str,
    env_factory,
    required_environment_version: str = DEFAULT_REQUIRED_ENVIRONMENT_VERSION,
    required_max_steps: int = DEFAULT_REPLAY_MAX_STEPS,
    prefix_action_count: int | None = None,
) -> dict[str, object]:
    """Verify a replay and release the environment on every leased path."""
    task_id, _, preflight_error = _preflight_replay_contract(
        trajectory,
        environment_manifest_sha256=environment_manifest_sha256,
        required_environment_version=required_environment_version,
        required_max_steps=required_max_steps,
        prefix_action_count=prefix_action_count,
    )
    if preflight_error is not None:
        return preflight_error
    try:
        env = env_factory()
    except Exception as exc:  # noqa: BLE001 - no server error text enters the report.
        return _result(
            task_id=task_id,
            verified=False,
            stage="create",
            reason=f"environment_create_error:{exc.__class__.__name__}",
            verified_transitions=0,
        )
    try:
        try:
            result = verify_replay_trajectory(
                env,
                trajectory,
                environment_manifest_sha256=environment_manifest_sha256,
                required_environment_version=required_environment_version,
                required_max_steps=required_max_steps,
                prefix_action_count=prefix_action_count,
            )
        except Exception as exc:  # noqa: BLE001 - fail closed and still release.
            result = _result(
                task_id=task_id,
                verified=False,
                stage="verification",
                reason=f"verification_error:{exc.__class__.__name__}",
                verified_transitions=0,
            )
    finally:
        try:
            env.release()
        except Exception as exc:  # noqa: BLE001 - release failure invalidates verification.
            release_error = exc.__class__.__name__
        else:
            release_error = None
    if release_error is not None:
        result.update(
            {
                "verified": False,
                "stage": "release",
                "reason": f"release_error:{release_error}",
            }
        )
    result["release_ok"] = release_error is None
    return result
