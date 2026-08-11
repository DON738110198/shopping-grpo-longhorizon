import copy
import logging
import math
import os
import sys
import time
import uuid
from typing import Any

from flask import Flask, Response, jsonify, request

sys.path.append("../")
from shop_agent import shop_agent
from slot_lease_pool import (
    LEASE_CONTRACT_V2,
    LeaseCancelledError,
    LeaseOwnershipError,
    SlotLeasePool,
)
from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv
from web_agent_site.utils import DEBUG_PROD_SIZE


def _positive_float_environment(name: str, default: str) -> float:
    value = float(os.environ.get(name, default))
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return value


def _positive_int_environment(name: str, default: str) -> int:
    value = int(os.environ.get(name, default))
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _boolean_environment(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean")


# Constants
LOG_FILE = "shop_agent.log"
MAX_RETRIES = 5
RETRY_DELAY_SECONDS = 5
DEFAULT_ENV_MAX_NUM = int(os.environ.get("SHOPSIM_ENV_SLOTS", "20"))
SERVER_HOST = "0.0.0.0"
SERVER_PORT = int(os.environ.get("SHOPSIM_PORT", "5000"))
LEASE_TTL_SECONDS = _positive_float_environment("SHOPSIM_LEASE_TTL_SECONDS", "900")
RESET_RETRY_WAIT_SECONDS = _positive_float_environment("SHOPSIM_RESET_RETRY_WAIT_SECONDS", "60")
RETIRED_TOKEN_FILTER_BYTES = _positive_int_environment(
    "SHOPSIM_RETIRED_TOKEN_FILTER_BYTES", "1048576"
)
ALLOW_LEGACY_LEASES = _boolean_environment("SHOPSIM_ALLOW_LEGACY_LEASES")
ALLOW_RELEASE_ALL = _boolean_environment("SHOPSIM_ALLOW_RELEASE_ALL")
ENVIRONMENT_MANIFEST_SHA256 = os.environ.get("SHOPSIM_ENVIRONMENT_MANIFEST_SHA256", "")

# Global variables
envs: list[Any] = []
env_max_num: int = DEFAULT_ENV_MAX_NUM
slot_pool = SlotLeasePool(
    env_max_num,
    lease_ttl_seconds=LEASE_TTL_SECONDS,
    retired_token_filter_bytes=RETIRED_TOKEN_FILTER_BYTES,
)

# Configure logging format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

app = Flask(__name__)


def _lease_token_from_request(data: dict[str, Any]) -> str | None:
    """Return a canonical v2 UUID token, or ``None`` for a legacy request."""
    token = data.get("lease_token")
    contract = data.get("lease_contract")
    if token is None and contract is None:
        return None
    if token is None or contract != LEASE_CONTRACT_V2:
        raise ValueError(
            f"tokenized requests require lease_contract={LEASE_CONTRACT_V2!r} and lease_token"
        )
    if not isinstance(token, str):
        raise TypeError("lease_token must be a canonical UUID string")
    try:
        canonical = str(uuid.UUID(token))
    except (AttributeError, ValueError) as exc:
        raise ValueError("lease_token must be a canonical UUID string") from exc
    if canonical != token:
        raise ValueError("lease_token must be a canonical UUID string")
    return token


def _acquire_legacy_slot():
    for retry_count in range(MAX_RETRIES):
        grant = slot_pool.acquire_legacy_reset()
        if grant is not None:
            return grant
        logger.info(
            "[Retry %s/%s] No available environment index, retrying in %s seconds...",
            retry_count + 1,
            MAX_RETRIES,
            RETRY_DELAY_SECONDS,
        )
        time.sleep(RETRY_DELAY_SECONDS)
    return None


def _acquire_tokenized_slot(token: str, owner: object):
    for retry_count in range(MAX_RETRIES):
        grant = slot_pool.acquire_token(token, owner=owner)
        if grant is not None:
            return grant
        logger.info(
            "[Retry %s/%s] No available environment index, retrying in %s seconds...",
            retry_count + 1,
            MAX_RETRIES,
            RETRY_DELAY_SECONDS,
        )
        time.sleep(RETRY_DELAY_SECONDS)
    return None


def _with_manifest(result: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(result, dict):
        raise TypeError("shop_agent result must be a JSON object")
    result = copy.deepcopy(result)
    if ENVIRONMENT_MANIFEST_SHA256:
        result["environment_manifest_sha256"] = ENVIRONMENT_MANIFEST_SHA256
    return result


def _tokenized_reset_result(
    result: dict[str, Any], *, token: str, recovered: bool
) -> dict[str, Any]:
    response_result = copy.deepcopy(result)
    response_result.update(
        {
            "lease_contract": LEASE_CONTRACT_V2,
            "lease_token": token,
            "lease_ttl_seconds": slot_pool.lease_ttl_seconds,
            "lease_recovered": bool(recovered),
        }
    )
    return response_result


def _release_new_reset_after_failure(grant, *, completed: bool) -> None:
    try:
        if completed:
            token = None if grant.token.startswith("legacy:") else grant.token
            slot_pool.release(grant.slot, token=token)
        else:
            slot_pool.fail_reset(grant.token, grant.generation)
        logger.info("[Release] Environment %s released after reset failure", grant.slot)
    except Exception:
        logger.exception(
            "[Release] Failed to release environment %s after reset failure",
            grant.slot,
        )


def _serve_reset(*, lease_token: str | None, idx: Any, response: Any) -> Response:
    owner = ("reset", idx)
    grant = (
        _acquire_legacy_slot()
        if lease_token is None
        else _acquire_tokenized_slot(lease_token, owner)
    )
    if grant is None:
        raise RuntimeError(
            "Unable to get available environment resource, please try again later"
        )

    created_by_request = not grant.recovered
    reset_completed = False
    try:
        if grant.recovered and grant.reset_in_progress:
            grant = slot_pool.wait_for_reset(
                lease_token,
                owner=owner,
                timeout=RESET_RETRY_WAIT_SECONDS,
            )

        if grant.recovered:
            result = grant.reset_result
            if not isinstance(result, dict):
                raise RuntimeError("recovered reset lease is missing its cached result")
        else:
            result = _with_manifest(
                shop_agent(envs[grant.slot], grant.slot, "reset", idx, response)
            )
            if result.get("env_idx") != grant.slot:
                raise ValueError("reset result env_idx does not match leased slot")
            reset_completed = slot_pool.complete_reset(
                grant.token,
                grant.generation,
                result,
            )
            if not reset_completed:
                raise LeaseCancelledError("reset completed after its lease was canceled")

        response_result = (
            result
            if lease_token is None
            else _tokenized_reset_result(
                result,
                token=lease_token,
                recovered=grant.recovered,
            )
        )
        return jsonify({"result": response_result})
    except Exception:
        if created_by_request:
            _release_new_reset_after_failure(grant, completed=reset_completed)
        raise


@app.route("/api/shop_agent", methods=["POST"])
def api_some_function() -> Response:
    """
    API endpoint for shop agent operations.

    Handles three types of actions:
    - release_all: Release all environments
    - release_one: Release a specific environment
    - reset/interact: Process shop agent actions

    Returns:
        JSON response with result or error message
    """
    data = request.json
    if data is None:
        logger.error("[Error] No JSON data provided in request")
        return jsonify({"result": {"error": "No JSON data provided"}})
    if not isinstance(data, dict):
        logger.error("[Error] JSON request body is not an object")
        return jsonify({"result": {"error": "JSON request body must be an object"}})

    action = data.get("action")
    env_idx = data.get("env_idx")
    response = data.get("response")
    idx = data.get("idx")
    try:
        lease_token = _lease_token_from_request(data)

        # Release all environments
        if action == "release_all":
            if not ALLOW_RELEASE_ALL:
                raise PermissionError("release_all is disabled")
            if not slot_pool.reset_if_idle(env_max_num):
                raise LeaseOwnershipError("release_all is forbidden while leases are active")
            logger.info("[Init] Idle lease pool has been initialized")
            return jsonify({"result": {"message": "Idle lease pool has been initialized"}})

        if lease_token is None and not ALLOW_LEGACY_LEASES:
            raise PermissionError(
                f"legacy index-only leases are disabled; use {LEASE_CONTRACT_V2}"
            )

        # Release one environment
        if action == "release_one":
            if lease_token is not None:
                if env_idx is not None and not isinstance(env_idx, int):
                    raise ValueError("environment index must be an integer")
                release = slot_pool.release(env_idx, token=lease_token)
                display_slot = env_idx if env_idx is not None else "for token"
            else:
                if not isinstance(env_idx, int):
                    raise ValueError("No valid environment index provided")
                release = slot_pool.release(env_idx)
                display_slot = env_idx
            if release.pending:
                logger.info("[Release] Environment %s release is pending", display_slot)
                message = f"Environment {display_slot} release is pending"
            elif release.accepted:
                logger.info("[Release] Environment %s has been released", display_slot)
                message = f"Environment {display_slot} has been released"
            else:
                logger.warning("[Release] Environment %s is already free", display_slot)
                message = f"Environment {display_slot} is already free"
            return jsonify(
                {
                    "result": {
                        "message": message,
                        "released": release.accepted,
                        "release_pending": release.pending,
                    }
                }
            )

        if action == "reset":
            if idx is None:
                raise ValueError("reset action requires idx parameter")
            if env_idx is not None:
                raise ValueError("reset requests must not provide env_idx")

            return _serve_reset(lease_token=lease_token, idx=idx, response=response)

        if action != "interact":
            raise ValueError(f"unsupported action: {action!r}")
        if not isinstance(env_idx, int):
            raise TypeError("interact requires an integer env_idx")
        operation = slot_pool.begin_operation(env_idx, token=lease_token)
        try:
            result = shop_agent(envs[env_idx], env_idx, action, idx, response)
        except Exception:
            slot_pool.finish_operation(operation)
            raise
        if not slot_pool.finish_operation(operation):
            raise LeaseCancelledError(
                "environment operation completed after its lease was canceled"
            )

        # The caller owns the lease until release_one.  Auto-releasing here
        # races with the caller's finally-release: another worker can lease
        # this slot between the two releases and then have its active lease
        # accidentally freed by the previous worker.
        if result.get("over"):
            logger.info(f"[Task Over] Environment {env_idx} is awaiting explicit release")
        return jsonify({"result": result})

    except Exception as e:
        logger.exception("[Exception] Exception occurred while processing request")
        return jsonify({"result": {"error": str(e)}})


def initialize_environments() -> None:
    """
    Initialize all environments and add them to the free environment index.
    """
    global envs

    envs = []
    slot_pool.reset(env_max_num)

    shared_server = None
    for i in range(env_max_num):
        logger.info(f"Environment {i} is being initialized")
        env = WebAgentTextEnv(
            observation_mode="text",
            split="train",
            num_products=DEBUG_PROD_SIZE,
            server=shared_server,
            session_prefix=f"slot-{i}",
        )
        if shared_server is None:
            shared_server = env.server
        envs.append(env)


if __name__ == "__main__":
    initialize_environments()
    app.run(host=SERVER_HOST, port=SERVER_PORT)
