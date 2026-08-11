import importlib.util
import logging
import os
import sys
import threading
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from shop_env import slot_lease_pool as slot_lease_pool_module
from shop_env.slot_lease_pool import LEASE_CONTRACT_V2, SlotLeasePool

TOKEN_A = "11111111-1111-4111-8111-111111111111"
TOKEN_B = "22222222-2222-4222-8222-222222222222"


def token_request(action, *, token=TOKEN_A, **fields):
    return {
        "action": action,
        "lease_contract": LEASE_CONTRACT_V2,
        "lease_token": token,
        **fields,
    }


def complete(pool, grant):
    pool.complete_reset(
        grant.token,
        grant.generation,
        {"env_idx": grant.slot},
    )


class _FakeFlask:
    def __init__(self, *args, **kwargs):
        del args, kwargs

    def route(self, *args, **kwargs):
        del args, kwargs

        def decorator(function):
            return function

        return decorator

    def run(self, *args, **kwargs):  # pragma: no cover - import must not start a server.
        raise AssertionError((args, kwargs))


def _load_pack_api(extra_environment=None):
    flask = types.ModuleType("flask")
    flask.Flask = _FakeFlask
    flask.request = types.SimpleNamespace(json=None)
    flask.jsonify = lambda payload: payload
    flask.Response = object

    shop_agent = types.ModuleType("shop_agent")
    shop_agent.shop_agent = Mock()
    web_agent_site = types.ModuleType("web_agent_site")
    web_agent_site.__path__ = []
    web_agent_utils = types.ModuleType("web_agent_site.utils")
    web_agent_utils.DEBUG_PROD_SIZE = 1
    web_agent_envs = types.ModuleType("web_agent_site.envs")
    web_agent_envs.__path__ = []
    web_agent_text_env = types.ModuleType("web_agent_site.envs.web_agent_text_env")
    web_agent_text_env.WebAgentTextEnv = object

    stubs = {
        "flask": flask,
        "shop_agent": shop_agent,
        "slot_lease_pool": slot_lease_pool_module,
        "web_agent_site": web_agent_site,
        "web_agent_site.utils": web_agent_utils,
        "web_agent_site.envs": web_agent_envs,
        "web_agent_site.envs.web_agent_text_env": web_agent_text_env,
    }
    path = Path(__file__).resolve().parents[1] / "shop_env" / "pack_api.py"
    spec = importlib.util.spec_from_file_location("pack_api_lease_cleanup_test", path)
    module = importlib.util.module_from_spec(spec)
    environment = {"SHOPSIM_ENVIRONMENT_MANIFEST_SHA256": "a" * 64}
    environment.update(extra_environment or {})
    with (
        patch.dict(sys.modules, stubs),
        patch.dict(os.environ, environment, clear=True),
        patch.object(logging, "basicConfig"),
        patch.object(logging, "FileHandler", return_value=logging.NullHandler()),
    ):
        spec.loader.exec_module(module)
    return module


class PackApiLeaseCleanupTest(unittest.TestCase):
    def setUp(self):
        self.module = _load_pack_api()
        self.module.env_max_num = 1
        self.module.envs = [object()]
        self.module.slot_pool = SlotLeasePool(1)

    def test_formal_defaults_disable_legacy_and_release_all(self):
        self.module.request.json = {"action": "reset", "idx": 7}
        legacy = self.module.api_some_function()
        self.module.request.json = {"action": "release_all"}
        release_all = self.module.api_some_function()

        self.assertIn("legacy index-only leases are disabled", legacy["result"]["error"])
        self.assertEqual(release_all, {"result": {"error": "release_all is disabled"}})
        self.module.shop_agent.assert_not_called()

    def test_server_lease_storage_is_configurable(self):
        module = _load_pack_api(
            {
                "SHOPSIM_LEASE_TTL_SECONDS": "12.5",
                "SHOPSIM_RETIRED_TOKEN_FILTER_BYTES": "4096",
            }
        )

        self.assertEqual(module.LEASE_TTL_SECONDS, 12.5)
        self.assertEqual(module.slot_pool.lease_ttl_seconds, 12.5)
        self.assertEqual(module.slot_pool.retired_token_filter_bytes, 4096)

    def test_release_all_opt_in_still_rejects_active_leases(self):
        module = _load_pack_api({"SHOPSIM_ALLOW_RELEASE_ALL": "true"})
        module.env_max_num = 1
        module.envs = [object()]
        module.slot_pool = SlotLeasePool(1)
        grant = module.slot_pool.acquire_token(TOKEN_A, owner=("reset", 7))
        complete(module.slot_pool, grant)
        module.request.json = {"action": "release_all"}

        active = module.api_some_function()
        module.slot_pool.release(grant.slot, token=TOKEN_A)
        idle = module.api_some_function()

        self.assertIn("forbidden while leases are active", active["result"]["error"])
        self.assertEqual(
            idle,
            {"result": {"message": "Idle lease pool has been initialized"}},
        )

    def test_failed_reset_releases_the_request_owned_slot(self):
        self.module.request.json = token_request("reset", idx=999999)
        self.module.shop_agent = Mock(side_effect=IndexError("invalid task"))

        result = self.module.api_some_function()

        self.assertEqual(result, {"result": {"error": "invalid task"}})
        self.assertEqual(self.module.slot_pool.free_slots(), frozenset({0}))
        with self.assertRaisesRegex(Exception, "expired or already released"):
            self.module.slot_pool.acquire_token(TOKEN_A, owner=("reset", 999999))

    def test_successful_reset_keeps_the_slot_leased_for_the_client(self):
        self.module.request.json = token_request("reset", idx=7)
        self.module.shop_agent = Mock(return_value={"env_idx": 0})

        result = self.module.api_some_function()

        self.assertEqual(result["result"]["env_idx"], 0)
        self.assertEqual(result["result"]["lease_token"], TOKEN_A)
        self.assertEqual(result["result"]["lease_contract"], LEASE_CONTRACT_V2)
        self.assertEqual(result["result"]["environment_manifest_sha256"], "a" * 64)
        self.assertFalse(result["result"]["lease_recovered"])
        self.assertEqual(self.module.slot_pool.free_slots(), frozenset())

    def test_reset_response_serialization_failure_releases_the_slot(self):
        self.module.request.json = token_request("reset", idx=7)
        self.module.shop_agent = Mock(return_value={"env_idx": 0})

        def fail_success_response(payload):
            if "env_idx" in payload.get("result", {}):
                raise TypeError("cannot serialize response")
            return payload

        self.module.jsonify = fail_success_response
        result = self.module.api_some_function()

        self.assertEqual(result, {"result": {"error": "cannot serialize response"}})
        self.assertEqual(self.module.slot_pool.free_slots(), frozenset({0}))

    def test_failed_interact_finishes_guard_without_releasing_lease(self):
        grant = self.module.slot_pool.acquire_token(TOKEN_A, owner=("reset", 7))
        complete(self.module.slot_pool, grant)
        self.module.request.json = token_request(
            "interact",
            env_idx=grant.slot,
            response="search[mug]",
        )
        self.module.shop_agent = Mock(side_effect=RuntimeError("step failed"))

        result = self.module.api_some_function()

        self.assertEqual(result, {"result": {"error": "step failed"}})
        operation = self.module.slot_pool.begin_operation(grant.slot, token=TOKEN_A)
        self.assertTrue(self.module.slot_pool.finish_operation(operation))

    def test_duplicate_reset_recovers_cached_result_without_resetting_again(self):
        payload = token_request("reset", idx=7)
        self.module.shop_agent = Mock(return_value={"env_idx": 0, "instruction": "goal"})

        self.module.request.json = payload
        first = self.module.api_some_function()
        self.module.request.json = payload
        recovered = self.module.api_some_function()

        self.assertFalse(first["result"]["lease_recovered"])
        self.assertTrue(recovered["result"]["lease_recovered"])
        self.assertEqual(first["result"]["env_idx"], recovered["result"]["env_idx"])
        self.module.shop_agent.assert_called_once()

    def test_concurrent_duplicate_reset_waits_for_and_recovers_first_result(self):
        entered = threading.Event()
        resume = threading.Event()
        results = []

        def blocking_reset(_env, env_idx, *_args):
            entered.set()
            self.assertTrue(resume.wait(timeout=2))
            return {"env_idx": env_idx, "instruction": "goal"}

        self.module.shop_agent = Mock(side_effect=blocking_reset)
        self.module.request.json = token_request("reset", idx=7)
        first = threading.Thread(
            target=lambda: results.append(self.module.api_some_function())
        )
        second = threading.Thread(
            target=lambda: results.append(self.module.api_some_function())
        )
        first.start()
        self.assertTrue(entered.wait(timeout=2))
        second.start()
        resume.set()
        first.join(timeout=2)
        second.join(timeout=2)

        self.assertFalse(first.is_alive())
        self.assertFalse(second.is_alive())
        self.assertEqual(len(results), 2)
        self.assertEqual(
            sorted(item["result"]["lease_recovered"] for item in results),
            [False, True],
        )
        self.assertEqual(
            {item["result"]["env_idx"] for item in results},
            {0},
        )
        self.module.shop_agent.assert_called_once()

    def test_wrong_token_cannot_interact_with_or_release_active_slot(self):
        grant = self.module.slot_pool.acquire_token(TOKEN_A, owner=("reset", 7))
        complete(self.module.slot_pool, grant)
        self.module.shop_agent = Mock(return_value={"done": False})

        self.module.request.json = token_request(
            "interact",
            token=TOKEN_B,
            env_idx=grant.slot,
            response="search[mug]",
        )
        interact = self.module.api_some_function()
        self.module.request.json = token_request(
            "release_one",
            token=TOKEN_B,
            env_idx=grant.slot,
        )
        release = self.module.api_some_function()

        self.assertIn("does not own", interact["result"]["error"])
        self.assertFalse(release["result"]["released"])
        self.module.shop_agent.assert_not_called()
        self.assertEqual(self.module.slot_pool.active_lease_count(), 1)

    def test_expired_token_release_does_not_free_reassigned_slot(self):
        now = [0.0]
        self.module.slot_pool = SlotLeasePool(
            1,
            lease_ttl_seconds=10,
            clock=lambda: now[0],
        )
        first = self.module.slot_pool.acquire_token(TOKEN_A, owner=("reset", 7))
        complete(self.module.slot_pool, first)
        now[0] = 11.0
        second = self.module.slot_pool.acquire_token(TOKEN_B, owner=("reset", 8))
        complete(self.module.slot_pool, second)
        self.module.request.json = token_request(
            "release_one",
            env_idx=first.slot,
        )

        result = self.module.api_some_function()

        self.assertFalse(result["result"]["released"])
        self.assertEqual(self.module.slot_pool.active_lease_count(), 1)

    def test_legacy_opt_in_quarantines_slot_against_stale_release(self):
        module = _load_pack_api({"SHOPSIM_ALLOW_LEGACY_LEASES": "true"})
        module.env_max_num = 2
        module.envs = [object(), object()]
        module.slot_pool = SlotLeasePool(2)
        module.shop_agent = Mock(
            side_effect=lambda _env, env_idx, *_args: {"env_idx": env_idx}
        )

        module.request.json = {"action": "reset", "idx": 7}
        first = module.api_some_function()["result"]["env_idx"]
        module.request.json = {"action": "release_one", "env_idx": first}
        module.api_some_function()
        module.request.json = {"action": "reset", "idx": 8}
        second = module.api_some_function()["result"]["env_idx"]
        module.request.json = {"action": "release_one", "env_idx": first}
        stale = module.api_some_function()

        self.assertNotEqual(first, second)
        self.assertFalse(stale["result"]["released"])
        self.assertEqual(module.slot_pool.active_lease_count(), 1)

    def test_concurrent_reset_cancel_defers_reuse_until_reset_returns(self):
        entered = threading.Event()
        resume = threading.Event()
        thread_result = []

        def blocking_reset(_env, env_idx, *_args):
            entered.set()
            self.assertTrue(resume.wait(timeout=2))
            return {"env_idx": env_idx}

        self.module.shop_agent = Mock(side_effect=blocking_reset)
        self.module.request.json = token_request("reset", idx=7)
        worker = threading.Thread(
            target=lambda: thread_result.append(self.module.api_some_function())
        )
        worker.start()
        self.assertTrue(entered.wait(timeout=2))

        self.module.request.json = token_request("release_one")
        release = self.module.api_some_function()
        self.assertTrue(release["result"]["released"])
        self.assertTrue(release["result"]["release_pending"])
        self.assertIsNone(
            self.module.slot_pool.acquire_token(TOKEN_B, owner=("reset", 8))
        )

        resume.set()
        worker.join(timeout=2)
        self.assertFalse(worker.is_alive())
        self.assertIn("canceled", thread_result[0]["result"]["error"])
        self.assertEqual(self.module.slot_pool.free_slots(), frozenset({0}))

    def test_concurrent_interact_release_defers_reuse_until_interact_returns(self):
        grant = self.module.slot_pool.acquire_token(TOKEN_A, owner=("reset", 7))
        complete(self.module.slot_pool, grant)
        entered = threading.Event()
        resume = threading.Event()
        thread_result = []

        def blocking_interact(*_args):
            entered.set()
            self.assertTrue(resume.wait(timeout=2))
            return {"done": False}

        self.module.shop_agent = Mock(side_effect=blocking_interact)
        self.module.request.json = token_request(
            "interact",
            env_idx=grant.slot,
            response="search[mug]",
        )
        worker = threading.Thread(
            target=lambda: thread_result.append(self.module.api_some_function())
        )
        worker.start()
        self.assertTrue(entered.wait(timeout=2))

        self.module.request.json = token_request("release_one", env_idx=grant.slot)
        release = self.module.api_some_function()
        self.assertTrue(release["result"]["release_pending"])
        self.assertIsNone(
            self.module.slot_pool.acquire_token(TOKEN_B, owner=("reset", 8))
        )

        resume.set()
        worker.join(timeout=2)
        self.assertFalse(worker.is_alive())
        self.assertIn("canceled", thread_result[0]["result"]["error"])
        replacement = self.module.slot_pool.acquire_token(TOKEN_B, owner=("reset", 8))
        self.assertEqual(replacement.slot, grant.slot)


if __name__ == "__main__":
    unittest.main()
