import importlib.util
import logging
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from shop_env.slot_lease_pool import SlotLeasePool


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


def _load_pack_api():
    flask = types.ModuleType("flask")
    flask.Flask = _FakeFlask
    flask.request = types.SimpleNamespace(json=None)
    flask.jsonify = lambda payload: payload
    flask.Response = object

    shop_agent = types.ModuleType("shop_agent")
    shop_agent.shop_agent = Mock()
    lease_pool = types.ModuleType("slot_lease_pool")
    lease_pool.SlotLeasePool = SlotLeasePool
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
        "slot_lease_pool": lease_pool,
        "web_agent_site": web_agent_site,
        "web_agent_site.utils": web_agent_utils,
        "web_agent_site.envs": web_agent_envs,
        "web_agent_site.envs.web_agent_text_env": web_agent_text_env,
    }
    path = Path(__file__).resolve().parents[1] / "shop_env" / "pack_api.py"
    spec = importlib.util.spec_from_file_location("pack_api_lease_cleanup_test", path)
    module = importlib.util.module_from_spec(spec)
    with (
        patch.dict(sys.modules, stubs),
        patch.dict(
            "os.environ",
            {"SHOPSIM_ENVIRONMENT_MANIFEST_SHA256": "a" * 64},
        ),
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

    def test_failed_reset_releases_the_request_owned_slot(self):
        self.module.request.json = {"action": "reset", "idx": 999999}
        self.module.shop_agent = Mock(side_effect=IndexError("invalid task"))

        result = self.module.api_some_function()

        self.assertEqual(result, {"result": {"error": "invalid task"}})
        self.assertEqual(self.module.slot_pool.free_slots(), frozenset({0}))
        self.assertEqual(self.module.slot_pool.acquire(), 0)

    def test_successful_reset_keeps_the_slot_leased_for_the_client(self):
        self.module.request.json = {"action": "reset", "idx": 7}
        self.module.shop_agent = Mock(return_value={"env_idx": 0})

        result = self.module.api_some_function()

        self.assertEqual(
            result,
            {
                "result": {
                    "env_idx": 0,
                    "environment_manifest_sha256": "a" * 64,
                }
            },
        )
        self.assertEqual(self.module.slot_pool.free_slots(), frozenset())

    def test_reset_response_serialization_failure_releases_the_slot(self):
        self.module.request.json = {"action": "reset", "idx": 7}
        self.module.shop_agent = Mock(return_value={"env_idx": 0})

        def fail_success_response(payload):
            if "env_idx" in payload.get("result", {}):
                raise TypeError("cannot serialize response")
            return payload

        self.module.jsonify = fail_success_response
        result = self.module.api_some_function()

        self.assertEqual(result, {"result": {"error": "cannot serialize response"}})
        self.assertEqual(self.module.slot_pool.free_slots(), frozenset({0}))

    def test_failed_interact_does_not_release_a_client_owned_slot(self):
        slot = self.module.slot_pool.acquire()
        self.module.request.json = {
            "action": "interact",
            "env_idx": slot,
            "response": "search[mug]",
        }
        self.module.shop_agent = Mock(side_effect=RuntimeError("step failed"))

        result = self.module.api_some_function()

        self.assertEqual(result, {"result": {"error": "step failed"}})
        self.assertEqual(self.module.slot_pool.free_slots(), frozenset())


if __name__ == "__main__":
    unittest.main()
