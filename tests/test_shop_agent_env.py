import http.client
import inspect
import json
import unittest

from shopping_grpo.environment import client as shop_http_env

TOKEN = "11111111-1111-4111-8111-111111111111"


def reset_response(env_idx, *, recovered=False, **fields):
    return {
        "result": {
            "env_idx": env_idx,
            "lease_contract": shop_http_env.ShopAgentEnv.LEASE_CONTRACT,
            "lease_token": TOKEN,
            "lease_ttl_seconds": 900.0,
            "lease_recovered": recovered,
            **fields,
        }
    }


class FakeTransport:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, url, payload, timeout):
        self.calls.append((url, payload, timeout))
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


class ShopAgentEnvTest(unittest.TestCase):
    def test_structured_api_environment_is_exposed(self):
        self.assertTrue(hasattr(shop_http_env, "ShopAgentEnv"))

    def test_structured_api_environment_exposes_lifecycle_methods(self):
        for name in ("reset", "step", "release"):
            self.assertTrue(hasattr(shop_http_env.ShopAgentEnv, name))

    def test_transport_can_be_injected_for_api_tests(self):
        parameters = inspect.signature(shop_http_env.ShopAgentEnv).parameters
        self.assertIn("transport", parameters)
        self.assertIn("token_factory", parameters)

    def test_reset_step_and_release_use_structured_api(self):
        transport = FakeTransport(
            [
                reset_response(7, instruction="找乳胶枕", idx=0),
                {"result": {"instruction": "搜索结果", "done": False, "reward": 0.0}},
                {
                    "result": {
                        "instruction": "购买完成",
                        "done": True,
                        "reward": 0.0,
                        "reward_detail": {"r_type": 0, "r_att": 0, "r_option": 0, "r_price": 0},
                        "purchase": {"asin": "wrong"},
                        "goal": {"instruction_text": "hidden"},
                    }
                },
                {"result": {"message": "Environment 7 is already free"}},
            ]
        )
        env = shop_http_env.ShopAgentEnv(
            "http://shop.test",
            timeout=12,
            transport=transport,
            token_factory=lambda: TOKEN,
        )

        reset = env.reset(0)
        search = env.step("search[乳胶枕]")
        terminal = env.step("click[Buy Now]")
        env.release()

        self.assertEqual(reset["instruction"], "找乳胶枕")
        self.assertFalse(search["done"])
        self.assertTrue(terminal["done"])
        self.assertEqual(terminal["reward"], 0.0)
        self.assertEqual(terminal["purchase"], {"asin": "wrong"})
        self.assertEqual(terminal["goal"], {"instruction_text": "hidden"})
        self.assertIsNone(env.env_idx)
        self.assertIsNone(env.lease_token)
        self.assertEqual(
            [payload for _, payload, _ in transport.calls],
            [
                {
                    "action": "reset",
                    "idx": 0,
                    "lease_contract": shop_http_env.ShopAgentEnv.LEASE_CONTRACT,
                    "lease_token": TOKEN,
                },
                {
                    "action": "interact",
                    "env_idx": 7,
                    "response": "search[乳胶枕]",
                    "lease_contract": shop_http_env.ShopAgentEnv.LEASE_CONTRACT,
                    "lease_token": TOKEN,
                },
                {
                    "action": "interact",
                    "env_idx": 7,
                    "response": "click[Buy Now]",
                    "lease_contract": shop_http_env.ShopAgentEnv.LEASE_CONTRACT,
                    "lease_token": TOKEN,
                },
                {
                    "action": "release_one",
                    "lease_contract": shop_http_env.ShopAgentEnv.LEASE_CONTRACT,
                    "lease_token": TOKEN,
                    "env_idx": 7,
                },
            ],
        )

    def test_zero_reward_terminal_purchase_stops_further_steps(self):
        transport = FakeTransport(
            [
                reset_response(2, instruction="找乳胶枕", idx=0),
                {
                    "result": {
                        "instruction": "购买完成",
                        "done": True,
                        "reward": 0.0,
                        "reward_detail": {"r_type": 0, "r_att": 0, "r_option": 0, "r_price": 0},
                        "purchase": {"asin": "wrong"},
                        "goal": {"instruction_text": "hidden"},
                    }
                },
            ]
        )
        env = shop_http_env.ShopAgentEnv(
            transport=transport,
            token_factory=lambda: TOKEN,
        )

        terminal = env.reset(0)
        terminal = env.step("click[Buy Now]")

        self.assertTrue(terminal["done"])
        self.assertEqual(terminal["reward"], 0.0)
        with self.assertRaises(shop_http_env.ShopEnvironmentStateError):
            env.step("search[乳胶枕]")

    def test_environment_error_is_distinct_from_http_error(self):
        env = shop_http_env.ShopAgentEnv(
            transport=FakeTransport(
                [
                    {"result": {"error": "reset action requires idx parameter"}},
                    {"result": {"released": False, "release_pending": False}},
                ]
            ),
            token_factory=lambda: TOKEN,
        )

        with self.assertRaises(shop_http_env.ShopEnvironmentError):
            env.reset(0)
        self.assertIsNone(env.lease_token)

    def test_context_manager_releases_after_http_error(self):
        transport = FakeTransport(
            [
                reset_response(3, instruction="找乳胶枕", idx=0),
                OSError("connection reset"),
                {"result": {"message": "Environment 3 has been released"}},
            ]
        )

        with (
            self.assertRaises(shop_http_env.ShopHttpError),
            shop_http_env.ShopAgentEnv(
                transport=transport,
                token_factory=lambda: TOKEN,
            ) as env,
        ):
            env.reset(0)
            env.step("search[乳胶枕]")

        self.assertEqual(
            [payload["action"] for _, payload, _ in transport.calls],
            ["reset", "interact", "release_one"],
        )
        self.assertIsNone(env.env_idx)

    def test_failed_release_keeps_lease_for_recovery(self):
        """释放请求未送达时，客户端必须保留租约编号供上层恢复。"""
        transport = FakeTransport(
            [
                reset_response(3, instruction="找乳胶枕", idx=0),
                OSError("connection reset"),
            ]
        )
        env = shop_http_env.ShopAgentEnv(
            transport=transport,
            token_factory=lambda: TOKEN,
        )
        env.reset(0)

        with self.assertRaises(shop_http_env.ShopHttpError):
            env.release()

        self.assertEqual(env.env_idx, 3)
        self.assertEqual(env.lease_token, TOKEN)

    def test_reset_http_retry_reuses_token_and_recovers_same_lease(self):
        transport = FakeTransport(
            [
                OSError("response lost"),
                reset_response(5, recovered=True, instruction="goal"),
            ]
        )
        env = shop_http_env.ShopAgentEnv(
            transport=transport,
            token_factory=lambda: TOKEN,
        )

        result = env.reset(17)

        self.assertEqual(result, {"env_idx": 5, "instruction": "goal"})
        self.assertEqual(len(transport.calls), 2)
        first_payload = transport.calls[0][1]
        second_payload = transport.calls[1][1]
        self.assertEqual(first_payload, second_payload)
        self.assertEqual(first_payload["lease_token"], TOKEN)

    def test_ambiguous_reset_is_automatically_canceled_by_token(self):
        transport = FakeTransport(
            [
                OSError("response lost"),
                OSError("response lost again"),
                {"result": {"message": "Environment for token has been released"}},
            ]
        )
        env = shop_http_env.ShopAgentEnv(
            transport=transport,
            token_factory=lambda: TOKEN,
        )

        with self.assertRaises(shop_http_env.ShopHttpError):
            env.reset(17)
        self.assertIsNone(env.env_idx)
        self.assertEqual(
            transport.calls[-1][1],
            {
                "action": "release_one",
                "lease_contract": shop_http_env.ShopAgentEnv.LEASE_CONTRACT,
                "lease_token": TOKEN,
            },
        )
        self.assertIsNone(env.lease_token)

    def test_failed_ambiguous_reset_cancel_keeps_token_for_outer_finally(self):
        transport = FakeTransport(
            [
                OSError("response lost"),
                OSError("response lost again"),
                OSError("cancel response lost"),
            ]
        )
        env = shop_http_env.ShopAgentEnv(
            transport=transport,
            token_factory=lambda: TOKEN,
        )

        with self.assertRaises(shop_http_env.ShopHttpError):
            env.reset(17)

        self.assertIsNone(env.env_idx)
        self.assertEqual(env.lease_token, TOKEN)

    def test_reset_rejects_mismatched_response_token_and_cancels_lease(self):
        response = reset_response(5)
        response["result"]["lease_token"] = "22222222-2222-4222-8222-222222222222"
        env = shop_http_env.ShopAgentEnv(
            transport=FakeTransport(
                [response, {"result": {"released": True, "release_pending": False}}]
            ),
            token_factory=lambda: TOKEN,
        )

        with self.assertRaisesRegex(shop_http_env.ShopProtocolError, "does not match"):
            env.reset(17)

        self.assertIsNone(env.env_idx)
        self.assertIsNone(env.lease_token)

    def test_truncated_or_malformed_reset_response_retries_with_same_token(self):
        failures = (
            http.client.IncompleteRead(b'{"result":', 10),
            json.JSONDecodeError("truncated", '{"result":', 10),
        )
        for failure in failures:
            with self.subTest(failure=type(failure).__name__):
                transport = FakeTransport(
                    [failure, reset_response(5, recovered=True, instruction="goal")]
                )
                env = shop_http_env.ShopAgentEnv(
                    transport=transport,
                    token_factory=lambda: TOKEN,
                )

                result = env.reset(17)

                self.assertEqual(result["instruction"], "goal")
                self.assertEqual(transport.calls[0][1], transport.calls[1][1])


if __name__ == "__main__":
    unittest.main()
