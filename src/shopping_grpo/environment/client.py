"""ShopSimulator 的最小 HTTP 客户端。

每个 ``ShopAgentEnv`` 对象只负责一条 trajectory：先租用一个环境实例，
反复执行动作，最后释放租约。训练和评测都通过这个生命周期访问商店。
"""

import http.client
import json
import math
import uuid
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from shopping_grpo import __version__
from shopping_grpo.environment.manifest import LEASE_CONTRACT as MANIFEST_LEASE_CONTRACT


class ShopHttpError(RuntimeError):
    """The HTTP request did not reach a usable ShopSimulator response."""


class ShopEnvironmentError(RuntimeError):
    """ShopSimulator accepted HTTP request but reported an environment error."""


class ShopProtocolError(RuntimeError):
    """ShopSimulator response did not match its structured API contract."""


class ShopEnvironmentStateError(RuntimeError):
    """The client lifecycle was used out of order."""


class ShopAgentEnv:
    """一条 trajectory 独占的 ShopSimulator API 租约。"""

    LEASE_CONTRACT = MANIFEST_LEASE_CONTRACT
    _RESET_METADATA_KEYS = frozenset(
        {
            "lease_contract",
            "lease_token",
            "lease_ttl_seconds",
            "lease_recovered",
        }
    )

    def __init__(
        self,
        base_url="http://127.0.0.1:5700",
        timeout=60,
        transport=None,
        reset_attempts=2,
        token_factory=None,
    ):
        reset_attempts = int(reset_attempts)
        if reset_attempts <= 0:
            raise ValueError("reset_attempts must be positive")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.transport = transport
        self.reset_attempts = reset_attempts
        self._token_factory = token_factory or uuid.uuid4
        self.env_idx = None
        self.lease_token = None
        self.lease_ttl_seconds = None
        self._lease_task_id = None
        self.done = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            self.release()
        except Exception:
            if exc_type is None:
                raise
        return False

    def reset(self, task_id):
        """为任务申请环境实例，并保存服务端返回的 ``env_idx``。"""
        if self.env_idx is not None:
            raise ShopEnvironmentStateError(
                "Environment is already leased; release it before reset"
            )

        task_id = int(task_id)
        if self.lease_token is None:
            self.lease_token = self._new_lease_token()
            self._lease_task_id = task_id
        elif self._lease_task_id != task_id:
            raise ShopEnvironmentStateError(
                "An ambiguous reset lease must be released before changing task"
            )

        # reset 只负责建立租约；真正的购物动作统一走 step，便于上层记录轨迹。
        payload = {
            "action": "reset",
            "idx": task_id,
            "lease_contract": self.LEASE_CONTRACT,
            "lease_token": self.lease_token,
        }
        for attempt in range(self.reset_attempts):
            try:
                result = self._call(payload)
                break
            except ShopHttpError:
                if attempt + 1 == self.reset_attempts:
                    self._cancel_ambiguous_reset()
                    raise
            except (ShopEnvironmentError, ShopProtocolError):
                self._cancel_ambiguous_reset()
                raise

        try:
            env_idx = result.get("env_idx")
            if not isinstance(env_idx, int):
                raise ShopProtocolError("reset response is missing integer env_idx")
            if result.get("lease_contract") != self.LEASE_CONTRACT:
                raise ShopProtocolError("reset response has an unsupported lease contract")
            if result.get("lease_token") != self.lease_token:
                raise ShopProtocolError("reset response lease_token does not match the request")
            try:
                lease_ttl_seconds = float(result.get("lease_ttl_seconds"))
            except (TypeError, ValueError) as exc:
                raise ShopProtocolError(
                    "reset response is missing positive lease_ttl_seconds"
                ) from exc
            if not math.isfinite(lease_ttl_seconds) or lease_ttl_seconds <= 0:
                raise ShopProtocolError(
                    "reset response is missing positive lease_ttl_seconds"
                )
            if not isinstance(result.get("lease_recovered"), bool):
                raise ShopProtocolError("reset response is missing boolean lease_recovered")
        except ShopProtocolError:
            self._cancel_ambiguous_reset()
            raise
        self.env_idx = env_idx
        self.lease_ttl_seconds = lease_ttl_seconds
        self.done = False
        return {key: value for key, value in result.items() if key not in self._RESET_METADATA_KEYS}

    def step(self, action):
        """执行一个已经转换好的环境动作，并更新终局状态。"""
        if not isinstance(action, str) or not action:
            raise ValueError("action must be a non-empty string")
        if self.done:
            raise ShopEnvironmentStateError("Environment is already done; release it before reset")
        result = self._call(
            {
                "action": "interact",
                "env_idx": self._leased_env_idx(),
                "response": action,
                "lease_contract": self.LEASE_CONTRACT,
                "lease_token": self.lease_token,
            }
        )
        self.done = bool(result.get("done", False))
        return result

    def release(self):
        """释放当前租约；重复 release 是安全的空操作。"""
        if self.lease_token is None:
            return None

        env_idx = self.env_idx
        payload = {
            "action": "release_one",
            "lease_contract": self.LEASE_CONTRACT,
            "lease_token": self.lease_token,
        }
        if env_idx is not None:
            payload["env_idx"] = env_idx
        result = self._call(payload)
        self.env_idx = None
        self.lease_token = None
        self.lease_ttl_seconds = None
        self._lease_task_id = None
        self.done = False
        return result

    def _leased_env_idx(self):
        if self.env_idx is None or self.lease_token is None:
            raise ShopEnvironmentStateError("reset must succeed before step")
        return self.env_idx

    def _new_lease_token(self):
        token = str(self._token_factory())
        try:
            canonical = str(uuid.UUID(token))
        except (AttributeError, ValueError) as exc:
            raise ValueError("token_factory must return a UUID") from exc
        if token != canonical:
            raise ValueError("token_factory must return a canonical UUID")
        return token

    def _cancel_ambiguous_reset(self):
        try:
            self.release()
        except (ShopHttpError, ShopEnvironmentError, ShopProtocolError):
            # Preserve the original reset failure. A failed cancellation retains
            # the token so an outer finally block can retry it safely.
            return

    def _call(self, payload):
        """统一处理 HTTP、环境错误和结构化协议错误。"""
        try:
            response = self._send(payload)
        except (
            HTTPError,
            URLError,
            OSError,
            http.client.HTTPException,
            json.JSONDecodeError,
            UnicodeError,
        ) as exc:
            raise ShopHttpError(f"ShopSimulator HTTP request failed: {exc}") from exc

        if not isinstance(response, dict):
            raise ShopProtocolError("ShopSimulator response must be a JSON object")
        result = response.get("result")
        if not isinstance(result, dict):
            raise ShopProtocolError("ShopSimulator response is missing object result")
        if result.get("error"):
            raise ShopEnvironmentError(str(result["error"]))
        return result

    def _send(self, payload):
        endpoint = f"{self.base_url}/api/shop_agent"
        if self.transport is not None:
            return self.transport(endpoint, payload, self.timeout)

        body = json.dumps(payload).encode("utf-8")
        request = Request(
            endpoint,
            data=body,
            headers={
                "Content-Type": "application/json",
                "User-Agent": f"shopping-grpo/{__version__}",
            },
            method="POST",
        )
        with urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))
