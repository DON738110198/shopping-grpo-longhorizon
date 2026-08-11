"""Thread-safe, token-bound leases for ShopSimulator worker slots."""

from __future__ import annotations

import copy
import hashlib
import math
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

LEASE_CONTRACT_V2 = "tokenized-ttl-release-v2"


class LeaseError(ValueError):
    """Base class for lease protocol violations."""


class LeaseOwnershipError(LeaseError):
    """A request does not own the referenced slot."""


class LeaseTokenRetiredError(LeaseError):
    """A completed or expired request token cannot create another lease."""


class LeaseCancelledError(LeaseError):
    """A lease was canceled while an operation was still in flight."""


@dataclass(frozen=True)
class LeaseGrant:
    slot: int
    generation: int
    token: str
    recovered: bool
    reset_in_progress: bool
    reset_result: dict[str, Any] | None
    expires_at: float


@dataclass(frozen=True)
class LeaseOperation:
    slot: int
    generation: int
    token: str


@dataclass(frozen=True)
class LeaseRelease:
    accepted: bool
    pending: bool
    slot: int | None

    def __bool__(self) -> bool:
        return self.accepted


@dataclass
class _Lease:
    slot: int
    generation: int
    token: str
    owner: object
    legacy: bool
    expires_at: float
    reset_in_progress: bool
    reset_result: dict[str, Any] | None = None
    inflight_operations: int = 0
    cancel_requested: bool = False


class _RetiredTokenFilter:
    """Fixed-memory Bloom filter; false positives fail closed, never unsafe-open."""

    def __init__(self, storage_bytes: int, hash_count: int = 7):
        storage_bytes = int(storage_bytes)
        hash_count = int(hash_count)
        if storage_bytes <= 0 or hash_count <= 0:
            raise ValueError("retired token filter dimensions must be positive")
        self.storage_bytes = storage_bytes
        self._bits = bytearray(storage_bytes)
        self._bit_count = storage_bytes * 8
        self._hash_count = hash_count

    def _indices(self, token: str):
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=32).digest()
        first = int.from_bytes(digest[:16], "big")
        step = int.from_bytes(digest[16:], "big") | 1
        for index in range(self._hash_count):
            yield (first + index * step) % self._bit_count

    def add(self, token: str) -> None:
        for bit_index in self._indices(token):
            byte_index, offset = divmod(bit_index, 8)
            self._bits[byte_index] |= 1 << offset

    def __contains__(self, token: str) -> bool:
        return all(
            self._bits[byte_index] & (1 << offset)
            for byte_index, offset in (
                divmod(bit_index, 8) for bit_index in self._indices(token)
            )
        )


class SlotLeasePool:
    """Lease slots with idempotent reset recovery and operation-safe reclamation."""

    def __init__(
        self,
        size: int,
        *,
        lease_ttl_seconds: float = 900.0,
        retired_token_filter_bytes: int = 1024 * 1024,
        clock: Callable[[], float] | None = None,
    ):
        ttl = float(lease_ttl_seconds)
        if not math.isfinite(ttl) or ttl <= 0:
            raise ValueError("lease_ttl_seconds must be positive and finite")
        filter_bytes = int(retired_token_filter_bytes)
        if filter_bytes <= 0:
            raise ValueError("retired_token_filter_bytes must be positive")
        self.lease_ttl_seconds = ttl
        self.retired_token_filter_bytes = filter_bytes
        self._clock = clock or time.monotonic
        self._condition = threading.Condition(threading.Lock())
        self._size = 0
        self._next_generation = 0
        self._free: set[int] = set()
        self._legacy_quarantined: set[int] = set()
        self._by_slot: dict[int, _Lease] = {}
        self._by_token: dict[str, _Lease] = {}
        self._retired_tokens = _RetiredTokenFilter(filter_bytes)
        self.reset(size)

    def _now(self) -> float:
        return float(self._clock())

    def _grant_locked(self, lease: _Lease, *, recovered: bool) -> LeaseGrant:
        return LeaseGrant(
            slot=lease.slot,
            generation=lease.generation,
            token=lease.token,
            recovered=recovered,
            reset_in_progress=lease.reset_in_progress,
            reset_result=copy.deepcopy(lease.reset_result),
            expires_at=lease.expires_at,
        )

    def _drop_locked(self, lease: _Lease, *, retire: bool) -> None:
        self._by_slot.pop(lease.slot, None)
        self._by_token.pop(lease.token, None)
        if lease.legacy:
            # Index-only clients cannot prove a generation. Never reuse their slot
            # in this server process, so a delayed legacy release cannot hit a new owner.
            self._legacy_quarantined.add(lease.slot)
        else:
            self._free.add(lease.slot)
            if retire:
                self._retired_tokens.add(lease.token)
        self._condition.notify_all()

    def _reclaim_expired_locked(self, now: float) -> list[int]:
        expired = [
            lease
            for lease in self._by_slot.values()
            if lease.expires_at <= now
            and not lease.reset_in_progress
            and lease.inflight_operations == 0
        ]
        for lease in expired:
            self._drop_locked(lease, retire=True)
        return sorted(lease.slot for lease in expired)

    def _new_lease_locked(
        self,
        *,
        token: str,
        owner: object,
        legacy: bool,
        reset_in_progress: bool,
        now: float,
    ) -> _Lease | None:
        if not self._free:
            return None
        slot = self._free.pop()
        self._next_generation += 1
        lease = _Lease(
            slot=slot,
            generation=self._next_generation,
            token=token,
            owner=owner,
            legacy=legacy,
            expires_at=now + self.lease_ttl_seconds,
            reset_in_progress=reset_in_progress,
        )
        self._by_slot[slot] = lease
        self._by_token[token] = lease
        return lease

    def _lease_for_generation_locked(
        self, token: str, generation: int
    ) -> _Lease | None:
        lease = self._by_token.get(token)
        if lease is None:
            return None
        if lease.generation != int(generation):
            raise LeaseOwnershipError("lease generation no longer owns this environment")
        return lease

    def reset(self, size: int) -> None:
        """Initialize an idle pool; destructive reset of active leases is forbidden."""
        size = int(size)
        if size < 0:
            raise ValueError("slot pool size must be non-negative")
        with self._condition:
            if self._by_slot:
                raise LeaseOwnershipError("cannot reset a pool with active leases")
            self._size = size
            self._free = set(range(size))
            self._legacy_quarantined = set()
            self._by_slot = {}
            self._by_token = {}
            self._retired_tokens = _RetiredTokenFilter(
                self.retired_token_filter_bytes
            )
            self._condition.notify_all()

    def reset_if_idle(self, size: int | None = None) -> bool:
        """Administrative reset that preserves replay and legacy quarantine state."""
        with self._condition:
            self._reclaim_expired_locked(self._now())
            if self._by_slot:
                return False
            if size is not None:
                size = int(size)
                if size < 0:
                    raise ValueError("slot pool size must be non-negative")
                self._size = size
                self._legacy_quarantined.intersection_update(range(size))
            self._free = set(range(self._size)) - self._legacy_quarantined
            self._condition.notify_all()
            return True

    def acquire(self) -> int | None:
        """Acquire an idle legacy lease for low-level backward compatibility."""
        with self._condition:
            now = self._now()
            self._reclaim_expired_locked(now)
            lease = self._new_lease_locked(
                token=f"legacy:{uuid.uuid4()}",
                owner=None,
                legacy=True,
                reset_in_progress=False,
                now=now,
            )
            return lease.slot if lease is not None else None

    def acquire_legacy_reset(self) -> LeaseGrant | None:
        with self._condition:
            now = self._now()
            self._reclaim_expired_locked(now)
            lease = self._new_lease_locked(
                token=f"legacy:{uuid.uuid4()}",
                owner=None,
                legacy=True,
                reset_in_progress=True,
                now=now,
            )
            return self._grant_locked(lease, recovered=False) if lease else None

    def acquire_token(self, token: str, *, owner: object) -> LeaseGrant | None:
        """Acquire once, or recover the same active lease for a duplicate reset."""
        if not isinstance(token, str) or not token:
            raise ValueError("lease token must be a non-empty string")
        with self._condition:
            now = self._now()
            self._reclaim_expired_locked(now)
            existing = self._by_token.get(token)
            if existing is not None:
                if existing.legacy or existing.owner != owner:
                    raise LeaseOwnershipError(
                        "lease token is already bound to a different reset request"
                    )
                if existing.cancel_requested:
                    raise LeaseCancelledError("lease cancellation is pending")
                existing.expires_at = now + self.lease_ttl_seconds
                return self._grant_locked(existing, recovered=True)
            if token in self._retired_tokens:
                raise LeaseTokenRetiredError("lease token is expired or already released")
            lease = self._new_lease_locked(
                token=token,
                owner=owner,
                legacy=False,
                reset_in_progress=True,
                now=now,
            )
            return self._grant_locked(lease, recovered=False) if lease else None

    def complete_reset(
        self,
        token: str,
        generation: int,
        result: dict[str, Any],
    ) -> bool:
        if not isinstance(result, dict):
            raise TypeError("reset result must be an object")
        with self._condition:
            now = self._now()
            lease = self._lease_for_generation_locked(token, generation)
            if lease is None:
                raise LeaseOwnershipError("reset lease is no longer active")
            if not lease.reset_in_progress:
                raise LeaseOwnershipError("environment reset is not in progress")
            lease.reset_in_progress = False
            if lease.cancel_requested:
                self._drop_locked(lease, retire=True)
                return False
            lease.reset_result = copy.deepcopy(result)
            lease.expires_at = now + self.lease_ttl_seconds
            self._condition.notify_all()
            return True

    def fail_reset(self, token: str, generation: int) -> bool:
        with self._condition:
            lease = self._lease_for_generation_locked(token, generation)
            if lease is None:
                return False
            if not lease.reset_in_progress:
                return False
            lease.reset_in_progress = False
            self._drop_locked(lease, retire=True)
            return True

    def wait_for_reset(self, token: str, *, owner: object, timeout: float) -> LeaseGrant:
        deadline = time.monotonic() + max(float(timeout), 0.0)
        with self._condition:
            while True:
                lease = self._by_token.get(token)
                if lease is None:
                    raise LeaseTokenRetiredError("reset lease expired or failed")
                if lease.legacy or lease.owner != owner:
                    raise LeaseOwnershipError(
                        "lease token is already bound to a different reset request"
                    )
                if lease.cancel_requested:
                    raise LeaseCancelledError("lease cancellation is pending")
                if not lease.reset_in_progress:
                    return self._grant_locked(lease, recovered=True)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("duplicate reset is still in progress")
                self._condition.wait(timeout=remaining)

    def begin_operation(self, slot: int, *, token: str | None) -> LeaseOperation:
        slot = int(slot)
        with self._condition:
            now = self._now()
            self._reclaim_expired_locked(now)
            lease = self._by_slot.get(slot)
            if lease is None:
                raise LeaseOwnershipError(f"environment {slot} is not leased")
            if lease.legacy:
                if token is not None:
                    raise LeaseOwnershipError("legacy lease does not accept a lease token")
            elif token != lease.token:
                raise LeaseOwnershipError("lease token does not own this environment")
            if lease.cancel_requested:
                raise LeaseCancelledError("lease cancellation is pending")
            if lease.reset_in_progress:
                raise LeaseOwnershipError("environment reset is still in progress")
            if lease.inflight_operations:
                raise LeaseOwnershipError("another environment operation is in progress")
            lease.inflight_operations = 1
            lease.expires_at = now + self.lease_ttl_seconds
            return LeaseOperation(
                slot=lease.slot,
                generation=lease.generation,
                token=lease.token,
            )

    def finish_operation(self, operation: LeaseOperation) -> bool:
        with self._condition:
            lease = self._by_slot.get(operation.slot)
            if lease is None:
                raise LeaseOwnershipError("operation lease is no longer active")
            if (
                lease.generation != operation.generation
                or lease.token != operation.token
                or lease.inflight_operations != 1
            ):
                raise LeaseOwnershipError("operation generation no longer owns this environment")
            lease.inflight_operations = 0
            if lease.cancel_requested:
                self._drop_locked(lease, retire=True)
                return False
            lease.expires_at = self._now() + self.lease_ttl_seconds
            self._condition.notify_all()
            return True

    def release(self, slot: int | None = None, *, token: str | None = None) -> LeaseRelease:
        with self._condition:
            self._reclaim_expired_locked(self._now())
            if token is not None:
                lease = self._by_token.get(token)
                if lease is None:
                    return LeaseRelease(False, False, slot)
                if lease.legacy:
                    raise LeaseOwnershipError("token does not own a tokenized lease")
                if slot is not None and int(slot) != lease.slot:
                    raise LeaseOwnershipError("lease token and environment index disagree")
            else:
                if slot is None:
                    raise ValueError("legacy release requires an environment index")
                slot = int(slot)
                if slot < 0 or slot >= self._size:
                    raise ValueError(f"invalid environment index: {slot}")
                lease = self._by_slot.get(slot)
                if lease is None:
                    return LeaseRelease(False, False, slot)
                if not lease.legacy:
                    raise LeaseOwnershipError("tokenized lease requires its lease token")

            if lease.reset_in_progress or lease.inflight_operations:
                lease.cancel_requested = True
                self._condition.notify_all()
                return LeaseRelease(True, True, lease.slot)
            released_slot = lease.slot
            self._drop_locked(lease, retire=True)
            return LeaseRelease(True, False, released_slot)

    def reclaim_expired(self) -> tuple[int, ...]:
        with self._condition:
            return tuple(self._reclaim_expired_locked(self._now()))

    def free_slots(self) -> frozenset[int]:
        with self._condition:
            self._reclaim_expired_locked(self._now())
            return frozenset(self._free)

    def legacy_quarantined_slots(self) -> frozenset[int]:
        with self._condition:
            return frozenset(self._legacy_quarantined)

    def active_lease_count(self) -> int:
        with self._condition:
            self._reclaim_expired_locked(self._now())
            return len(self._by_slot)
