import unittest
import uuid

from shop_env.slot_lease_pool import (
    LeaseCancelledError,
    LeaseOperation,
    LeaseOwnershipError,
    LeaseTokenRetiredError,
    SlotLeasePool,
)

TOKEN_A = "11111111-1111-4111-8111-111111111111"
TOKEN_B = "22222222-2222-4222-8222-222222222222"


def complete(pool, grant, result=None):
    return pool.complete_reset(
        grant.token,
        grant.generation,
        result or {"env_idx": grant.slot},
    )


class SlotLeasePoolTest(unittest.TestCase):
    def test_ttl_must_be_positive_and_finite(self):
        for value in (0, -1, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "finite"):
                SlotLeasePool(1, lease_ttl_seconds=value)

    def test_tokenized_owner_keeps_slot_until_explicit_release(self):
        pool = SlotLeasePool(1)
        grant = pool.acquire_token(TOKEN_A, owner=("reset", 7))
        complete(pool, grant)

        self.assertIsNone(pool.acquire_token(TOKEN_B, owner=("reset", 8)))
        release = pool.release(grant.slot, token=TOKEN_A)
        self.assertTrue(release.accepted)
        self.assertFalse(release.pending)
        self.assertEqual(
            pool.acquire_token(TOKEN_B, owner=("reset", 8)).slot,
            grant.slot,
        )

    def test_legacy_slots_are_quarantined_to_make_stale_release_harmless(self):
        pool = SlotLeasePool(2)
        first = pool.acquire()
        self.assertTrue(pool.release(first))
        second = pool.acquire()

        self.assertIsNotNone(second)
        self.assertNotEqual(second, first)
        self.assertFalse(pool.release(first))
        self.assertEqual(pool.active_lease_count(), 1)
        self.assertEqual(pool.legacy_quarantined_slots(), frozenset({first}))

    def test_administrative_reset_rejects_active_leases_and_preserves_quarantine(self):
        pool = SlotLeasePool(2)
        legacy_slot = pool.acquire()
        with self.assertRaisesRegex(LeaseOwnershipError, "active leases"):
            pool.reset(2)
        pool.release(legacy_slot)

        self.assertTrue(pool.reset_if_idle(2))
        self.assertNotIn(legacy_slot, pool.free_slots())

    def test_duplicate_reset_token_recovers_slot_and_cached_result(self):
        pool = SlotLeasePool(1)
        first = pool.acquire_token(TOKEN_A, owner=("reset", 7))
        self.assertFalse(first.recovered)
        complete(pool, first, {"env_idx": first.slot, "nested": {"value": 1}})

        recovered = pool.acquire_token(TOKEN_A, owner=("reset", 7))
        self.assertEqual(recovered.slot, first.slot)
        self.assertEqual(recovered.generation, first.generation)
        self.assertTrue(recovered.recovered)
        recovered.reset_result["nested"]["value"] = 99
        recovered_again = pool.acquire_token(TOKEN_A, owner=("reset", 7))
        self.assertEqual(recovered_again.reset_result["nested"]["value"], 1)

    def test_token_cannot_be_rebound_to_a_different_task(self):
        pool = SlotLeasePool(1)
        pool.acquire_token(TOKEN_A, owner=("reset", 7))

        with self.assertRaisesRegex(LeaseOwnershipError, "different reset"):
            pool.acquire_token(TOKEN_A, owner=("reset", 8))

    def test_expired_idle_token_is_retired_and_stale_release_is_safe(self):
        now = [0.0]
        pool = SlotLeasePool(1, lease_ttl_seconds=10, clock=lambda: now[0])
        first = pool.acquire_token(TOKEN_A, owner=("reset", 7))
        complete(pool, first)

        now[0] = 11.0
        second = pool.acquire_token(TOKEN_B, owner=("reset", 8))
        self.assertEqual(second.slot, first.slot)
        self.assertFalse(pool.release(first.slot, token=TOKEN_A))
        self.assertEqual(pool.free_slots(), frozenset())
        with self.assertRaisesRegex(LeaseTokenRetiredError, "expired"):
            pool.acquire_token(TOKEN_A, owner=("reset", 7))

    def test_wrong_active_token_cannot_operate_or_release_a_slot(self):
        pool = SlotLeasePool(2)
        first = pool.acquire_token(TOKEN_A, owner=("reset", 7))
        second = pool.acquire_token(TOKEN_B, owner=("reset", 8))
        complete(pool, first)
        complete(pool, second)

        with self.assertRaisesRegex(LeaseOwnershipError, "does not own"):
            pool.begin_operation(first.slot, token=TOKEN_B)
        with self.assertRaisesRegex(LeaseOwnershipError, "disagree"):
            pool.release(first.slot, token=TOKEN_B)
        self.assertEqual(pool.free_slots(), frozenset())

    def test_inflight_operation_blocks_ttl_and_release_until_finish(self):
        now = [0.0]
        pool = SlotLeasePool(1, lease_ttl_seconds=10, clock=lambda: now[0])
        grant = pool.acquire_token(TOKEN_A, owner=("reset", 7))
        complete(pool, grant)
        operation = pool.begin_operation(grant.slot, token=TOKEN_A)

        now[0] = 100.0
        self.assertEqual(pool.reclaim_expired(), ())
        release = pool.release(grant.slot, token=TOKEN_A)
        self.assertTrue(release.accepted)
        self.assertTrue(release.pending)
        self.assertIsNone(pool.acquire_token(TOKEN_B, owner=("reset", 8)))

        self.assertFalse(pool.finish_operation(operation))
        replacement = pool.acquire_token(TOKEN_B, owner=("reset", 8))
        self.assertEqual(replacement.slot, grant.slot)
        self.assertGreater(replacement.generation, grant.generation)

    def test_reset_cancel_waits_for_reset_handler_and_releases_atomically(self):
        now = [0.0]
        pool = SlotLeasePool(1, lease_ttl_seconds=10, clock=lambda: now[0])
        grant = pool.acquire_token(TOKEN_A, owner=("reset", 7))

        now[0] = 100.0
        self.assertEqual(pool.reclaim_expired(), ())
        release = pool.release(grant.slot, token=TOKEN_A)
        self.assertTrue(release.pending)
        self.assertIsNone(pool.acquire_token(TOKEN_B, owner=("reset", 8)))
        with self.assertRaises(LeaseCancelledError):
            pool.acquire_token(TOKEN_A, owner=("reset", 7))

        self.assertFalse(complete(pool, grant))
        self.assertEqual(
            pool.acquire_token(TOKEN_B, owner=("reset", 8)).slot,
            grant.slot,
        )

    def test_operation_generation_mismatch_cannot_finish_newer_owner(self):
        pool = SlotLeasePool(1)
        grant = pool.acquire_token(TOKEN_A, owner=("reset", 7))
        complete(pool, grant)
        operation = pool.begin_operation(grant.slot, token=TOKEN_A)
        forged = LeaseOperation(
            slot=operation.slot,
            generation=operation.generation + 1,
            token=operation.token,
        )

        with self.assertRaisesRegex(LeaseOwnershipError, "generation"):
            pool.finish_operation(forged)
        self.assertTrue(pool.finish_operation(operation))

    def test_duplicate_reset_and_operation_completion_renew_ttl(self):
        now = [0.0]
        pool = SlotLeasePool(1, lease_ttl_seconds=10, clock=lambda: now[0])
        grant = pool.acquire_token(TOKEN_A, owner=("reset", 7))
        complete(pool, grant)

        now[0] = 9.0
        recovered = pool.acquire_token(TOKEN_A, owner=("reset", 7))
        self.assertEqual(recovered.expires_at, 19.0)
        operation = pool.begin_operation(grant.slot, token=TOKEN_A)
        now[0] = 30.0
        self.assertTrue(pool.finish_operation(operation))
        now[0] = 39.0
        self.assertIsNone(pool.acquire_token(TOKEN_B, owner=("reset", 8)))

    def test_retired_token_filter_has_fixed_storage_and_no_false_negatives(self):
        pool = SlotLeasePool(1, retired_token_filter_bytes=64 * 1024)
        first_token = None
        for index in range(200):
            token = str(uuid.UUID(int=index + 1))
            first_token = first_token or token
            grant = pool.acquire_token(token, owner=("reset", index))
            complete(pool, grant)
            pool.release(grant.slot, token=token)

        self.assertEqual(pool.retired_token_filter_bytes, 64 * 1024)
        with self.assertRaises(LeaseTokenRetiredError):
            pool.acquire_token(first_token, owner=("reset", 0))


if __name__ == "__main__":
    unittest.main()
