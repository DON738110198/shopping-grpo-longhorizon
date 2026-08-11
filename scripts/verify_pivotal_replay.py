#!/usr/bin/env python3
"""Live-verify public pivotal prefixes against the frozen ShopSimulator."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from shopping_grpo.environment.client import ShopAgentEnv
from shopping_grpo.environment.manifest import sha256_file, validate_manifest
from shopping_grpo.training.grpo.pivotal_states import (
    validate_event_training_contract,
)
from shopping_grpo.training.grpo.replay import verify_replay_with_factory


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--environment-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:5700")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--max-branches", type=int, default=100)
    return parser.parse_args()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_validated_manifest(path: Path) -> dict[str, object]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        return validate_manifest(manifest)
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid environment manifest: {exc}") from exc


def _candidates(path: Path):
    seen = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            for trajectory_index, trajectory in enumerate(record.get("trajectories", [])):
                events = trajectory.get("decision_trace", trajectory.get("action_trace", []))
                for event_index, event in enumerate(events):
                    valid, reason = validate_event_training_contract(trajectory, event)
                    if not valid:
                        continue
                    key = (
                        str(event["branch_uid"]),
                        int(event["prefix_action_count"]),
                    )
                    if key in seen:
                        continue
                    seen.add(key)
                    yield {
                        "trajectory": trajectory,
                        "event": event,
                        "source": {
                            "line": line_number,
                            "trajectory_index": trajectory_index,
                            "event_index": event_index,
                            "contract_reason": reason,
                        },
                    }


def main():
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    manifest_path = args.environment_manifest.expanduser().resolve()
    if not input_path.is_file():
        raise SystemExit(f"sampling audit does not exist: {input_path}")
    if not manifest_path.is_file():
        raise SystemExit(f"environment manifest does not exist: {manifest_path}")
    if args.max_branches < 1:
        raise SystemExit("--max-branches must be positive")
    try:
        manifest = _load_validated_manifest(manifest_path)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    manifest_hash = sha256_file(manifest_path)
    required_environment_version = str(
        manifest.get("environment_version", "shopsimulator-environment-v2.1")
    )
    required_max_steps = int(manifest["max_steps"])
    results = []
    for candidate in _candidates(input_path):
        if len(results) >= args.max_branches:
            break
        trajectory = candidate["trajectory"]
        event = candidate["event"]
        verification = verify_replay_with_factory(
            trajectory,
            environment_manifest_sha256=manifest_hash,
            required_environment_version=required_environment_version,
            required_max_steps=required_max_steps,
            env_factory=lambda: ShopAgentEnv(
                base_url=args.base_url,
                timeout=args.timeout,
            ),
            prefix_action_count=int(event["prefix_action_count"]),
        )
        expected_state_id = str(event["replay_state_id"])
        if (
            verification.get("verified")
            and verification.get("recomputed_replay_state_id") != expected_state_id
        ):
            verification.update(
                {
                    "verified": False,
                    "stage": "state_identity",
                    "reason": "recomputed_replay_state_id_mismatch",
                }
            )
        results.append(
            {
                "task_id": int(trajectory["task_id"]),
                "branch_uid": str(event["branch_uid"]),
                "expected_replay_state_id": expected_state_id,
                "prefix_action_count": int(event["prefix_action_count"]),
                "source": candidate["source"],
                "verification": verification,
            }
        )
    verified = sum(bool(result["verification"]["verified"]) for result in results)
    output = {
        "schema_version": "shopping-pivotal-live-replay-audit-v1",
        "provenance": {
            "input": str(input_path),
            "input_sha256": _file_sha256(input_path),
            "environment_manifest": str(manifest_path),
            "environment_manifest_sha256": manifest_hash,
            "required_environment_version": required_environment_version,
            "required_max_steps": required_max_steps,
            "base_url": str(args.base_url),
        },
        "aggregate": {
            "attempted_branches": len(results),
            "verified_branches": verified,
            "verification_rate": verified / len(results) if results else 0.0,
        },
        "safety": {
            "uses_hidden_goal": False,
            "raw_environment_payloads_saved": False,
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output["aggregate"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
