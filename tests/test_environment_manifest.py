import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.check_grpo_runtime import CURRENT_RUNTIME_FILES
from shopping_grpo.environment.manifest import (
    MANIFEST_VERSION,
    shopsimulator_source_commit,
    validate_manifest,
)


class EnvironmentManifestTest(unittest.TestCase):
    def test_frozen_runtime_hashes_match_embedded_sources(self):
        root = Path(__file__).resolve().parents[1]
        manifest = json.loads(
            (root / "data/environment.json").read_text(encoding="utf-8")
        )

        for name, relative_path in CURRENT_RUNTIME_FILES.items():
            source = (root / relative_path).read_bytes().replace(b"\r\n", b"\n")
            actual = hashlib.sha256(source).hexdigest()
            self.assertEqual(actual, manifest["runtime_files_sha256"][name], name)

    def test_current_environment_contract_is_validated(self):
        manifest = {
            "manifest_version": MANIFEST_VERSION,
            "environment_version": "shopsimulator-environment-v2.1",
            "shopsimulator_commit": "a" * 40,
            "product_data_sha256": "c" * 64,
            "search": {
                "version": "shopsimulator-multifield-bm25-v2",
                "page_size": 20,
            },
            "reward": {"version": "shopsimulator-reward-v3"},
            "observation_version": "shopping-observation-v2",
            "tool_version": "shopping-tools-v2",
            "max_steps": 35,
            "seed": 20260726,
        }
        self.assertIs(validate_manifest(manifest), manifest)

    def test_page_size_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "missing"):
            validate_manifest({})

    def test_current_environment_requires_reward_v3(self):
        manifest = {
            "manifest_version": MANIFEST_VERSION,
            "environment_version": "shopsimulator-environment-v2.1",
            "shopsimulator_commit": "a" * 40,
            "product_data_sha256": "c" * 64,
            "search": {
                "version": "shopsimulator-multifield-bm25-v2",
                "page_size": 20,
            },
            "reward": {"version": "shopsimulator-reward-v3"},
            "observation_version": "shopping-observation-v2",
            "tool_version": "shopping-tools-v2",
            "max_steps": 35,
            "seed": 20260726,
        }
        self.assertIs(validate_manifest(manifest), manifest)
        manifest["reward"] = {"version": "unsupported-reward"}
        with self.assertRaisesRegex(ValueError, "requires shopsimulator-reward-v3"):
            validate_manifest(manifest)

    def test_wrong_tool_contract_is_rejected(self):
        manifest = {
            "manifest_version": MANIFEST_VERSION,
            "shopsimulator_commit": "a" * 40,
            "product_data_sha256": "c" * 64,
            "search": {
                "version": "shopsimulator-multifield-bm25-v2",
                "page_size": 20,
            },
            "reward": {"version": "shopsimulator-reward-v3"},
            "observation_version": "shopping-observation-v2",
            "tool_version": "unsupported-tools",
            "max_steps": 35,
            "seed": 20260726,
        }
        with self.assertRaisesRegex(ValueError, "Tool v2"):
            validate_manifest(manifest)

    def test_embedded_shopsimulator_commit_is_read_without_nested_git(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "EMBEDDED_SOURCE.json").write_text(
                json.dumps({"source_commit": "e" * 40}),
                encoding="utf-8",
            )
            self.assertEqual(shopsimulator_source_commit(root), "e" * 40)


if __name__ == "__main__":
    unittest.main()
