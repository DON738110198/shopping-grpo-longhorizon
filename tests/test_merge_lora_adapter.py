"""验证 LoRA 合并入口的纯配置逻辑，不需要本地下载模型。"""

import unittest

from scripts.merge_lora_adapter import (
    build_merge_manifest,
    choose_model_class,
    scale_lora_adapters,
)


class _Config:
    def __init__(self, model_type):
        self.model_type = model_type


class _Module:
    def __init__(self, scaling=None):
        if scaling is not None:
            self.scaling = scaling


class _Model:
    def __init__(self, modules):
        self._modules = modules

    def modules(self):
        return iter(self._modules)


class MergeLoraAdapterTest(unittest.TestCase):
    def test_qwen35_uses_multimodal_model_class(self):
        self.assertEqual(choose_model_class(_Config("qwen3_5"), "causal", "multimodal"), "multimodal")
        self.assertEqual(choose_model_class(_Config("qwen3"), "causal", "multimodal"), "causal")

    def test_merge_manifest_is_auditable(self):
        manifest = build_merge_manifest(
            base_model="Qwen/Qwen3.5-2B",
            adapter_path="checkpoints/sft",
            output_path="checkpoints/sft_merged",
            model_type="qwen3_5",
        )
        self.assertEqual(manifest["operation"], "peft_merge_and_unload")
        self.assertEqual(manifest["source"]["adapter"], "checkpoints/sft")
        self.assertEqual(manifest["output"], "checkpoints/sft_merged")
        self.assertEqual(manifest["adapter_scale"], 1.0)

    def test_scales_every_lora_adapter_entry(self):
        first = _Module({"default": 2.0})
        second = _Module({"default": 4.0, "alternate": 1.0})
        ignored = _Module()
        count = scale_lora_adapters(_Model([first, second, ignored]), 0.25)
        self.assertEqual(count, 3)
        self.assertEqual(first.scaling, {"default": 0.5})
        self.assertEqual(second.scaling, {"default": 1.0, "alternate": 0.25})

    def test_rejects_invalid_scale_or_missing_lora(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            scale_lora_adapters(_Model([_Module({"default": 2.0})]), 0)
        with self.assertRaisesRegex(ValueError, "no LoRA"):
            scale_lora_adapters(_Model([_Module()]), 0.5)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
