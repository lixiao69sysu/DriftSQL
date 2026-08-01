from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from driftsql.training import export_lora_adapter


class CheckpointExportTest(unittest.TestCase):
    def test_exports_standard_peft_adapter(self) -> None:
        # Some shared filesystems populate PEFT output directories asynchronously.
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            hf_dir = root / "checkpoint" / "huggingface"
            hf_dir.mkdir(parents=True)
            shard = "model-00001-of-00001.safetensors"
            prefix = "base_model.model.model.layers.0.self_attn.q_proj"
            state = {
                f"{prefix}.lora_A.default.weight": torch.ones(2, 4),
                f"{prefix}.lora_B.default.weight": torch.ones(4, 2),
                f"{prefix}.base_layer.weight": torch.zeros(4, 4),
            }
            save_file(state, hf_dir / shard)
            index = {"weight_map": {key: shard for key in state}}
            (hf_dir / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")
            (hf_dir.parent / "lora_train_meta.json").write_text(
                json.dumps({"r": 2, "lora_alpha": 4, "task_type": "CAUSAL_LM"}),
                encoding="utf-8",
            )

            output = root / "adapter"
            result = export_lora_adapter(hf_dir, output, root / "base-model")

            config = json.loads((output / "adapter_config.json").read_text(encoding="utf-8"))
            tensors = load_file(output / "adapter_model.safetensors")
            self.assertEqual(result["tensor_count"], 2)
            self.assertEqual(config["r"], 2)
            self.assertEqual(config["lora_alpha"], 4)
            self.assertEqual(config["target_modules"], ["q_proj"])
            self.assertEqual(len(tensors), 2)
            self.assertTrue(all(".default." not in key for key in tensors))


if __name__ == "__main__":
    unittest.main()
