"""Utilities for turning VERL LoRA checkpoints into portable PEFT adapters."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _jsonable(value: Any) -> Any:
    if hasattr(value, "value"):
        return value.value
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def export_lora_adapter(
    checkpoint_hf_dir: str | Path,
    output_dir: str | Path,
    base_model_path: str | Path,
    lora_meta_path: str | Path | None = None,
) -> dict[str, Any]:
    """Extract LoRA tensors from a VERL HF export as a standard PEFT adapter."""
    from peft import LoraConfig, TaskType
    from safetensors import safe_open
    from safetensors.torch import save_file

    checkpoint_hf_dir = Path(checkpoint_hf_dir).resolve()
    output_dir = Path(output_dir).resolve()
    base_model_path = Path(base_model_path).resolve()
    index_path = checkpoint_hf_dir / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index["weight_map"]

    selected = {
        key: shard
        for key, shard in weight_map.items()
        if ".lora_A.default.weight" in key or ".lora_B.default.weight" in key
    }
    if not selected:
        raise ValueError(f"No LoRA tensors found in {index_path}")

    by_shard: dict[str, list[str]] = defaultdict(list)
    for key, shard in selected.items():
        by_shard[shard].append(key)

    adapter_state = {}
    target_modules = set()
    for shard, keys in sorted(by_shard.items()):
        with safe_open(checkpoint_hf_dir / shard, framework="pt", device="cpu") as handle:
            for key in keys:
                adapter_key = key.replace(".default.weight", ".weight")
                adapter_state[adapter_key] = handle.get_tensor(key).contiguous()
                target_modules.add(adapter_key.split(".")[-3])

    if lora_meta_path is None:
        lora_meta_path = checkpoint_hf_dir.parent / "lora_train_meta.json"
    meta = json.loads(Path(lora_meta_path).read_text(encoding="utf-8"))

    output_dir.mkdir(parents=True, exist_ok=True)
    config = LoraConfig(
        base_model_name_or_path=str(base_model_path),
        bias="none",
        inference_mode=True,
        lora_alpha=int(meta["lora_alpha"]),
        r=int(meta["r"]),
        target_modules=sorted(target_modules),
        task_type=TaskType.CAUSAL_LM,
    )
    config_dict = _jsonable(config.to_dict())
    (output_dir / "adapter_config.json").write_text(
        json.dumps(config_dict, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    adapter_path = output_dir / "adapter_model.safetensors"
    save_file(adapter_state, adapter_path)

    return {
        "adapter_path": str(adapter_path),
        "tensor_count": len(adapter_state),
        "target_modules": sorted(target_modules),
        "rank": int(meta["r"]),
        "alpha": int(meta["lora_alpha"]),
        "size_bytes": adapter_path.stat().st_size,
    }
