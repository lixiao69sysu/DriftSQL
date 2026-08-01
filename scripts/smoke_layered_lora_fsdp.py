#!/usr/bin/env python3
"""Two-GPU smoke test for layered LoRA collection under FSDP1."""

from __future__ import annotations

import json
import os

import torch
import torch.distributed
from peft import LoraConfig, get_peft_model
from torch.distributed import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from transformers import AutoModelForCausalLM, Qwen2Config
from verl.utils.fsdp_utils import get_fsdp_wrap_policy, layered_summon_lora_params


def main() -> None:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group("nccl")
    mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("fsdp",))

    config = Qwen2Config(
        vocab_size=128,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        tie_word_embeddings=False,
    )
    model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16).cuda()
    model = get_peft_model(
        model,
        LoraConfig(
            r=8,
            lora_alpha=8,
            target_modules="all-linear",
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.to(torch.bfloat16)

    mixed_precision = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        buffer_dtype=torch.float32,
    )
    model = FSDP(
        model,
        auto_wrap_policy=get_fsdp_wrap_policy(model, is_lora=True),
        device_id=local_rank,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=mixed_precision,
        device_mesh=mesh,
        use_orig_params=False,
    )

    # FSDP1 initializes its unsharded parameter buffers lazily on the first
    # forward. The production actor has already run this path before syncing
    # LoRA weights, so mirror that lifecycle in the smoke test.
    with torch.no_grad():
        model(input_ids=torch.tensor([[1, 2, 3, 4]], device=local_rank))

    tensors = layered_summon_lora_params(model)
    bad_shapes = {key: list(value.shape) for key, value in tensors.items() if value.ndim != 2}
    if not tensors or bad_shapes:
        raise RuntimeError(f"Invalid layered LoRA result: count={len(tensors)}, bad_shapes={bad_shapes}")
    if rank == 0:
        print(
            json.dumps(
                {
                    "world_size": world_size,
                    "tensor_count": len(tensors),
                    "all_rank_two": True,
                    "sample_shapes": {key: list(value.shape) for key, value in list(tensors.items())[:4]},
                },
                indent=2,
            )
        )
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
