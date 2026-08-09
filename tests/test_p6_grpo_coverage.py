from __future__ import annotations

import pytest
import numpy as np
import torch

from driftsql.training.grpo_coverage import coverage_plan, unique_first_order
from verl.trainer.ppo.core_algos import agg_loss, compute_grpo_outcome_advantage


def test_unique_first_order_maximizes_independent_task_prefix() -> None:
    instance_ids = ["a", "a", "b", "c", "c", "d"]
    order = unique_first_order(instance_ids, seed=7)
    reordered = [instance_ids[index] for index in order]
    assert sorted(order) == list(range(len(instance_ids)))
    assert len(set(reordered[:4])) == 4
    assert set(reordered[:4]) == {"a", "b", "c", "d"}


def test_coverage_plan_matches_verl_drop_last_semantics() -> None:
    complete = coverage_plan(1000, 8, 125)
    assert complete.steps_per_epoch == 125
    assert complete.planned_prompt_slots == 1000
    assert complete.dropped_rows_per_epoch == 0
    assert complete.full_row_coverage

    partial = coverage_plan(1000, 8, 50)
    assert partial.planned_prompt_slots == 400
    assert not partial.full_row_coverage

    dropped = coverage_plan(1001, 8, 126)
    assert dropped.dropped_rows_per_epoch == 1
    assert not dropped.full_row_coverage


@pytest.mark.parametrize("rows,batch,steps", [(0, 8, 1), (8, 0, 1), (8, 8, 0)])
def test_coverage_plan_rejects_non_positive_inputs(rows: int, batch: int, steps: int) -> None:
    with pytest.raises(ValueError):
        coverage_plan(rows, batch, steps)


def test_grpo_episode_advantage_is_broadcast_to_every_assistant_token() -> None:
    rewards = torch.tensor(
        [
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    response_mask = torch.tensor(
        [
            [1.0, 1.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 0.0],
        ]
    )
    advantages, _ = compute_grpo_outcome_advantage(
        token_level_rewards=rewards,
        response_mask=response_mask,
        index=np.array(["prompt-a", "prompt-a"]),
    )
    assert torch.allclose(advantages[0, response_mask[0].bool()], advantages[0, 0].expand(3))
    assert advantages[0, 2] == 0
    assert advantages[1, 1:].abs().sum() == 0


def test_sequence_mean_token_mean_gives_each_episode_equal_weight() -> None:
    losses = torch.tensor([[2.0, 2.0, 2.0], [4.0, 0.0, 0.0]])
    mask = torch.tensor([[1.0, 1.0, 1.0], [1.0, 0.0, 0.0]])
    result = agg_loss(
        loss_mat=losses,
        loss_mask=mask,
        loss_agg_mode="seq-mean-token-mean",
        global_batch_size=2,
    )
    assert result.item() == pytest.approx(3.0)
