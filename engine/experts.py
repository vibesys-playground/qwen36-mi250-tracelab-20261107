"""The MoE experts implementation the baseline registers for ROCm.

Transformers 5.22 defaults to ``grouped_mm`` (``torch._grouped_mm``), which the
torch 2.9.1+rocm6.4 build cannot run, and its ``eager`` fallback synchronizes
with the host once per selected expert. This implementation uses only ordinary
matmuls:

- a few tokens (decode steps): Transformers' own ``batched_mm``, which gathers
  each token's selected expert weights and runs one batched matmul, with no
  host synchronization;
- many tokens (prefill chunks), where gathering weights per token would need
  gigabytes: tokens sorted by expert, one host synchronization for the group
  sizes, then one padded batched matmul per projection (per-expert loop only for
  extremely skewed routing, see ``PADDED_MAX_ROWS``).

``register`` adds it to Transformers' experts registry under ``NAME``.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F
from transformers.integrations.moe import ExpertsInterface, batched_mm_experts_forward

NAME = "baseline_matmul"
# Up to this many tokens per call, gather per-token weights (decode); above it, loop over experts.
BATCHED_TOKENS = 16
# Padded bmm computes num_experts * C rows (C = largest per-expert token count). If that exceeds
# this many rows (highly skewed routing), fall back to the per-expert loop for that call.
# 131072 rows = 256 experts x 512 tokens: <= 0.5 GB input buffer at hidden 2048 in BF16.
PADDED_MAX_ROWS = 131072


def _sorted_experts_forward(
    self: torch.nn.Module,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    num_tokens, top_k = top_k_index.shape
    expert_ids = top_k_index.reshape(-1)
    order = torch.argsort(expert_ids, stable=True)
    counts = torch.bincount(expert_ids, minlength=self.num_experts).tolist()  # the one host sync
    inputs = hidden_states[order // top_k]
    outputs = torch.empty_like(inputs)
    start = 0
    for expert, count in enumerate(counts):
        if count == 0:
            continue
        rows = slice(start, start + count)
        gate, up = F.linear(inputs[rows], self.gate_up_proj[expert]).chunk(2, dim=-1)
        outputs[rows] = F.linear(self.act_fn(gate) * up, self.down_proj[expert])
        start += count
    weighted = outputs.float() * top_k_weights.reshape(-1)[order, None].float()
    unsorted = torch.empty_like(weighted)
    unsorted[order] = weighted
    return unsorted.view(num_tokens, top_k, -1).sum(dim=1).to(hidden_states.dtype)


def _padded_experts_forward(
    self: torch.nn.Module,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    """Prefill experts as one padded batched matmul per projection (no per-expert launches).

    Same routing, BF16 weights and fp32 weighting/summation order as ``_sorted_experts_forward``;
    padding rows are zeros and are never read back.
    """
    num_tokens, top_k = top_k_index.shape
    num_experts = self.num_experts
    expert_ids = top_k_index.reshape(-1)
    order = torch.argsort(expert_ids, stable=True)
    counts = torch.bincount(expert_ids, minlength=num_experts)
    capacity = int(counts.max())  # the one host sync
    if capacity * num_experts > PADDED_MAX_ROWS:
        return _sorted_experts_forward(self, hidden_states, top_k_index, top_k_weights)
    sorted_ids = expert_ids[order]
    offsets = torch.cumsum(counts, dim=0) - counts
    slot = torch.arange(order.numel(), device=order.device) - offsets[sorted_ids]
    buf = hidden_states.new_zeros(num_experts, capacity, hidden_states.shape[-1])
    buf[sorted_ids, slot] = hidden_states[order // top_k]
    gate, up = torch.bmm(buf, self.gate_up_proj.transpose(1, 2)).chunk(2, dim=-1)
    padded = torch.bmm(self.act_fn(gate) * up, self.down_proj.transpose(1, 2))
    outputs = padded[sorted_ids, slot]
    weighted = outputs.float() * top_k_weights.reshape(-1)[order, None].float()
    unsorted = torch.empty_like(weighted)
    unsorted[order] = weighted
    return unsorted.view(num_tokens, top_k, -1).sum(dim=1).to(hidden_states.dtype)


def experts_forward(
    self: torch.nn.Module,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    """Batched gather for decode-sized inputs, a padded batched matmul for prefill-sized ones."""
    if hidden_states.shape[0] <= BATCHED_TOKENS:
        return batched_mm_experts_forward(self, hidden_states, top_k_index, top_k_weights)
    return _padded_experts_forward(self, hidden_states, top_k_index, top_k_weights)


def register() -> str:
    """Make ``NAME`` a valid ``experts_implementation``; return it."""
    ExpertsInterface.register(NAME, experts_forward)
    return NAME
