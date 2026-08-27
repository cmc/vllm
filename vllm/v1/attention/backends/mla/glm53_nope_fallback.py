# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness-first BF16 sparse-MLA fallback for GLM-5.3 on SM120.

Benefit: this preserves the model's 512-wide absorbed NoPE latent layout with
no RoPE tail and its full 2048-token sparse selection without requiring an
unavailable fused kernel.

Cost: BF16 KV storage, indexed gathers, temporary tensors, Python chunking,
and torch.bmm are less memory- and throughput-efficient than packed FP8
FlashInfer sparse MLA. This is a compatibility reference, not the preferred
long-term performance path.
"""

import torch


def bf16_nope_sparse_attention(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    physical_indices: torch.Tensor,
    valid_counts: torch.Tensor,
    scale: float,
    *,
    chunk_size: int = 128,
) -> torch.Tensor:
    """Attend to per-token physical KV rows without a fused SM120 NoPE kernel."""
    if query.dtype != torch.bfloat16 or kv_cache.dtype != torch.bfloat16:
        raise TypeError("GLM-5.3 NoPE fallback requires BF16 query and KV cache")
    if query.ndim != 3 or kv_cache.ndim != 3 or physical_indices.ndim != 2:
        raise ValueError("expected query [T,H,D], cache [B,S,D], indices [T,K]")
    if query.shape[0] != physical_indices.shape[0]:
        raise ValueError("query and physical index token counts must match")
    if valid_counts.shape != (query.shape[0],):
        raise ValueError("valid_counts must contain one entry per query token")
    if query.shape[-1] != kv_cache.shape[-1]:
        raise ValueError("query and KV cache latent dimensions must match")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    num_tokens = query.shape[0]
    topk_width = physical_indices.shape[1]
    flat_cache = kv_cache.reshape(-1, kv_cache.shape[-1])
    offsets = torch.arange(topk_width, device=query.device)
    output = torch.empty_like(query)

    for start in range(0, num_tokens, chunk_size):
        end = min(start + chunk_size, num_tokens)
        chunk_query = query[start:end]
        chunk_indices = physical_indices[start:end]
        chunk_counts = valid_counts[start:end]
        valid = offsets[None, :] < chunk_counts[:, None]
        nonempty = chunk_counts > 0
        safe_indices = torch.where(valid, chunk_indices, 0).long()
        keys = flat_cache[safe_indices]

        scores = torch.bmm(
            chunk_query,
            keys.transpose(1, 2),
            out_dtype=torch.float32,
        )
        scores.mul_(scale)
        scores.masked_fill_(~valid[:, None, :], float("-inf"))
        scores = torch.where(
            nonempty[:, None, None],
            scores,
            torch.zeros_like(scores),
        )
        probabilities = torch.softmax(scores, dim=-1).to(torch.bfloat16)
        chunk_output = torch.bmm(
            probabilities,
            keys,
            out_dtype=torch.float32,
        ).to(query.dtype)
        chunk_output.masked_fill_(~nonempty[:, None, None], 0.0)
        output[start:end].copy_(chunk_output)

    return output
