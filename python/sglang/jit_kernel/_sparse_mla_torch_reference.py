"""Pure-PyTorch reference implementation of V4-Flash sparse MLA decode.

This is "Oracle A" for the Phase 6 Triton sparse-MLA kernel. It mirrors the
existing torch fallback's contract (Oracle B at
`sglang.srt.layers.attention.debug_flash_mla_adapter.flash_mla_with_kvcache_torch`)
and the upstream reference at
`sglang.srt.flashmla_tests.ref.ref_sparse_attn_decode`:

  - plain ``logsumexp`` over masked logits (NOT ``logaddexp(logsumexp, sink)``);
  - sink-scaled output: ``O *= 1 / (1 + exp(sink - lse))``;
  - lonely-query correction: rows with no valid tokens get ``O = 0``,
    ``lse = +inf``;
  - returns ``(output: bf16 [B, s_q, h_q, d_v], lse: fp32 [B, h_q, s_q])``.

The signature matches the live call site at
``deepseek_v4_backend_radix.py:1074-1089`` so the oracle is a drop-in
replacement for any backend in ``flash_mla_with_kvcache_entrypoint``.

K/V cache byte layout (V4-Flash MODEL1_FP8Sparse, per-page split):

  Per page of ``P`` tokens, total ``P*584`` payload bytes:

    [0,    P*576) - per-token NoPE (FP8 E4M3, 448 elements) followed by
                    per-token RoPE (BF16, 64 elements = 128 bytes), tightly
                    packed: token i occupies bytes [i*576, (i+1)*576).
    [P*576, P*584) - per-token UE8M0 scale bytes (7 active + 1 pad):
                     token i occupies bytes [P*576 + i*8, P*576 + (i+1)*8).

  NoPE has 7 scale groups of 64 elements each. Group g of token i uses
  scale byte ``P*576 + i*8 + g``; the dequantized value is
  ``nope_fp32 * (2.0 ** (scale_byte - 127.0))``.

The ``[num_pages, P, 1, 584]`` tensor shape used by the live call site is
*shape metadata* for the kernel, NOT a memory-layout claim. The host-side
reference path calls ``quant_k_cache.view(num_blocks, -1)`` and indexes the
split layout directly; this oracle does the same via the canonical
``flashmla_tests.quant.dequantize_k_cache``.

Indices contract: ``indices`` carries flat token IDs
``page_id * P + row_in_page`` (NOT page IDs), with ``-1`` for invalid
slots. ``topk_length[b]`` is the number of valid leading slots in
``indices[b]``. The reference at ``flashmla_tests/ref.py:81-87`` clamps
``min(indices, 0)`` before ``index_select``, so positive out-of-range values
will fault both this oracle and the kernel; tests must respect that
contract.
"""
from __future__ import annotations

from typing import Any, Optional, Tuple

import torch

from sglang.srt.flashmla_tests import quant as _flashmla_quant


def _dequantize_kv_cache(k_cache: torch.Tensor) -> torch.Tensor:
    """Dequantize a ``[num_pages, P, 1, 584]`` uint8 cache view to bf16
    ``[num_pages, P, 1, 512]``.

    Wraps the canonical
    ``flashmla_tests.quant.dequantize_k_cache(..., MODEL1_FP8Sparse)``,
    which interprets the per-page split byte layout regardless of the
    intermediate ``[P, 1, 584]`` shape.
    """
    assert k_cache.ndim == 4, (
        f"k_cache must be [num_pages, P, 1, 584]; got {tuple(k_cache.shape)}"
    )
    assert k_cache.shape[2] == 1
    assert k_cache.shape[3] == 584
    return _flashmla_quant.dequantize_k_cache(
        k_cache.view(_flashmla_quant.FP8_DTYPE),
        _flashmla_quant.FP8KVCacheLayout.MODEL1_FP8Sparse,
    )


def _gather_kv_scope(
    blocked_k_fp32: torch.Tensor,
    indices: torch.Tensor,
    topk_length: Optional[torch.Tensor],
    d_qk: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Gather K rows for one (SWA or compressed) scope and produce
    ``(gathered_kv [B, s_q, topk, d_qk], invalid_mask [B, s_q, topk])``.

    Mirrors ``flashmla_tests/ref.py:78-94`` exactly.
    """
    B, s_q, topk = indices.shape
    indices_clamped = torch.clamp_min(indices, 0)
    gathered_kv = (
        blocked_k_fp32.view(-1, d_qk)
        .index_select(0, indices_clamped.view(-1))
        .view(B, s_q, topk, d_qk)
    )
    invalid_mask = indices == -1
    if topk_length is not None:
        invalid_mask = invalid_mask | (
            torch.arange(0, topk, device=invalid_mask.device).view(1, 1, topk)
            >= topk_length.view(B, 1, 1)
        )
    return gathered_kv, invalid_mask


def flash_mla_with_kvcache_torch_reference(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    *,
    indices: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
    attn_sink: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    head_dim_v: int = 512,
    extra_k_cache: Optional[torch.Tensor] = None,
    extra_indices_in_kvcache: Optional[torch.Tensor] = None,
    extra_topk_length: Optional[torch.Tensor] = None,
    # The following are accepted to mirror the live call site but must hold
    # the values listed below; the wrapper asserts them.
    block_table: Optional[torch.Tensor] = None,
    cache_seqlens: Optional[torch.Tensor] = None,
    tile_scheduler_metadata: Any = None,
    is_fp8_kvcache: bool = True,
    causal: bool = False,
    num_splits: Optional[int] = None,
    **_unused: Any,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pure-PyTorch sparse-MLA decode reference.

    Arguments mirror ``flash_mla_with_kvcache_torch`` at
    ``debug_flash_mla_adapter.py:48-65``. Returns
    ``(output bf16 [B, s_q, h_q, d_v], lse fp32 [B, h_q, s_q])``.

    Contract assertions are deliberately tight to fail loud on any caller
    drift; they match the live call-site values verified at
    ``deepseek_v4_backend_radix.py:1074-1089``.
    """
    assert block_table is None
    assert cache_seqlens is None
    assert is_fp8_kvcache is True
    assert not causal
    assert num_splits in (None, 1)
    assert indices is not None
    assert topk_length is not None
    assert attn_sink is not None
    assert q.ndim == 4, f"q must be [B, s_q, h_q, d_qk]; got {tuple(q.shape)}"
    B, s_q, h_q, d_qk = q.shape
    assert d_qk == 512
    assert head_dim_v == 512
    d_v = head_dim_v
    assert k_cache.ndim == 4 and k_cache.shape[2] == 1 and k_cache.shape[3] == 584
    assert k_cache.dtype == torch.uint8

    assert attn_sink is not None and attn_sink.shape == (h_q,)
    assert topk_length is not None and topk_length.ndim == 1
    assert topk_length.shape == (B,)
    assert topk_length.dtype == torch.int32
    assert indices.ndim == 3
    assert indices.shape[0] == B and indices.shape[1] == s_q
    assert indices.shape[-1] % 64 == 0
    assert indices.dtype == torch.int32

    if extra_k_cache is not None:
        assert extra_indices_in_kvcache is not None
        assert extra_topk_length is not None
        assert extra_k_cache.ndim == 4
        assert extra_k_cache.shape[2] == 1
        assert extra_k_cache.shape[3] == 584
        assert extra_k_cache.dtype == torch.uint8
        assert extra_indices_in_kvcache.ndim == 3
        assert extra_indices_in_kvcache.shape[0] == B
        assert extra_indices_in_kvcache.shape[1] == s_q
        assert extra_indices_in_kvcache.shape[-1] % 64 == 0
        assert extra_indices_in_kvcache.dtype == torch.int32
        assert extra_topk_length.shape == (B,)
        assert extra_topk_length.dtype == torch.int32
    else:
        assert extra_indices_in_kvcache is None
        assert extra_topk_length is None

    devices = {q.device, k_cache.device, indices.device, topk_length.device, attn_sink.device}
    if extra_k_cache is not None:
        devices.update({extra_k_cache.device, extra_indices_in_kvcache.device, extra_topk_length.device})
    assert len(devices) == 1, f"all tensors must share a device; got {devices}"

    if softmax_scale is None:
        softmax_scale = d_qk ** -0.5

    # Dequantize once per scope (host-side: lift bytes to bf16 KV rows).
    blocked_k = _dequantize_kv_cache(k_cache)  # [num_pages, P, 1, 512] bf16

    # Mirror ref_sparse_attn_decode's process_kv_scope.
    blocked_k_fp32 = blocked_k.view(-1, d_qk).float()
    gathered_kv, invalid_mask = _gather_kv_scope(
        blocked_k_fp32=blocked_k_fp32,
        indices=indices,
        topk_length=topk_length,
        d_qk=d_qk,
    )

    if extra_k_cache is not None:
        extra_blocked_k = _dequantize_kv_cache(extra_k_cache)
        extra_blocked_k_fp32 = extra_blocked_k.view(-1, d_qk).float()
        extra_gathered_kv, extra_invalid_mask = _gather_kv_scope(
            blocked_k_fp32=extra_blocked_k_fp32,
            indices=extra_indices_in_kvcache,
            topk_length=extra_topk_length,
            d_qk=d_qk,
        )
        gathered_kv = torch.cat([gathered_kv, extra_gathered_kv], dim=2)
        invalid_mask = torch.cat([invalid_mask, extra_invalid_mask], dim=2)

    # Squash the gathered KV per ref_sparse_attn_decode:108-109. Replace
    # NaN dequant artifacts with 0 (the kernel's masking will discard them
    # via the invalid_mask, but the float-multiply must not propagate NaN).
    gathered_kv = gathered_kv.view(B * s_q, -1, d_qk).float()
    gathered_kv[gathered_kv != gathered_kv] = 0.0

    q_fp32 = q.float().view(B * s_q, h_q, d_qk)
    attn_weight = q_fp32 @ gathered_kv.transpose(-1, -2)
    attn_weight *= softmax_scale
    attn_weight[
        invalid_mask.view(B * s_q, 1, -1).broadcast_to(
            B * s_q, h_q, invalid_mask.size(-1)
        )
    ] = float("-inf")

    lse = attn_weight.logsumexp(dim=-1)  # [B*s_q, h_q]
    attn_weight = torch.exp(attn_weight - lse.unsqueeze(-1))
    output = attn_weight @ gathered_kv[..., :d_v]  # [B*s_q, h_q, d_v]
    output = output.view(B, s_q, h_q, d_v)
    lse = lse.view(B, s_q, h_q)

    # Sink-scaled output (NOT applied to LSE).
    output = output * (
        1.0 / (1.0 + torch.exp(attn_sink.view(1, 1, h_q) - lse))
    ).unsqueeze(-1)

    # Lonely-query correction: zero output, +inf lse.
    lonely_q_mask = lse == float("-inf")
    output = output.masked_fill(
        lonely_q_mask.unsqueeze(-1).broadcast_to(B, s_q, h_q, d_v), 0.0
    )
    lse = lse.masked_fill(lonely_q_mask, float("+inf"))

    return output.to(torch.bfloat16), lse.transpose(1, 2)
