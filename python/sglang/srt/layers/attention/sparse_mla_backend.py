"""Backend selector for sparse MLA decode (V4 / V4-Flash).

Phase 6.3: introduces an enum-based per-call selector for the
`flash_mla_with_kvcache(..., indices=...)` path. Replaces the legacy
string-keyed dispatch in `debug_flash_mla_adapter.flash_mla_with_kvcache_entrypoint`
with a five-valued enum:

    TORCH         — pure-torch reference (debug_flash_mla_adapter)
    TILELANG      — HIP gfx95 tilelang kernel
    COMPARISON    — dual-run (TORCH + FLASH_MLA) with tolerance checks
    TRITON_SM120  — sm_120 desktop Blackwell Triton kernel (Phase 6.2)
    FLASH_MLA     — upstream `flash_mla.flash_mla_with_kvcache`

Auto-dispatch on sm_120 routes SWA layers (extra_k_cache is None) to
TRITON_SM120 and compressed C4/C128 layers (extra_k_cache is not None)
to TORCH until Phase 6.7 lands the compressed kernel.

`SGLANG_FLASHMLA_BACKEND` (new) and `SGLANG_HACK_FLASHMLA_BACKEND`
(legacy) are recognised. Both default-only env reads are ignored — the
selector uses `is_set()` to distinguish "user set the env" from "env
has its declared default" (legacy declares default `"kernel"`, which
must NOT silently win over auto-dispatch).
"""
from __future__ import annotations

from enum import Enum, auto
from typing import Any, Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.utils import is_hip


class SparseMLADecodeBackend(Enum):
    TORCH = auto()
    TILELANG = auto()
    COMPARISON = auto()
    TRITON_SM120 = auto()
    FLASH_MLA = auto()


_LEGACY_STRING_MAP = {
    "torch": SparseMLADecodeBackend.TORCH,
    "tilelang": SparseMLADecodeBackend.TILELANG,
    "comparison": SparseMLADecodeBackend.COMPARISON,
    "triton_sm120": SparseMLADecodeBackend.TRITON_SM120,
    "kernel": SparseMLADecodeBackend.FLASH_MLA,
}


def parse_backend_string(s: str) -> SparseMLADecodeBackend:
    if s not in _LEGACY_STRING_MAP:
        raise ValueError(
            f"unknown sparse-MLA backend string {s!r}; "
            f"known: {sorted(_LEGACY_STRING_MAP)}"
        )
    return _LEGACY_STRING_MAP[s]


def _explicit_override() -> Optional[SparseMLADecodeBackend]:
    """Return the explicitly-configured backend, or None if unset.

    Distinguishes "user set the env var" from "env var has its declared
    default" via `is_set()`. The legacy var declares default `"kernel"`,
    so a naive `.get()` would always look set.
    """
    if envs.SGLANG_FLASHMLA_BACKEND.is_set():
        return parse_backend_string(envs.SGLANG_FLASHMLA_BACKEND.get())
    if envs.SGLANG_HACK_FLASHMLA_BACKEND.is_set():
        return parse_backend_string(envs.SGLANG_HACK_FLASHMLA_BACKEND.get())
    return None


def _is_sm120() -> bool:
    if is_hip() or not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major == 12


def _triton_supported(
    *,
    q,
    k_cache,
    extra_k_cache,
    extra_indices_in_kvcache,
    extra_topk_length,
    num_splits,
    block_table,
    cache_seqlens,
    causal,
    is_fp8_kvcache,
    attn_sink,
    topk_length,
    indices,
    head_dim_v,
) -> bool:
    """Full Phase 6.2 Triton-wrapper precondition check.

    Mirrors the asserts in `flash_mla_with_kvcache_triton_sm120` so the
    dispatcher routes any unsupported call elsewhere instead of letting
    it land on a hard assert. Wrapper still asserts as defence-in-depth.
    """
    if q is None or q.ndim != 4:
        return False
    if k_cache is None or k_cache.ndim != 4:
        return False
    # Compressed scope: all three must be present together (Phase 6.7) or
    # all three None (Phase 6.2 SWA-only).
    if extra_k_cache is None:
        if extra_indices_in_kvcache is not None or extra_topk_length is not None:
            return False
    else:
        if extra_indices_in_kvcache is None or extra_topk_length is None:
            return False
        if extra_k_cache.ndim != 4:
            return False
        if extra_k_cache.shape[2] != 1 or extra_k_cache.shape[3] != 584:
            return False
        if extra_k_cache.dtype != torch.uint8:
            return False
        e_page_stride = extra_k_cache.stride(0) * extra_k_cache.element_size()
        P_extra = extra_k_cache.shape[1]
        if e_page_stride % 576 != 0 or e_page_stride < P_extra * 584:
            return False
        if extra_indices_in_kvcache.ndim != 3:
            return False
        if extra_indices_in_kvcache.shape[0] != q.shape[0]:
            return False
        if extra_indices_in_kvcache.shape[1] != 1:
            return False
        if extra_indices_in_kvcache.shape[-1] % 64 != 0:
            return False
        if extra_indices_in_kvcache.dtype != torch.int32:
            return False
        if extra_indices_in_kvcache.stride(2) != 1:
            return False
        if extra_topk_length.ndim != 1:
            return False
        if extra_topk_length.shape != (q.shape[0],):
            return False
        if extra_topk_length.dtype != torch.int32:
            return False
        if not (extra_k_cache.is_cuda and extra_indices_in_kvcache.is_cuda
                and extra_topk_length.is_cuda):
            return False
        if not (extra_k_cache.device == q.device
                and extra_indices_in_kvcache.device == q.device
                and extra_topk_length.device == q.device):
            return False
    if not (q.shape[1] == 1 and q.shape[2] % 16 == 0):
        return False
    if q.shape[3] != 512 or head_dim_v != 512:
        return False
    if q.dtype != torch.bfloat16:
        return False
    if q.stride(3) != 1:
        return False
    if k_cache.dtype != torch.uint8:
        return False
    if k_cache.shape[2] != 1 or k_cache.shape[3] != 584:
        return False
    P = k_cache.shape[1]
    page_byte_stride = k_cache.stride(0) * k_cache.element_size()
    if page_byte_stride % 576 != 0:
        return False
    if page_byte_stride < P * 584:
        return False
    if is_fp8_kvcache is not True:
        return False
    # Phase 7.3: supported set is {1, 2, 4, 8}. Larger values would cause
    # the merge kernel to materialize an unmanageable [NUM_SPLITS, BLOCK_M,
    # BLOCK_DV] tile; redesign to stream-over-splits is required before
    # accepting a wider domain.
    if num_splits is not None and num_splits not in (1, 2, 4, 8):
        return False
    if block_table is not None or cache_seqlens is not None:
        return False
    if causal:
        return False
    if attn_sink is None or attn_sink.shape != (q.shape[2],):
        return False
    if topk_length is None or topk_length.ndim != 1:
        return False
    if topk_length.shape != (q.shape[0],) or topk_length.dtype != torch.int32:
        return False
    if indices is None or indices.ndim != 3:
        return False
    if indices.shape[0] != q.shape[0] or indices.shape[1] != 1:
        return False
    if indices.shape[-1] % 64 != 0 or indices.dtype != torch.int32:
        return False
    if indices.stride(2) != 1:
        return False
    if not (q.is_cuda and k_cache.is_cuda and indices.is_cuda
            and topk_length.is_cuda and attn_sink.is_cuda):
        return False
    if not (q.device == k_cache.device == indices.device
            == topk_length.device == attn_sink.device):
        return False
    return True


def _torch_supported(
    *,
    num_splits,
    block_table,
    cache_seqlens,
    causal,
    is_fp8_kvcache,
) -> bool:
    """Whether the torch fallback handles this call without silent error.

    The torch adapter accepts but ignores `causal`, `num_splits>1`,
    `is_fp8_kvcache=False`, `block_table`, `cache_seqlens`. Routing
    those there would silently return wrong semantics; route to
    NotImplementedError instead. Compressed (`extra_k_cache != None`)
    and `s_q > 1` ARE handled correctly by the torch path.
    """
    if num_splits not in (None, 1):
        return False
    if block_table is not None or cache_seqlens is not None:
        return False
    if causal:
        return False
    if is_fp8_kvcache is not True:
        return False
    return True


def get_sparse_mla_decode_backend(
    *,
    q,
    k_cache,
    extra_k_cache=None,
    extra_indices_in_kvcache=None,
    extra_topk_length=None,
    num_splits=None,
    block_table=None,
    cache_seqlens=None,
    causal: bool = False,
    is_fp8_kvcache: bool = True,
    softmax_scale=None,
    attn_sink=None,
    topk_length=None,
    indices=None,
    head_dim_v: int = 512,
    **_unused: Any,
) -> SparseMLADecodeBackend:
    """Per-call backend selection.

    Order of precedence:
      1. Explicit env override (new or legacy var, via `is_set()`).
      2. Auto-dispatch by GPU capability:
         - HIP → TORCH
         - sm_120: TRITON_SM120 if all preconditions pass; otherwise
           TORCH if torch-supported; otherwise NotImplementedError.
         - sm_90 / sm_100 → FLASH_MLA
         - other CUDA / no CUDA → NotImplementedError
    """
    explicit = _explicit_override()
    if explicit is not None:
        return explicit

    if is_hip():
        return SparseMLADecodeBackend.TORCH

    if _is_sm120():
        if _triton_supported(
            q=q, k_cache=k_cache, extra_k_cache=extra_k_cache,
            extra_indices_in_kvcache=extra_indices_in_kvcache,
            extra_topk_length=extra_topk_length,
            num_splits=num_splits, block_table=block_table,
            cache_seqlens=cache_seqlens, causal=causal,
            is_fp8_kvcache=is_fp8_kvcache, attn_sink=attn_sink,
            topk_length=topk_length, indices=indices,
            head_dim_v=head_dim_v,
        ):
            return SparseMLADecodeBackend.TRITON_SM120
        if not _torch_supported(
            num_splits=num_splits, block_table=block_table,
            cache_seqlens=cache_seqlens, causal=causal,
            is_fp8_kvcache=is_fp8_kvcache,
        ):
            raise NotImplementedError(
                "sparse-MLA decode call has features unsupported by both "
                "the sm_120 Triton path and the torch fallback"
            )
        return SparseMLADecodeBackend.TORCH

    if not torch.cuda.is_available():
        raise NotImplementedError("sparse-MLA decode requires CUDA")

    major, _ = torch.cuda.get_device_capability()
    if major in (9, 10):
        return SparseMLADecodeBackend.FLASH_MLA

    raise NotImplementedError(
        f"sparse-MLA decode has no auto-dispatch for compute capability {major}.x"
    )


def need_flashmla_metadata() -> bool:
    """True iff any forward path on this device might need
    `flash_mla.get_mla_metadata()` to be called.

    Phase 6.3: gates `init_flashmla_related()` / `_create_flashmla_metadata()`
    so sm_120 deployments don't import flash_mla just to build a metadata
    object that no per-call dispatch will ever consume.

    Over-approximates: returns True if explicit override selects a
    FlashMLA-using backend, OR if no override is set and the device
    auto-dispatches to FLASH_MLA / COMPARISON.
    """
    explicit = _explicit_override()
    if explicit is not None:
        return explicit in (
            SparseMLADecodeBackend.FLASH_MLA,
            SparseMLADecodeBackend.COMPARISON,
        )
    if is_hip() or not torch.cuda.is_available():
        return False
    if _is_sm120():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major in (9, 10)
