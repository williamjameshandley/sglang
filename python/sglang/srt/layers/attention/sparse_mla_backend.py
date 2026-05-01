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


def _triton_supported_reason(
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
) -> Optional[str]:
    """Returns None if Triton kernel preconditions hold, else a string
    naming the first failed precondition (for PCG diagnostic logging)."""
    if q is None: return "q is None"
    if q.ndim != 4: return f"q.ndim={q.ndim} != 4"
    if k_cache is None: return "k_cache is None"
    if k_cache.ndim != 4: return f"k_cache.ndim={k_cache.ndim} != 4"
    if extra_k_cache is None:
        if extra_indices_in_kvcache is not None:
            return "extra_k_cache is None but extra_indices_in_kvcache is not None"
        if extra_topk_length is not None:
            return "extra_k_cache is None but extra_topk_length is not None"
    else:
        if extra_indices_in_kvcache is None: return "extra_indices_in_kvcache is None with extra_k_cache present"
        if extra_topk_length is None: return "extra_topk_length is None with extra_k_cache present"
        if extra_k_cache.ndim != 4: return f"extra_k_cache.ndim={extra_k_cache.ndim} != 4"
        if extra_k_cache.shape[2] != 1: return f"extra_k_cache.shape[2]={extra_k_cache.shape[2]} != 1"
        if extra_k_cache.shape[3] != 584: return f"extra_k_cache.shape[3]={extra_k_cache.shape[3]} != 584"
        if extra_k_cache.dtype != torch.uint8: return f"extra_k_cache.dtype={extra_k_cache.dtype} != uint8"
        e_page_stride = extra_k_cache.stride(0) * extra_k_cache.element_size()
        P_extra = extra_k_cache.shape[1]
        if e_page_stride % 576 != 0: return f"extra_k_cache page_byte_stride={e_page_stride} % 576 != 0"
        if e_page_stride < P_extra * 584: return f"extra_k_cache page_byte_stride={e_page_stride} < P_extra*584={P_extra*584}"
        if extra_indices_in_kvcache.ndim != 3: return f"extra_indices_in_kvcache.ndim={extra_indices_in_kvcache.ndim} != 3"
        if extra_indices_in_kvcache.shape[0] != q.shape[0]: return f"extra_indices_in_kvcache.shape[0]={extra_indices_in_kvcache.shape[0]} != q.shape[0]={q.shape[0]}"
        if extra_indices_in_kvcache.shape[1] != 1: return f"extra_indices_in_kvcache.shape[1]={extra_indices_in_kvcache.shape[1]} != 1"
        if extra_indices_in_kvcache.shape[-1] % 64 != 0: return f"extra_indices_in_kvcache.shape[-1]={extra_indices_in_kvcache.shape[-1]} % 64 != 0"
        if extra_indices_in_kvcache.dtype != torch.int32: return f"extra_indices_in_kvcache.dtype={extra_indices_in_kvcache.dtype} != int32"
        if extra_indices_in_kvcache.stride(2) != 1: return f"extra_indices_in_kvcache.stride(2)={extra_indices_in_kvcache.stride(2)} != 1"
        if extra_topk_length.ndim != 1: return f"extra_topk_length.ndim={extra_topk_length.ndim} != 1"
        if extra_topk_length.shape != (q.shape[0],): return f"extra_topk_length.shape={tuple(extra_topk_length.shape)} != ({q.shape[0]},)"
        if extra_topk_length.dtype != torch.int32: return f"extra_topk_length.dtype={extra_topk_length.dtype} != int32"
        if not extra_k_cache.is_cuda: return "extra_k_cache not CUDA"
        if not extra_indices_in_kvcache.is_cuda: return "extra_indices_in_kvcache not CUDA"
        if not extra_topk_length.is_cuda: return "extra_topk_length not CUDA"
        if extra_k_cache.device != q.device: return f"extra_k_cache.device={extra_k_cache.device} != q.device={q.device}"
        if extra_indices_in_kvcache.device != q.device: return f"extra_indices_in_kvcache.device != q.device"
        if extra_topk_length.device != q.device: return f"extra_topk_length.device != q.device"
    if q.shape[1] != 1: return f"q.shape[1]={q.shape[1]} != 1 (s_q must be 1; Contract A flattens)"
    if q.shape[2] % 16 != 0: return f"q.shape[2]={q.shape[2]} % 16 != 0 (h_q)"
    if q.shape[3] != 512: return f"q.shape[3]={q.shape[3]} != 512 (head_dim_qk)"
    if head_dim_v != 512: return f"head_dim_v={head_dim_v} != 512"
    if q.dtype != torch.bfloat16: return f"q.dtype={q.dtype} != bfloat16"
    if q.stride(3) != 1: return f"q.stride(3)={q.stride(3)} != 1"
    if k_cache.dtype != torch.uint8: return f"k_cache.dtype={k_cache.dtype} != uint8"
    if k_cache.shape[2] != 1: return f"k_cache.shape[2]={k_cache.shape[2]} != 1"
    if k_cache.shape[3] != 584: return f"k_cache.shape[3]={k_cache.shape[3]} != 584"
    P = k_cache.shape[1]
    page_byte_stride = k_cache.stride(0) * k_cache.element_size()
    if page_byte_stride % 576 != 0: return f"k_cache page_byte_stride={page_byte_stride} % 576 != 0"
    if page_byte_stride < P * 584: return f"k_cache page_byte_stride={page_byte_stride} < P*584={P*584}"
    if is_fp8_kvcache is not True: return f"is_fp8_kvcache={is_fp8_kvcache} (must be True)"
    if num_splits is not None and num_splits not in (1, 2, 4, 8):
        return f"num_splits={num_splits} not in (None, 1, 2, 4, 8)"
    if block_table is not None: return "block_table is not None (sparse-MLA path requires None)"
    if cache_seqlens is not None: return "cache_seqlens is not None"
    if causal: return "causal=True (sparse-MLA path requires causal=False)"
    if attn_sink is None: return "attn_sink is None"
    if attn_sink.shape != (q.shape[2],): return f"attn_sink.shape={tuple(attn_sink.shape)} != ({q.shape[2]},)"
    if topk_length is None: return "topk_length is None"
    if topk_length.ndim != 1: return f"topk_length.ndim={topk_length.ndim} != 1"
    if topk_length.shape != (q.shape[0],): return f"topk_length.shape={tuple(topk_length.shape)} != ({q.shape[0]},)"
    if topk_length.dtype != torch.int32: return f"topk_length.dtype={topk_length.dtype} != int32"
    if indices is None: return "indices is None"
    if indices.ndim != 3: return f"indices.ndim={indices.ndim} != 3"
    if indices.shape[0] != q.shape[0]: return f"indices.shape[0]={indices.shape[0]} != q.shape[0]={q.shape[0]}"
    if indices.shape[1] != 1: return f"indices.shape[1]={indices.shape[1]} != 1"
    if indices.shape[-1] % 64 != 0: return f"indices.shape[-1]={indices.shape[-1]} % 64 != 0"
    if indices.dtype != torch.int32: return f"indices.dtype={indices.dtype} != int32"
    if indices.stride(2) != 1: return f"indices.stride(2)={indices.stride(2)} != 1"
    if not q.is_cuda: return "q not CUDA"
    if not k_cache.is_cuda: return "k_cache not CUDA"
    if not indices.is_cuda: return "indices not CUDA"
    if not topk_length.is_cuda: return "topk_length not CUDA"
    if not attn_sink.is_cuda: return "attn_sink not CUDA"
    if q.device != k_cache.device: return f"q.device={q.device} != k_cache.device={k_cache.device}"
    if q.device != indices.device: return f"q.device != indices.device"
    if q.device != topk_length.device: return f"q.device != topk_length.device"
    if q.device != attn_sink.device: return f"q.device != attn_sink.device"
    return None


def _triton_supported(**kwargs) -> bool:
    """Bool wrapper around `_triton_supported_reason` for callers that
    don't need diagnostic strings."""
    return _triton_supported_reason(**kwargs) is None


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
        triton_reason = _triton_supported_reason(
            q=q, k_cache=k_cache, extra_k_cache=extra_k_cache,
            extra_indices_in_kvcache=extra_indices_in_kvcache,
            extra_topk_length=extra_topk_length,
            num_splits=num_splits, block_table=block_table,
            cache_seqlens=cache_seqlens, causal=causal,
            is_fp8_kvcache=is_fp8_kvcache, attn_sink=attn_sink,
            topk_length=topk_length, indices=indices,
            head_dim_v=head_dim_v,
        )
        if triton_reason is None:
            return SparseMLADecodeBackend.TRITON_SM120
        # Triton precondition failed. Under Dynamo/PCG compile, the TORCH
        # fallback is graph-hostile (Python reference calls method on an
        # untraceable kvcache_layout object) — fail loud with the exact
        # reason rather than letting the fallback explode inside Dynamo.
        if torch._dynamo.is_compiling():
            raise RuntimeError(
                "sm_120 sparse-MLA Triton precondition failed during "
                f"Dynamo/PCG compile: {triton_reason}. "
                "TORCH fallback is not PCG-safe; fix the metadata/shape "
                "so _triton_supported holds."
            )
        if not _torch_supported(
            num_splits=num_splits, block_table=block_table,
            cache_seqlens=cache_seqlens, causal=causal,
            is_fp8_kvcache=is_fp8_kvcache,
        ):
            raise NotImplementedError(
                "sparse-MLA decode call has features unsupported by both "
                f"the sm_120 Triton path ({triton_reason}) and the torch "
                "fallback"
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
