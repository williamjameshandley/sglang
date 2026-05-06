from typing import Any, Optional, Union

import torch

from sglang.srt.layers.attention.sparse_mla_backend import (
    SparseMLADecodeBackend,
    parse_backend_string,
)
from sglang.srt.layers.quantization.fp8_kernel import is_fp8_fnuz
from sglang.srt.utils import is_hip

FP8_DTYPE = torch.float8_e4m3fnuz if is_fp8_fnuz() else torch.float8_e4m3fn


def flash_mla_with_kvcache_entrypoint(
    backend: Union[SparseMLADecodeBackend, str],
    **kwargs,
):
    """Dispatch a sparse-MLA decode call to the chosen backend.

    `backend` accepts either the new `SparseMLADecodeBackend` enum
    (preferred; emitted by `get_sparse_mla_decode_backend`) or a legacy
    string (`torch` / `tilelang` / `comparison` / `kernel` /
    `triton_sm120`). All imports are scoped per-branch so unused backends
    don't drag in heavy dependencies (e.g. `flash_mla` on sm_120, where
    the upstream library has no compatible kernels).
    """
    if isinstance(backend, str):
        backend = parse_backend_string(backend)

    if backend is SparseMLADecodeBackend.TORCH:
        return flash_mla_with_kvcache_torch(**kwargs)

    if backend is SparseMLADecodeBackend.TRITON_SM120:
        from sglang.jit_kernel.deepseek_v4 import (
            flash_mla_with_kvcache_triton_sm120,
        )
        return flash_mla_with_kvcache_triton_sm120(**kwargs)

    if backend is SparseMLADecodeBackend.TILELANG:
        from sglang.srt.layers.attention.nsa.tilelang_kernel import (
            dpsk_v4_fp8_attention_fwd,
        )
        return dpsk_v4_fp8_attention_fwd(**kwargs)

    if backend is SparseMLADecodeBackend.FLASH_MLA:
        import flash_mla
        return flash_mla.flash_mla_with_kvcache(**kwargs)

    if backend is SparseMLADecodeBackend.COMPARISON:
        if kwargs.get("attn_sink") is not None:
            raise NotImplementedError(
                "COMPARISON backend disabled when attn_sink is non-None: the "
                "torch adapter returns plain logsumexp while FlashMLA returns "
                "sink-included LSE; the comparison would diverge legitimately."
            )
        import flash_mla
        pack_ref = flash_mla_with_kvcache_torch(**kwargs)
        pack_fast = flash_mla.flash_mla_with_kvcache(**kwargs)
        _assert_close(pack_ref=pack_ref, pack_fast=pack_fast)
        return pack_ref

    raise NotImplementedError(f"unhandled sparse-MLA backend: {backend}")


def flash_mla_with_kvcache_torch(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    block_table: Optional[torch.Tensor] = None,
    cache_seqlens: Optional[torch.Tensor] = None,
    head_dim_v: int = 512,
    tile_scheduler_metadata: Any = None,
    num_splits: None = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    is_fp8_kvcache: bool = False,
    indices: Optional[torch.Tensor] = None,
    attn_sink: Optional[torch.Tensor] = None,
    extra_k_cache: Optional[torch.Tensor] = None,
    extra_indices_in_kvcache: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
    extra_topk_length: Optional[torch.Tensor] = None,
):

    from sglang.srt.flashmla_tests import quant as flashmla_quant
    from sglang.srt.flashmla_tests.lib import (
        ExtraTestParamForDecode,
        KVScope,
        TestcaseForDecode,
        TestParam,
    )
    from sglang.srt.flashmla_tests.ref import ref_sparse_attn_decode

    assert block_table is None
    assert cache_seqlens is None
    assert is_fp8_kvcache

    b, s_q, h_q, d_qk = q.shape
    d_v = head_dim_v

    fp8_layout = flashmla_quant.FP8KVCacheLayout.MODEL1_FP8Sparse

    p = TestParam(
        s_q=s_q,
        s_kv="unused",
        topk="unused",
        h_q=h_q,
        h_kv=1,
        d_qk=d_qk,
        d_v=d_v,
        decode=ExtraTestParamForDecode(
            b=b,
            is_varlen="unused",
            have_zero_seqlen_k="unused",
            extra_s_k="unused",
            extra_topk="unused",
            extra_block_size="unused",
            have_extra_topk_length="unused",
        ),
        seed=-1,
        check_correctness=True,
        is_all_indices_invalid=False,
        num_runs=10,
        have_attn_sink=True,
        have_topk_length=True,
    )

    blocked_k_quantized = k_cache
    blocked_k = flashmla_quant.dequantize_k_cache(
        blocked_k_quantized.view(FP8_DTYPE), fp8_layout
    )
    kv_scope = KVScope(
        t="unused",
        cache_seqlens="unused",
        block_table="unused",
        blocked_k=blocked_k,
        blocked_k_quantized=blocked_k_quantized,
        abs_indices="unused",
        indices_in_kvcache=indices,
        topk_length=topk_length,
    )

    extra_kv_scope = None
    if extra_k_cache is not None:
        extra_blocked_k_quantized = extra_k_cache
        extra_blocked_k = flashmla_quant.dequantize_k_cache(
            extra_blocked_k_quantized.view(FP8_DTYPE), fp8_layout
        )
        extra_kv_scope = KVScope(
            t="unused",
            cache_seqlens="unused",
            block_table="unused",
            blocked_k=extra_blocked_k,
            blocked_k_quantized=extra_blocked_k_quantized,
            abs_indices="unused",
            indices_in_kvcache=extra_indices_in_kvcache,
            topk_length=extra_topk_length,
        )

    t = TestcaseForDecode(
        p="unused",
        q=q,
        attn_sink=attn_sink,
        sm_scale=softmax_scale,
        kv_scope=kv_scope,
        extra_kv_scope=extra_kv_scope,
    )

    pack_ref = ref_sparse_attn_decode(p, t)
    return pack_ref


def _assert_close(pack_ref, pack_fast):
    import sglang.srt.flashmla_tests.kernelkit as kk

    out_ref, lse_ref = pack_ref
    out_fast, lse_fast = pack_fast

    is_out_correct = kk.check_is_allclose(
        "out", out_fast, out_ref, abs_tol=1e-2, rel_tol=10.0, cos_diff_tol=5e-6
    )
    is_lse_correct = kk.check_is_allclose(
        "lse", lse_fast, lse_ref, abs_tol=1e-6, rel_tol=8.01 / 65536
    )

    assert is_out_correct and is_lse_correct, f"{is_out_correct=} {is_lse_correct=}"
