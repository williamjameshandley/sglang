from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, NamedTuple, Optional, Tuple, Union

import torch
import triton
import triton.language as tl

from sglang.jit_kernel.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)
from sglang.srt.debug_utils.deepseek_v4_debug_utils import (
    deepseek_v4_moe_code_path_checker,
)
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from tvm_ffi.module import Module


def make_name(name: str) -> str:
    return f"dpsk_v4_{name}"


@cache_once
def _jit_common_module() -> Module:
    return load_jit(
        make_name(f"common"),
        cuda_files=[f"deepseek_v4/common.cuh"],
        cuda_wrappers=[("plan_compress_prefill", "plan_compress_prefill")],
    )


@cache_once
def _jit_topk_module() -> Module:
    args = make_cpp_args(is_arch_support_pdl())
    return load_jit(
        make_name("topk"),
        *args,
        cuda_files=["deepseek_v4/topk.cuh"],
        cuda_wrappers=[("topk_transform", f"TopK512Kernel<{args}>::transform")],
    )


@cache_once
def _jit_topk_v2_module() -> Module:
    return load_jit(
        make_name("topk_v2"),
        cuda_files=["deepseek_v4/topk_v2.cuh"],
        cuda_wrappers=[("topk_transform", "TopK512Kernel::transform")],
    )


@cache_once
def _jit_hash_topk_module() -> Module:
    args = make_cpp_args("act_sqrt_softplus", is_arch_support_pdl())
    return load_jit(
        make_name("hash_topk"),
        *args,
        cuda_files=["deepseek_v4/hash_topk.cuh"],
        cuda_wrappers=[("hash_topk", f"HashTopKKernel<{args}>::run")],
    )


@cache_once
def _jit_compress_module(
    head_dim: int,
    dtype_in: torch.dtype,
    dtype_out: torch.dtype,
    ratio: Literal[4, 128],
) -> Module:
    args = make_cpp_args(head_dim, dtype_in, dtype_out, is_arch_support_pdl())
    kernel_class = f"FlashCompress{ratio}Kernel<{args}>"
    return load_jit(
        make_name(f"compress_{ratio}"),
        *args,
        cuda_files=[f"deepseek_v4/c{ratio}.cuh"],
        cuda_wrappers=[
            ("decode", f"{kernel_class}::run_decode"),
            ("prefill", f"{kernel_class}::run_prefill"),
        ],
    )


@cache_once
def _jit_fused_rope_module() -> Module:
    args = make_cpp_args(is_arch_support_pdl())
    return load_jit(
        make_name("fused_rope"),
        *args,
        cuda_files=["deepseek_v4/rope.cuh"],
        cuda_wrappers=[("forward", f"FusedQKRopeKernel<{args}>::forward")],
    )


@cache_once
def _jit_norm_rope_module(
    dtype: torch.dtype,
    head_dim: int,
    rope_dim: int,
) -> Module:
    args = make_cpp_args(dtype, head_dim, rope_dim, is_arch_support_pdl())
    return load_jit(
        make_name(f"fused_norm_rope"),
        *args,
        cuda_files=[f"deepseek_v4/fused_norm_rope.cuh"],
        cuda_wrappers=[
            ("forward", f"FusedNormRopeKernel<{args}>::forward"),
        ],
    )


@cache_once
def _jit_fused_store_module(
    name: Literal["flashmla", "indexer"],
    input_dtype: torch.dtype,
    index_dtype: torch.dtype,
    page_size: int,
) -> Module:
    args = make_cpp_args(input_dtype, index_dtype, page_size, is_arch_support_pdl())
    cname = "FlashMLA" if name == "flashmla" else "Indexer"
    kernel_class = f"FusedStoreCache{cname}Kernel<{args}>"
    return load_jit(
        make_name("store_" + name),
        *args,
        cuda_files=["deepseek_v4/store.cuh"],
        cuda_wrappers=[("run", f"{kernel_class}::run")],
    )


@cache_once
def _jit_metadata_module():
    return load_jit(
        make_name("metadata"),
        cuda_files=["deepseek_v4/paged_mqa_metadata.cuh"],
        cuda_wrappers=[("run", "IndexerMetadataKernel::run")],
    )


@register_custom_op(
    op_name="deepseek_v4_topk_transform_512_v1_page_only_",
    mutates_args=["out_page_indices"],
)
def _topk_transform_512_v1_page_only_(
    scores: torch.Tensor,
    seq_lens: torch.Tensor,
    page_tables: torch.Tensor,
    out_page_indices: torch.Tensor,
    page_size: int,
) -> None:
    _jit_topk_module().topk_transform(
        scores, seq_lens, page_tables, out_page_indices, page_size, None,
    )


@register_custom_op(
    op_name="deepseek_v4_topk_transform_512_v1_raw_",
    mutates_args=["out_page_indices", "out_raw_indices"],
)
def _topk_transform_512_v1_raw_(
    scores: torch.Tensor,
    seq_lens: torch.Tensor,
    page_tables: torch.Tensor,
    out_page_indices: torch.Tensor,
    out_raw_indices: torch.Tensor,
    page_size: int,
) -> None:
    _jit_topk_module().topk_transform(
        scores, seq_lens, page_tables, out_page_indices, page_size, out_raw_indices,
    )


@register_custom_op(
    op_name="deepseek_v4_topk_transform_512_v2_page_only_",
    mutates_args=["out_page_indices"],
)
def _topk_transform_512_v2_page_only_(
    scores: torch.Tensor,
    seq_lens: torch.Tensor,
    page_tables: torch.Tensor,
    out_page_indices: torch.Tensor,
    page_size: int,
) -> None:
    _jit_topk_v2_module().topk_transform(
        scores, seq_lens, page_tables, out_page_indices, page_size, None,
    )


def topk_transform_512(
    scores: torch.Tensor,
    seq_lens: torch.Tensor,
    page_tables: torch.Tensor,
    out_page_indices: torch.Tensor,
    page_size: int,
    out_raw_indices: Optional[torch.Tensor] = None,
    ver: Literal[1, 2] = 1,
) -> None:
    """Output to page_indices tensor, optionally also output raw abs position indices.

    `ver=2` does not support `out_raw_indices` — the C++ wrapper at
    csrc/deepseek_v4/topk_v2.cuh:347-374 rejects a non-None sixth arg.
    """
    if ver == 2:
        if out_raw_indices is not None:
            raise NotImplementedError(
                "topk_transform_512 ver=2 does not support out_raw_indices"
            )
        _topk_transform_512_v2_page_only_(
            scores, seq_lens, page_tables, out_page_indices, page_size,
        )
    else:
        if out_raw_indices is None:
            _topk_transform_512_v1_page_only_(
                scores, seq_lens, page_tables, out_page_indices, page_size,
            )
        else:
            _topk_transform_512_v1_raw_(
                scores, seq_lens, page_tables, out_page_indices, out_raw_indices, page_size,
            )


@register_custom_op(
    op_name="deepseek_v4_hash_topk_fill_",
    mutates_args=["topk_weights", "topk_ids"],
)
def _hash_topk_fill_(
    router_logits: torch.Tensor,
    input_ids: torch.Tensor,
    tid2eid: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    routed_scaling_factor: float,
) -> None:
    _jit_hash_topk_module().hash_topk(
        router_logits, input_ids, tid2eid,
        topk_weights, topk_ids, routed_scaling_factor,
    )


def hash_topk(
    router_logits: torch.Tensor,
    input_ids: torch.Tensor,
    tid2eid: torch.Tensor,
    num_fused_shared_experts: int = 0,
    routed_scaling_factor: float = 1.0,
    scoring_func: str = "sqrtsoftplus",
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert scoring_func == "sqrtsoftplus"
    num_tokens = router_logits.size(0)
    topk_routed = tid2eid.size(1)
    topk_fused = topk_routed + num_fused_shared_experts
    topk_ids = torch.empty(
        (num_tokens, topk_fused), dtype=torch.int32, device=router_logits.device
    )
    topk_weights = torch.empty(
        (num_tokens, topk_fused), dtype=torch.float32, device=router_logits.device
    )
    _hash_topk_fill_(
        router_logits, input_ids, tid2eid,
        topk_weights, topk_ids, routed_scaling_factor,
    )
    return topk_weights, topk_ids


class CompressorPrefillPlan(NamedTuple):
    compress_ratio: int
    compress_plan: torch.Tensor
    write_plan: torch.Tensor

    def copy_(self, other: CompressorPrefillPlan) -> None:
        assert self.compress_ratio == other.compress_ratio
        self.compress_plan.copy_(other.compress_plan)
        self.write_plan.copy_(other.write_plan)

    @staticmethod
    def generate(
        compress_ratio: Literal[4, 128],
        num_q_tokens: int,
        seq_lens: torch.Tensor,
        extend_lens: torch.Tensor,
        device: torch.device,
        use_cuda_graph: bool = False,
    ) -> CompressorPrefillPlan:
        assert seq_lens.device == extend_lens.device
        seq_lens = seq_lens.to(torch.int64)
        extend_lens = extend_lens.to(torch.int64)
        plan_tensor = torch.empty(
            (2, num_q_tokens, 16),
            dtype=torch.uint8,
            device=seq_lens.device,
            pin_memory=seq_lens.is_cpu,
        )
        module = _jit_common_module()
        is_overlap = compress_ratio == 4
        # NOTE: when seq_lens on CUDA device or use_cuda_graph = True,
        # the C++/CUDA implementation will pad up to num_q_tokens
        plan_lens = module.plan_compress_prefill(
            extend_lens,
            seq_lens,
            plan_tensor[0],
            plan_tensor[1],
            compress_ratio,
            is_overlap,
            use_cuda_graph,
        )
        return CompressorPrefillPlan(
            compress_ratio,
            plan_tensor[0, : plan_lens[0]].to(device, non_blocking=True),
            plan_tensor[1, : plan_lens[1]].to(device, non_blocking=True),
        )


# NOTE: only decode plan is compatible with cuda graph
class CompressorDecodePlan(NamedTuple):
    compress_ratio: int
    seq_lens: torch.Tensor

    def copy_(self, other: CompressorDecodePlan) -> None:
        assert self.compress_ratio == other.compress_ratio
        self.seq_lens.copy_(other.seq_lens)


def compress_plan(
    compress_ratio: Literal[4, 128],
    num_q_tokens: int,
    seq_lens: torch.Tensor,
    extend_lens: Optional[torch.Tensor],
    device: torch.device,
) -> Union[CompressorDecodePlan, CompressorPrefillPlan]:
    if extend_lens is not None:
        return CompressorPrefillPlan.generate(
            compress_ratio,
            num_q_tokens,
            seq_lens,
            extend_lens,
            device,
        )
    else:
        assert num_q_tokens == len(seq_lens)
        seq_lens = seq_lens.to(device, non_blocking=True)
        return CompressorDecodePlan(compress_ratio, seq_lens)


@register_custom_op(
    op_name="deepseek_v4_compress_forward_decode_fill_",
    mutates_args=["kv_score_buffer", "out"],
)
def _compress_forward_decode_fill_(
    kv_score_buffer: torch.Tensor,
    kv_score_input: torch.Tensor,
    out: torch.Tensor,
    ape: torch.Tensor,
    indices: torch.Tensor,
    seq_lens: torch.Tensor,
    extra_data: Optional[torch.Tensor],
    head_dim: int,
    compress_ratio: int,
) -> None:
    _jit_compress_module(
        head_dim, kv_score_input.dtype, out.dtype, compress_ratio,
    ).decode(
        kv_score_buffer, kv_score_input, out, ape, indices, seq_lens, extra_data,
    )


@register_custom_op(
    op_name="deepseek_v4_compress_forward_prefill_fill_",
    mutates_args=["kv_score_buffer", "out"],
)
def _compress_forward_prefill_fill_(
    kv_score_buffer: torch.Tensor,
    kv_score_input: torch.Tensor,
    out: torch.Tensor,
    ape: torch.Tensor,
    indices: torch.Tensor,
    compress_plan_tensor: torch.Tensor,
    write_plan_tensor: torch.Tensor,
    extra_data: Optional[torch.Tensor],
    head_dim: int,
    compress_ratio: int,
) -> None:
    _jit_compress_module(
        head_dim, kv_score_input.dtype, out.dtype, compress_ratio,
    ).prefill(
        kv_score_buffer, kv_score_input, out, ape, indices,
        compress_plan_tensor, write_plan_tensor, extra_data,
    )


def compress_forward(
    kv_score_buffer: torch.Tensor,
    kv_score_input: torch.Tensor,
    ape: torch.Tensor,
    indices: torch.Tensor,
    plan: Union[CompressorDecodePlan, CompressorPrefillPlan, None] = None,
    extra_data: Optional[torch.Tensor] = None,
    *,
    head_dim: int,
    compress_ratio: Literal[4, 128],
    out: Optional[torch.Tensor] = None,
    seq_lens: Optional[torch.Tensor] = None,
    extend_lens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    # TODO(dark): support dynamic plan and dispatch for decode kernel
    # Currently, there's no load-balancing for compression kernel
    # In worst cases, few SM will be overloaded with most compression work.
    # For C4, this may not be a big issue, since the compression is fast enough,
    # and the compression is quite common (with an probability of 1/4 in average).
    # For C128, the compression involves CTA reduction, which is relatively slow,
    # and the compression is rare (with an probability of 1/128 in average).
    # We may need to implement dynamic dispatch to better balance the load among SMs.
    # We may need some interface like `module.plan(...)` to prepare before forward pass.
    assert head_dim % 128 == 0
    num_q_tokens = kv_score_input.shape[0]
    if out is None:
        out = kv_score_input.new_empty((num_q_tokens, head_dim))
    if plan is None:
        # compress_plan() invokes _jit_common_module().plan_compress_prefill(...)
        # whose returned plan_lens drive data-dependent slicing — incompatible
        # with PCG/Dynamo capture. Fail loud here rather than surface as an
        # opaque graph-break in the C++ binding.
        if torch._dynamo.is_compiling():
            raise RuntimeError(
                "compress_forward(plan=None) is not PCG-safe; precompute the "
                "plan in eager Python (via compress_plan / make_compressor_plan) "
                "before calling compress_forward inside a Dynamo-compiled "
                "forward."
            )
        assert seq_lens is not None
        plan = compress_plan(
            compress_ratio,
            num_q_tokens,
            seq_lens,
            extend_lens,
            kv_score_input.device,
        )
    assert plan.compress_ratio == compress_ratio, "Mismatched compress ratio in plan!"
    if isinstance(plan, CompressorDecodePlan):
        _compress_forward_decode_fill_(
            kv_score_buffer, kv_score_input, out, ape, indices,
            plan.seq_lens, extra_data, head_dim, int(compress_ratio),
        )
    else:
        _compress_forward_prefill_fill_(
            kv_score_buffer, kv_score_input, out, ape, indices,
            plan.compress_plan, plan.write_plan,
            extra_data, head_dim, int(compress_ratio),
        )
    return out




@register_custom_op(
    op_name="deepseek_v4_compress_fused_norm_rope_decode_",
    mutates_args=["kv"],
)
def _compress_fused_norm_rope_decode_(
    kv: torch.Tensor,
    weight: torch.Tensor,
    plan_tensor: torch.Tensor,
    freq_real: torch.Tensor,
    eps: float,
    compress_ratio: int,
) -> None:
    _jit_norm_rope_module(kv.dtype, kv.shape[-1], freq_real.shape[-1]).forward(
        kv, weight, plan_tensor, freq_real, 1, eps, compress_ratio,
    )


@register_custom_op(
    op_name="deepseek_v4_compress_fused_norm_rope_prefill_",
    mutates_args=["kv"],
)
def _compress_fused_norm_rope_prefill_(
    kv: torch.Tensor,
    weight: torch.Tensor,
    plan_tensor: torch.Tensor,
    freq_real: torch.Tensor,
    eps: float,
    compress_ratio: int,
) -> None:
    _jit_norm_rope_module(kv.dtype, kv.shape[-1], freq_real.shape[-1]).forward(
        kv, weight, plan_tensor, freq_real, 0, eps, compress_ratio,
    )


@register_custom_op(
    op_name="deepseek_v4_fused_norm_rope_",
    mutates_args=["kv"],
)
def _fused_norm_rope_(
    kv: torch.Tensor,
    weight: torch.Tensor,
    positions: torch.Tensor,
    freq_real: torch.Tensor,
    eps: float,
) -> None:
    _jit_norm_rope_module(kv.dtype, kv.shape[-1], freq_real.shape[-1]).forward(
        kv, weight, positions, freq_real, 2, eps, 0,
    )


def compress_fused_norm_rope_inplace(
    kv: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    freq_cis: torch.Tensor,
    plan: Union[CompressorDecodePlan, CompressorPrefillPlan],
) -> None:
    freq_real = torch.view_as_real(freq_cis).flatten(-2)
    if isinstance(plan, CompressorDecodePlan):
        _compress_fused_norm_rope_decode_(
            kv, weight, plan[1], freq_real, eps, plan.compress_ratio,
        )
    else:
        _compress_fused_norm_rope_prefill_(
            kv, weight, plan[1], freq_real, eps, plan.compress_ratio,
        )


def fused_norm_rope_inplace(
    kv: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    freq_cis: torch.Tensor,
    positions: torch.Tensor,
) -> None:
    freq_real = torch.view_as_real(freq_cis).flatten(-2)
    _fused_norm_rope_(kv, weight, positions, freq_real, eps)


@register_custom_op(op_name="deepseek_v4_fused_rope_q_", mutates_args=["q"])
def _fused_rope_q_(
    q: torch.Tensor,
    freqs_real: torch.Tensor,
    positions: torch.Tensor,
    inverse: bool,
) -> None:
    _jit_fused_rope_module().forward(q, None, freqs_real, positions, inverse)


@register_custom_op(op_name="deepseek_v4_fused_rope_qk_", mutates_args=["q", "k"])
def _fused_rope_qk_(
    q: torch.Tensor,
    k: torch.Tensor,
    freqs_real: torch.Tensor,
    positions: torch.Tensor,
    inverse: bool,
) -> None:
    _jit_fused_rope_module().forward(q, k, freqs_real, positions, inverse)


def fused_rope(
    q: torch.Tensor,
    k: Optional[torch.Tensor],
    freqs_cis: torch.Tensor,
    positions: torch.Tensor,
    inverse: bool = False,
) -> None:
    """Apply rotary embeddings to both Q and K in a single fused CUDA kernel.

    Args:
        q: [batch_size, num_q_heads, rope_dim] bfloat16
        k: [batch_size, num_k_heads, rope_dim] bfloat16 or None
        freqs_cis: [max_seq_len, rope_dim // 2] complex64 (full table)
        positions: [batch_size] int32 or int64, indices into freqs_cis
        inverse: if True, apply inverse rotation (conjugate freqs)
    """
    from sglang.srt.utils import is_hip

    if is_hip():
        from sglang.srt.layers.deepseek_v4_rope import apply_rotary_emb_triton

        apply_rotary_emb_triton(q, freqs_cis, positions=positions, inverse=inverse)
        if k is not None:
            apply_rotary_emb_triton(k, freqs_cis, positions=positions, inverse=inverse)
        return

    freqs_real = torch.view_as_real(freqs_cis).flatten(-2).contiguous()
    if k is None:
        _fused_rope_q_(q, freqs_real, positions, inverse)
    else:
        _fused_rope_qk_(q, k, freqs_real, positions, inverse)


@cache_once
def _tilelang_make_swa_indices_kernel(swa_window_size: int, threads: int = 128) -> Any:
    import tilelang
    import tilelang.language as T

    batch_size = T.dynamic("batch_size")
    batch_size_plus_1 = T.dynamic("batch_size_plus_1")
    num_q_tokens = T.dynamic("num_q_tokens")
    num_warps = threads // 32
    assert swa_window_size % 32 == 0

    @tilelang.jit
    def make_swa_prefill_indices(
        seq_lens_k: T.Tensor[(batch_size,), T.int32],
        seq_lens_q: T.Tensor[(batch_size,), T.int32],
        cu_seqlens_q: T.Tensor[(batch_size_plus_1,), T.int32],
        swa_indices: T.Tensor[(num_q_tokens, swa_window_size), T.int32],
    ):
        _ = batch_size_plus_1  # unused, but don't remove it
        with T.Kernel(T.ceildiv(num_q_tokens, num_warps), threads=threads) as bx:
            # each warp handles 1 q token
            tx = T.get_thread_binding()
            warp_id = tx // 32
            lane_id = tx % 32
            s_batch_id = T.alloc_shared((num_warps,), dtype=T.int32)

            token_id = warp_id + bx * num_warps
            if token_id >= num_q_tokens:
                return
            for i in T.serial(0, batch_size, step=32):
                j = i + lane_id
                if cu_seqlens_q[j] <= token_id < cu_seqlens_q[j + 1]:
                    s_batch_id[warp_id] = j
            T.sync_warp()

            seq_idx = s_batch_id[warp_id]
            kv_len = seq_lens_k[seq_idx]
            qo_len = seq_lens_q[seq_idx]
            cum_qo_len = cu_seqlens_q[seq_idx]
            prefix_len = kv_len - qo_len
            curr_seq_qo_idx = token_id - cum_qo_len
            end_abs_pos = prefix_len + curr_seq_qo_idx + 1
            start_abs_pos = T.max(end_abs_pos - swa_window_size, 0)
            old_kv_start = seq_idx * swa_window_size
            new_kv_start = batch_size * swa_window_size + cum_qo_len

            for i in T.unroll(0, swa_window_size, step=32):
                j = i + lane_id
                abs_pos = start_abs_pos + j
                swa_indices[token_id, j] = T.if_then_else(
                    abs_pos < end_abs_pos,
                    T.if_then_else(
                        abs_pos < prefix_len,
                        old_kv_start + abs_pos % swa_window_size,
                        new_kv_start + (abs_pos - prefix_len),
                    ),
                    -1,
                )

    return make_swa_prefill_indices


def tilelang_make_swa_prefill_indices(
    seq_lens_k: torch.Tensor,
    seq_lens_q: torch.Tensor,
    swa_indices: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if cu_seqlens_q is None:
        cu_seqlens_q = torch.cumsum(seq_lens_q, dim=0, dtype=torch.int32)
        cu_seqlens_q = torch.nn.functional.pad(cu_seqlens_q, (1, 0), value=0)
    swa_window_size = swa_indices.shape[1]
    kernel = _tilelang_make_swa_indices_kernel(swa_window_size)
    kernel(seq_lens_k, seq_lens_q, cu_seqlens_q, swa_indices)
    return swa_indices


@triton.jit
def create_paged_compress_data_kernel(
    req_pool_indices_ptr,  # int32 [batch]
    seq_lens_ptr,  # int32 [batch]
    extend_seq_lens_ptr,  # int32 [batch]
    req_to_token_ptr,  # int32 [A, B]
    full_to_swa_index_mapping_ptr,  # int32 [C]
    out_0_ptr,  # int32 [batch]
    out_1_ptr,  # int32 [batch, out_dim]
    batch_size,
    stride_req_to_token_0,
    stride_req_to_token_1: tl.constexpr,  # 1
    stride_out_1_0,
    stride_out_1_1: tl.constexpr,  # 1
    compress_ratio: tl.constexpr,
    is_overlap: tl.constexpr,  # 0/1
    swa_page_size: tl.constexpr,
    ring_size: tl.constexpr,
    BLOCK: tl.constexpr,
) -> None:
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < batch_size

    # load per-batch
    rid = tl.load(req_pool_indices_ptr + offs, mask=mask, other=0).to(tl.int32)
    seq_len = tl.load(seq_lens_ptr + offs, mask=mask, other=0).to(tl.int32)
    extend_len = tl.load(extend_seq_lens_ptr + offs, mask=mask, other=0).to(tl.int32)
    prefix_len = seq_len - extend_len

    cr = compress_ratio
    write_pos = ((seq_len - 1) // cr) * cr
    load_pos = ((prefix_len - 1) // cr) * cr
    write_overlap_pos = write_pos - cr
    load_overlap_pos = load_pos - cr
    v0 = tl.zeros([BLOCK], tl.int32)
    v1 = tl.zeros([BLOCK], tl.int32)
    v2 = tl.zeros([BLOCK], tl.int32)
    v3 = tl.zeros([BLOCK], tl.int32)

    for i in tl.static_range(4):
        if i == 0:
            pos = load_pos
        elif i == 1:
            pos = write_pos
        elif i == 2:
            pos = load_overlap_pos
        else:
            pos = write_overlap_pos
        pos = tl.maximum(pos, 0)
        # req_to_token[rid, pos]
        loc = tl.load(
            req_to_token_ptr
            + rid * stride_req_to_token_0
            + pos * stride_req_to_token_1,
            mask=mask,
            other=0,
        ).to(tl.int32)
        swa_loc = tl.load(full_to_swa_index_mapping_ptr + loc, mask=mask, other=0).to(
            tl.int32
        )
        swa_page = swa_loc // swa_page_size
        state_loc = swa_page * ring_size + (swa_loc % ring_size)
        state_loc = state_loc // cr
        if i == 0:
            v0 = state_loc
        elif i == 1:
            v1 = state_loc
        elif i == 2:
            v2 = state_loc
        else:
            v3 = state_loc

    tl.store(out_0_ptr + offs, v1, mask=mask)

    if is_overlap:
        base = out_1_ptr + offs * stride_out_1_0
        tl.store(base + 0 * stride_out_1_1, v2, mask=mask)
        tl.store(base + 1 * stride_out_1_1, v0, mask=mask)
        tl.store(base + 2 * stride_out_1_1, v3, mask=mask)
        tl.store(base + 3 * stride_out_1_1, write_pos.to(tl.int32), mask=mask)
    else:
        base = out_1_ptr + offs * stride_out_1_0
        tl.store(base + 0 * stride_out_1_1, v0, mask=mask)


def triton_create_paged_compress_data(
    *,
    compress_ratio: int,
    is_overlap: bool,
    swa_page_size: int,
    ring_size: int,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    extend_seq_lens: torch.Tensor,
    req_to_token: torch.Tensor,
    full_to_swa_index_mapping: torch.Tensor,
    block: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size = req_pool_indices.shape[0]
    out_dim = 4 if is_overlap else 1
    device_args: dict = dict(device=req_pool_indices.device, dtype=torch.int32)
    out_0 = torch.empty((batch_size,), **device_args)
    out_1 = torch.empty((batch_size, out_dim), **device_args)
    grid = (triton.cdiv(batch_size, block),)
    create_paged_compress_data_kernel[grid](
        req_pool_indices,
        seq_lens,
        extend_seq_lens,
        req_to_token,
        full_to_swa_index_mapping,
        out_0,
        out_1,
        batch_size=batch_size,  # type: ignore
        stride_req_to_token_0=req_to_token.stride(0),  # type: ignore
        stride_req_to_token_1=req_to_token.stride(1),  # type: ignore
        stride_out_1_0=out_1.stride(0),  # type: ignore
        stride_out_1_1=out_1.stride(1),  # type: ignore
        compress_ratio=compress_ratio,  # type: ignore
        is_overlap=1 if is_overlap else 0,  # type: ignore
        swa_page_size=swa_page_size,  # type: ignore
        ring_size=ring_size,  # type: ignore
        BLOCK=block,  # type: ignore
    )
    if not is_overlap:
        out_1.squeeze_(1)
    return out_0, out_1


@register_custom_op(
    op_name="deepseek_v4_fused_store_cache_flashmla_",
    mutates_args=["cache"],
)
def _fused_store_cache_flashmla_(
    input: torch.Tensor,
    cache: torch.Tensor,
    indices: torch.Tensor,
    page_size: int,
) -> None:
    _jit_fused_store_module(
        name="flashmla",
        input_dtype=input.dtype,
        index_dtype=indices.dtype,
        page_size=page_size,
    ).run(input, cache, indices)


@register_custom_op(
    op_name="deepseek_v4_fused_store_cache_indexer_",
    mutates_args=["cache"],
)
def _fused_store_cache_indexer_(
    input: torch.Tensor,
    cache: torch.Tensor,
    indices: torch.Tensor,
    page_size: int,
) -> None:
    _jit_fused_store_module(
        name="indexer",
        input_dtype=input.dtype,
        index_dtype=indices.dtype,
        page_size=page_size,
    ).run(input, cache, indices)


def fused_store_cache(
    input: torch.Tensor,
    cache: torch.Tensor,
    indices: torch.Tensor,
    *,
    page_size: int,
    type: Literal["flashmla", "indexer"],
) -> None:
    if type == "flashmla":
        _fused_store_cache_flashmla_(input, cache, indices, page_size)
    elif type == "indexer":
        _fused_store_cache_indexer_(input, cache, indices, page_size)
    else:
        raise ValueError(f"unknown fused_store_cache type {type!r}")


@cache_once
def _jit_silu_mul_quant_module(
    quant_group_size: int, scale_ue8m0: bool, apply_swiglu_limit: bool
) -> Module:
    args = make_cpp_args(
        quant_group_size, scale_ue8m0, is_arch_support_pdl(), apply_swiglu_limit
    )
    return load_jit(
        make_name("silu_mul_quant"),
        *args,
        cuda_files=["deepseek_v4/silu_and_mul_masked_post_quant.cuh"],
        cuda_wrappers=[("run", f"SiluAndMulMaskedPostQuantKernel<{args}>::run")],
    )


@register_custom_op(
    op_name="deepseek_v4_silu_mul_quant_plain_",
    mutates_args=["output", "output_scale"],
)
def _silu_mul_quant_plain_(
    input: torch.Tensor,
    output: torch.Tensor,
    output_scale: torch.Tensor,
    masked_m: torch.Tensor,
    quant_group_size: int,
    scale_ue8m0: bool,
    topk: int,
    transposed: bool,
) -> None:
    _jit_silu_mul_quant_module(quant_group_size, scale_ue8m0, False).run(
        input, output, output_scale, masked_m, topk, transposed, 0.0,
    )


@register_custom_op(
    op_name="deepseek_v4_silu_mul_quant_swiglu_",
    mutates_args=["output", "output_scale"],
)
def _silu_mul_quant_swiglu_(
    input: torch.Tensor,
    output: torch.Tensor,
    output_scale: torch.Tensor,
    masked_m: torch.Tensor,
    quant_group_size: int,
    scale_ue8m0: bool,
    topk: int,
    transposed: bool,
    swiglu_limit: float,
) -> None:
    _jit_silu_mul_quant_module(quant_group_size, scale_ue8m0, True).run(
        input, output, output_scale, masked_m, topk, transposed, swiglu_limit,
    )


def silu_and_mul_masked_post_quant(
    input: torch.Tensor,
    output: torch.Tensor,
    output_scale: torch.Tensor,
    quant_group_size: int,
    masked_m: torch.Tensor,
    scale_ue8m0: bool = False,
    topk: int = 8,
    transposed: bool = False,
    swiglu_limit: Optional[float] = None,
) -> None:
    """
    Fused SiLU-and-mul with per-group FP8 quantization for expert-parallel MoE.

    input shape:        [expert_num, token_num_padded, hidden_dim]
    output shape:       [expert_num, token_num_padded, hidden_dim // 2], dtype fp8_e4m3
    output_scale shape: [expert_num, token_num_padded, hidden_dim // 2 // quant_group_size], dtype float32
    masked_m shape:     [expert_num], dtype int32. i.e. actual token count per expert
    topk:               max routed experts per token (grid = token_num_padded * topk blocks)
    swiglu_limit:       Optional. When None (default), use the original fast path (no clamp).
                        When set, JIT-compiles a separate kernel variant that clamps gate to
                        [-inf, L] and up to [-L, L] before silu (fused).
    """
    if swiglu_limit is None:
        _silu_mul_quant_plain_(
            input, output, output_scale, masked_m,
            quant_group_size, scale_ue8m0, topk, transposed,
        )
    else:
        deepseek_v4_moe_code_path_checker.observed += 1
        _silu_mul_quant_swiglu_(
            input, output, output_scale, masked_m,
            quant_group_size, scale_ue8m0, topk, transposed, float(swiglu_limit),
        )


@register_custom_op(
    op_name="deepseek_v4_paged_mqa_logits_metadata_fill_",
    mutates_args=["metadata"],
)
def _paged_mqa_logits_metadata_fill_(
    seq_lens: torch.Tensor,
    metadata: torch.Tensor,
) -> None:
    _jit_metadata_module().run(seq_lens, metadata)


def get_paged_mqa_logits_metadata(seq_lens: torch.Tensor, page_size: int, num_sm: int):
    assert page_size == 64
    seq_lens = seq_lens.to(torch.int32)
    metadata = seq_lens.new_empty(num_sm + 1, 2)
    _paged_mqa_logits_metadata_fill_(seq_lens, metadata)
    return metadata


@cache_once
def _jit_torch_cublas_bf16_fp32() -> Any:
    import torch.utils.cpp_extension

    source = """
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cublas_v2.h>

torch::Tensor linear_bf16_fp32(
    torch::Tensor X,
    torch::Tensor W)
{
    int batch = X.size(0);
    int in_features = X.size(1);
    int out_features = W.size(0);

    auto Y = torch::empty(
        {batch, out_features},
        torch::dtype(torch::kFloat32).device(X.device()));

    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();

    float alpha = 1.0f;
    float beta = 0.0f;

    cublasGemmEx(
        handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        out_features,
        batch,
        in_features,
        &alpha,
        W.data_ptr(), CUDA_R_16BF, in_features,
        X.data_ptr(), CUDA_R_16BF, in_features,
        &beta,
        Y.data_ptr(), CUDA_R_32F, out_features,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    );

    return Y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("linear_bf16_fp32", &linear_bf16_fp32, "BF16xBF16 -> FP32 linear (no bias)");
}
"""
    module = torch.utils.cpp_extension.load_inline(
        name="linear_bf16_fp32",
        cpp_sources="",
        cuda_sources=source,
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        verbose=False,
    )
    return module


def linear_bf16_fp32(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    from sglang.srt.environ import envs

    algo = envs.SGLANG_OPT_BF16_FP32_GEMM_ALGO.get()

    if algo == "cublas":
        module = _jit_torch_cublas_bf16_fp32()
        return module.linear_bf16_fp32(x, y)
    elif algo == "deep_gemm":
        import deep_gemm

        z = x.new_empty(x.size(0), y.size(0), dtype=torch.float32)
        deep_gemm.bf16_gemm_nt(x, y, z)
        return z
    else:  # fall back to torch fp32 GEMM
        return torch.nn.functional.linear(x.float(), y.float())


def _compile_one(*input_tuple) -> None:
    name, job_fn, *args = input_tuple
    print(f"Compiling {name}...", flush=True)
    job_fn(*args)
    print(f"Finished compiling {name}.", flush=True)


def compile_aot():
    c_dtype = torch.float32  # compress uses float32
    jobs = [
        ("cublas", _jit_torch_cublas_bf16_fp32),
        ("common", _jit_common_module),
        ("topk", _jit_topk_module),
        ("hash_topk", _jit_hash_topk_module),
        ("rope", _jit_fused_rope_module),
        ("metadata", _jit_metadata_module),
        (
            "compress_128_4",
            _jit_compress_module,
            128,
            c_dtype,
            c_dtype,
            4,
        ),
        (
            "compress_512_4",
            _jit_compress_module,
            512,
            c_dtype,
            c_dtype,
            4,
        ),
        (
            "compress_512_128",
            _jit_compress_module,
            512,
            c_dtype,
            c_dtype,
            128,
        ),
        (
            "norm_rope_128_64",
            _jit_norm_rope_module,
            c_dtype,
            128,
            64,
        ),
        (
            "norm_rope_512_64",
            _jit_norm_rope_module,
            c_dtype,
            512,
            64,
        ),
        (
            "store_flashmla_bf16_swa_256",
            _jit_fused_store_module,
            "flashmla",
            torch.bfloat16,
            torch.int32,
            256,
        ),
        (
            "store_flashmla_fp32_c4_64",
            _jit_fused_store_module,
            "flashmla",
            torch.float32,
            torch.int32,
            64,
        ),
        (
            "store_flashmla_fp32_c128_2",
            _jit_fused_store_module,
            "flashmla",
            torch.float32,
            torch.int32,
            2,
        ),
        (
            "store_indexer_fp32_c4_64",
            _jit_fused_store_module,
            "indexer",
            torch.float32,
            torch.int32,
            64,
        ),
    ]
    # use multiprocess to speed up compilation
    import multiprocessing

    max_parallel_jobs = min(len(jobs), multiprocessing.cpu_count())
    with multiprocessing.Pool(processes=max_parallel_jobs) as pool:
        pool.starmap(_compile_one, jobs)


# ---------------------------------------------------------------------------
# fp8_paged_mqa_logits — Triton kernel for sm_120 (CD-3)
# ---------------------------------------------------------------------------


@triton.jit
def _fp8_paged_mqa_logits_kernel(
    Q_ptr,            # [B, NUM_HEADS, HEAD_DIM] fp8
    KV_ptr,           # kvcache reinterpreted as fp8 elements
    KV_F32_ptr,       # same buffer reinterpreted as fp32
    W_ptr,            # [B, NUM_HEADS] fp32
    SL_ptr,           # [B] int32
    PT_ptr,           # [B, MAX_PAGES] int32
    Out_ptr,          # [B, MAX_SEQ_LEN] fp32 (pre-zeroed)
    NUM_HEADS,
    MAX_PAGES,
    MAX_SEQ_LEN,
    BLOCK_SIZE: tl.constexpr,           # 64 compressed positions per page
    HEAD_DIM: tl.constexpr,             # 128
    NUM_HEADS_PAD: tl.constexpr,        # next pow-2 of NUM_HEADS
    PAGE_BYTES: tl.constexpr,           # BLOCK_SIZE * (HEAD_DIM + 4)
    K_BYTES_PER_PAGE: tl.constexpr,     # BLOCK_SIZE * HEAD_DIM
    PAGE_F32: tl.constexpr,             # PAGE_BYTES // 4
    SCALE_F32_BASE: tl.constexpr,       # K_BYTES_PER_PAGE // 4
    BLOCK_S: tl.constexpr,              # positions per program
):
    """One program covers `BLOCK_S` positions for a single batch.

    Per position computes
        score = sum_h(weight[b,h] * relu(<q[b,h], k>)) * k_scale.

    KV layout (per page, matching `fp8_paged_mqa_logits_torch` at
    `compressed/indexer.py:68-84`):
      [BLOCK_SIZE * HEAD_DIM bytes of FP8 K] then
      [BLOCK_SIZE * 4 bytes of FP32 scale],
    not interleaved per position.

      K   byte offset = page * PAGE_BYTES + in_page * HEAD_DIM + d
      scl fp32 index  = page * PAGE_F32   + SCALE_F32_BASE + in_page
    """
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)

    pos_offs = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    seq_len = tl.load(SL_ptr + pid_b)
    in_seq = pos_offs < seq_len

    page_idx = pos_offs // BLOCK_SIZE
    in_page = pos_offs % BLOCK_SIZE
    page_idx_clamped = tl.where(page_idx < MAX_PAGES, page_idx, 0)
    page_id = tl.load(
        PT_ptr + pid_b * MAX_PAGES + page_idx_clamped,
        mask=in_seq, other=0,
    )
    page_id = tl.maximum(page_id, 0)

    h_offs = tl.arange(0, NUM_HEADS_PAD)
    h_mask = h_offs < NUM_HEADS
    d_offs = tl.arange(0, HEAD_DIM)
    q_offs = pid_b * NUM_HEADS * HEAD_DIM + h_offs[:, None] * HEAD_DIM + d_offs[None, :]
    q = tl.load(Q_ptr + q_offs, mask=h_mask[:, None], other=0.0).to(tl.float32)

    w = tl.load(W_ptr + pid_b * NUM_HEADS + h_offs, mask=h_mask, other=0.0).to(tl.float32)

    # K byte offsets: split layout — all-K then all-scale per page.
    kv_byte_offs = (
        page_id[:, None] * PAGE_BYTES
        + in_page[:, None] * HEAD_DIM
        + d_offs[None, :]
    )
    kv = tl.load(KV_ptr + kv_byte_offs, mask=in_seq[:, None], other=0.0).to(tl.float32)

    # Scale fp32 indices: page * PAGE_F32 + SCALE_F32_BASE + in_page.
    scale_idx = page_id * PAGE_F32 + SCALE_F32_BASE + in_page
    scale = tl.load(KV_F32_ptr + scale_idx, mask=in_seq, other=0.0)

    # Tensor-core dot: [BLOCK_S, HEAD_DIM] @ [HEAD_DIM, NUM_HEADS_PAD].
    q_t = tl.trans(q)
    dots = tl.dot(kv, q_t)
    dots = tl.maximum(dots, 0.0)
    dots = tl.where(h_mask[None, :], dots, 0.0)
    weighted = dots * w[None, :]
    head_sum = tl.sum(weighted, axis=1)
    score = head_sum * scale
    score = tl.where(in_seq, score, 0.0)

    out_offs = pid_b * MAX_SEQ_LEN + pos_offs
    tl.store(Out_ptr + out_offs, score, mask=pos_offs < MAX_SEQ_LEN)


def fp8_paged_mqa_logits_triton(
    q_fp8: torch.Tensor,
    kvcache_fp8: torch.Tensor,
    weight: torch.Tensor,
    seq_lens: torch.Tensor,
    page_table: torch.Tensor,
    deep_gemm_metadata: Any,
    max_seq_len: int,
    clean_logits: bool = False,
) -> torch.Tensor:
    """sm_120 Triton replacement for `deep_gemm.fp8_paged_mqa_logits`.

    Shape contract matches `fp8_paged_mqa_logits_torch`
    (`compressed/indexer.py`):
      * input  q_fp8       [B, 1, NUM_HEADS, 128]
      * input  kvcache_fp8 [num_pages, 64, 1, 132]
      * input  weight      [B, NUM_HEADS] fp32
      * input  seq_lens    [B] int32
      * input  page_table  [B, MAX_PAGES]
      * output             [B, max_seq_len] fp32, zero-fill past seq_lens[b]
        and past page_table.shape[1] * 64.
    """
    assert clean_logits is False, "clean_logits=True not implemented"
    _ = deep_gemm_metadata  # not consumed by Triton path

    B, _, num_heads, head_dim = q_fp8.shape
    block_size = kvcache_fp8.shape[1]
    assert head_dim == 128
    assert block_size == 64
    assert kvcache_fp8.shape[1:] == (block_size, 1, head_dim + 4)
    assert q_fp8.shape == (B, 1, num_heads, head_dim)
    assert weight.shape == (B, num_heads)
    assert seq_lens.shape == (B,)
    assert page_table.shape[0] == B

    PAGE_BYTES = block_size * (head_dim + 4)        # 8448 for the V4 layout
    K_BYTES_PER_PAGE = block_size * head_dim        # 8192
    PAGE_F32 = PAGE_BYTES // 4                      # 2112
    SCALE_F32_BASE = K_BYTES_PER_PAGE // 4          # 2048

    # Phase 6.11 capture-trace + warmed-spec guard. Spec captures all Triton
    # constexpr meta values + pointer dtypes that affect specialization.
    fp8_spec = (
        int(num_heads),
        int(block_size),
        int(head_dim),
        int(page_table.shape[1]),
        int(max_seq_len),
        str(q_fp8.dtype),
        str(kvcache_fp8.dtype),
        str(weight.dtype),
        str(seq_lens.dtype),
        str(page_table.dtype),
    )
    fp8_capturing = _capture_trace_once(
        "fp8_paged_mqa_logits", fp8_spec,
        warmed=_FP8_MQA_WARMED_SPECS, phase="enter",
    )
    if fp8_capturing and fp8_spec not in _FP8_MQA_WARMED_SPECS:
        import sys as _sys
        _sys.stderr.write(
            f"[capture-trace] FATAL unwarmed fp8_paged_mqa_logits spec={fp8_spec} "
            f"warmed={sorted(_FP8_MQA_WARMED_SPECS)}\n"
        )
        _sys.stderr.flush()
        raise RuntimeError(
            f"fp8_paged_mqa_logits_triton: spec {fp8_spec} hit CUDA graph "
            f"capture without prior warmup."
        )

    cache_contig = kvcache_fp8.contiguous()
    # Reinterpret cache bytes as FP8 (for K loads) and fp32 (for scale).
    # The torch reference at compressed/indexer.py:78 does the same view; the
    # underlying storage is byte-aligned, so both views are valid no-ops if
    # cache_contig is already FP8 / uint8. This Triton path is only entered on
    # CUDA sm_120 (the HIP fnuz case routes to TORCH in paged_mqa_backend.py),
    # so torch.float8_e4m3fn is the right typed view.
    kv_fp8_flat = cache_contig.view(torch.float8_e4m3fn).view(-1)
    kv_f32_flat = cache_contig.view(torch.float32).view(-1)

    # Pre-zero output so masked positions and the [padded_seq_len, max_seq_len)
    # tail are zero without the kernel having to touch them.
    scores = torch.zeros((B, max_seq_len), dtype=torch.float32, device=q_fp8.device)

    max_pages = page_table.shape[1]
    padded_seq_len = max_pages * block_size
    BLOCK_S = 64
    NUM_HEADS_PAD = triton.next_power_of_2(num_heads)

    grid = (B, triton.cdiv(padded_seq_len, BLOCK_S))
    _fp8_paged_mqa_logits_kernel[grid](
        q_fp8,
        kv_fp8_flat,
        kv_f32_flat,
        weight,
        seq_lens,
        page_table,
        scores,
        num_heads,
        max_pages,
        max_seq_len,
        BLOCK_SIZE=block_size,
        HEAD_DIM=head_dim,
        NUM_HEADS_PAD=NUM_HEADS_PAD,
        PAGE_BYTES=PAGE_BYTES,
        K_BYTES_PER_PAGE=K_BYTES_PER_PAGE,
        PAGE_F32=PAGE_F32,
        SCALE_F32_BASE=SCALE_F32_BASE,
        BLOCK_S=BLOCK_S,
    )

    if not fp8_capturing:
        _FP8_MQA_WARMED_SPECS.add(fp8_spec)
    _capture_trace_once(
        "fp8_paged_mqa_logits", fp8_spec,
        warmed=_FP8_MQA_WARMED_SPECS, phase="exit",
    )
    return scores


# ---------------------------------------------------------------------------
# Sparse MLA decode — Triton kernel for sm_120 (Phase 6.2 / 6.7)
# ---------------------------------------------------------------------------

# Module-global set of (has_extra, topk, extra_topk) specs that have launched
# the Triton kernel outside CUDA graph capture (and therefore been JIT-compiled).
# Phase 6.11 capture-safety: any spec entering capture without an entry here is
# guaranteed to invalidate the stream when Triton compiles its first kernel
# inside the captured region. The wrapper raises a clear Python error in that
# case instead of letting CUDA report cudaErrorStreamCaptureInvalidated at
# capture_end.
_SPARSE_MLA_WARMED_SPECS: set = set()
_FP8_MQA_WARMED_SPECS: set = set()
# Per-(name, spec, capturing, phase) one-shot trace log
_CAPTURE_TRACE_LOGGED: set = set()


def _capture_trace_once(name: str, spec, warmed=None, phase: str = "enter"):
    """Phase-6.11 stderr trace at Triton wrapper entry/exit.

    Logs (once per unique (name, spec, capturing, phase)) whether the
    wrapper is reached during graph capture and whether the spec was
    pre-warmed. Used to disambiguate "wrapper not called during capture"
    from "wrapper called and silently OK" without a separate rebuild.

    Caller must call once at entry (`phase="enter"`) and once after the
    kernel launch (`phase="exit"`); a missing exit while the enter ran
    under capture means the wrapper raised between the two.

    No-op under Dynamo/PCG trace — `torch.cuda.is_current_stream_capturing()`
    returns `bool`, which Dynamo can't lift into the FX graph. The
    instrumentation is for eager-vs-CUDA-graph diagnostics; PCG warmup
    runs through Dynamo first, where the tracing helper itself would
    derail the compile.
    """
    if torch._dynamo.is_compiling():
        return False
    import sys
    capturing = torch.cuda.is_current_stream_capturing()
    key = (name, spec, capturing, phase)
    if key not in _CAPTURE_TRACE_LOGGED:
        warmed_str = ""
        if warmed is not None:
            warmed_str = f" warmed={spec in warmed}"
        sys.stderr.write(
            f"[capture-trace] {phase} {name} spec={spec} "
            f"capturing={capturing}{warmed_str}\n"
        )
        sys.stderr.flush()
        _CAPTURE_TRACE_LOGGED.add(key)
    return capturing


@triton.jit
def _sparse_mla_decode_kernel(
    Q_ptr,            # [B, 1, h_q, 512] bf16
    KV_FP8_ptr,       # primary cache reinterpreted as fp8_e4m3fn
    KV_BF16_ptr,      # primary cache reinterpreted as bfloat16
    KV_U8_ptr,        # primary cache reinterpreted as uint8
    Indices_ptr,      # [B, 1, topk] int32  (flat token IDs page*P+row, -1 invalid)
    TopkLen_ptr,      # [B] int32
    Sink_ptr,         # [h_q] fp32
    Out_ptr,          # [B, 1, h_q, 512] bf16 (pre-allocated)
    Lse_ptr,          # [B, h_q, 1] fp32 (pre-allocated)
    # Optional second (compressed C4/C128) cache for Phase 6.7 combined-scope.
    # When HAS_EXTRA is False these may be the primary tensors (unused) — the
    # second loop is gated out at compile time.
    EXTRA_KV_FP8_ptr,
    EXTRA_KV_BF16_ptr,
    EXTRA_KV_U8_ptr,
    EXTRA_Indices_ptr,
    EXTRA_TopkLen_ptr,
    softmax_scale,
    page_byte_stride,        # bytes per padded page of the primary cache
    P,                       # tokens per primary page
    extra_page_byte_stride,  # bytes per padded page of the compressed cache
    P_EXTRA,                 # tokens per compressed page (page_size//4 or //128)
    h_q,
    stride_q_b, stride_q_h,  # element strides into Q
    stride_idx_b,            # element stride into primary Indices
    stride_extra_idx_b,      # element stride into compressed Indices
    stride_o_b, stride_o_h,  # element strides into Out
    stride_lse_b, stride_lse_h,  # element strides into Lse
    BLOCK_M: tl.constexpr,         # 16
    KV_CHUNK: tl.constexpr,        # 32
    HEAD_DIM_NOPE: tl.constexpr,   # 448
    HEAD_DIM_ROPE: tl.constexpr,   # 64
    HEAD_DIM_QK: tl.constexpr,     # 512 (= NoPE + RoPE)
    BLOCK_DV: tl.constexpr,        # 128 (output-dim tile; HEAD_DIM_QK/BLOCK_DV programs along z)
    TOPK: tl.constexpr,            # padded primary topk (constexpr; multiple of KV_CHUNK)
    EXTRA_TOPK: tl.constexpr,      # padded compressed topk (0 when HAS_EXTRA is False)
    HAS_EXTRA: tl.constexpr,       # True iff compressed scope present
    NOPE_GROUPS: tl.constexpr,     # 7
    GROUP_SIZE: tl.constexpr,      # 64 (NoPE elements per scale group)
):
    """One program covers BLOCK_M heads of one (b, s_q=0) decode row.

    Layout per page (V4-Flash MODEL1_FP8Sparse, MUST match
    `flashmla_tests.quant.dequantize_k_cache`):

      Per token i in page:
        bytes [i*576, i*576 + 448): NoPE FP8 E4M3
        bytes [i*576 + 448, (i+1)*576): RoPE BF16 (64 elements)
      Per token i (scale region, starts at byte P*576):
        bytes [P*576 + i*8, P*576 + i*8 + 7): 7 UE8M0 group scales
        byte  [P*576 + i*8 + 7]: pad

    NoPE has 7 groups of 64 elements each. Element d in [0, 448) belongs to
    group g = d // 64; dequantized value is fp8_val * 2**(byte_g - 127).

    Indices: flat token IDs (page_id * P + row_in_page); -1 marks invalid.

    Output dim is tiled along program-z: each (pid_dv) covers BLOCK_DV columns
    of the [BLOCK_M, HEAD_DIM_QK] output. Q@K^T (and softmax) is recomputed
    per program-z (each loads K_full once, redundant compute amortised by
    avoiding cross-program reduction); only V@P writes the [BLOCK_M, BLOCK_DV]
    accumulator slice. LSE is identical across z and is stored only by
    pid_dv == 0.
    """
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_dv = tl.program_id(2)

    h_offs = pid_h * BLOCK_M + tl.arange(0, BLOCK_M)
    h_mask = h_offs < h_q

    d_qk = tl.arange(0, HEAD_DIM_QK)              # 512 (full K width)
    is_nope_d = d_qk < HEAD_DIM_NOPE              # bool [512]
    nope_d_safe = tl.where(is_nope_d, d_qk, 0)
    nope_group_d = nope_d_safe // GROUP_SIZE      # [512] in [0, 7]; 0 outside NoPE
    rope_d_within = d_qk - HEAD_DIM_NOPE
    rope_d_safe = tl.maximum(rope_d_within, 0)

    # DV slice (output columns this program writes)
    dv_offs = pid_dv * BLOCK_DV + tl.arange(0, BLOCK_DV)
    is_nope_dv = dv_offs < HEAD_DIM_NOPE
    nope_dv_safe = tl.where(is_nope_dv, dv_offs, 0)
    nope_group_dv = nope_dv_safe // GROUP_SIZE
    rope_dv_within = dv_offs - HEAD_DIM_NOPE
    rope_dv_safe = tl.maximum(rope_dv_within, 0)

    # Load Q [BLOCK_M, HEAD_DIM_QK] bf16. Q is contiguous [B, 1, h_q, 512]
    # (s_q=1, so we ignore s_q stride).
    q_ptrs = (Q_ptr
              + pid_b * stride_q_b
              + h_offs[:, None] * stride_q_h
              + d_qk[None, :])
    q_bf = tl.load(q_ptrs, mask=h_mask[:, None], other=0.0)

    # Online softmax state — acc is the BLOCK_DV slice only.
    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DV], dtype=tl.float32)

    topk_len = tl.load(TopkLen_ptr + pid_b)

    # Iterate over selected positions in chunks of KV_CHUNK.
    # TOPK is constexpr (padded multiple of KV_CHUNK), so this loop unrolls
    # at compile time per Triton's static_range semantics for `range` over
    # a constexpr.
    for chunk_start in range(0, TOPK, KV_CHUNK):
        kv_pos = chunk_start + tl.arange(0, KV_CHUNK)
        in_topk = kv_pos < TOPK

        flat_idx = tl.load(
            Indices_ptr + pid_b * stride_idx_b + kv_pos,
            mask=in_topk, other=-1,
        )
        # Position is valid iff (a) within topk_length, (b) flat index >= 0.
        # The reference (`flashmla_tests/ref.py:81-87`) clamps min to 0 before
        # gather; we replicate that to keep the address in-range, then mask
        # the logit to -inf so it contributes nothing.
        valid = (kv_pos < topk_len) & (flat_idx >= 0) & in_topk
        flat_safe = tl.maximum(flat_idx, 0)

        page_id = flat_safe // P
        row = flat_safe % P
        # int64 byte arithmetic — page_id * page_byte_stride can exceed 2^31
        # for large pools (page_byte_stride ~150 KB).
        page_id64 = page_id.to(tl.int64)
        row64 = row.to(tl.int64)
        token_byte_base = page_id64 * page_byte_stride + row64 * 576
        scale_byte_base = page_id64 * page_byte_stride + P * 576 + row64 * 8

        # ===== Build full K [KV_CHUNK, 512] for QK^T =====
        # NoPE FP8 (masked outside NoPE; those positions become 0 then are
        # overwritten by the RoPE branch via additive merge).
        nope_byte_offs = token_byte_base[:, None] + nope_d_safe[None, :]
        nope_load_mask = valid[:, None] & is_nope_d[None, :]
        nope_fp32 = tl.load(
            KV_FP8_ptr + nope_byte_offs,
            mask=nope_load_mask, other=0.0,
        ).to(tl.float32)

        # Direct per-element scale gather: scale_byte[k, d] is at
        #   uint8 offset = scale_byte_base[k] + nope_group_d[d]
        # No tl.dot needed — this is a true gather, not a matmul.
        scale_byte_offs = scale_byte_base[:, None] + nope_group_d[None, :]
        scale_bytes = tl.load(
            KV_U8_ptr + scale_byte_offs,
            mask=nope_load_mask, other=0,
        ).to(tl.float32)
        nope_scale = tl.exp2(scale_bytes - 127.0)  # [KV_CHUNK, 512]
        nope_deq = nope_fp32 * nope_scale          # 0 outside NoPE (mask)

        # RoPE BF16. bf16 element offset = (byte offset) >> 1.
        rope_elem_base = (token_byte_base + HEAD_DIM_NOPE) >> 1
        rope_elem_offs = rope_elem_base[:, None] + rope_d_safe[None, :]
        rope_load_mask = valid[:, None] & (~is_nope_d)[None, :]
        rope_fp32 = tl.load(
            KV_BF16_ptr + rope_elem_offs,
            mask=rope_load_mask, other=0.0,
        ).to(tl.float32)

        k_full_bf = (nope_deq + rope_fp32).to(tl.bfloat16)

        # QK^T: [BLOCK_M, 512] @ [512, KV_CHUNK] → [BLOCK_M, KV_CHUNK]
        qk = tl.dot(q_bf, tl.trans(k_full_bf), out_dtype=tl.float32)
        qk = qk * softmax_scale
        qk = tl.where(valid[None, :], qk, -float("inf"))

        # --- Online softmax update ---
        m_chunk = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_chunk)
        # Guard against m_new == -inf (no valid tokens yet) → exp(-inf - -inf) = NaN.
        m_safe = tl.where(m_new == -float("inf"), 0.0, m_new)
        alpha = tl.exp(m_i - m_safe)        # rescales prior accumulator
        p = tl.exp(qk - m_safe[:, None])    # qk = -inf masked → p = 0

        l_i = l_i * alpha + tl.sum(p, axis=1)

        # ===== Build K_slice [KV_CHUNK, BLOCK_DV] for V@P =====
        nope_dv_byte_offs = token_byte_base[:, None] + nope_dv_safe[None, :]
        nope_dv_load_mask = valid[:, None] & is_nope_dv[None, :]
        nope_dv_fp32 = tl.load(
            KV_FP8_ptr + nope_dv_byte_offs,
            mask=nope_dv_load_mask, other=0.0,
        ).to(tl.float32)
        scale_dv_byte_offs = scale_byte_base[:, None] + nope_group_dv[None, :]
        scale_dv_bytes = tl.load(
            KV_U8_ptr + scale_dv_byte_offs,
            mask=nope_dv_load_mask, other=0,
        ).to(tl.float32)
        nope_dv_deq = nope_dv_fp32 * tl.exp2(scale_dv_bytes - 127.0)

        rope_dv_elem_offs = rope_elem_base[:, None] + rope_dv_safe[None, :]
        rope_dv_load_mask = valid[:, None] & (~is_nope_dv)[None, :]
        rope_dv_fp32 = tl.load(
            KV_BF16_ptr + rope_dv_elem_offs,
            mask=rope_dv_load_mask, other=0.0,
        ).to(tl.float32)

        k_slice_bf = (nope_dv_deq + rope_dv_fp32).to(tl.bfloat16)

        # acc += p @ K_slice → [BLOCK_M, BLOCK_DV]
        acc = acc * alpha[:, None] + tl.dot(
            p.to(tl.bfloat16), k_slice_bf, out_dtype=tl.float32,
        )
        m_i = m_new

    # ===== Phase 6.7: compressed scope (combined online softmax) =====
    # Iterate over compressed-cache selected tokens with the SAME m_i/l_i/acc
    # state. Mathematically this is one combined logsumexp over the
    # concatenation of SWA-selected and compressed-selected tokens, exactly
    # matching `flashmla_tests/ref.py:96-104,120-130`.
    if HAS_EXTRA:
        extra_topk_len = tl.load(EXTRA_TopkLen_ptr + pid_b)

        for chunk_start in range(0, EXTRA_TOPK, KV_CHUNK):
            kv_pos = chunk_start + tl.arange(0, KV_CHUNK)
            in_topk = kv_pos < EXTRA_TOPK

            flat_idx = tl.load(
                EXTRA_Indices_ptr + pid_b * stride_extra_idx_b + kv_pos,
                mask=in_topk, other=-1,
            )
            valid = (kv_pos < extra_topk_len) & (flat_idx >= 0) & in_topk
            flat_safe = tl.maximum(flat_idx, 0)

            page_id = flat_safe // P_EXTRA
            row = flat_safe % P_EXTRA
            page_id64 = page_id.to(tl.int64)
            row64 = row.to(tl.int64)
            token_byte_base = page_id64 * extra_page_byte_stride + row64 * 576
            scale_byte_base = (
                page_id64 * extra_page_byte_stride + P_EXTRA * 576 + row64 * 8
            )

            # === Build full K [KV_CHUNK, 512] for QK^T from compressed cache ===
            nope_byte_offs = token_byte_base[:, None] + nope_d_safe[None, :]
            nope_load_mask = valid[:, None] & is_nope_d[None, :]
            nope_fp32 = tl.load(
                EXTRA_KV_FP8_ptr + nope_byte_offs,
                mask=nope_load_mask, other=0.0,
            ).to(tl.float32)

            scale_byte_offs = scale_byte_base[:, None] + nope_group_d[None, :]
            scale_bytes = tl.load(
                EXTRA_KV_U8_ptr + scale_byte_offs,
                mask=nope_load_mask, other=0,
            ).to(tl.float32)
            nope_scale = tl.exp2(scale_bytes - 127.0)
            nope_deq = nope_fp32 * nope_scale

            rope_elem_base = (token_byte_base + HEAD_DIM_NOPE) >> 1
            rope_elem_offs = rope_elem_base[:, None] + rope_d_safe[None, :]
            rope_load_mask = valid[:, None] & (~is_nope_d)[None, :]
            rope_fp32 = tl.load(
                EXTRA_KV_BF16_ptr + rope_elem_offs,
                mask=rope_load_mask, other=0.0,
            ).to(tl.float32)

            k_full_bf = (nope_deq + rope_fp32).to(tl.bfloat16)

            qk = tl.dot(q_bf, tl.trans(k_full_bf), out_dtype=tl.float32)
            qk = qk * softmax_scale
            qk = tl.where(valid[None, :], qk, -float("inf"))

            m_chunk = tl.max(qk, axis=1)
            m_new = tl.maximum(m_i, m_chunk)
            m_safe = tl.where(m_new == -float("inf"), 0.0, m_new)
            alpha = tl.exp(m_i - m_safe)
            p = tl.exp(qk - m_safe[:, None])

            l_i = l_i * alpha + tl.sum(p, axis=1)

            # === Build K_slice [KV_CHUNK, BLOCK_DV] for V@P from compressed ===
            nope_dv_byte_offs = token_byte_base[:, None] + nope_dv_safe[None, :]
            nope_dv_load_mask = valid[:, None] & is_nope_dv[None, :]
            nope_dv_fp32 = tl.load(
                EXTRA_KV_FP8_ptr + nope_dv_byte_offs,
                mask=nope_dv_load_mask, other=0.0,
            ).to(tl.float32)
            scale_dv_byte_offs = scale_byte_base[:, None] + nope_group_dv[None, :]
            scale_dv_bytes = tl.load(
                EXTRA_KV_U8_ptr + scale_dv_byte_offs,
                mask=nope_dv_load_mask, other=0,
            ).to(tl.float32)
            nope_dv_deq = nope_dv_fp32 * tl.exp2(scale_dv_bytes - 127.0)

            rope_dv_elem_offs = rope_elem_base[:, None] + rope_dv_safe[None, :]
            rope_dv_load_mask = valid[:, None] & (~is_nope_dv)[None, :]
            rope_dv_fp32 = tl.load(
                EXTRA_KV_BF16_ptr + rope_dv_elem_offs,
                mask=rope_dv_load_mask, other=0.0,
            ).to(tl.float32)

            k_slice_bf = (nope_dv_deq + rope_dv_fp32).to(tl.bfloat16)

            acc = acc * alpha[:, None] + tl.dot(
                p.to(tl.bfloat16), k_slice_bf, out_dtype=tl.float32,
            )
            m_i = m_new

    # --- Epilogue: sink-scaled output, lonely-query correction ---
    sink = tl.load(Sink_ptr + h_offs, mask=h_mask, other=0.0).to(tl.float32)
    lonely = l_i == 0.0  # no valid tokens for this row

    safe_l = tl.where(lonely, 1.0, l_i)
    lse = tl.where(lonely, float("inf"), m_i + tl.log(safe_l))

    o = acc / safe_l[:, None]
    sink_factor = 1.0 / (1.0 + tl.exp(sink - lse))    # safe even when lse=+inf
    o = o * sink_factor[:, None]
    o = tl.where(lonely[:, None], 0.0, o)             # lonely → zero output

    # Store output slice
    o_ptrs = (Out_ptr
              + pid_b * stride_o_b
              + h_offs[:, None] * stride_o_h
              + dv_offs[None, :])
    tl.store(o_ptrs, o.to(tl.bfloat16), mask=h_mask[:, None])

    # Store LSE only from pid_dv == 0 (identical across z).
    if pid_dv == 0:
        lse_ptrs = Lse_ptr + pid_b * stride_lse_b + h_offs * stride_lse_h
        tl.store(lse_ptrs, lse, mask=h_mask)


# ---------------------------------------------------------------------------
# Phase 7.3: Split-KV partial + merge kernels
# ---------------------------------------------------------------------------
#
# Partial kernel: clone of `_sparse_mla_decode_kernel` parameterised by
# NUM_SPLITS along grid axis 0. Each (b, split) pair processes a contiguous
# slice of the unified `[0, TOPK + EXTRA_TOPK)` selected-token axis.
# Splits that span the SWA/compressed boundary handle both halves via the
# per-loop `in_split` mask. Output is partial_O (no sink applied, no
# lonely +inf correction) and partial_LSE (-inf for empty splits).
#
# Merge kernel: per (b, h_block, dv_block), reads NUM_SPLITS partials,
# applies safe-max logsumexp + safe-denominator merge weights + sink +
# lonely-query correction, writes final Out + LSE.

@triton.jit
def _sparse_mla_decode_partial_kernel(
    Q_ptr,
    KV_FP8_ptr, KV_BF16_ptr, KV_U8_ptr,
    Indices_ptr, TopkLen_ptr,
    EXTRA_KV_FP8_ptr, EXTRA_KV_BF16_ptr, EXTRA_KV_U8_ptr,
    EXTRA_Indices_ptr, EXTRA_TopkLen_ptr,
    PartialO_ptr,     # [B, num_splits, h_q, num_dv_blocks, BLOCK_DV] fp32
    PartialLse_ptr,   # [B, num_splits, h_q] fp32
    softmax_scale,
    page_byte_stride, P,
    extra_page_byte_stride, P_EXTRA,
    h_q,
    stride_q_b, stride_q_h,
    stride_idx_b,
    stride_extra_idx_b,
    stride_po_b, stride_po_split, stride_po_h, stride_po_dv,
    stride_pl_b, stride_pl_split, stride_pl_h,
    BLOCK_M: tl.constexpr,
    KV_CHUNK: tl.constexpr,
    HEAD_DIM_NOPE: tl.constexpr,
    HEAD_DIM_ROPE: tl.constexpr,
    HEAD_DIM_QK: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    TOPK: tl.constexpr,
    EXTRA_TOPK: tl.constexpr,
    HAS_EXTRA: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    NOPE_GROUPS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    # Decompose grid axis 0 into (pid_b, pid_split). Folded together for
    # capture stability and launch coalescing.
    pid0 = tl.program_id(0)
    pid_split = pid0 % NUM_SPLITS
    pid_b = pid0 // NUM_SPLITS
    pid_h = tl.program_id(1)
    pid_dv = tl.program_id(2)

    # Static split bounds over the unified [0, TOPK + EXTRA_TOPK) axis.
    # All inputs are constexpr → boundaries are compile-time and capture-stable.
    TOTAL_TOPK: tl.constexpr = TOPK + EXTRA_TOPK
    CHUNKS_TOTAL: tl.constexpr = (TOTAL_TOPK + KV_CHUNK - 1) // KV_CHUNK
    CHUNKS_PER_SPLIT: tl.constexpr = (CHUNKS_TOTAL + NUM_SPLITS - 1) // NUM_SPLITS
    split_chunk_begin = pid_split * CHUNKS_PER_SPLIT
    split_chunk_end = pid_split * CHUNKS_PER_SPLIT + CHUNKS_PER_SPLIT
    logical_begin = split_chunk_begin * KV_CHUNK
    # Cap end at TOTAL_TOPK so trailing splits with NUM_SPLITS > CHUNKS_TOTAL
    # produce zero-width [logical_begin, logical_end) and exit cleanly via
    # the in_split mask below.
    logical_end = tl.minimum(TOTAL_TOPK, split_chunk_end * KV_CHUNK)

    h_offs = pid_h * BLOCK_M + tl.arange(0, BLOCK_M)
    h_mask = h_offs < h_q

    d_qk = tl.arange(0, HEAD_DIM_QK)
    is_nope_d = d_qk < HEAD_DIM_NOPE
    nope_d_safe = tl.where(is_nope_d, d_qk, 0)
    nope_group_d = nope_d_safe // GROUP_SIZE
    rope_d_within = d_qk - HEAD_DIM_NOPE
    rope_d_safe = tl.maximum(rope_d_within, 0)

    dv_offs = pid_dv * BLOCK_DV + tl.arange(0, BLOCK_DV)
    is_nope_dv = dv_offs < HEAD_DIM_NOPE
    nope_dv_safe = tl.where(is_nope_dv, dv_offs, 0)
    nope_group_dv = nope_dv_safe // GROUP_SIZE
    rope_dv_within = dv_offs - HEAD_DIM_NOPE
    rope_dv_safe = tl.maximum(rope_dv_within, 0)

    q_ptrs = (Q_ptr
              + pid_b * stride_q_b
              + h_offs[:, None] * stride_q_h
              + d_qk[None, :])
    q_bf = tl.load(q_ptrs, mask=h_mask[:, None], other=0.0)

    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DV], dtype=tl.float32)

    topk_len = tl.load(TopkLen_ptr + pid_b)

    # ===== Primary (SWA) loop, gated to this split's logical range =====
    for chunk_start in range(0, TOPK, KV_CHUNK):
        kv_pos = chunk_start + tl.arange(0, KV_CHUNK)
        in_topk = kv_pos < TOPK
        # Logical position equals kv_pos for the primary scope.
        in_split = (kv_pos >= logical_begin) & (kv_pos < logical_end)

        flat_idx = tl.load(
            Indices_ptr + pid_b * stride_idx_b + kv_pos,
            mask=in_topk & in_split, other=-1,
        )
        valid = (kv_pos < topk_len) & (flat_idx >= 0) & in_topk & in_split
        flat_safe = tl.maximum(flat_idx, 0)

        page_id = flat_safe // P
        row = flat_safe % P
        page_id64 = page_id.to(tl.int64)
        row64 = row.to(tl.int64)
        token_byte_base = page_id64 * page_byte_stride + row64 * 576
        scale_byte_base = page_id64 * page_byte_stride + P * 576 + row64 * 8

        nope_byte_offs = token_byte_base[:, None] + nope_d_safe[None, :]
        nope_load_mask = valid[:, None] & is_nope_d[None, :]
        nope_fp32 = tl.load(
            KV_FP8_ptr + nope_byte_offs, mask=nope_load_mask, other=0.0,
        ).to(tl.float32)
        scale_byte_offs = scale_byte_base[:, None] + nope_group_d[None, :]
        scale_bytes = tl.load(
            KV_U8_ptr + scale_byte_offs, mask=nope_load_mask, other=0,
        ).to(tl.float32)
        nope_scale = tl.exp2(scale_bytes - 127.0)
        nope_deq = nope_fp32 * nope_scale

        rope_elem_base = (token_byte_base + HEAD_DIM_NOPE) >> 1
        rope_elem_offs = rope_elem_base[:, None] + rope_d_safe[None, :]
        rope_load_mask = valid[:, None] & (~is_nope_d)[None, :]
        rope_fp32 = tl.load(
            KV_BF16_ptr + rope_elem_offs, mask=rope_load_mask, other=0.0,
        ).to(tl.float32)
        k_full_bf = (nope_deq + rope_fp32).to(tl.bfloat16)

        qk = tl.dot(q_bf, tl.trans(k_full_bf), out_dtype=tl.float32)
        qk = qk * softmax_scale
        qk = tl.where(valid[None, :], qk, -float("inf"))

        m_chunk = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_chunk)
        m_safe = tl.where(m_new == -float("inf"), 0.0, m_new)
        alpha = tl.exp(m_i - m_safe)
        p = tl.exp(qk - m_safe[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)

        nope_dv_byte_offs = token_byte_base[:, None] + nope_dv_safe[None, :]
        nope_dv_load_mask = valid[:, None] & is_nope_dv[None, :]
        nope_dv_fp32 = tl.load(
            KV_FP8_ptr + nope_dv_byte_offs,
            mask=nope_dv_load_mask, other=0.0,
        ).to(tl.float32)
        scale_dv_byte_offs = scale_byte_base[:, None] + nope_group_dv[None, :]
        scale_dv_bytes = tl.load(
            KV_U8_ptr + scale_dv_byte_offs,
            mask=nope_dv_load_mask, other=0,
        ).to(tl.float32)
        nope_dv_deq = nope_dv_fp32 * tl.exp2(scale_dv_bytes - 127.0)

        rope_dv_elem_offs = rope_elem_base[:, None] + rope_dv_safe[None, :]
        rope_dv_load_mask = valid[:, None] & (~is_nope_dv)[None, :]
        rope_dv_fp32 = tl.load(
            KV_BF16_ptr + rope_dv_elem_offs,
            mask=rope_dv_load_mask, other=0.0,
        ).to(tl.float32)
        k_slice_bf = (nope_dv_deq + rope_dv_fp32).to(tl.bfloat16)

        acc = acc * alpha[:, None] + tl.dot(
            p.to(tl.bfloat16), k_slice_bf, out_dtype=tl.float32,
        )
        m_i = m_new

    # ===== Compressed scope loop, gated to this split's logical range =====
    if HAS_EXTRA:
        extra_topk_len = tl.load(EXTRA_TopkLen_ptr + pid_b)

        for chunk_start in range(0, EXTRA_TOPK, KV_CHUNK):
            kv_pos = chunk_start + tl.arange(0, KV_CHUNK)
            in_topk = kv_pos < EXTRA_TOPK
            # Compressed tokens occupy logical positions [TOPK, TOPK + EXTRA_TOPK).
            logical_pos = TOPK + kv_pos
            in_split = (logical_pos >= logical_begin) & (logical_pos < logical_end)

            flat_idx = tl.load(
                EXTRA_Indices_ptr + pid_b * stride_extra_idx_b + kv_pos,
                mask=in_topk & in_split, other=-1,
            )
            valid = (
                (kv_pos < extra_topk_len) & (flat_idx >= 0)
                & in_topk & in_split
            )
            flat_safe = tl.maximum(flat_idx, 0)

            page_id = flat_safe // P_EXTRA
            row = flat_safe % P_EXTRA
            page_id64 = page_id.to(tl.int64)
            row64 = row.to(tl.int64)
            token_byte_base = page_id64 * extra_page_byte_stride + row64 * 576
            scale_byte_base = (
                page_id64 * extra_page_byte_stride + P_EXTRA * 576 + row64 * 8
            )

            nope_byte_offs = token_byte_base[:, None] + nope_d_safe[None, :]
            nope_load_mask = valid[:, None] & is_nope_d[None, :]
            nope_fp32 = tl.load(
                EXTRA_KV_FP8_ptr + nope_byte_offs,
                mask=nope_load_mask, other=0.0,
            ).to(tl.float32)
            scale_byte_offs = scale_byte_base[:, None] + nope_group_d[None, :]
            scale_bytes = tl.load(
                EXTRA_KV_U8_ptr + scale_byte_offs,
                mask=nope_load_mask, other=0,
            ).to(tl.float32)
            nope_scale = tl.exp2(scale_bytes - 127.0)
            nope_deq = nope_fp32 * nope_scale

            rope_elem_base = (token_byte_base + HEAD_DIM_NOPE) >> 1
            rope_elem_offs = rope_elem_base[:, None] + rope_d_safe[None, :]
            rope_load_mask = valid[:, None] & (~is_nope_d)[None, :]
            rope_fp32 = tl.load(
                EXTRA_KV_BF16_ptr + rope_elem_offs,
                mask=rope_load_mask, other=0.0,
            ).to(tl.float32)
            k_full_bf = (nope_deq + rope_fp32).to(tl.bfloat16)

            qk = tl.dot(q_bf, tl.trans(k_full_bf), out_dtype=tl.float32)
            qk = qk * softmax_scale
            qk = tl.where(valid[None, :], qk, -float("inf"))

            m_chunk = tl.max(qk, axis=1)
            m_new = tl.maximum(m_i, m_chunk)
            m_safe = tl.where(m_new == -float("inf"), 0.0, m_new)
            alpha = tl.exp(m_i - m_safe)
            p = tl.exp(qk - m_safe[:, None])
            l_i = l_i * alpha + tl.sum(p, axis=1)

            nope_dv_byte_offs = token_byte_base[:, None] + nope_dv_safe[None, :]
            nope_dv_load_mask = valid[:, None] & is_nope_dv[None, :]
            nope_dv_fp32 = tl.load(
                EXTRA_KV_FP8_ptr + nope_dv_byte_offs,
                mask=nope_dv_load_mask, other=0.0,
            ).to(tl.float32)
            scale_dv_byte_offs = scale_byte_base[:, None] + nope_group_dv[None, :]
            scale_dv_bytes = tl.load(
                EXTRA_KV_U8_ptr + scale_dv_byte_offs,
                mask=nope_dv_load_mask, other=0,
            ).to(tl.float32)
            nope_dv_deq = nope_dv_fp32 * tl.exp2(scale_dv_bytes - 127.0)

            rope_dv_elem_offs = rope_elem_base[:, None] + rope_dv_safe[None, :]
            rope_dv_load_mask = valid[:, None] & (~is_nope_dv)[None, :]
            rope_dv_fp32 = tl.load(
                EXTRA_KV_BF16_ptr + rope_dv_elem_offs,
                mask=rope_dv_load_mask, other=0.0,
            ).to(tl.float32)
            k_slice_bf = (nope_dv_deq + rope_dv_fp32).to(tl.bfloat16)

            acc = acc * alpha[:, None] + tl.dot(
                p.to(tl.bfloat16), k_slice_bf, out_dtype=tl.float32,
            )
            m_i = m_new

    # --- Partial epilogue: NO sink, NO lonely-query +inf correction ---
    # Empty splits write partial_LSE = -inf so the merge kernel's safe-max
    # logsumexp masks them out cleanly.
    empty = l_i == 0.0
    safe_l = tl.where(empty, 1.0, l_i)
    partial_lse_val = tl.where(empty, -float("inf"), m_i + tl.log(safe_l))
    partial_o = acc / safe_l[:, None]
    partial_o = tl.where(empty[:, None], 0.0, partial_o)

    # Store partial_O[pid_b, pid_split, h_offs, pid_dv, dv_offs]
    po_ptrs = (PartialO_ptr
               + pid_b * stride_po_b
               + pid_split * stride_po_split
               + h_offs[:, None] * stride_po_h
               + pid_dv * stride_po_dv
               + tl.arange(0, BLOCK_DV)[None, :])
    tl.store(po_ptrs, partial_o, mask=h_mask[:, None])

    # Store partial_LSE[pid_b, pid_split, h_offs] only from pid_dv == 0.
    if pid_dv == 0:
        pl_ptrs = (PartialLse_ptr
                   + pid_b * stride_pl_b
                   + pid_split * stride_pl_split
                   + h_offs * stride_pl_h)
        tl.store(pl_ptrs, partial_lse_val, mask=h_mask)


@triton.jit
def _sparse_mla_merge_kernel(
    PartialO_ptr,    # [B, num_splits, h_q, num_dv_blocks, BLOCK_DV] fp32
    PartialLse_ptr,  # [B, num_splits, h_q] fp32
    Sink_ptr,        # [h_q] fp32 or bf16
    Out_ptr,         # [B, 1, h_q, HEAD_DIM_QK] bf16
    Lse_ptr,         # [B, h_q, 1] fp32
    h_q,
    stride_po_b, stride_po_split, stride_po_h, stride_po_dv,
    stride_pl_b, stride_pl_split, stride_pl_h,
    stride_o_b, stride_o_h,
    stride_lse_b, stride_lse_h,
    BLOCK_M: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
):
    """Merge per-split partial outputs into final attention output.

    Stage 1: safe-max logsumexp over splits (handles -inf partials cleanly).
    Stage 2: safe-denominator merge weights via partial_LSE - lse_total.
    Stage 3: weighted output sum across splits.
    Stage 4: post-merge sink scaling + lonely-query +inf correction.
    """
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_dv = tl.program_id(2)

    h_offs = pid_h * BLOCK_M + tl.arange(0, BLOCK_M)
    h_mask = h_offs < h_q
    dv_offs = pid_dv * BLOCK_DV + tl.arange(0, BLOCK_DV)
    split_offs = tl.arange(0, NUM_SPLITS)

    # Load partial LSE [NUM_SPLITS, BLOCK_M].
    pl_ptrs = (PartialLse_ptr
               + pid_b * stride_pl_b
               + split_offs[:, None] * stride_pl_split
               + h_offs[None, :] * stride_pl_h)
    partial_lse = tl.load(pl_ptrs, mask=h_mask[None, :], other=-float("inf"))

    # Stage 1: safe-max logsumexp across splits (axis 0).
    m = tl.max(partial_lse, axis=0)                       # [BLOCK_M]
    valid = m != -float("inf")
    safe_m = tl.where(valid, m, 0.0)
    sumexp = tl.sum(tl.exp(partial_lse - safe_m[None, :]), axis=0)  # [BLOCK_M]
    lse_total = tl.where(valid, safe_m + tl.log(sumexp), -float("inf"))

    # Stage 2: safe-denominator merge weights.
    safe_lse = tl.where(valid, lse_total, 0.0)
    delta = partial_lse - safe_lse[None, :]               # [NUM_SPLITS, BLOCK_M]
    weights = tl.where(valid[None, :], tl.exp(delta), 0.0)

    # Stage 3: weighted output sum. Load partial_O
    # [NUM_SPLITS, BLOCK_M, BLOCK_DV].
    po_ptrs = (PartialO_ptr
               + pid_b * stride_po_b
               + split_offs[:, None, None] * stride_po_split
               + h_offs[None, :, None] * stride_po_h
               + pid_dv * stride_po_dv
               + tl.arange(0, BLOCK_DV)[None, None, :])
    po_mask = h_mask[None, :, None]
    partial_o = tl.load(po_ptrs, mask=po_mask, other=0.0)

    o_merged = tl.sum(
        weights[:, :, None] * partial_o, axis=0
    )                                                     # [BLOCK_M, BLOCK_DV]

    # Stage 4: post-merge sink + lonely-query.
    sink = tl.load(Sink_ptr + h_offs, mask=h_mask, other=0.0).to(tl.float32)
    lonely = ~valid
    lse_final = tl.where(lonely, float("inf"), lse_total)
    sink_factor = 1.0 / (1.0 + tl.exp(sink - lse_final))
    o = o_merged * sink_factor[:, None]
    o = tl.where(lonely[:, None], 0.0, o)

    # Store output slice.
    o_ptrs = (Out_ptr
              + pid_b * stride_o_b
              + h_offs[:, None] * stride_o_h
              + dv_offs[None, :])
    tl.store(o_ptrs, o.to(tl.bfloat16), mask=h_mask[:, None])

    # Store final LSE only from pid_dv == 0.
    if pid_dv == 0:
        lse_ptrs = Lse_ptr + pid_b * stride_lse_b + h_offs * stride_lse_h
        tl.store(lse_ptrs, lse_final, mask=h_mask)


def flash_mla_with_kvcache_triton_sm120(
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
    block_table: Optional[torch.Tensor] = None,
    cache_seqlens: Optional[torch.Tensor] = None,
    tile_scheduler_metadata: Any = None,
    is_fp8_kvcache: bool = True,
    causal: bool = False,
    num_splits: Optional[int] = None,
    **_unused: Any,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """sm_120 Triton sparse MLA decode (s_q==1).

    Phase 6.2: SWA-only (extra_k_cache None).
    Phase 6.7: combined SWA + compressed C4/C128 (extra_k_cache non-None).

    Drop-in for `flash_mla.flash_mla_with_kvcache(..., indices=...)` matching
    the live call site at
    `deepseek_v4_backend_radix.py:1074-1089`. Returns
    `(output bf16 [B, 1, h_q, 512], lse fp32 [B, h_q, 1])`.

    All contract assertions match the wrapper preconditions enumerated in
    `sparse_mla_backend.get_sparse_mla_decode_backend`'s `_triton_supported`
    predicate; the dispatcher routes incompatible calls elsewhere, but the
    asserts here are defence-in-depth.

    SPECULATIVE DECODING NOTE (Phase 7 analysis): the `s_q == 1` assert
    is structurally satisfied for `speculative_num_draft_tokens > 1` and
    target-verify code paths. The main sparse-MLA Q path flattens all
    leading dimensions in `MQALayer._compute_q_b` and `_forward_prepare`
    via `q = q.view(-1, n_local_heads, head_dim)`
    (`models/deepseek_v4.py:1414-1416` and `:1527-1528`), so q is always
    3-D `[bs_effective, n_heads, head_dim]` for the flattened
    multi-token draft path (`speculative_num_draft_tokens > 1`). The
    conditional
    unsqueeze at `deepseek_v4_backend_radix.py:1065-1067` then yields
    `[bs_effective, 1, n_heads, head_dim]` — always s_q==1 with
    effective batch `B*qo_len`.

    This proves the contract for flattened multi-token speculative
    decoding via `speculative_num_draft_tokens > 1`. It does NOT cover
    `speculative_eagle_topk > 1`, which has a separate cap at
    `deepseek_v4_backend_radix.py:412` and would need its own analysis
    of how multi-branch EAGLE constructs query tensors. Phase 7 does
    not lift that cap.
    """
    # Contract: live call site never passes these.
    assert block_table is None
    assert cache_seqlens is None
    assert is_fp8_kvcache is True
    assert not causal
    # Phase 7.3: num_splits > 1 enables split-KV (partial + merge two-stage
    # launch). num_splits=None means auto-pick=1 for production stability.
    # `SGLANG_SPARSE_MLA_NUM_SPLITS` env var allows explicit override.
    if num_splits is None:
        from sglang.srt.environ import envs as _envs
        env_splits = _envs.SGLANG_SPARSE_MLA_NUM_SPLITS.get()
        num_splits = env_splits if env_splits is not None else 1
    # The merge kernel materializes [NUM_SPLITS, BLOCK_M, BLOCK_DV] fp32
    # tiles. Beyond ~8 splits this becomes a register-spill catastrophe
    # without redesigning the merge to stream over splits. Restrict to a
    # small tested set of power-of-2 values; anything else needs a kernel
    # redesign before being safe.
    assert num_splits in (1, 2, 4, 8), (
        f"num_splits={num_splits} not in supported set (1, 2, 4, 8). "
        f"Larger values would require redesigning the merge kernel to "
        f"stream over splits rather than materialize a [NUM_SPLITS, "
        f"BLOCK_M, BLOCK_DV] tile."
    )

    # Compressed-scope contract: all-or-nothing trio.
    has_extra = extra_k_cache is not None
    if has_extra:
        assert extra_indices_in_kvcache is not None
        assert extra_topk_length is not None
    else:
        assert extra_indices_in_kvcache is None
        assert extra_topk_length is None

    assert q.ndim == 4, f"q must be [B, s_q, h_q, d_qk]; got {tuple(q.shape)}"
    B, s_q, h_q, d_qk = q.shape
    assert s_q == 1, (
        "Triton sparse-MLA path expects Contract A flattened decode "
        "([B_effective, 1, H, D]); true s_q>1 is not implemented"
    )
    assert d_qk == 512
    assert head_dim_v == 512
    assert h_q % 16 == 0, f"h_q must be divisible by BLOCK_M=16; got {h_q}"
    assert q.dtype == torch.bfloat16
    # Kernel relies on contiguous innermost stride for Q (per-element pointer
    # arithmetic over d_qk uses unit stride implicitly).
    assert q.stride(3) == 1, f"q must be contiguous on last dim; got stride {q.stride(3)}"

    assert k_cache.ndim == 4 and k_cache.shape[2] == 1 and k_cache.shape[3] == 584
    assert k_cache.dtype == torch.uint8
    P = k_cache.shape[1]
    page_byte_stride = k_cache.stride(0) * k_cache.element_size()
    assert page_byte_stride % 576 == 0, (
        f"page byte stride {page_byte_stride} must be a multiple of 576"
    )
    assert page_byte_stride >= P * 584, (
        f"page byte stride {page_byte_stride} too small for P={P}"
    )
    assert page_byte_stride % 2 == 0  # required for bf16 view alignment

    assert indices is not None and indices.ndim == 3
    assert indices.shape == (B, 1, indices.shape[-1])
    topk = indices.shape[-1]
    assert topk % 64 == 0
    assert indices.dtype == torch.int32
    # Kernel uses unit-stride access along the topk dim.
    assert indices.stride(2) == 1, (
        f"indices must be contiguous on last dim; got stride {indices.stride(2)}"
    )

    assert topk_length is not None
    assert topk_length.shape == (B,)
    assert topk_length.dtype == torch.int32

    assert attn_sink is not None
    assert attn_sink.shape == (h_q,)

    devices = {q.device, k_cache.device, indices.device,
               topk_length.device, attn_sink.device}
    if has_extra:
        assert extra_k_cache.ndim == 4 and extra_k_cache.shape[2] == 1
        assert extra_k_cache.shape[3] == 584
        assert extra_k_cache.dtype == torch.uint8
        P_extra = extra_k_cache.shape[1]
        extra_page_byte_stride = (
            extra_k_cache.stride(0) * extra_k_cache.element_size()
        )
        assert extra_page_byte_stride % 576 == 0
        assert extra_page_byte_stride >= P_extra * 584
        assert extra_page_byte_stride % 2 == 0

        assert extra_indices_in_kvcache.ndim == 3
        assert extra_indices_in_kvcache.shape[0] == B
        assert extra_indices_in_kvcache.shape[1] == 1
        extra_topk = extra_indices_in_kvcache.shape[-1]
        assert extra_topk % 64 == 0
        assert extra_indices_in_kvcache.dtype == torch.int32
        assert extra_indices_in_kvcache.stride(2) == 1

        assert extra_topk_length.shape == (B,)
        assert extra_topk_length.dtype == torch.int32

        devices.update({
            extra_k_cache.device,
            extra_indices_in_kvcache.device,
            extra_topk_length.device,
        })
    else:
        P_extra = 1
        extra_page_byte_stride = 0
        extra_topk = 0

    assert len(devices) == 1, f"all tensors must share a device; got {devices}"
    assert q.is_cuda

    if softmax_scale is None:
        softmax_scale = float(d_qk) ** -0.5
    softmax_scale = float(softmax_scale)

    # Reinterpret the cache storage at three dtypes (no copy).
    kv_fp8 = k_cache.view(torch.float8_e4m3fn)
    kv_bf16 = k_cache.view(torch.bfloat16)
    kv_u8 = k_cache  # already uint8

    if has_extra:
        ekv_fp8 = extra_k_cache.view(torch.float8_e4m3fn)
        ekv_bf16 = extra_k_cache.view(torch.bfloat16)
        ekv_u8 = extra_k_cache
        eindices = extra_indices_in_kvcache
        etopk_len = extra_topk_length
        stride_extra_idx_b = extra_indices_in_kvcache.stride(0)
    else:
        # Pass primary tensors as placeholders; the kernel's HAS_EXTRA=False
        # branch never dereferences them.
        ekv_fp8 = kv_fp8
        ekv_bf16 = kv_bf16
        ekv_u8 = kv_u8
        eindices = indices
        etopk_len = topk_length
        stride_extra_idx_b = indices.stride(0)

    # Pre-allocate outputs.
    output = torch.empty((B, 1, h_q, head_dim_v), dtype=torch.bfloat16, device=q.device)
    lse = torch.empty((B, h_q, 1), dtype=torch.float32, device=q.device)

    # Strides (in element units of the relevant tensor).
    stride_q_b = q.stride(0)
    stride_q_h = q.stride(2)
    stride_idx_b = indices.stride(0)
    stride_o_b = output.stride(0)
    stride_o_h = output.stride(2)
    stride_lse_b = lse.stride(0)
    stride_lse_h = lse.stride(1)

    BLOCK_M = 16
    KV_CHUNK = 32
    HEAD_DIM_NOPE = 448
    HEAD_DIM_ROPE = 64
    HEAD_DIM_QK = 512
    BLOCK_DV = 256
    assert HEAD_DIM_QK % BLOCK_DV == 0

    # The kernel reads attn_sink and upcasts to fp32 internally
    # (`tl.load(Sink_ptr).to(tl.float32)`), so we don't need to allocate
    # an fp32 tensor here. Allocating inside the captured forward used to
    # add a per-layer aten::_to_copy op that was a needless alloc/copy.
    sink = attn_sink

    # Spec-tracking guard + capture trace. Triton specializations include
    # constexpr meta values AND pointer dtypes — all included so that a JIT
    # mismatch between warmup and capture is caught explicitly.
    spec = (
        # Phase 7.3: split mode is part of the Triton specialization. The
        # num_splits=1 fast path launches `_sparse_mla_decode_kernel`; the
        # >1 path launches `_sparse_mla_decode_partial_kernel` plus
        # `_sparse_mla_merge_kernel`. Keep them separate in the spec so
        # capture-warmup tracking distinguishes them.
        "split" if num_splits > 1 else "single",
        int(num_splits),
        bool(has_extra),
        int(topk),
        int(extra_topk),
        str(q.dtype),
        str(k_cache.dtype),
        str(indices.dtype),
        str(topk_length.dtype),
        str(attn_sink.dtype),
        str(extra_k_cache.dtype) if has_extra else None,
        str(extra_indices_in_kvcache.dtype) if has_extra else None,
        str(extra_topk_length.dtype) if has_extra else None,
    )
    capturing = _capture_trace_once(
        "sparse_mla", spec, warmed=_SPARSE_MLA_WARMED_SPECS, phase="enter",
    )
    if capturing and spec not in _SPARSE_MLA_WARMED_SPECS:
        import sys
        sys.stderr.write(
            f"[capture-trace] FATAL unwarmed sparse_mla spec={spec} "
            f"warmed={sorted(_SPARSE_MLA_WARMED_SPECS)}\n"
        )
        sys.stderr.flush()
        raise RuntimeError(
            f"sparse_mla_triton_sm120: spec {spec} hit CUDA graph capture "
            f"without prior warmup. Known warmed specs: "
            f"{sorted(_SPARSE_MLA_WARMED_SPECS)}."
        )

    if num_splits == 1:
        # Fast path: existing single-kernel implementation, unchanged.
        _launch_single_split(
            q, kv_fp8, kv_bf16, kv_u8, indices, topk_length, sink, output, lse,
            ekv_fp8, ekv_bf16, ekv_u8, eindices, etopk_len,
            softmax_scale, page_byte_stride, P, extra_page_byte_stride, P_extra,
            h_q, stride_q_b, stride_q_h, stride_idx_b, stride_extra_idx_b,
            stride_o_b, stride_o_h, stride_lse_b, stride_lse_h,
            B, BLOCK_M, KV_CHUNK, HEAD_DIM_NOPE, HEAD_DIM_ROPE,
            HEAD_DIM_QK, BLOCK_DV, topk, extra_topk, has_extra,
        )
        if not capturing:
            _SPARSE_MLA_WARMED_SPECS.add(spec)
        _capture_trace_once(
            "sparse_mla", spec, warmed=_SPARSE_MLA_WARMED_SPECS, phase="exit",
        )
        return output, lse

    # Phase 7.3 split-KV path: num_splits > 1 → partial + merge two-stage launch.
    num_dv_blocks = HEAD_DIM_QK // BLOCK_DV
    partial_o = torch.empty(
        (B, num_splits, h_q, num_dv_blocks, BLOCK_DV),
        dtype=torch.float32, device=q.device,
    )
    partial_lse = torch.empty(
        (B, num_splits, h_q),
        dtype=torch.float32, device=q.device,
    )
    stride_po_b = partial_o.stride(0)
    stride_po_split = partial_o.stride(1)
    stride_po_h = partial_o.stride(2)
    stride_po_dv = partial_o.stride(3)
    stride_pl_b = partial_lse.stride(0)
    stride_pl_split = partial_lse.stride(1)
    stride_pl_h = partial_lse.stride(2)

    partial_grid = (B * num_splits, h_q // BLOCK_M, num_dv_blocks)
    _sparse_mla_decode_partial_kernel[partial_grid](
        q, kv_fp8, kv_bf16, kv_u8, indices, topk_length,
        ekv_fp8, ekv_bf16, ekv_u8, eindices, etopk_len,
        partial_o, partial_lse,
        softmax_scale,
        page_byte_stride, P, extra_page_byte_stride, P_extra,
        h_q,
        stride_q_b, stride_q_h,
        stride_idx_b, stride_extra_idx_b,
        stride_po_b, stride_po_split, stride_po_h, stride_po_dv,
        stride_pl_b, stride_pl_split, stride_pl_h,
        BLOCK_M=BLOCK_M, KV_CHUNK=KV_CHUNK,
        HEAD_DIM_NOPE=HEAD_DIM_NOPE, HEAD_DIM_ROPE=HEAD_DIM_ROPE,
        HEAD_DIM_QK=HEAD_DIM_QK, BLOCK_DV=BLOCK_DV,
        TOPK=topk, EXTRA_TOPK=extra_topk, HAS_EXTRA=has_extra,
        NUM_SPLITS=num_splits,
        NOPE_GROUPS=7, GROUP_SIZE=64,
    )

    merge_grid = (B, h_q // BLOCK_M, num_dv_blocks)
    _sparse_mla_merge_kernel[merge_grid](
        partial_o, partial_lse, sink, output, lse,
        h_q,
        stride_po_b, stride_po_split, stride_po_h, stride_po_dv,
        stride_pl_b, stride_pl_split, stride_pl_h,
        stride_o_b, stride_o_h,
        stride_lse_b, stride_lse_h,
        BLOCK_M=BLOCK_M, BLOCK_DV=BLOCK_DV,
        NUM_SPLITS=num_splits,
    )

    if not capturing:
        _SPARSE_MLA_WARMED_SPECS.add(spec)
    _capture_trace_once(
        "sparse_mla", spec, warmed=_SPARSE_MLA_WARMED_SPECS, phase="exit",
    )
    return output, lse


def _launch_single_split(
    q, kv_fp8, kv_bf16, kv_u8, indices, topk_length, sink, output, lse,
    ekv_fp8, ekv_bf16, ekv_u8, eindices, etopk_len,
    softmax_scale, page_byte_stride, P, extra_page_byte_stride, P_extra,
    h_q, stride_q_b, stride_q_h, stride_idx_b, stride_extra_idx_b,
    stride_o_b, stride_o_h, stride_lse_b, stride_lse_h,
    B, BLOCK_M, KV_CHUNK, HEAD_DIM_NOPE, HEAD_DIM_ROPE,
    HEAD_DIM_QK, BLOCK_DV, topk, extra_topk, has_extra,
):
    """num_splits=1 fast-path: single-kernel launch with the existing
    `_sparse_mla_decode_kernel`. No partial buffers, no merge kernel."""
    grid = (B, h_q // BLOCK_M, HEAD_DIM_QK // BLOCK_DV)
    _sparse_mla_decode_kernel[grid](
        q,
        kv_fp8,
        kv_bf16,
        kv_u8,
        indices,
        topk_length,
        sink,
        output,
        lse,
        ekv_fp8,
        ekv_bf16,
        ekv_u8,
        eindices,
        etopk_len,
        softmax_scale,
        page_byte_stride,
        P,
        extra_page_byte_stride,
        P_extra,
        h_q,
        stride_q_b, stride_q_h,
        stride_idx_b,
        stride_extra_idx_b,
        stride_o_b, stride_o_h,
        stride_lse_b, stride_lse_h,
        BLOCK_M=BLOCK_M,
        KV_CHUNK=KV_CHUNK,
        HEAD_DIM_NOPE=HEAD_DIM_NOPE,
        HEAD_DIM_ROPE=HEAD_DIM_ROPE,
        HEAD_DIM_QK=HEAD_DIM_QK,
        BLOCK_DV=BLOCK_DV,
        TOPK=topk,
        EXTRA_TOPK=extra_topk,
        HAS_EXTRA=has_extra,
        NOPE_GROUPS=7,
        GROUP_SIZE=64,
    )


if __name__ == "__main__":
    compile_aot()
