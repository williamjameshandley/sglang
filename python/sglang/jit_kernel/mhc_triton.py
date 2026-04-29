"""Triton replacements for the V4-Flash MHC TileLang kernels.

Phase 7.4 of the V4-Flash sm_120 plan: replace the PCG-reachable
TileLang kernels with Triton equivalents so that piecewise CUDA graph
capture (which uses Dynamo to find graph boundaries) does not trip
TileLang's eager-builder `inspect.getsourcelines()` introspection.

Kernels in this module:

  * `_mhc_pre_gemm_sqrsum_splitk_stage_0_kernel` /
    `_mhc_pre_gemm_sqrsum_splitk_stage_1_kernel` (Phase 7.4.1):
    split-K replacement for the `num_tokens <= 2048` MHC pre path.
    Stage 0 computes per-split partial outputs (GEMM + sum-of-squares);
    Stage 1 reduces across splits.

  * `_mhc_pre_gemm_sqrsum_kernel` (Phase 7.4.2): simple variant for
    `num_tokens > 2048`.

  * Phase 7.4.3-7.4.6: big_fuse, mhc_post, sinkhorn — added in
    subsequent steps.

The reference oracle for all of these is the torch fallback in
`python/sglang/srt/models/deepseek_v4.py` (`hc_pre_torch_impl`,
`hc_post_torch_impl`), NOT the TileLang implementation.
"""
from __future__ import annotations

from typing import Tuple

import torch
import triton
import triton.language as tl


@triton.jit
def _mhc_pre_gemm_sqrsum_splitk_stage_0_kernel(
    X_ptr,             # [num_tokens, hc_hidden] bf16
    Fn_ptr,            # [hc_mult3, hc_hidden] fp32
    OutPartial_ptr,    # [split_k, num_tokens, 32] fp32 (32 = padded hc_mult3)
    SqrPartial_ptr,    # [split_k, num_tokens] fp32
    num_tokens,
    stride_x_t, stride_x_h,
    stride_fn_n, stride_fn_h,
    stride_op_split, stride_op_t, stride_op_n,
    stride_sp_split, stride_sp_t,
    HC_MULT3: tl.constexpr,    # actual columns; padded to 32 in workspace
    HIDDEN: tl.constexpr,      # full hidden dim per token
    SPLIT_K: tl.constexpr,
    HIDDEN_BLOCK: tl.constexpr,
    TOKEN_BLOCK: tl.constexpr,
):
    """Stage 0: per-(token_block, split) partial GEMM + sum-of-squares.

    Each program covers TOKEN_BLOCK tokens and one split (a contiguous
    slice of the hidden dim). Loops over the slice in HIDDEN_BLOCK chunks
    accumulating both `out += x @ fn[hc_mult3].T` and
    `sqrsum += sum(x*x along hidden)`.

    Unique writes per (token_block, split): no atomics needed.
    """
    pid_tok = tl.program_id(0)
    pid_split = tl.program_id(1)

    SPLIT_SIZE: tl.constexpr = HIDDEN // SPLIT_K  # caller asserts evenly divisible

    tok_offs = pid_tok * TOKEN_BLOCK + tl.arange(0, TOKEN_BLOCK)
    tok_mask = tok_offs < num_tokens

    # Padded HC_MULT3 to 32 to match the TileLang workspace contract.
    # Triton's tl.arange requires power-of-2 size.
    n_offs = tl.arange(0, 32)
    n_mask = n_offs < HC_MULT3

    out_acc = tl.zeros([TOKEN_BLOCK, 32], dtype=tl.float32)
    sqr_acc = tl.zeros([TOKEN_BLOCK], dtype=tl.float32)

    h_block_offs = tl.arange(0, HIDDEN_BLOCK)

    k_base = pid_split * SPLIT_SIZE
    num_h_chunks = SPLIT_SIZE // HIDDEN_BLOCK

    for pz in range(0, num_h_chunks):
        h_offs = k_base + pz * HIDDEN_BLOCK + h_block_offs

        # Load x [TOKEN_BLOCK, HIDDEN_BLOCK] bf16
        x_ptrs = (X_ptr
                  + tok_offs[:, None] * stride_x_t
                  + h_offs[None, :] * stride_x_h)
        x_bf = tl.load(x_ptrs, mask=tok_mask[:, None], other=0.0)
        x_f32 = x_bf.to(tl.float32)

        # Per-tile sum of squares
        sqr_acc += tl.sum(x_f32 * x_f32, axis=1)

        # Load fn[:, slice] [32, HIDDEN_BLOCK] fp32 (rows beyond hc_mult3 = 0)
        fn_ptrs = (Fn_ptr
                   + n_offs[:, None] * stride_fn_n
                   + h_offs[None, :] * stride_fn_h)
        fn_f = tl.load(fn_ptrs, mask=n_mask[:, None], other=0.0)

        # Tensor-core dot: bf16 x bf16 -> fp32. Cast fn to bf16 for the dot;
        # the fp32 fn values are reproducible in bf16 within the precision the
        # TileLang reference also gets (it casts x to fp32 separately and
        # accumulates the same product).
        fn_bf = fn_f.to(tl.bfloat16)
        out_acc += tl.dot(x_bf, tl.trans(fn_bf), out_dtype=tl.float32)

    # Store partials. Each (pid_tok, pid_split) program writes a unique slab.
    op_ptrs = (OutPartial_ptr
               + pid_split * stride_op_split
               + tok_offs[:, None] * stride_op_t
               + n_offs[None, :] * stride_op_n)
    tl.store(op_ptrs, out_acc, mask=tok_mask[:, None])

    sp_ptrs = (SqrPartial_ptr
               + pid_split * stride_sp_split
               + tok_offs * stride_sp_t)
    tl.store(sp_ptrs, sqr_acc, mask=tok_mask)


@triton.jit
def _mhc_pre_gemm_sqrsum_splitk_stage_1_kernel(
    OutPartial_ptr,    # [split_k, num_tokens, 32] fp32
    SqrPartial_ptr,    # [split_k, num_tokens] fp32
    Out_ptr,           # [num_tokens, hc_mult3] fp32
    Sqrsum_ptr,        # [num_tokens] fp32
    num_tokens,
    stride_op_split, stride_op_t, stride_op_n,
    stride_sp_split, stride_sp_t,
    stride_o_t, stride_o_n,
    stride_s_t,
    HC_MULT3: tl.constexpr,
    SPLIT_K: tl.constexpr,    # power-of-2
    BLOCK_T: tl.constexpr,    # tokens per program
):
    """Stage 1: reduce per-split partials into final out / sqrsum.

    Each program covers BLOCK_T tokens. Sums across SPLIT_K partials.
    SPLIT_K is constexpr power-of-2; if the caller's actual split_k is
    smaller, it pads to the next pow-2 (with zero-init partials).
    """
    pid_t = tl.program_id(0)

    tok_offs = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    tok_mask = tok_offs < num_tokens

    split_offs = tl.arange(0, SPLIT_K)

    # Reduce sqrsum across splits.
    sp_ptrs = (SqrPartial_ptr
               + split_offs[:, None] * stride_sp_split
               + tok_offs[None, :] * stride_sp_t)
    sp = tl.load(sp_ptrs, mask=tok_mask[None, :], other=0.0)
    sqr_total = tl.sum(sp, axis=0)

    s_ptrs = Sqrsum_ptr + tok_offs * stride_s_t
    tl.store(s_ptrs, sqr_total, mask=tok_mask)

    # Reduce out partials across splits, but only the first HC_MULT3 cols
    # are meaningful (partials beyond that are zero from the masked store).
    n_offs = tl.arange(0, 32)
    n_mask = n_offs < HC_MULT3

    op_ptrs = (OutPartial_ptr
               + split_offs[:, None, None] * stride_op_split
               + tok_offs[None, :, None] * stride_op_t
               + n_offs[None, None, :] * stride_op_n)
    op = tl.load(
        op_ptrs,
        mask=tok_mask[None, :, None] & n_mask[None, None, :],
        other=0.0,
    )
    out_total = tl.sum(op, axis=0)

    o_ptrs = (Out_ptr
              + tok_offs[:, None] * stride_o_t
              + n_offs[None, :] * stride_o_n)
    tl.store(
        o_ptrs, out_total,
        mask=tok_mask[:, None] & n_mask[None, :],
    )


def mhc_pre_gemm_sqrsum_splitk_triton(
    x: torch.Tensor,
    fn: torch.Tensor,
    hc_mult3: int,
    n_splits_pre: int = 32,
    token_block: int = 32,
    hidden_block: int = 256,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Triton replacement for `mhc_pre_gemm_sqrsum_splitk_kernel` factory
    (the `num_tokens <= 2048` path).

    Args:
        x:   `[num_tokens, hc_hidden]` BF16 (contiguous on last dim).
        fn:  `[hc_mult3, hc_hidden]` FP32.
        hc_mult3: actual rows of fn (≤ 32).
        n_splits_pre: split-K factor along hidden dim. Must divide
            `hc_hidden` evenly and yield `split_size % hidden_block == 0`.

    Returns:
        out: `[num_tokens, hc_mult3]` FP32.
        sqrsum: `[num_tokens]` FP32.

    Empty-input fast path: `num_tokens == 0` returns correctly-shaped
    empty tensors WITHOUT launching any kernel.
    """
    num_tokens, hc_hidden = x.shape
    assert fn.ndim == 2 and fn.shape == (hc_mult3, hc_hidden), (
        f"fn shape {tuple(fn.shape)} must be ({hc_mult3}, {hc_hidden})"
    )
    assert hc_mult3 <= 32, f"hc_mult3={hc_mult3} exceeds workspace pad of 32"
    assert hc_hidden % n_splits_pre == 0, (
        f"hc_hidden={hc_hidden} must be divisible by n_splits_pre={n_splits_pre}"
    )
    split_size = hc_hidden // n_splits_pre
    assert split_size % hidden_block == 0, (
        f"split_size={split_size} must be divisible by hidden_block={hidden_block}"
    )
    assert x.dtype == torch.bfloat16
    assert fn.dtype == torch.float32
    assert x.is_cuda and fn.is_cuda
    assert x.device == fn.device

    # Empty-input fast path: no kernel launch, return correctly-shaped zeros.
    if num_tokens == 0:
        out = torch.empty((0, hc_mult3), dtype=torch.float32, device=x.device)
        sqrsum = torch.empty((0,), dtype=torch.float32, device=x.device)
        return out, sqrsum

    # SPLIT_K must be pow-2 for tl.arange; pad up if needed.
    split_k_pad = 1
    while split_k_pad < n_splits_pre:
        split_k_pad *= 2
    assert split_k_pad == n_splits_pre, (
        f"n_splits_pre={n_splits_pre} must be a power of 2 (got pad {split_k_pad})"
    )

    out_partial = torch.zeros(
        (n_splits_pre, num_tokens, 32), dtype=torch.float32, device=x.device,
    )
    sqr_partial = torch.zeros(
        (n_splits_pre, num_tokens), dtype=torch.float32, device=x.device,
    )

    grid_0 = (triton.cdiv(num_tokens, token_block), n_splits_pre)
    _mhc_pre_gemm_sqrsum_splitk_stage_0_kernel[grid_0](
        x, fn, out_partial, sqr_partial,
        num_tokens,
        x.stride(0), x.stride(1),
        fn.stride(0), fn.stride(1),
        out_partial.stride(0), out_partial.stride(1), out_partial.stride(2),
        sqr_partial.stride(0), sqr_partial.stride(1),
        HC_MULT3=hc_mult3,
        HIDDEN=hc_hidden,
        SPLIT_K=n_splits_pre,
        HIDDEN_BLOCK=hidden_block,
        TOKEN_BLOCK=token_block,
    )

    out = torch.empty((num_tokens, hc_mult3), dtype=torch.float32, device=x.device)
    sqrsum = torch.empty((num_tokens,), dtype=torch.float32, device=x.device)

    BLOCK_T_STAGE1 = 32
    grid_1 = (triton.cdiv(num_tokens, BLOCK_T_STAGE1),)
    _mhc_pre_gemm_sqrsum_splitk_stage_1_kernel[grid_1](
        out_partial, sqr_partial, out, sqrsum,
        num_tokens,
        out_partial.stride(0), out_partial.stride(1), out_partial.stride(2),
        sqr_partial.stride(0), sqr_partial.stride(1),
        out.stride(0), out.stride(1),
        sqrsum.stride(0),
        HC_MULT3=hc_mult3,
        SPLIT_K=n_splits_pre,
        BLOCK_T=BLOCK_T_STAGE1,
    )

    return out, sqrsum


@triton.jit
def _mhc_pre_gemm_sqrsum_kernel(
    X_ptr,             # [num_tokens, hc_hidden] bf16
    Fn_ptr,            # [hc_mult3, hc_hidden] fp32
    Out_ptr,           # [num_tokens, hc_mult3] fp32
    Sqrsum_ptr,        # [num_tokens] fp32
    num_tokens,
    stride_x_t, stride_x_h,
    stride_fn_n, stride_fn_h,
    stride_o_t, stride_o_n,
    stride_s_t,
    HC_MULT3: tl.constexpr,
    HIDDEN: tl.constexpr,
    HIDDEN_BLOCK: tl.constexpr,
    TOKEN_BLOCK: tl.constexpr,
):
    """Single-stage GEMM + sqrsum (no split-K).

    One program per token block. Loops over the full hidden dim in
    HIDDEN_BLOCK chunks. Mirrors `mhc_pre_gemm_sqrsum_tilelang` at
    `srt/layers/mhc.py:268-339` for `num_tokens > 2048`.
    """
    pid_tok = tl.program_id(0)

    tok_offs = pid_tok * TOKEN_BLOCK + tl.arange(0, TOKEN_BLOCK)
    tok_mask = tok_offs < num_tokens

    n_offs = tl.arange(0, 32)
    n_mask = n_offs < HC_MULT3

    out_acc = tl.zeros([TOKEN_BLOCK, 32], dtype=tl.float32)
    sqr_acc = tl.zeros([TOKEN_BLOCK], dtype=tl.float32)

    h_block_offs = tl.arange(0, HIDDEN_BLOCK)
    num_h_chunks = HIDDEN // HIDDEN_BLOCK

    for pz in range(0, num_h_chunks):
        h_offs = pz * HIDDEN_BLOCK + h_block_offs

        x_ptrs = (X_ptr
                  + tok_offs[:, None] * stride_x_t
                  + h_offs[None, :] * stride_x_h)
        x_bf = tl.load(x_ptrs, mask=tok_mask[:, None], other=0.0)
        x_f32 = x_bf.to(tl.float32)

        sqr_acc += tl.sum(x_f32 * x_f32, axis=1)

        fn_ptrs = (Fn_ptr
                   + n_offs[:, None] * stride_fn_n
                   + h_offs[None, :] * stride_fn_h)
        fn_f = tl.load(fn_ptrs, mask=n_mask[:, None], other=0.0)
        fn_bf = fn_f.to(tl.bfloat16)
        out_acc += tl.dot(x_bf, tl.trans(fn_bf), out_dtype=tl.float32)

    o_ptrs = (Out_ptr
              + tok_offs[:, None] * stride_o_t
              + n_offs[None, :] * stride_o_n)
    tl.store(o_ptrs, out_acc, mask=tok_mask[:, None] & n_mask[None, :])

    s_ptrs = Sqrsum_ptr + tok_offs * stride_s_t
    tl.store(s_ptrs, sqr_acc, mask=tok_mask)


def mhc_pre_gemm_sqrsum_triton(
    x: torch.Tensor,
    fn: torch.Tensor,
    hc_mult3: int,
    token_block: int = 32,
    hidden_block: int = 256,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Triton replacement for `mhc_pre_gemm_sqrsum_tilelang` (the
    `num_tokens > 2048` path)."""
    num_tokens, hc_hidden = x.shape
    assert fn.ndim == 2 and fn.shape == (hc_mult3, hc_hidden), (
        f"fn shape {tuple(fn.shape)} must be ({hc_mult3}, {hc_hidden})"
    )
    assert hc_mult3 <= 32
    assert hc_hidden % hidden_block == 0, (
        f"hc_hidden={hc_hidden} must be divisible by hidden_block={hidden_block}"
    )
    assert x.dtype == torch.bfloat16
    assert fn.dtype == torch.float32
    assert x.is_cuda and fn.is_cuda
    assert x.device == fn.device

    if num_tokens == 0:
        out = torch.empty((0, hc_mult3), dtype=torch.float32, device=x.device)
        sqrsum = torch.empty((0,), dtype=torch.float32, device=x.device)
        return out, sqrsum

    out = torch.empty((num_tokens, hc_mult3), dtype=torch.float32, device=x.device)
    sqrsum = torch.empty((num_tokens,), dtype=torch.float32, device=x.device)

    grid = (triton.cdiv(num_tokens, token_block),)
    _mhc_pre_gemm_sqrsum_kernel[grid](
        x, fn, out, sqrsum,
        num_tokens,
        x.stride(0), x.stride(1),
        fn.stride(0), fn.stride(1),
        out.stride(0), out.stride(1),
        sqrsum.stride(0),
        HC_MULT3=hc_mult3,
        HIDDEN=hc_hidden,
        HIDDEN_BLOCK=hidden_block,
        TOKEN_BLOCK=token_block,
    )

    return out, sqrsum


@triton.jit
def _mhc_post_kernel(
    A_ptr,        # [n, hc, hc] fp32  (comb_res_mix)
    B_ptr,        # [n, hc, h]  bf16  (residual)
    C_ptr,        # [n, hc]     fp32  (post_layer_mix)
    D_ptr,        # [n, h]      bf16  (attention output x)
    Out_ptr,      # [n, hc, h]  bf16
    n_tokens,
    stride_a_n, stride_a_co, stride_a_ci,
    stride_b_n, stride_b_c, stride_b_h,
    stride_c_n, stride_c_c,
    stride_d_n, stride_d_h,
    stride_o_n, stride_o_c, stride_o_h,
    HC: tl.constexpr,           # padded power-of-2; mask via HC_REAL
    HC_REAL: tl.constexpr,
    HIDDEN: tl.constexpr,
    H_BLOCK: tl.constexpr,
):
    """One program per (token, hidden tile). Each program loads
    a[t]: [HC, HC], c[t]: [HC], and a tile of b[t]: [HC, H_BLOCK]
    and d[t]: [H_BLOCK]; computes
        out[i, j] = c[i] * d[j] + sum_k a[k, i] * b[k, j]
    in fp32, casts to bf16, stores.
    """
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)

    co = tl.arange(0, HC)
    ci = tl.arange(0, HC)
    co_mask = co < HC_REAL
    ci_mask = ci < HC_REAL

    h_offs = pid_h * H_BLOCK + tl.arange(0, H_BLOCK)
    h_mask = h_offs < HIDDEN

    # Load a [HC, HC] fp32 (rows beyond HC_REAL = 0 via mask)
    a_ptrs = (A_ptr + pid_t * stride_a_n
              + ci[:, None] * stride_a_ci
              + co[None, :] * stride_a_co)
    a_tile = tl.load(a_ptrs, mask=ci_mask[:, None] & co_mask[None, :], other=0.0)

    # Load c [HC] fp32
    c_ptrs = C_ptr + pid_t * stride_c_n + co * stride_c_c
    c_tile = tl.load(c_ptrs, mask=co_mask, other=0.0)

    # Load b [HC, H_BLOCK] bf16 -> fp32
    b_ptrs = (B_ptr + pid_t * stride_b_n
              + ci[:, None] * stride_b_c
              + h_offs[None, :] * stride_b_h)
    b_tile = tl.load(b_ptrs,
                     mask=ci_mask[:, None] & h_mask[None, :],
                     other=0.0).to(tl.float32)

    # Load d [H_BLOCK] bf16 -> fp32
    d_ptrs = D_ptr + pid_t * stride_d_n + h_offs * stride_d_h
    d_tile = tl.load(d_ptrs, mask=h_mask, other=0.0).to(tl.float32)

    # out[co, h] = c[co] * d[h] + sum_ci a[ci, co] * b[ci, h]
    # = (c[:, None] * d[None, :]) + (a.T @ b)
    cd = c_tile[:, None] * d_tile[None, :]                             # [HC, H_BLOCK]
    # a is [HC_ci, HC_co]; want sum over ci of a[ci, co] * b[ci, h]
    # tl.dot expects 2D-2D bf16/fp32. Cast a to bf16 to use tensor cores.
    a_bf = a_tile.to(tl.bfloat16)
    b_bf = b_tile.to(tl.bfloat16)
    ab = tl.dot(tl.trans(a_bf), b_bf, out_dtype=tl.float32)            # [HC_co, H_BLOCK]

    out_tile = (cd + ab).to(tl.bfloat16)

    o_ptrs = (Out_ptr + pid_t * stride_o_n
              + co[:, None] * stride_o_c
              + h_offs[None, :] * stride_o_h)
    tl.store(o_ptrs, out_tile, mask=co_mask[:, None] & h_mask[None, :])


def mhc_post_triton(
    x: torch.Tensor,                # [n, h] bf16  (attention output)
    residual: torch.Tensor,         # [n, hc, h] bf16
    post_layer_mix: torch.Tensor,   # [n, hc] or [n, hc, 1] fp32
    comb_res_mix: torch.Tensor,     # [n, hc, hc] fp32
) -> torch.Tensor:
    """Triton replacement for `mhc_post_tilelang`.

    Computes per-token, per-head, per-position:
        out[i, c, j] = post[i, c] * x[i, j]
                     + sum_k comb[i, k, c] * residual[i, k, j]

    Returns `[n, hc, h]` bf16 matching the existing torch fallback at
    `models/deepseek_v4.py:1875-1880`.
    """
    if post_layer_mix.dim() == 3 and post_layer_mix.shape[-1] == 1:
        post_layer_mix = post_layer_mix.squeeze(-1)

    assert x.dim() == 2 and residual.dim() == 3
    n, hidden = x.shape
    assert residual.shape[0] == n and residual.shape[2] == hidden
    hc = residual.shape[1]
    assert post_layer_mix.shape == (n, hc)
    assert comb_res_mix.shape == (n, hc, hc)
    assert x.dtype == torch.bfloat16
    assert residual.dtype == torch.bfloat16
    assert post_layer_mix.dtype == torch.float32
    assert comb_res_mix.dtype == torch.float32
    assert x.is_cuda and residual.is_cuda
    assert post_layer_mix.is_cuda and comb_res_mix.is_cuda

    out = torch.empty((n, hc, hidden), dtype=torch.bfloat16, device=x.device)
    if n == 0:
        return out

    # tl.arange wants pow-2; pad HC to 16 so V4-Flash hc=4 fits with margin.
    hc_pad = 1
    while hc_pad < max(hc, 16):
        hc_pad *= 2
    # tl.dot requires inner dim ≥ 16 in many backends; the cast to bf16
    # already lifted it to ≥16 with HC=16 padding for V4-Flash hc=4.

    H_BLOCK = 256
    grid = (n, triton.cdiv(hidden, H_BLOCK))
    # comb_res_mix has shape (n, hc_ci, hc_co): axis 1 is the inner head
    # ("k" in the oracle's `sum_k comb[i, k, c]`), axis 2 is the output head.
    # Pass stride(1)=ci stride, stride(2)=co stride to match kernel param order.
    _mhc_post_kernel[grid](
        comb_res_mix, residual, post_layer_mix, x, out,
        n,
        comb_res_mix.stride(0), comb_res_mix.stride(2), comb_res_mix.stride(1),
        residual.stride(0), residual.stride(1), residual.stride(2),
        post_layer_mix.stride(0), post_layer_mix.stride(1),
        x.stride(0), x.stride(1),
        out.stride(0), out.stride(1), out.stride(2),
        HC=hc_pad,
        HC_REAL=hc,
        HIDDEN=hidden,
        H_BLOCK=H_BLOCK,
    )
    return out
