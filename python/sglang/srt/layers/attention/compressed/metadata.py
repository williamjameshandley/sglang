from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Any, List, Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.utils import is_hip

if TYPE_CHECKING:
    from flash_mla.flash_mla_interface import FlashMLASchedMeta


"""
Some comments on the common terms used in DeepSeekV4Backend:

topk_lengths:
    NOTE: TL;DR: topk_lengths == seq_lens
    The FlashMLA sparse decode kernel will attend to `k` tokens for each query.
    `topk_lengths` indicates how many tokens each query will attend to.
    This should be named as `seq_lens`, but we simply follow the naming convention.

page_table:
    The page table indicates which pages each request is assigned to.
    Each value in the page table is the page index in the TokenToKVPool.
    This page index is irrelevant to the actual `page_size`.

page_indices:
    The real indices used to index into the KV cache.
    This can be computed from the `page_table` and `page_size`.
    e.g. page_indices[i, j] = page_table[i, j // page_size] * page_size + (j % page_size)
    For sparse C4 top-512 attention, the indices will be selected from the C4 page indices.
    In implementation, we don't materialize the full C4 `page_indices`,
    but calculate them from `page_table` on-the-fly in the attention kernel.

positions:
    The position of the last token for each request.
    For compress token, the positions must be times of compress ratio.
    For example, for C4, raw_position=11 will trigger a compression,
    But the RoPE's position, during compression, must be 8 instead of 11.

Some other notes:
    c4_ / c128_: means "compressed by 4" / "compressed by 128".
    c4_page_size: page_size // 4
    c4_seq_lens: seq_lens // 4, but bounded by at least 1, due to flash_mla requirement.
    c4_sparse: means "compressed by 4" but only attend to top-512 tokens.
               all related length will be clipped to 512.
"""


def copy_metadata(
    *,
    src,
    dst,
    check_eq_fields: List[str],
    copy_fields: List[str],
    assign_fields: Optional[List[str]] = None,
):
    assign_fields = assign_fields or []

    for field_name in check_eq_fields:
        src_val = getattr(src, field_name)
        dst_val = getattr(dst, field_name)
        assert src_val == dst_val, f"{field_name=} {src_val=} {dst_val=}"

    for field_name in copy_fields:
        src_val = getattr(src, field_name)
        dst_val = getattr(dst, field_name)
        assert dst_val is not None, f"{field_name=} {src_val=} {dst_val=}"
        dst_val.copy_(src_val)

    for field_name in assign_fields:
        setattr(dst, field_name, getattr(src, field_name))

    provided_fields = check_eq_fields + copy_fields + assign_fields
    assert len(provided_fields) == len(
        set(provided_fields)
    ), f"{provided_fields=} has dup"
    all_fields = {f.name for f in fields(src)}
    assert set(provided_fields) == all_fields, f"{provided_fields=} {all_fields=}"


def create_flashmla_metadata():
    # Phase 6.3: replaced env-var-string check with the per-call backend
    # selector's `need_flashmla_metadata()` predicate so HIP, sm_120, and
    # any explicit non-FlashMLA backend share a single source of truth.
    from sglang.srt.layers.attention.sparse_mla_backend import (
        need_flashmla_metadata,
    )

    if not need_flashmla_metadata():
        return None

    import flash_mla

    return flash_mla.get_mla_metadata()[0]


@dataclass
class CoreMetadata:
    positions: torch.Tensor  # needed for sliding window and others
    # NOTE: swa_out_loc only applies to indices that needs to be written
    # to the swa_kv_pool. For prefill, we will take a slicing Tensor
    # that selects the k/v values that needs to be written.
    swa_slice: Optional[torch.Tensor]
    swa_out_loc_sliced: torch.Tensor
    # NOTE: c4/c128 out_loc will mask the invalid write locations to 0.
    # When no compression happens, out_loc will be 0, which is the "padded slot"
    c4_out_loc: torch.Tensor
    c128_out_loc: torch.Tensor

    def init_swa_slice(self, swa_slice: torch.Tensor):
        assert self.swa_slice is None, "can only update once"
        self.swa_slice = swa_slice
        self.swa_out_loc_sliced = self.swa_out_loc_sliced[swa_slice]

    def copy_(self, other):
        raise NotImplementedError


@dataclass
class IndexerMetadata:
    def copy_(self, other):
        raise NotImplementedError


@dataclass
class PagedIndexerMetadata(IndexerMetadata):
    page_size: int
    page_table: torch.Tensor
    c4_seq_lens: torch.Tensor
    deep_gemm_metadata: Any = field(init=False, repr=False)

    def __post_init__(self):
        from sglang.srt.layers.attention.compressed.paged_mqa_backend import (
            get_paged_mqa_logits_backend,
            uses_deep_gemm_metadata,
        )

        backend = get_paged_mqa_logits_backend()
        if not uses_deep_gemm_metadata(backend):
            # torch fallback / HIP / sm_120 Triton / TileLang: no DeepGEMM
            # metadata is consumed downstream.
            self.deep_gemm_metadata = None
        else:
            import deep_gemm

            if envs.SGLANG_OPT_DG_PAGED_MQA_LOGITS_CHUNK_SIZE.get() != -1:
                from sglang.srt.layers.deep_gemm_wrapper.paged_mqa_logits import (
                    get_paged_mqa_logits_metadata_chunked as get_paged_mqa_logits_metadata,
                )
            else:
                from deep_gemm import get_paged_mqa_logits_metadata

            # PR #324 csrc/apis/attention.hpp asserts context_lens.dim() == 2.
            # Pass a 2-D [B, 1] view for the DeepGEMM metadata path; the
            # downstream kernel call site reshapes the same way.
            context_lens_2d = self.c4_seq_lens.to(torch.int32).view(-1, 1)
            self.deep_gemm_metadata = get_paged_mqa_logits_metadata(
                context_lens_2d,
                self.c4_page_size,
                deep_gemm.get_num_sms(),
            )

            if envs.SGLANG_OPT_DG_PAGED_MQA_LOGITS_CHUNK_SIZE.get() != -1:
                pass
            else:
                # It is a tensor, thus our CUDA graph replay copy will be easier (just copy it)
                assert isinstance(self.deep_gemm_metadata, torch.Tensor)

        assert self.page_size == 256

    @property
    def c4_page_size(self) -> int:
        return self.page_size // 4

    @property
    def max_seq_len(self) -> int:
        return self.page_table.shape[1] * self.page_size

    @property
    def c4_max_seq_len(self) -> int:
        """Compressed-page length expected by the paged-MQA logits kernel.

        The C4 indexer cache uses `c4_page_size = page_size // 4 = 64`
        bytes per compressed token, so the kernel-side max context length
        is `page_table.shape[1] * c4_page_size`. The Triton sm_120 wrapper
        used `max_seq_len` (raw `page_size`) for output-allocation only
        and computed over `max_pages * block_size` internally; deepgemm's
        `fp8_paged_mqa_logits` uses `max_context_len` for grid scheduling,
        so passing the raw-page value 4× too large stalls the kernel.
        """
        return self.page_table.shape[1] * self.c4_page_size

    def copy_(self, other: "PagedIndexerMetadata"):
        from sglang.srt.layers.attention.compressed.paged_mqa_backend import (
            get_paged_mqa_logits_backend,
            uses_deep_gemm_metadata,
        )

        if uses_deep_gemm_metadata(get_paged_mqa_logits_backend()):
            copy_metadata(
                src=other,
                dst=self,
                check_eq_fields=["page_size"],
                copy_fields=["page_table", "c4_seq_lens", "deep_gemm_metadata"],
            )
        else:
            # torch / sm_120 Triton / TileLang / HIP: deep_gemm_metadata is
            # always None on both sides; check equality rather than copy.
            copy_metadata(
                src=other,
                dst=self,
                check_eq_fields=["page_size", "deep_gemm_metadata"],
                copy_fields=["page_table", "c4_seq_lens"],
            )


@dataclass
class PagedCoreMetadata(CoreMetadata):
    page_table: torch.Tensor
    # sliding window attention (core)
    swa_page_indices: torch.Tensor  # at most (sum_qo_len, 128)
    swa_topk_lengths: torch.Tensor  # clipped to 128
    # C128 dense attention (core)
    c128_page_indices: torch.Tensor
    c128_topk_lengths_clamp1: torch.Tensor
    # C4 sparse attention (core)
    c4_topk_lengths_raw: torch.Tensor
    c4_topk_lengths_clamp1: torch.Tensor  # i.e. c4_seq_lens
    c4_sparse_topk: int  # must be 512
    c4_sparse_topk_lengths: torch.Tensor = field(init=False)  # clipped to 512
    c4_sparse_page_indices: torch.Tensor = field(init=False)  # (bs, 512)
    # FlashMLA
    c1_flashmla_metadata: FlashMLASchedMeta = field(init=False, repr=False)
    c4_flashmla_metadata: FlashMLASchedMeta = field(init=False, repr=False)
    c128_flashmla_metadata: FlashMLASchedMeta = field(init=False, repr=False)

    def get_flashmla_metadata(self, compress_ratio: int):
        if compress_ratio == 0:
            return self.c1_flashmla_metadata
        elif compress_ratio == 4:
            return self.c4_flashmla_metadata
        elif compress_ratio == 128:
            return self.c128_flashmla_metadata
        else:
            raise ValueError(f"invalid {compress_ratio=}")

    def __post_init__(self):
        assert self.c4_sparse_topk == 512
        self.c4_sparse_topk_lengths = torch.clamp(
            self.c4_topk_lengths_clamp1, max=self.c4_sparse_topk
        )
        self.c4_sparse_page_indices = torch.full(
            (self.c4_topk_lengths_clamp1.size(0), self.c4_sparse_topk),
            -1,
            dtype=torch.int32,
            device=self.c4_topk_lengths_clamp1.device,
        )
        self.c1_flashmla_metadata = create_flashmla_metadata()
        self.c4_flashmla_metadata = create_flashmla_metadata()
        self.c128_flashmla_metadata = create_flashmla_metadata()

    def copy_(self, other: PagedCoreMetadata) -> None:
        copy_metadata(
            src=other,
            dst=self,
            check_eq_fields=["c4_sparse_topk", "swa_slice"],
            copy_fields=[
                "positions",
                "swa_out_loc_sliced",
                "c4_out_loc",
                "c128_out_loc",
                "page_table",
                "swa_page_indices",
                "swa_topk_lengths",
                "c128_page_indices",
                "c128_topk_lengths_clamp1",
                "c4_topk_lengths_raw",
                "c4_topk_lengths_clamp1",
                "c4_sparse_topk_lengths",
                "c4_sparse_page_indices",
            ],
            assign_fields=[
                # For the new API, the metadata has the following lifecycle:
                #
                # Graph capture warmup forward pass:
                # (ignore, we will reset to brand new object after such passes)
                #
                # Graph capture real-capture forward pass:
                # * Layer 0: Set python & tensor objects to metadata
                # * Layer >=1: Read them from metadata
                #
                # Graph replay:
                # * Layer 0: The kernels are in "generate metadata" mode
                # * Layer >=1: The kernels are in "non-generate metadata" mode
                #
                # Thus this field can be ignored.
                # However, to allow running replay w/o in real cuda graph, we do an assignment.
                # (Do we really need that? If no, we can change this field to skip-copy mode)
                "c1_flashmla_metadata",
                "c4_flashmla_metadata",
                "c128_flashmla_metadata",
            ],
        )


# TODO: implement the ragged metadata


@dataclass
class RaggedCoreMetadata(CoreMetadata):
    swa_ragged_indices: torch.Tensor
    swa_c4_ragged_indices: torch.Tensor
    swa_c128_ragged_indices: torch.Tensor


@dataclass
class RaggedIndexerMetadata(IndexerMetadata):
    c4_k_start: torch.Tensor
    c4_k_finish: torch.Tensor


@dataclass
class DeepseekV4Metadata:
    core_metadata: CoreMetadata
    indexer_metadata: IndexerMetadata
    debug_seq_lens_expanded: torch.Tensor

    def copy_(self, other: "DeepseekV4Metadata"):
        self.core_metadata.copy_(other.core_metadata)
        self.indexer_metadata.copy_(other.indexer_metadata)


def maybe_copy_inplace(dst, *, src) -> None:
    assert type(src) == type(dst)
    if dst is not None:
        dst.copy_(src)
