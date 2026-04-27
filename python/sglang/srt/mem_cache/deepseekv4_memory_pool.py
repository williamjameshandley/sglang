from __future__ import annotations

import logging
from contextlib import nullcontext
from typing import List, Literal, NamedTuple, Optional, Tuple, Union

import torch
from sgl_kernel.kvcacheio import transfer_kv_all_layer_mla

from sglang.jit_kernel.deepseek_v4 import fused_store_cache
from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.environ import envs
from sglang.srt.layers.attention.nsa import index_buf_accessor, index_buf_accessor_v4
from sglang.srt.layers.attention.nsa.index_buf_accessor_v4 import NopeFp8RopeBf16Pack
from sglang.srt.mem_cache.compress_state import (
    CompressStatePool,
    DeepSeekV4CompressState,
    KVAndScore,
)
from sglang.srt.mem_cache.memory_pool import KVCache
from sglang.srt.mem_cache.swa_memory_pool import SWATokenToKVPoolAllocator
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import ceil_div

logger = logging.getLogger(__name__)


def get_compress_state_ring_size(
    compress_ratio: int, is_speculative: bool = False
) -> int:
    """Get ring size for given compression ratio.

    This is the single source of truth for ring size calculation.
    All other code should call this function instead of duplicating the logic.

    Args:
        compress_ratio: Compression ratio (4 or 128)
        is_speculative: Whether speculative decoding is enabled

    Returns:
        Ring size for the given compression ratio
    """
    assert compress_ratio in [4, 128], f"Unsupported {compress_ratio = }"
    if is_speculative:
        return 8 if compress_ratio == 4 else 128
    else:
        return 16 if compress_ratio == 4 else 256


class DeepSeekV4SingleKVPool(KVCache):
    # FIXME: rename to something like PartialRoPEKVPool and combine with NSA KVPool
    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
        is_swa_pool: Optional[bool] = False,
    ):
        super().__init__(
            size,
            page_size,
            dtype,
            layer_num,
            device,
            enable_memory_saver,
            start_layer,
            end_layer,
        )
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim

        self.scale_pad = 1
        self.quantize_block_size = 64
        self.rope_storage_dtype = torch.bfloat16
        self.k_with_scale_buffer_dtype = torch.int8
        self.is_swa_pool = is_swa_pool
        self._create_buffers()

    @property
    def page_size(self):
        if self.is_swa_pool:
            assert (
                envs.SGLANG_OPT_DPSK_V4_RADIX.get()
                and (self._page_size == 256)
                or not envs.SGLANG_OPT_DPSK_V4_RADIX.get()
                and (self._page_size == 128)
            ), "SWA KV pool page size not correct!"

        return self._page_size

    @page_size.setter
    def page_size(self, value: int):
        self._page_size = value

    def _create_buffers(self):
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            with (
                torch.cuda.use_mem_pool(self.custom_mem_pool)
                if self.custom_mem_pool
                else nullcontext()
            ):
                self.kv_buffer = [
                    self.create_buffer(
                        num_pages=(self.size + self.page_size + 1) // self.page_size,
                    )
                    for _ in range(self.layer_num)
                ]

    def get_bytes_per_token(self) -> int:
        # The padded slot 0 is used for writing dummy outputs from padded tokens.
        # Layout:
        #     shape: (num_pages, page_size * (nope_dim 448 + rope_dim 128 * 2) +
        #            page_size * (nope_dim / quant_block_size + scale_pad) * fp32_nbytes 4)
        #     data: for page i,
        #         * buf[i, :page_size * head_dim] for fp8 data
        #         * buf[i, page_size * head_dim:].view(float32) for scale
        #
        # Raw description from FlashMLA flash_mla_with_kvcache:
        #     head_dim should be 512 while head_dim_v is also 512.
        #     In FP8+sparse mode, every block can be divided into two parts.
        #     The first parts stores NoPE0, RoPE0, NoPE1, RoPE1, ...
        #     while the second part stores scale factors: 7xue8m0, 1Bpad, 7xue8m0, 1Bpad, ...
        dim_per_token = (
            self.qk_nope_head_dim
            + self.qk_rope_head_dim * self.rope_storage_dtype.itemsize
            + self.qk_nope_head_dim // self.quantize_block_size
            + self.scale_pad
        )
        return dim_per_token

    def create_buffer(self, *, num_pages: int):
        bytes_per_token = self.get_bytes_per_token()
        self.kv_cache_total_dim = bytes_per_token
        bytes_per_page_non_padded = self.page_size * bytes_per_token
        self.bytes_per_page_padded = ceil_div(bytes_per_page_non_padded, 576) * 576

        assert bytes_per_token == 448 + 64 * 2 + 8
        assert self.store_dtype == torch.uint8

        return torch.zeros(
            num_pages,
            self.bytes_per_page_padded,
            dtype=self.store_dtype,
            device=self.device,
        )

    def set_key_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache_nope_fp8_rope_bf16_pack: NopeFp8RopeBf16Pack,
    ):
        index_buf_accessor_v4.SetKAndS.execute(
            pool=self,
            buf=self.kv_buffer[layer_id],
            loc=loc,
            nope_fp8_rope_bf16_pack=cache_nope_fp8_rope_bf16_pack,
        )

    def set_key_buffer_fused(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
    ) -> None:
        return fused_store_cache(
            input=cache_k,
            cache=self.kv_buffer[layer_id],
            indices=loc,
            page_size=self.page_size,
            type="flashmla",
        )

    def get_key_buffer(self, layer_id: int):
        if self.store_dtype != self.dtype:
            return self.kv_buffer[layer_id - self.start_layer].view(self.dtype)

        return self.kv_buffer[layer_id]

    def set_kv_buffer(self, *args, **kwargs) -> None:
        raise NotImplementedError()

    def get_value_buffer(self, layer_id: int) -> torch.Tensor:
        raise NotImplementedError("Use get_key_buffer instead.")

    def get_kv_buffer(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("Use get_key_buffer instead.")


class HiSparseC4DevicePool(DeepSeekV4SingleKVPool):

    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        start_layer: int | None = None,
        end_layer: int | None = None,
    ):
        super().__init__(
            size,
            page_size,
            dtype,
            qk_nope_head_dim,
            qk_rope_head_dim,
            layer_num,
            device,
            enable_memory_saver,
            start_layer,
            end_layer,
        )

        self.data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.kv_buffer],
            dtype=torch.uint64,
            device=self.device,
        )
        self.compress_ratio = 4

    def register_mapping(self, full_to_hisparse_device_index_mapping: torch.Tensor):
        self.full_to_hisparse_device_index_mapping = (
            full_to_hisparse_device_index_mapping
        )

    def translate_loc_from_full_to_compressed(self, full_indices: torch.Tensor):
        mask = (full_indices + 1) % self.compress_ratio == 0
        compressed_indices = full_indices[mask] // self.compress_ratio
        return compressed_indices

    def translate_loc_from_compressed_to_hisparse_device(
        self, compressed_indices: torch.Tensor
    ):
        return self.full_to_hisparse_device_index_mapping[compressed_indices].to(
            torch.int32
        )

    def _translate_loc_from_compressed_to_hisparse_device(
        self, compressed_indices: torch.Tensor
    ):
        return self.full_to_hisparse_device_index_mapping[compressed_indices]

    def translate_loc_from_full_to_hisparse_device(self, full_indices: torch.Tensor):
        return self._translate_loc_from_compressed_to_hisparse_device(
            self.translate_loc_from_full_to_compressed(full_indices)
        )

    def set_key_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache_nope_fp8_rope_bf16_pack,
    ):
        loc = self.translate_loc_from_compressed_to_hisparse_device(loc)
        super().set_key_buffer(layer_id, loc, cache_nope_fp8_rope_bf16_pack)

    def transfer_values_on_device(self, dst_indices, src_indices):
        # FIXME, page padding to be handled in the custom op
        transfer_kv_all_layer_mla(
            src_layers=self.data_ptrs,
            dst_layers=self.data_ptrs,
            src_indices=src_indices,
            dst_indices=dst_indices,
            item_size=self.kv_cache_total_dim,
            num_layers=self.layer_num,
        )

    def get_cpu_copy(self, indices):
        raise NotImplementedError("HiSparseC4DevicePool does not support get_cpu_copy")

    def load_cpu_copy(self, kv_cache_cpu, indices):
        raise NotImplementedError("HiSparseC4DevicePool does not support load_cpu_copy")


class DeepSeekV4IndexerPool(KVCache):
    quant_block_size = 128
    index_k_with_scale_buffer_dtype = torch.uint8

    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        index_head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
    ):
        super().__init__(
            size,
            page_size,
            dtype,
            layer_num,
            device,
            enable_memory_saver,
            start_layer,
            end_layer,
        )
        self.index_head_dim = index_head_dim

        self._create_buffer()

    def _create_buffer(self):
        num_scales_per_token = self.index_head_dim // self.quant_block_size
        # NOTE: weight in fp8, and scale in fp32
        page_bytes = self.page_size * self.index_head_dim
        page_bytes += self.page_size * num_scales_per_token * 4
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            with (
                torch.cuda.use_mem_pool(self.custom_mem_pool)
                if self.custom_mem_pool
                else nullcontext()
            ):
                self.index_k_with_scale_buffer = [
                    torch.zeros(
                        (self.size + self.page_size + 1) // self.page_size,
                        page_bytes,
                        dtype=self.index_k_with_scale_buffer_dtype,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]

    def get_kv_buffer(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError()

    def get_key_buffer(self, layer_id: int) -> torch.Tensor:
        raise NotImplementedError()

    def get_value_buffer(self, layer_id: int) -> torch.Tensor:
        raise NotImplementedError()

    def set_kv_buffer(self, *args, **kwargs) -> None:
        raise NotImplementedError()

    def get_index_k_with_scale_buffer(self, layer_id: int) -> torch.Tensor:
        return self.index_k_with_scale_buffer[layer_id]

    # copied from NSATokenToKVPool, theoretically can be directly reused
    def get_index_k_scale_buffer(
        self,
        layer_id: int,
        seq_len: int,
        page_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fused method to get both index K and scale data in a single call using Triton.
        More efficient than calling get_index_k_continuous and get_index_k_scale_continuous separately.

        :param layer_id: Layer index
        :param seq_len: Sequence length
        :param page_indices: Page indices tensor
        :return: tuple of (k_fp8, k_scale) where
                 k_fp8: (seq_len, index_head_dim), uint8
                 k_scale: (seq_len, 4), uint8
        """
        buf = self.index_k_with_scale_buffer[layer_id]
        return index_buf_accessor.GetKAndS.execute(
            self, buf, seq_len=seq_len, page_indices=page_indices
        )

    def set_index_k_scale_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        index_k: torch.Tensor,
        index_k_scale: torch.Tensor,
    ) -> None:
        buf = self.index_k_with_scale_buffer[layer_id - self.start_layer]
        index_buf_accessor.SetKAndS.execute(
            pool=self, buf=buf, loc=loc, index_k=index_k, index_k_scale=index_k_scale
        )

    def set_index_fused(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
    ) -> None:
        return fused_store_cache(
            input=cache_k,
            cache=self.index_k_with_scale_buffer[layer_id - self.start_layer],
            indices=loc,
            page_size=self.page_size,
            type="indexer",
        )


class DeepSeekV4LayerItem(NamedTuple):
    compress_ratio: Literal[0, 4, 128]
    compress_layer_id: int
    compress_kv_pool: Optional[DeepSeekV4SingleKVPool] = None


class DeepSeekV4TokenToKVPool(KVCache):

    def __init__(
        self,
        max_num_reqs: int,
        swa_size: int,
        c4_size: int,
        c128_size: int,
        c4_state_pool_size: int,
        c128_state_pool_size: int,
        page_size: int,
        swa_page_size: int,
        dtype: torch.dtype,
        state_dtype: torch.dtype,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        indexer_head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        compression_ratios: List[int],
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
        enable_hisparse: bool = False,
    ):
        super().__init__(
            swa_size,
            page_size,
            dtype,
            layer_num,
            device,
            enable_memory_saver,
            start_layer,
            end_layer,
        )

        logger.info(
            "Initialize DeepSeekV4TokenToKVPool with "
            f"{max_num_reqs=} {swa_size=} {c4_size=} {c128_size=} "
            f"{c4_state_pool_size=} {c128_state_pool_size=}"
        )

        self.max_num_reqs = max_num_reqs
        self.c4_size = c4_size
        self.c128_size = c128_size
        self.c4_state_pool_size = c4_state_pool_size
        self.c128_state_pool_size = c128_state_pool_size
        self.state_dtype = state_dtype
        self.compression_ratios = compression_ratios

        assert page_size % swa_page_size == 0

        self.swa_size = swa_size
        self.swa_window_size = swa_page_size
        self.swa_page_size = swa_page_size
        self.scale_pad = 1

        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.indexer_head_dim = indexer_head_dim

        c4_layer_num = sum(1 for r in compression_ratios if r == 4)
        c128_layer_num = sum(1 for r in compression_ratios if r == 128)
        c4_page_size = page_size // 4
        c128_page_size = page_size // 128
        self.swa_kv_pool = DeepSeekV4SingleKVPool(
            swa_size,
            swa_page_size,
            dtype,
            qk_nope_head_dim,
            qk_rope_head_dim,
            layer_num,
            device,
            enable_memory_saver,
            is_swa_pool=True,
        )

        c4_kv_pool_type = DeepSeekV4SingleKVPool
        if enable_hisparse:
            c4_kv_pool_type = HiSparseC4DevicePool
        self.c4_kv_pool = c4_kv_pool_type(
            c4_size,
            c4_page_size,
            dtype,
            qk_nope_head_dim,
            qk_rope_head_dim,
            c4_layer_num,
            device,
            enable_memory_saver,
        )

        self.c128_kv_pool = DeepSeekV4SingleKVPool(
            c128_size,
            c128_page_size,
            dtype,
            qk_nope_head_dim,
            qk_rope_head_dim,
            c128_layer_num,
            device,
            enable_memory_saver,
        )

        self.c4_indexer_kv_pool = DeepSeekV4IndexerPool(
            c4_size,
            c4_page_size,
            dtype,  # indexer kv: fp8 + fp32 scale
            indexer_head_dim,
            c4_layer_num,
            device,
            enable_memory_saver,
        )

        self._init_compressed_layer_mapping()

        if envs.SGLANG_OPT_DPSK_V4_RADIX.get():
            self._init_paged_compress_states()
        else:
            self._init_compress_states()

        self._should_cache_swa = envs.SGLANG_OPT_CACHE_SWA_TRANSLATION.get()
        self.swa_loc: Optional[torch.Tensor] = None

    def register_mapping(self, full_to_swa_index_mapping: torch.Tensor):
        self.full_to_swa_index_mapping = full_to_swa_index_mapping

    def set_swa_loc(self, loc: Optional[torch.Tensor]) -> None:
        """Stash an already-SWA-translated loc tensor that subsequent
        per-layer set_swa_key_buffer_radix[_fused] calls will use directly,
        skipping the per-call translate_loc_from_full_to_swa.

        Mirrors SWAKVPool.set_swa_loc (mem_cache/swa_memory_pool.py:146).
        Called by model_runner._forward_raw (eager) and
        cuda_graph_runner.capture_one_batch_size with
        forward_batch.out_cache_loc_swa, which is already SWA-space.
        """
        self.swa_loc = loc

    def get_ring_size(self, compress_ratio: int) -> int:
        server_args = get_global_server_args()
        is_speculative = server_args.speculative_algorithm is not None
        return get_compress_state_ring_size(compress_ratio, is_speculative)

    def translate_loc_from_full_to_swa(self, kv_indices: torch.Tensor):
        assert self.full_to_swa_index_mapping is not None

        # Note: kv_indices could have -1 values (from alloc_extend), which will be mapped to -1
        # since the last item of full_to_swa_index_mapping is -1.
        return self.full_to_swa_index_mapping[kv_indices].to(torch.int32)

    def get_contiguous_buf_infos(self) -> Tuple[List[int], List[int], List[int]]:
        """Channel 1: C4 KV + C4 indexer + C128 KV (source page indices)."""
        data_ptrs: List[int] = []
        data_lens: List[int] = []
        item_lens: List[int] = []

        for bufs in [
            self.c4_kv_pool.kv_buffer,
            self.c4_indexer_kv_pool.index_k_with_scale_buffer,
            self.c128_kv_pool.kv_buffer,
        ]:
            for buf in bufs:
                assert buf.ndim == 2, f"expected 2D buffer, got {buf.ndim}D"
                data_ptrs.append(buf.data_ptr())
                data_lens.append(buf.nbytes)
                item_lens.append(buf[0].nbytes)

        return data_ptrs, data_lens, item_lens

    def get_state_buf_infos(self) -> Tuple[List[int], List[int], List[int]]:
        """Channel 2: SWA KV + compress states + indexer compress states (SWA page indices).

        Compress state ring buffers are bundled per SWA page:
        item_lens = ring_size * per_slot_bytes, so one SWA page index
        copies the entire ring region for that page.
        """
        data_ptrs: List[int] = []
        data_lens: List[int] = []
        item_lens: List[int] = []

        for buf in self.swa_kv_pool.kv_buffer:
            assert buf.ndim == 2, f"expected 2D buffer, got {buf.ndim}D"
            data_ptrs.append(buf.data_ptr())
            data_lens.append(buf.nbytes)
            item_lens.append(buf[0].nbytes)

        for pools in [
            self.compress_state_pools,
            self.indexer_compress_state_pools,
        ]:
            for pool in pools:
                if pool is None:
                    continue
                t = pool.kv_score_buffer.kv_score
                assert t.ndim == 2, f"expected 2D buffer, got {t.ndim}D"
                data_ptrs.append(t.data_ptr())
                data_lens.append(t.nbytes)
                item_lens.append(t[0].nbytes * pool.ring_size)

        return data_ptrs, data_lens, item_lens

    def _init_paged_compress_states(self):
        # Use pre-calculated pool sizes from memory profiler
        c4_state_pool_size = self.c4_state_pool_size
        c128_state_pool_size = self.c128_state_pool_size
        self.compress_state_pools: List[CompressStatePool] = []
        self.indexer_compress_state_pools: List[CompressStatePool] = []

        for ratio in self.compression_ratios:
            overlap = ratio == 4
            compress_state_pool = indexer_compress_state_pool = None
            size = c4_state_pool_size if ratio == 4 else c128_state_pool_size
            ring_size = self.get_ring_size(ratio) if ratio != 0 else 0

            # NOTE: c1 layer has no compress state
            if ratio != 0:
                compress_state_pool = CompressStatePool(
                    size=size,
                    swa_page_size=self.swa_page_size,
                    ring_size=ring_size,
                    overlap=overlap,
                    head_dim=self.qk_nope_head_dim + self.qk_rope_head_dim,
                    dtype=self.state_dtype,
                    device=self.device,
                    enable_memory_saver=False,
                    ratio=ratio,
                )

            if ratio == 4:
                indexer_compress_state_pool = CompressStatePool(
                    size=size,
                    swa_page_size=self.swa_page_size,
                    ring_size=ring_size,
                    overlap=overlap,
                    head_dim=self.indexer_head_dim,
                    device=self.device,
                    dtype=self.state_dtype,
                    enable_memory_saver=False,
                    ratio=ratio,
                )

            self.compress_state_pools.append(compress_state_pool)
            self.indexer_compress_state_pools.append(indexer_compress_state_pool)

    def _init_compressed_layer_mapping(self):
        c1_cnt, c4_cnt, c128_cnt = 0, 0, 0
        self.layer_mapping: List[DeepSeekV4LayerItem] = []

        for ratio in self.compression_ratios:
            if ratio == 0:
                self.layer_mapping.append(
                    DeepSeekV4LayerItem(
                        compress_ratio=0,
                        compress_layer_id=c1_cnt,
                    )
                )
                c1_cnt += 1
            elif ratio == 4:
                self.layer_mapping.append(
                    DeepSeekV4LayerItem(
                        compress_ratio=4,
                        compress_layer_id=c4_cnt,
                        compress_kv_pool=self.c4_kv_pool,
                    )
                )
                c4_cnt += 1
            elif ratio == 128:
                self.layer_mapping.append(
                    DeepSeekV4LayerItem(
                        compress_ratio=128,
                        compress_layer_id=c128_cnt,
                        compress_kv_pool=self.c128_kv_pool,
                    )
                )
                c128_cnt += 1
            else:
                raise ValueError(f"Unsupported compression ratio: {ratio}")

    def _init_compress_states(self):
        self.compress_states: List[Optional[DeepSeekV4CompressState]] = []
        self.indexer_compress_states: List[Optional[DeepSeekV4CompressState]] = []
        for ratio in self.compression_ratios:
            overlap = ratio == 4
            attn_kv_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
            compress_state = indexer_compress_state = None
            # NOTE: c1 layer has no compress state
            if ratio != 0:
                compress_state = DeepSeekV4CompressState(
                    max_num_reqs=self.max_num_reqs,
                    ratio=ratio,
                    overlap=overlap,
                    head_dim=attn_kv_head_dim,
                    device=self.device,
                    dtype=self.state_dtype,
                )
            # NOTE: only c4 needs indexer
            if ratio == 4:
                indexer_compress_state = DeepSeekV4CompressState(
                    max_num_reqs=self.max_num_reqs,
                    ratio=ratio,
                    overlap=overlap,
                    head_dim=self.indexer_head_dim,
                    device=self.device,
                    dtype=self.state_dtype,
                )
            self.compress_states.append(compress_state)
            self.indexer_compress_states.append(indexer_compress_state)

    def get_attention_compress_states(self, layer_id: int) -> KVAndScore:
        if envs.SGLANG_OPT_DPSK_V4_RADIX.get():
            compress_state_pool = self.compress_state_pools[layer_id]
            assert (
                compress_state_pool is not None
            ), "Only c4/c128 layers have attention states."
            return compress_state_pool
        else:
            compress_state = self.compress_states[layer_id]
            assert (
                compress_state is not None
            ), "Only c4/c128 layers have attention states."
            return compress_state.get_state()

    def get_indexer_compress_states(
        self, layer_id: int
    ) -> Union[KVAndScore, CompressStatePool]:
        if envs.SGLANG_OPT_DPSK_V4_RADIX.get():
            indexer_compress_state_pool = self.indexer_compress_state_pools[layer_id]
            assert (
                indexer_compress_state_pool is not None
            ), "Only c4 layers have indexer states."
            return indexer_compress_state_pool
        else:
            compress_state = self.indexer_compress_states[layer_id]
            assert compress_state is not None, "Only c4 layers have indexer states."
            return compress_state.get_state()

    def get_swa_key_buffer(self, layer_id: int) -> torch.Tensor:
        return self.swa_kv_pool.get_key_buffer(layer_id)

    # TODO seems no need to have this and can remove
    # def get_swa_key_buffer_by_loc(
    #     self, layer_id: int, loc: torch.Tensor
    # ) -> torch.Tensor:
    #     return self.swa_kv_pool.get_key_buffer_by_loc(layer_id, loc)

    def set_swa_key_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache_nope_fp8_rope_bf16_pack: NopeFp8RopeBf16Pack,
    ) -> None:
        self.swa_kv_pool.set_key_buffer(layer_id, loc, cache_nope_fp8_rope_bf16_pack)

    def get_extra_key_buffer(self, layer_id: int) -> torch.Tensor | None:
        # c4/c128 -> extra_cache_k
        _, compress_layer_id, compress_kv_pool = self.layer_mapping[layer_id]
        assert compress_kv_pool is not None
        return compress_kv_pool.get_key_buffer(compress_layer_id)

    def set_extra_key_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache_nope_fp8_rope_bf16_pack: NopeFp8RopeBf16Pack,
    ) -> None:
        _, compress_layer_id, compress_kv_pool = self.layer_mapping[layer_id]
        assert compress_kv_pool is not None
        compress_kv_pool.set_key_buffer(
            compress_layer_id, loc, cache_nope_fp8_rope_bf16_pack
        )

    def get_index_k_with_scale_buffer(self, layer_id: int) -> torch.Tensor:
        compress_ratio, compress_layer_id, _ = self.layer_mapping[layer_id]
        assert compress_ratio == 4, f"only c4 has indexer, got {compress_ratio = }"
        return self.c4_indexer_kv_pool.get_index_k_with_scale_buffer(compress_layer_id)

    def get_index_k_scale_buffer(
        self,
        layer_id: int,
        seq_len: int,
        page_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        compress_ratio, compress_layer_id, _ = self.layer_mapping[layer_id]
        assert compress_ratio == 4, f"only c4 has indexer, got {compress_ratio = }"
        return self.c4_indexer_kv_pool.get_index_k_scale_buffer(
            compress_layer_id, seq_len, page_indices
        )

    def set_index_k_scale_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        index_k: torch.Tensor,
        index_k_scale: torch.Tensor,
    ) -> None:
        compress_ratio, compress_layer_id, _ = self.layer_mapping[layer_id]
        assert compress_ratio == 4, f"only c4 has indexer, got {compress_ratio = }"
        self.c4_indexer_kv_pool.set_index_k_scale_buffer(
            compress_layer_id, loc, index_k, index_k_scale
        )

    def get_key_buffer(self, layer_id: int) -> torch.Tensor:
        raise NotImplementedError()

    def get_value_buffer(self, layer_id: int) -> torch.Tensor:
        raise NotImplementedError()

    def get_kv_buffer(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError()

    def set_kv_buffer(self, *args, **kwargs) -> None:
        raise NotImplementedError()

    # ---- APIs for radix cache compatible branch ----
    def set_swa_key_buffer_radix(
        self,
        layer_id: int,
        raw_loc: torch.Tensor,
        cache_nope_fp8_rope_bf16_pack: NopeFp8RopeBf16Pack,
    ) -> None:
        if self.swa_loc is not None:
            swa_loc = self.swa_loc
        else:
            swa_loc = self.translate_loc_from_full_to_swa(raw_loc)
        self.swa_kv_pool.set_key_buffer(
            layer_id, swa_loc, cache_nope_fp8_rope_bf16_pack
        )

    def get_swa_key_buffer_radix(self, layer_id: int) -> torch.Tensor:
        return self.swa_kv_pool.get_key_buffer(layer_id)

    # --- Fused APIs of setting key buffers ----
    def set_swa_key_buffer_radix_fused(
        self,
        layer_id: int,
        raw_loc: torch.Tensor,
        cache_k: torch.Tensor,
    ) -> None:
        if self.swa_loc is not None:
            swa_loc = self.swa_loc
        elif self._should_cache_swa:
            if layer_id == 0:
                self.cached_loc = self.translate_loc_from_full_to_swa(raw_loc)
            swa_loc = self.cached_loc
        else:
            swa_loc = self.translate_loc_from_full_to_swa(raw_loc)
        return self.swa_kv_pool.set_key_buffer_fused(layer_id, swa_loc, cache_k)

    def set_extra_key_buffer_fused(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
    ) -> None:
        _, compress_layer_id, compress_kv_pool = self.layer_mapping[layer_id]
        assert compress_kv_pool is not None
        return compress_kv_pool.set_key_buffer_fused(compress_layer_id, loc, cache_k)

    def set_index_k_fused(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
    ) -> None:
        compress_ratio, compress_layer_id, _ = self.layer_mapping[layer_id]
        assert compress_ratio == 4, f"only c4 has indexer, got {compress_ratio = }"
        return self.c4_indexer_kv_pool.set_index_fused(compress_layer_id, loc, cache_k)

    # final branch:
    # - DeepSeekV4TokenToKVPool
    #   - c4 / c128 / c4_indexer
    #   - swa_kv_pool: shape (num_pages, pages * bytes_per_page), where num_pages = max_num_reqs
    # - PagedTokenToKVAllocator


class DeepSeekV4TokenToKVPoolAllocator(SWATokenToKVPoolAllocator):
    """V4-aware allocator that owns separate full and SWA paged allocators
    plus the `full_to_swa_index_mapping` consumed by
    `DeepSeekV4TokenToKVPool.translate_loc_from_full_to_swa`.

    Mirrors :class:`SWATokenToKVPoolAllocator` (mem_cache/swa_memory_pool.py),
    retargeted to V4's pool layout: the SWA storage is the parent V4 pool's
    ``swa_kv_pool`` attribute rather than a separately managed pool.

    Both inner allocators share the same ``page_size`` so the canonical
    ``alloc_extend`` / ``alloc_decode`` page-alignment contracts hold. V4's
    parent pool internally subdivides its SWA storage by ``swa_page_size``
    (with ``page_size % swa_page_size == 0``); the allocator presents the
    coarser ``page_size`` view, which costs some SWA capacity but avoids
    any page-size mismatch in the alloc/free logic.
    """

    def __init__(
        self,
        size: int,
        size_swa: int,
        page_size: int,
        dtype: torch.dtype,
        device: str,
        kvcache: "DeepSeekV4TokenToKVPool",
        need_sort: bool,
    ):
        from sglang.srt.mem_cache.allocator import PagedTokenToKVPoolAllocator

        assert isinstance(kvcache, DeepSeekV4TokenToKVPool)
        # Both inner allocators share `page_size`; the SWA inner pool must
        # be paginated at the same size so allocator-level indices line up
        # with the underlying KV-cache page layout. V4 enforces
        # `page_size % swa_page_size == 0` at the pool level, but here we
        # require strict equality because we drive the SWA pool through
        # this allocator's page arithmetic.
        assert kvcache.swa_kv_pool.page_size == page_size, (
            f"DeepSeekV4TokenToKVPoolAllocator requires "
            f"kvcache.swa_kv_pool.page_size == page_size, got "
            f"{kvcache.swa_kv_pool.page_size} vs {page_size}"
        )
        self._size_full = size
        self._size_swa = size_swa
        self.dtype = dtype
        self.device = device
        self.page_size = page_size
        self.need_sort = need_sort

        self.full_attn_allocator = PagedTokenToKVPoolAllocator(
            size,
            page_size,
            dtype,
            device,
            kvcache,
            need_sort,
        )
        self.swa_attn_allocator = PagedTokenToKVPoolAllocator(
            size_swa,
            page_size,
            dtype,
            device,
            kvcache.swa_kv_pool,
            need_sort,
        )

        # Note: append one more item of value -1 in the end so -1 maps to -1.
        # Required because get_swa_page_indices() (deepseek_v4_backend_radix
        # line 1234-1239) marks invalid offsets as -1 before calling
        # translate_loc_from_full_to_swa, expecting -1 to stay -1.
        self.full_to_swa_index_mapping = torch.cat(
            [
                torch.zeros(size + page_size, dtype=torch.int64, device=device),
                torch.tensor([-1], dtype=torch.int64, device=device),
            ]
        )

        self.is_not_in_free_group = True
        self.free_group: List[torch.Tensor] = []

        self.clear()
        self._kvcache = kvcache
        self._kvcache.register_mapping(self.full_to_swa_index_mapping)

    @property
    def size(self) -> int:
        return min(self._size_full, self._size_swa)

    @property
    def size_full(self) -> int:
        return self._size_full

    @property
    def size_swa(self) -> int:
        return self._size_swa

    def available_size(self) -> int:
        return min(
            self.full_attn_allocator.available_size(),
            self.swa_attn_allocator.available_size(),
        )

    def full_available_size(self) -> int:
        return self.full_attn_allocator.available_size()

    def swa_available_size(self) -> int:
        return self.swa_attn_allocator.available_size()

    def get_kvcache(self) -> "DeepSeekV4TokenToKVPool":
        return self._kvcache

    def debug_print(self) -> str:
        return (
            f"#full-attn-available-size: {self.full_attn_allocator.available_size()}, "
            f"#swa-available-size: {self.swa_attn_allocator.available_size()}"
        )

    def translate_loc_from_full_to_swa(self, kv_indices: torch.Tensor) -> torch.Tensor:
        return self._kvcache.translate_loc_from_full_to_swa(kv_indices)

    def alloc(self, need_size: int):
        if need_size > self.full_attn_allocator.available_size():
            return None
        if need_size > self.swa_attn_allocator.available_size():
            return None

        alloc_full_indices = self.full_attn_allocator.alloc(need_size)
        alloc_swa_indices = self.swa_attn_allocator.alloc(need_size)
        assert alloc_full_indices is not None
        assert alloc_swa_indices is not None

        self.full_to_swa_index_mapping[alloc_full_indices] = alloc_swa_indices
        return alloc_full_indices

    def alloc_extend(
        self,
        prefix_lens: torch.Tensor,
        prefix_lens_cpu: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
        extend_num_tokens: int,
    ):
        from sglang.srt.mem_cache.allocator import get_num_new_pages

        num_new_pages = get_num_new_pages(
            seq_lens=seq_lens_cpu,
            page_size=self.page_size,
            prefix_lens=prefix_lens_cpu,
        )
        if num_new_pages > self.full_attn_allocator.available_size() // self.page_size:
            return None
        if num_new_pages > self.swa_attn_allocator.available_size() // self.page_size:
            return None

        swa_last_loc = self.translate_loc_from_full_to_swa(last_loc)

        alloc_full_indices = self.full_attn_allocator.alloc_extend(
            prefix_lens,
            prefix_lens_cpu,
            seq_lens,
            seq_lens_cpu,
            last_loc,
            extend_num_tokens,
        )
        alloc_swa_indices = self.swa_attn_allocator.alloc_extend(
            prefix_lens,
            prefix_lens_cpu,
            seq_lens,
            seq_lens_cpu,
            swa_last_loc,
            extend_num_tokens,
        )
        assert alloc_full_indices is not None
        assert alloc_swa_indices is not None

        self.full_to_swa_index_mapping[alloc_full_indices] = alloc_swa_indices
        return alloc_full_indices

    def alloc_decode(
        self,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
    ):
        swa_last_loc = self.translate_loc_from_full_to_swa(last_loc)

        alloc_full_indices = self.full_attn_allocator.alloc_decode(
            seq_lens, seq_lens_cpu, last_loc
        )
        alloc_swa_indices = self.swa_attn_allocator.alloc_decode(
            seq_lens, seq_lens_cpu, swa_last_loc
        )

        if alloc_full_indices is None or alloc_swa_indices is None:
            return None

        self.full_to_swa_index_mapping[alloc_full_indices] = alloc_swa_indices
        return alloc_full_indices

    def free(self, free_index: torch.Tensor) -> None:
        if free_index.numel() == 0:
            return

        if self.is_not_in_free_group:
            self.full_attn_allocator.free(free_index)
            self.free_swa(free_index)
        else:
            self.free_group.append(free_index)

        assert (
            self.full_attn_allocator.available_size()
            <= self.full_attn_allocator.size
        )
        assert (
            self.swa_attn_allocator.available_size()
            <= self.swa_attn_allocator.size
        )

    def free_swa(self, free_index: torch.Tensor) -> None:
        swa_indices = self.full_to_swa_index_mapping[free_index]
        swa_indices = swa_indices[swa_indices > 0]
        self.swa_attn_allocator.free(swa_indices)
        self.full_to_swa_index_mapping[free_index] = 0

    def free_group_begin(self) -> None:
        self.is_not_in_free_group = False
        self.free_group = []

    def free_group_end(self) -> None:
        self.is_not_in_free_group = True
        if self.free_group:
            self.free(torch.cat(self.free_group))
        self.free_group = []

    def backup_state(self):
        return [
            self.full_attn_allocator.backup_state(),
            self.swa_attn_allocator.backup_state(),
        ]

    def restore_state(self, state) -> None:
        assert len(state) == 2
        self.full_attn_allocator.restore_state(state[0])
        self.swa_attn_allocator.restore_state(state[1])

    def clear(self) -> None:
        self.full_attn_allocator.clear()
        self.swa_attn_allocator.clear()
        # Last item is the -1 sentinel; preserve it.
        self.full_to_swa_index_mapping[:-1].fill_(0)
        self.is_not_in_free_group = True
        self.free_group = []

    def get_cpu_copy(self, indices):
        return self._kvcache.get_cpu_copy(indices)

    def load_cpu_copy(self, kv_cache_cpu, indices):
        return self._kvcache.load_cpu_copy(kv_cache_cpu, indices)
