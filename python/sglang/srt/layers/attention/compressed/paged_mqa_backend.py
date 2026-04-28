"""Backend selection for `fp8_paged_mqa_logits`.

Single source of truth for which kernel implementation to use, shared
by:
  - the function dispatch in `compressed/indexer.py`,
  - the metadata construction in `compressed/metadata.py:__post_init__`,
  - the metadata replay-copy gate in `compressed/metadata.py:copy_`.

Lives in its own module rather than `indexer.py` so `metadata.py` can
import it without creating a cycle (`indexer.py` already imports from
`metadata.py`).
"""

from __future__ import annotations

from enum import Enum, auto
from functools import lru_cache

import torch

from sglang.srt.environ import envs
from sglang.srt.utils.common import is_cuda, is_hip


class PagedMQALogitsBackend(Enum):
    """The kernel actually used to compute fp8_paged_mqa_logits.

    Order is significant for `get_paged_mqa_logits_backend()` — the
    first matching condition wins.
    """

    TILELANG = auto()           # SGLANG_OPT_USE_TILELANG_INDEXER=1
    TORCH = auto()              # SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=1, also HIP path
    TRITON_SM120 = auto()       # CUDA sm_120 default
    DEEP_GEMM_CHUNKED = auto()  # SGLANG_OPT_DG_PAGED_MQA_LOGITS_CHUNK_SIZE != -1
    DEEP_GEMM = auto()          # CUDA sm_90/sm_100 default


@lru_cache(maxsize=1)
def _is_sm120() -> bool:
    if not is_cuda():
        return False
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    return major == 12 and minor == 0


def get_paged_mqa_logits_backend() -> PagedMQALogitsBackend:
    if envs.SGLANG_OPT_USE_TILELANG_INDEXER.get():
        return PagedMQALogitsBackend.TILELANG
    if envs.SGLANG_FP8_PAGED_MQA_LOGITS_TORCH.get():
        return PagedMQALogitsBackend.TORCH
    if is_hip():
        # The HIP path historically routes through aiter and does not use
        # DeepGEMM metadata; treat it as a non-DeepGEMM backend so the
        # metadata/copy gates collapse to the same arm as torch.
        return PagedMQALogitsBackend.TORCH
    if _is_sm120():
        return PagedMQALogitsBackend.TRITON_SM120
    if envs.SGLANG_OPT_DG_PAGED_MQA_LOGITS_CHUNK_SIZE.get() != -1:
        return PagedMQALogitsBackend.DEEP_GEMM_CHUNKED
    return PagedMQALogitsBackend.DEEP_GEMM


_USES_DEEP_GEMM_METADATA = {
    PagedMQALogitsBackend.DEEP_GEMM,
    PagedMQALogitsBackend.DEEP_GEMM_CHUNKED,
}


def uses_deep_gemm_metadata(backend: PagedMQALogitsBackend) -> bool:
    """Whether the chosen backend reads `PagedIndexerMetadata.deep_gemm_metadata`."""
    return backend in _USES_DEEP_GEMM_METADATA
