import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.sampler import Sampler
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestSamplerTpSync(unittest.TestCase):
    """A single TP rank must not all-reduce its sampled token ids: the reduction
    has nothing to reconcile, and NCCL's first-use buffer allocation can fail
    when the KV pool is sized to the card."""

    def _sync(self, world_size):
        sampler = object.__new__(Sampler)
        sampler.tp_sync_group = object()
        sampling_info = SimpleNamespace(grammars=[object()])
        with (
            patch("torch.distributed.get_world_size", return_value=world_size),
            patch("torch.distributed.all_reduce") as all_reduce,
        ):
            sampler._sync_token_ids_across_tp(
                torch.zeros(2, dtype=torch.int64), sampling_info
            )
        return all_reduce

    def test_single_rank_skips_the_all_reduce(self):
        self.assertFalse(self._sync(1).called)

    def test_multi_rank_syncs_grammar_tokens(self):
        self.assertEqual(self._sync(2).call_count, 1)


if __name__ == "__main__":
    unittest.main()
