"""Wire contract of /memory_occupation_status, which an external controller
reads to decide what to resume: a flat pause flag plus each rank's released
tags under the keys it expects."""

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.scheduler_components.internal_state import (  # noqa: E402
    SchedulerInternalStateController,
)
from sglang.srt.managers.tokenizer_manager import TokenizerManager  # noqa: E402

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestMemoryOccupationStatus(CustomTestCase):
    def test_internal_state_carries_released_tags(self):
        controller = SchedulerInternalStateController(
            scheduler=SimpleNamespace(
                weight_updater=SimpleNamespace(offload_tags={"weights", "kv_cache"})
            ),
            record_step_time=False,
        )
        state = {}
        controller._add_offload_tags(state)
        self.assertEqual(state["offload_tags"], ["kv_cache", "weights"])

    def test_status_reports_pause_flag_and_per_rank_tags(self):
        manager = TokenizerManager.__new__(TokenizerManager)
        manager._is_pause = True
        manager.get_internal_state = AsyncMock(
            return_value=[
                {"offload_tags": ["kv_cache", "weights"]},
                {"offload_tags": ["weights"]},
            ]
        )
        self.assertEqual(
            asyncio.run(manager.memory_occupation_status()),
            {
                "generation_paused": True,
                "offload_tags": [["kv_cache", "weights"], ["weights"]],
            },
        )


if __name__ == "__main__":
    unittest.main()
