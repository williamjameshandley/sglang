from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def test_internal_state_reports_sorted_offload_tags():
    scheduler = SimpleNamespace(
        metrics_reporter=SimpleNamespace(
            last_gen_throughput=0, spec_total_num_forward_ct=0
        ),
        tp_worker=SimpleNamespace(
            model_runner=SimpleNamespace(
                weight_load_mem_usage=0, graph_mem_usage=0
            )
        ),
        token_to_kv_pool_allocator=SimpleNamespace(
            get_kvcache=Mock(return_value=SimpleNamespace(mem_usage=0))
        ),
        max_total_num_tokens=1,
        max_running_requests=1,
        weight_updater=SimpleNamespace(offload_tags={"weights", "kv_cache"}),
        server_args=SimpleNamespace(elastic_ep_backend=None),
        spec_algorithm=SimpleNamespace(
            is_none=Mock(return_value=True), is_dspark=Mock(return_value=False)
        ),
        draft_worker=None,
    )
    with patch("sglang.srt.managers.scheduler.get_context") as get_context:
        get_context.return_value.resolved_server_args_dict.return_value = {}
        result = Scheduler.get_internal_state(scheduler, None)

    assert result.internal_state["offload_tags"] == ["kv_cache", "weights"]


@pytest.mark.asyncio
async def test_memory_occupation_status_reports_admission_and_rank_tags():
    manager = object.__new__(TokenizerManager)
    manager.is_pause = True
    manager.get_internal_state = AsyncMock(
        return_value=[
            {"offload_tags": ["kv_cache", "weights"]},
            {"offload_tags": ["weights"]},
        ]
    )

    assert await manager.memory_occupation_status() == {
        "generation_paused": True,
        "offload_tags": [["kv_cache", "weights"], ["weights"]],
    }
