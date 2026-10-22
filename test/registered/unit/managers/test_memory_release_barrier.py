"""An in-place pause is refused while requests are admitted, admission cannot
straddle a pause, a pause that fails to dispatch reopens the gate, an
abort-mode pause cannot deadlock with a queued admission, and a busy rank
refuses a memory release instead of crashing its scheduler."""

import asyncio
import unittest
from unittest.mock import AsyncMock, Mock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.io_struct import (  # noqa: E402
    PauseGenerationReqInput,
    ReleaseMemoryOccupationReqInput,
    ReleaseMemoryOccupationReqOutput,
)
from sglang.srt.managers.scheduler_components.weight_updater import (  # noqa: E402
    SchedulerWeightUpdaterManager,
)
from sglang.srt.managers.tokenizer_control_mixin import (  # noqa: E402
    TokenizerControlMixin,
)
from sglang.srt.managers.tokenizer_manager import TokenizerManager  # noqa: E402
from sglang.srt.utils.aio_rwlock import RWLock  # noqa: E402

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _bare_tokenizer_manager(is_pause=False):
    tm = TokenizerManager.__new__(TokenizerManager)
    tm._is_pause = is_pause
    tm.is_pause_cond = asyncio.Condition()
    tm.model_update_lock = RWLock()
    tm._async_dispatch_to_scheduler = AsyncMock()
    return tm


class TestPauseAdmissionBarrier(CustomTestCase):
    def test_in_place_pause_refuses_active_request_without_pausing(self):
        async def run():
            tm = _bare_tokenizer_manager()
            await tm.model_update_lock.acquire_reader()

            with self.assertRaisesRegex(RuntimeError, "in flight"):
                await tm.pause_generation(PauseGenerationReqInput(mode="in_place"))

            self.assertFalse(tm.is_pause)
            tm._async_dispatch_to_scheduler.assert_not_awaited()
            await tm.model_update_lock.release_reader()

        asyncio.run(run())

    def test_in_place_refusal_preserves_existing_pause(self):
        async def run():
            tm = _bare_tokenizer_manager(is_pause=True)
            await tm.model_update_lock.acquire_reader()

            with self.assertRaisesRegex(RuntimeError, "in flight"):
                await tm.pause_generation(PauseGenerationReqInput(mode="in_place"))

            self.assertTrue(tm.is_pause)
            await tm.model_update_lock.release_reader()

        asyncio.run(run())

    def test_request_admission_cannot_straddle_pause(self):
        """A request past the gate but not yet holding the reader lock must
        wait out a pause that closed the gate meanwhile, not enter."""

        async def run():
            tm = _bare_tokenizer_manager()
            original_acquire_reader = tm.model_update_lock.acquire_reader
            admission_entered = asyncio.Event()
            admission_release = asyncio.Event()

            async def delayed_acquire_reader():
                admission_entered.set()
                await admission_release.wait()
                await original_acquire_reader()

            tm.model_update_lock.acquire_reader = delayed_acquire_reader
            admission_task = asyncio.create_task(tm._acquire_generation_reader())
            await admission_entered.wait()

            pause_task = asyncio.create_task(
                tm.pause_generation(PauseGenerationReqInput(mode="in_place"))
            )
            await asyncio.wait_for(pause_task, timeout=1)
            tm._async_dispatch_to_scheduler.assert_awaited_once()

            admission_release.set()
            await asyncio.sleep(0)
            self.assertFalse(admission_task.done(), "request crossed the closed gate")
            self.assertFalse(await tm.model_update_lock.is_locked())

            async with tm.is_pause_cond:
                tm.is_pause = False
                tm.is_pause_cond.notify_all()
            reader_lock = await asyncio.wait_for(admission_task, timeout=1)
            self.assertTrue(await tm.model_update_lock.is_locked())
            await reader_lock.__aexit__(None, None, None)

        asyncio.run(run())

    def test_abort_pause_completes_when_blocked_admission_takes_the_reader(self):
        """An abort-mode pause holds the pause condition across its polling
        loop. A request that passed the gate and was queued behind a weight
        update must release the reader it then obtains without needing the
        condition, or the pause polls forever and the request never wakes."""

        async def run():
            tm = _bare_tokenizer_manager()
            tm.abort_request = Mock()
            await tm.model_update_lock.acquire_reader()  # the in-flight request
            writer_release = asyncio.Event()

            async def hold_writer():
                async with tm.model_update_lock.writer_lock:
                    await writer_release.wait()

            writer_task = asyncio.create_task(hold_writer())
            while tm.model_update_lock._waiting_writers == 0:
                await asyncio.sleep(0)
            admission_task = asyncio.create_task(tm._acquire_generation_reader())
            await asyncio.sleep(0)  # past the gate, queued behind the writer
            pause_task = asyncio.create_task(
                tm.pause_generation(PauseGenerationReqInput(mode="abort"))
            )
            await asyncio.sleep(0)  # gate closed, polling with the condition held
            self.assertTrue(tm.is_pause)
            tm.abort_request.assert_called_with(abort_all=True)

            await tm.model_update_lock.release_reader()  # aborted request finishes
            writer_release.set()  # weight update finishes; admission gets the reader
            await asyncio.wait_for(pause_task, timeout=5)
            self.assertFalse(admission_task.done())
            self.assertFalse(await tm.model_update_lock.is_locked())

            async with tm.is_pause_cond:
                tm.is_pause = False
                tm.is_pause_cond.notify_all()
            reader_lock = await asyncio.wait_for(admission_task, timeout=1)
            await reader_lock.__aexit__(None, None, None)
            await writer_task

        asyncio.run(run())

    def test_failed_in_place_pause_restores_gate(self):
        async def run():
            tm = _bare_tokenizer_manager()
            tm._async_dispatch_to_scheduler.side_effect = RuntimeError(
                "dispatch failed"
            )

            with self.assertRaisesRegex(RuntimeError, "dispatch failed"):
                await tm.pause_generation(PauseGenerationReqInput(mode="in_place"))
            self.assertFalse(tm.is_pause)
            reader_lock = await asyncio.wait_for(
                tm._acquire_generation_reader(), timeout=1
            )
            await reader_lock.__aexit__(None, None, None)

            tm.is_pause = True
            with self.assertRaisesRegex(RuntimeError, "dispatch failed"):
                await tm.pause_generation(PauseGenerationReqInput(mode="in_place"))
            self.assertTrue(tm.is_pause)

        asyncio.run(run())


class TestReleaseRefusal(CustomTestCase):
    def test_busy_release_returns_failure_instead_of_asserting(self):
        updater = object.__new__(SchedulerWeightUpdaterManager)
        updater.scheduler = None
        updater.is_fully_idle = Mock(return_value=False)

        result = updater.release_memory_occupation(
            ReleaseMemoryOccupationReqInput(tags=["kv_cache"])
        )

        self.assertFalse(result.success)
        self.assertIn("idle scheduler", result.message)

    def test_release_surfaces_fanout_failure(self):
        async def run():
            manager = TokenizerManager.__new__(TokenizerManager)
            manager.auto_create_handle_loop = Mock()
            manager.release_memory_occupation_communicator = AsyncMock(
                return_value=[
                    ReleaseMemoryOccupationReqOutput(),
                    ReleaseMemoryOccupationReqOutput(
                        success=False, message="rank busy"
                    ),
                ]
            )

            with self.assertRaisesRegex(RuntimeError, "rank busy"):
                await TokenizerControlMixin.release_memory_occupation(
                    manager, ReleaseMemoryOccupationReqInput(tags=["kv_cache"])
                )

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
