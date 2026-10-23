# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU mocks for CUDA event completion; no model or GPU is constructed."""

import json
import tempfile
import unittest
from threading import Event
from unittest.mock import Mock, patch

from vllm.v1.worker.batch_inspect import BatchInspector


def records(inspector):
    """Read JSONL after the writer has closed."""
    return [json.loads(line) for line in inspector.path.read_text().splitlines()]


class TestBatchTiming(unittest.TestCase):
    @patch("vllm.v1.worker.batch_inspect.envs.VLLM_BATCH_INSPECT_TIMING", True)
    def test_writer_waits_for_events_without_blocking_producer(self):
        ready, polled = Event(), Event()
        start, end, stream = Mock(), Mock(), Mock()
        start.elapsed_time.return_value = 2.5

        def query():
            polled.set()
            return ready.is_set()

        end.query.side_effect = query
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("torch.Event", side_effect=[start, end]),
            patch("torch.accelerator.current_stream", return_value=stream),
            patch("torch.accelerator.synchronize") as synchronize,
        ):
            inspector = BatchInspector(directory)
            try:
                timer = inspector.start_timing("cuda:0")
                inspector.end_timing(timer, step=42, phase="draft")
                self.assertTrue(polled.wait(2))
                start.elapsed_time.assert_not_called()
                start.record.assert_called_once_with(stream)
                end.record.assert_called_once_with(stream)
            finally:
                ready.set()
                self.assertTrue(inspector.close())
            synchronize.assert_not_called()
            start.synchronize.assert_not_called()
            end.synchronize.assert_not_called()
            row, summary = records(inspector)
            self.assertEqual((row["step"], row["phase"]), (42, "draft"))
            self.assertGreaterEqual(row["cpu_elapsed_ms"], 0)
            self.assertEqual(row["gpu_elapsed_ms"], 2.5)
            self.assertEqual(row["gpu_timing_status"], "complete")
            self.assertNotIn("_gpu_events", row)
            self.assertEqual(summary["dropped"], 0)

    @patch("vllm.v1.worker.batch_inspect.envs.VLLM_BATCH_INSPECT_TIMING", True)
    def test_full_queue_skips_event_creation_and_reports_loss(self):
        entered, release = Event(), Event()
        write_record = BatchInspector._write_record

        def slow_write(stream, record):
            if record.get("block"):
                entered.set()
                release.wait(5)
            write_record(stream, record)

        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(BatchInspector, "_write_record", side_effect=slow_write),
            patch("torch.Event") as event,
        ):
            inspector = BatchInspector(directory, queue_capacity=1)
            try:
                inspector.enqueue_statistics({"type": "test", "block": True})
                self.assertTrue(entered.wait(2))
                inspector.enqueue_statistics({"type": "test"})
                self.assertIsNone(inspector.start_timing("cuda:0"))
                event.assert_not_called()
            finally:
                release.set()
                self.assertTrue(inspector.close())
            self.assertEqual(records(inspector)[-1]["dropped"], 1)

    def test_incomplete_gpu_event_has_bounded_shutdown(self):
        polled = Event()
        start, end = Mock(), Mock()

        def query():
            polled.set()
            return False

        end.query.side_effect = query
        with tempfile.TemporaryDirectory() as directory:
            inspector = BatchInspector(directory)
            inspector.enqueue_statistics(
                {
                    "type": "timing",
                    "step": 7,
                    "phase": "verification",
                    "cpu_elapsed_ms": 1.0,
                    "_gpu_events": (start, end),
                }
            )
            self.assertTrue(polled.wait(2))
            self.assertFalse(inspector.close(timeout=0.01))
            inspector._thread.join(timeout=2)
            self.assertFalse(inspector._thread.is_alive())
            start.elapsed_time.assert_not_called()
            row, summary = records(inspector)
            self.assertIsNone(row["gpu_elapsed_ms"])
            self.assertEqual(row["gpu_timing_status"], "incomplete")
            self.assertEqual(summary["timings_incomplete"], 1)

    @patch("vllm.v1.worker.batch_inspect.envs.VLLM_BATCH_INSPECT_TIMING", True)
    def test_event_failure_keeps_metadata_writer_available(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("torch.Event", side_effect=RuntimeError("event failed")),
        ):
            inspector = BatchInspector(directory)
            with self.assertLogs("vllm.v1.worker.batch_inspect", level="WARNING"):
                self.assertIsNone(inspector.start_timing("cuda:0"))
            self.assertTrue(inspector.enqueue_statistics({"type": "test"}))
            self.assertFalse(inspector.close())
            self.assertEqual(records(inspector)[0]["type"], "test")


if __name__ == "__main__":
    unittest.main()
