"""Tests for the observer and the LLM helper, in Pipecat's own test style (pipecat.tests.utils.run_test)."""
import json
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

from pipecat.frames.frames import (
    FunctionCallInProgressFrame, FunctionCallResultFrame, LLMContextFrame, LLMFullResponseEndFrame,
    LLMFullResponseStartFrame, LLMTextFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.tests.utils import SleepFrame, run_test

from pipecat_trainer.llm import PolyvoiceLLMService
from pipecat_trainer.observer import PolyvoiceObserver


class FakeLLM(FrameProcessor):
    """Stands in for an LLM service: consumes LLMContextFrame, emits a reply and one tool call."""

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMContextFrame):
            await self.push_frame(LLMFullResponseStartFrame())
            await self.push_frame(LLMTextFrame("Hello "))
            await self.push_frame(LLMTextFrame("there."))
            await self.broadcast_frame(FunctionCallInProgressFrame, function_name="lookup",
                                       tool_call_id="c1", arguments={"q": "x"})
            await self.broadcast_frame(FunctionCallResultFrame, function_name="lookup",
                                       tool_call_id="c1", arguments={"q": "x"}, result={"ok": True})
            await self.push_frame(LLMFullResponseEndFrame())
        else:
            await self.push_frame(frame, direction)


class TestPolyvoiceObserver(unittest.IsolatedAsyncioTestCase):
    async def test_one_record_per_llm_turn(self):
        llm = FakeLLM()
        path = self.enterContext(tempfile.TemporaryDirectory()) + "/turns.jsonl"
        obs = PolyvoiceObserver(session_id="s1", path=path, llm=llm)
        ctx = LLMContext(messages=[{"role": "user", "content": "hi"}])
        await run_test(
            llm,
            frames_to_send=[LLMContextFrame(context=ctx), SleepFrame(sleep=0.1)],
            expected_down_frames=[LLMFullResponseStartFrame, LLMTextFrame, LLMTextFrame,
                                  FunctionCallInProgressFrame, FunctionCallResultFrame, LLMFullResponseEndFrame],
            observers=[obs],
        )
        records = [json.loads(line) for line in open(path)]
        self.assertEqual([r["type"] for r in records], ["turn", "session_end"])
        r = records[0]
        self.assertEqual((r["session_id"], r["turn"], r["speculation"]), ("s1", 1, False))
        self.assertEqual(r["messages"], [{"role": "user", "content": "hi"}])
        self.assertEqual(r["reply"], "Hello there.")
        self.assertEqual(r["tool_calls"], [{"id": "c1", "name": "lookup", "arguments": {"q": "x"},
                                            "result": {"ok": True}, "error": None}])
        self.assertIsNone(r["end_reason"])  # the turn closed before the session did
        self.assertEqual(records[1]["end_reason"], "end")  # run_test sends EndFrame() with no reason

    async def test_turn_type_header(self):
        with patch.object(PolyvoiceLLMService, "create_client"):
            svc = PolyvoiceLLMService(session_id="s1", settings=PolyvoiceLLMService.Settings(model="m"))
            svc._client = AsyncMock()
            svc.push_frame = AsyncMock()
            svc.push_error = AsyncMock()
            svc.start_processing_metrics = AsyncMock()
            svc.stop_processing_metrics = AsyncMock()
            seen = []

            async def fake_process_context(context):
                params = svc.build_chat_completion_params({"messages": [], "tools": [], "tool_choice": None})
                seen.append(params["extra_headers"]["X-Turn-Type"])

            svc._process_context = fake_process_context
            await svc.process_frame(LLMContextFrame(context=LLMContext(messages=[])), FrameDirection.DOWNSTREAM)
            await svc.process_frame(LLMContextFrame(context=LLMContext(messages=[]), speculation=True),
                                    FrameDirection.DOWNSTREAM)
            self.assertEqual(seen, ["main", "speculative"])
