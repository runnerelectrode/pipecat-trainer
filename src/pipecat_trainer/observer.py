"""PolyvoiceObserver: one JSONL record per LLM request, captured as a Pipecat observer.

Facts this relies on (pipecat dbdf21a): observer callbacks run on a per-observer queue and never block the
pipeline, but an exception inside one silently kills the observer for the session, so every handler is
wrapped; LLMContextFrame reaches the LLM from both directions (assistant aggregator re-runs upstream), so
dedupe on frame.id; reply/tool-call frames are broadcast as two instances, so count source-is-LLM DOWNSTREAM
copies only; a speculative turn may produce no reply frames at all.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from loguru import logger
from pipecat.frames.frames import (
    CancelFrame, EagerEndOfTurnCancelFrame, EndFrame, FunctionCallInProgressFrame, FunctionCallResultFrame,
    InterruptionFrame, LLMContextFrame, LLMFullResponseEndFrame, LLMTextFrame, MetricsFrame, StopFrame,
)
from pipecat.metrics.metrics import LLMUsageMetricsData, TTFBMetricsData
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.processors.aggregators.llm_context import LLMContext, LLMSpecificMessage
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.llm_service import LLMService
from pipecat.utils.types import NotGiven


def _json_default(o: Any):
    if isinstance(o, LLMSpecificMessage):
        return {"llm": o.llm, "message": o.message}
    if hasattr(o, "to_default_dict"):
        return o.to_default_dict()
    if hasattr(o, "model_dump"):
        return o.model_dump()
    return str(o)


def _tools_of(ctx: LLMContext) -> list[dict] | None:
    tools = ctx.tools
    if tools is None or isinstance(tools, NotGiven):
        return None
    return [t.to_default_dict() for t in tools.standard_tools]


@dataclass
class _Turn:
    turn: int
    speculation: bool
    messages: list
    tools: list[dict] | None
    tool_choice: Any
    started_at: float
    node: str | None = None
    reply: list[str] = field(default_factory=list)
    tool_calls: dict[str, dict] = field(default_factory=dict)
    ttfb_s: float | None = None
    usage: dict | None = None
    model: str | None = None
    was_interrupted: bool = False
    discarded: bool = False
    ended_at: float | None = None


class PolyvoiceObserver(BaseObserver):
    """Writes a JSONL record per LLM request. Never raises into the pipeline."""

    def __init__(self, *, session_id: str, path: str | Path, llm: FrameProcessor | None = None,
                 node_getter: Callable[[], str | None] | None = None, **kw):
        super().__init__(**kw)
        self._session_id = session_id
        self._path = Path(path)
        self._llm = llm                      # explicit identity, else isinstance(LLMService)
        self._node_getter = node_getter      # e.g. lambda: flow_manager.current_node (late-bound)
        self._turn_no = 0
        self._current: _Turn | None = None
        self._seen: set[int] = set()
        self._end_reason: str | None = None
        self._fh = None

    # ---- lifecycle ---------------------------------------------------------
    async def cleanup(self):
        try:
            if self._current is not None:
                self._finish(self._current, interrupted=True)
            self._write({"type": "session_end", "session_id": self._session_id, "turns": self._turn_no,
                         "end_reason": self._end_reason, "at": time.time()})
            if self._fh is not None:
                self._fh.close()
                self._fh = None
        finally:
            await super().cleanup()

    def _write(self, rec: dict) -> None:
        # Observer callbacks already run on their own task, off the pipeline's path; a small append+flush
        # here is cheaper and safer than a writer task that cleanup has to drain.
        if self._fh is None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = self._path.open("a", encoding="utf-8")
        self._fh.write(json.dumps(rec, default=_json_default, ensure_ascii=False) + "\n")
        self._fh.flush()

    def bind(self, worker) -> None:
        """Optional: attach the worker's turn tracker so conversational turn ends mark interruptions."""
        tto = getattr(worker, "turn_tracking_observer", None)
        if tto is None:
            return

        async def on_turn_ended(_obs, _n, _duration_s, was_interrupted):
            if self._current is not None and was_interrupted:
                self._current.was_interrupted = True

        tto.add_event_handler("on_turn_ended", on_turn_ended)

    # ---- helpers -----------------------------------------------------------
    def _is_llm(self, p: FrameProcessor) -> bool:
        return p is self._llm if self._llm is not None else isinstance(p, LLMService)

    def _emit(self, t: _Turn, end_reason: str | None):
        rec = {
            "type": "turn", "session_id": self._session_id, "turn": t.turn, "speculation": t.speculation, "discarded": t.discarded,
            "node": t.node, "messages": t.messages, "tools": t.tools,
            "tool_choice": None if isinstance(t.tool_choice, NotGiven) else t.tool_choice,
            "reply": "".join(t.reply), "tool_calls": list(t.tool_calls.values()),
            "was_interrupted": t.was_interrupted, "llm_ttfb_s": t.ttfb_s, "usage": t.usage, "model": t.model,
            "end_reason": end_reason, "started_at": t.started_at, "ended_at": t.ended_at or time.time(),
            "token_exact": False,
        }
        self._write(rec)

    def _finish(self, t: _Turn, *, interrupted: bool = False, discarded: bool = False):
        t.was_interrupted = t.was_interrupted or interrupted
        t.discarded = t.discarded or discarded
        t.ended_at = time.time()
        self._emit(t, self._end_reason)
        if self._current is t:
            self._current = None

    # ---- frames ------------------------------------------------------------
    async def on_push_frame(self, data: FramePushed):
        try:
            self._handle(data)
        except Exception as e:  # a raise here would silently kill this observer for the session
            logger.warning(f"{self}: dropped event {data.frame}: {e!r}")

    def _handle(self, data: FramePushed):
        f, src, dst, d = data.frame, data.source, data.destination, data.direction

        # 1) request: the exact context reaching the LLM (either direction; dedupe by frame id)
        if isinstance(f, LLMContextFrame) and self._is_llm(dst) and f.id not in self._seen:
            self._seen.add(f.id)
            if self._current is not None and self._current.ended_at is None:
                self._finish(self._current, interrupted=True)
            self._turn_no += 1
            ctx = f.context
            self._current = _Turn(
                turn=self._turn_no, speculation=f.speculation,
                messages=ctx.get_messages(truncate_large_values=True),  # deep copies; base64 media → placeholders
                tools=_tools_of(ctx), tool_choice=ctx.tool_choice, started_at=time.time(),
                node=self._node_getter() if self._node_getter else None,
            )
            return

        # 2) session-level frames from anywhere (dedupe: broadcast copies differ in id)
        if isinstance(f, (EndFrame, CancelFrame, StopFrame)) and f.id not in self._seen:
            self._seen.add(f.id)
            reason = getattr(f, "reason", None)
            self._end_reason = str(reason) if reason else ("end" if isinstance(f, EndFrame) else type(f).__name__)
            if self._current is not None:
                self._finish(self._current, interrupted=True)
            return
        if isinstance(f, InterruptionFrame) and f.id not in self._seen:
            self._seen.add(f.id)
            if self._current is not None:
                self._finish(self._current, interrupted=True)
            return

        # 3) everything below: source is the LLM service; count DOWNSTREAM copies only
        t = self._current
        if t is None or not self._is_llm(src) or d != FrameDirection.DOWNSTREAM:
            return
        if isinstance(f, LLMTextFrame):
            t.reply.append(f.text)
        elif isinstance(f, MetricsFrame):
            for m in f.data:
                if isinstance(m, TTFBMetricsData):
                    t.ttfb_s, t.model = m.value, m.model
                elif isinstance(m, LLMUsageMetricsData):
                    t.usage, t.model = m.value.model_dump(exclude_none=True), m.model
        elif isinstance(f, FunctionCallInProgressFrame):
            t.tool_calls[f.tool_call_id] = {"id": f.tool_call_id, "name": f.function_name,
                                            "arguments": f.arguments, "result": None, "error": None}
        elif isinstance(f, FunctionCallResultFrame):
            tc = t.tool_calls.setdefault(f.tool_call_id, {"id": f.tool_call_id, "name": f.function_name,
                                                          "arguments": f.arguments})
            tc["result"], tc["error"] = f.result, f.error
            if f.properties is not None and not f.properties.is_final:
                tc["partial"] = True
        elif isinstance(f, EagerEndOfTurnCancelFrame):
            self._finish(t, discarded=True)
        elif isinstance(f, LLMFullResponseEndFrame):
            self._finish(t)  # a tool-call turn re-runs the LLM with a new LLMContextFrame → new record
