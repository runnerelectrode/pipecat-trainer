"""PolyvoiceLLMService: OpenAILLMService that tags each request with X-Session-Id and X-Turn-Type, and by
default asks the server for whole turns (non-streaming) and feeds them to Pipecat as one chunk."""
from __future__ import annotations

import os
from typing import AsyncIterator

from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import Choice, ChoiceDelta, ChoiceDeltaToolCall, ChoiceDeltaToolCallFunction
from pipecat.adapters.services.open_ai_adapter import OpenAILLMInvocationParams
from pipecat.frames.frames import Frame, LLMContextFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.utils.types import assert_given


def completion_to_chunk(c: ChatCompletion) -> ChatCompletionChunk:
    """One non-streaming completion -> the single chunk Pipecat's stream loop expects."""
    choice = c.choices[0]
    msg = choice.message
    tool_calls = None
    if msg.tool_calls:
        tool_calls = [ChoiceDeltaToolCall(index=i, id=tc.id, type="function",
                                          function=ChoiceDeltaToolCallFunction(name=tc.function.name, arguments=tc.function.arguments))
                      for i, tc in enumerate(msg.tool_calls)]
    delta = ChoiceDelta(role="assistant", content=msg.content, tool_calls=tool_calls)
    return ChatCompletionChunk(id=c.id, object="chat.completion.chunk", created=c.created, model=c.model,
                               choices=[Choice(index=0, delta=delta, finish_reason=choice.finish_reason)], usage=c.usage)


class PolyvoiceLLMService(OpenAILLMService):
    # Most OpenAI-compatible servers (vLLM chat templates, hosted open models) do not know the `developer`
    # role; Pipecat's adapter rewrites it to `user` when this is False, which also matches what the
    # trainee's renderer will see.
    supports_developer_role = False

    def __init__(self, *, session_id: str, stream: bool | None = None, **kwargs):
        super().__init__(**kwargs)
        self._session_id = session_id
        self._turn_type = "main"
        # The polyloop proxy records whole turns and answers non-streaming; default to one request per turn.
        # POLYVOICE_STREAM=1 restores SSE for servers that stream. Non-streaming TTFB measures the whole reply.
        self._stream = stream if stream is not None else os.getenv("POLYVOICE_STREAM", "0") == "1"

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        if isinstance(frame, LLMContextFrame):  # captured before base_llm consumes the flag
            self._turn_type = "speculative" if frame.speculation else "main"
        await super().process_frame(frame, direction)

    def build_chat_completion_params(self, params_from_context: OpenAILLMInvocationParams) -> dict:
        params = super().build_chat_completion_params(params_from_context)  # settings.extra already merged
        headers = dict(params.get("extra_headers") or {})
        headers["X-Turn-Type"] = self._turn_type
        headers.setdefault("X-Session-Id", self._session_id)
        params["extra_headers"] = headers  # honoured by AsyncOpenAI.chat.completions.create, streaming too
        if not self._stream:
            params["stream"] = False
            params.pop("stream_options", None)
        return params

    async def get_chat_completions(self, context: LLMContext):
        if self._stream:
            return await super().get_chat_completions(context)
        adapter = self.get_llm_adapter()
        invocation = adapter.get_llm_invocation_params(
            context, system_instruction=assert_given(self._settings.system_instruction),
            convert_developer_to_user=not self.supports_developer_role)
        completion = await self._client.chat.completions.create(**self.build_chat_completion_params(invocation))

        async def one() -> AsyncIterator[ChatCompletionChunk]:
            yield completion_to_chunk(completion)

        return one()


def polyvoice_llm(session_id: str, *, base_url: str, model: str, system_instruction: str | None = None,
                  temperature: float | None = None, api_key: str = "polyvoice", stream: bool | None = None) -> OpenAILLMService:
    settings = PolyvoiceLLMService.Settings(model=model)  # unset fields stay NOT_GIVEN
    if system_instruction is not None:
        settings.system_instruction = system_instruction
    if temperature is not None:
        settings.temperature = temperature
    return PolyvoiceLLMService(session_id=session_id, stream=stream, base_url=base_url, api_key=api_key,
                               settings=settings, default_headers={"X-Session-Id": session_id})
