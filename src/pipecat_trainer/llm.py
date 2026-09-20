"""PolyvoiceLLMService: OpenAILLMService that tags each request with X-Session-Id and X-Turn-Type."""
from __future__ import annotations

from pipecat.adapters.services.open_ai_adapter import OpenAILLMInvocationParams
from pipecat.frames.frames import Frame, LLMContextFrame
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.openai.llm import OpenAILLMService


class PolyvoiceLLMService(OpenAILLMService):
    # Most OpenAI-compatible servers (vLLM chat templates, hosted open models) do not know the `developer`
    # role; Pipecat's adapter rewrites it to `user` when this is False, which also matches what the
    # trainee's renderer will see.
    supports_developer_role = False

    def __init__(self, *, session_id: str, **kwargs):
        super().__init__(**kwargs)
        self._session_id = session_id
        self._turn_type = "main"

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
        return params


def polyvoice_llm(session_id: str, *, base_url: str, model: str, system_instruction: str | None = None,
                  temperature: float | None = None, api_key: str = "polyvoice") -> OpenAILLMService:
    settings = PolyvoiceLLMService.Settings(model=model)  # unset fields stay NOT_GIVEN
    if system_instruction is not None:
        settings.system_instruction = system_instruction
    if temperature is not None:
        settings.temperature = temperature
    return PolyvoiceLLMService(session_id=session_id, base_url=base_url, api_key=api_key, settings=settings,
                               default_headers={"X-Session-Id": session_id})
