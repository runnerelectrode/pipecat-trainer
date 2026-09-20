"""The polyvoice bot template: a general-purpose Pipecat cascade bot whose LLM is the polyloop proxy.

Recipe-agnostic. A recipe supplies only its system prompt (POLYVOICE_SYSTEM_PROMPT) and its scenarios; the
bot layout follows pipecat-examples/phonellm, and the two polyvoice lines are the LLM constructor and the
observer. Copy it into a recipe when it needs tools or a different pipeline.

    python -m pipecat_trainer.bot -t eval --port 7900   # Pipecat Evals / polyvoice rollouts (text mode: no STT/TTS)
    python -m pipecat_trainer.bot -t webrtc             # browser client, Roark self-hosted simulations
    python -m pipecat_trainer.bot -t daily              # Daily room (Coval / Cekura / Roark via Pipecat Cloud)

Env:
    POLYVOICE_SYSTEM_PROMPT  path to the system prompt file (required)
    POLYVOICE_GREETING       developer instruction for the first turn (default: greet and ask how to help)
    POLYVOICE_PROXY_URL   OpenAI-compatible base URL (the polyloop proxy, or any server holding the adapter)
    POLYVOICE_MODEL       model name to request (default: polyvoice/live)
    POLYVOICE_API_KEY     bearer for the proxy (default: polyvoice)
    POLYVOICE_SESSION     session id override (the rollout driver sets one per bot process)
    POLYVOICE_TRACES_DIR  where the observer writes <session>.jsonl (default: ./traces)
    POLYVOICE_TEXT_ONLY   1 = never build STT/TTS (default: build them only if their keys are present)
    DEEPGRAM_API_KEY, CARTESIA_API_KEY, DAILY_API_KEY as in any Pipecat bot
"""
from __future__ import annotations

import os
import uuid
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.evals.transport import EvalTransportParams
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair, LLMUserAggregatorParams
from pipecat.runner.types import EvalRunnerArguments, RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.transports.base_transport import TransportParams
from pipecat.workers.runner import WorkerRunner

from pipecat_trainer import PolyvoiceObserver, polyvoice_llm

load_dotenv(override=True)


def _daily_params():
    from pipecat.transports.daily.transport import DailyParams
    return DailyParams(audio_in_enabled=True, audio_out_enabled=True, vad_analyzer=SileroVADAnalyzer())


transport_params = {
    "eval": lambda: EvalTransportParams(audio_in_enabled=True, audio_out_enabled=True),
    "webrtc": lambda: TransportParams(audio_in_enabled=True, audio_out_enabled=True, vad_analyzer=SileroVADAnalyzer()),
    "daily": _daily_params,
}


def _audio_services(runner_args: RunnerArguments):
    """STT and TTS only when the run needs audio and the keys exist. Under `-t eval` the harness sends text
    turns (RTVI send-text → LLMMessagesAppendFrame) and marks LLM output skip_tts, so neither is required."""
    text_only = os.getenv("POLYVOICE_TEXT_ONLY") == "1" or isinstance(runner_args, EvalRunnerArguments)
    if text_only or not (os.getenv("DEEPGRAM_API_KEY") and os.getenv("CARTESIA_API_KEY")):
        return None, None
    from pipecat.services.cartesia.tts import CartesiaTTSService
    from pipecat.services.deepgram.stt import DeepgramSTTService
    stt = DeepgramSTTService(api_key=os.environ["DEEPGRAM_API_KEY"])
    tts = CartesiaTTSService(api_key=os.environ["CARTESIA_API_KEY"])
    return stt, tts


async def bot(runner_args: RunnerArguments):
    session_id = os.getenv("POLYVOICE_SESSION") or runner_args.session_id or str(uuid.uuid4())
    transport = await create_transport(runner_args, transport_params)
    stt, tts = _audio_services(runner_args)

    llm = polyvoice_llm(
        session_id,
        base_url=os.environ["POLYVOICE_PROXY_URL"],
        model=os.getenv("POLYVOICE_MODEL", "polyvoice/live"),
        api_key=os.getenv("POLYVOICE_API_KEY", "polyvoice"),
        system_instruction=Path(os.environ["POLYVOICE_SYSTEM_PROMPT"]).read_text(),
        temperature=float(os.getenv("POLYVOICE_TEMPERATURE", "0.7")),
    )
    context = LLMContext()
    user_agg, assistant_agg = LLMContextAggregatorPair(
        context, user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()))

    stages = [transport.input(), stt, user_agg, llm, tts, transport.output(), assistant_agg]
    pipeline = Pipeline([s for s in stages if s is not None])

    traces_dir = Path(os.getenv("POLYVOICE_TRACES_DIR", "traces"))
    observer = PolyvoiceObserver(session_id=session_id, path=traces_dir / f"{session_id}.jsonl", llm=llm)
    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
        conversation_id=session_id,
        observers=[observer],
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
    )
    observer.bind(worker)
    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)
    await runner.add_workers(worker)

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"session {session_id}: client connected")
        greeting = os.getenv("POLYVOICE_GREETING", "Answer the call: greet the caller briefly and ask how you can help.")
        context.add_message({"role": "developer", "content": greeting})
        await worker.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        await runner.cancel(reason="client-disconnected")

    await runner.run()


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
