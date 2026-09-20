"""A standard Pipecat voice bot (Gradium STT + TTS, Smart Turn, WebRTC) made trainable with two polyvoice lines.

    uv run bot.py -t webrtc          # browser at http://localhost:7860  (speak to it)
    uv run bot.py -t eval --port 7900  # Pipecat Evals / polyvoice sandbox rollouts, text mode, no audio keys needed

LLM = PhoneLLM behind the polyloop proxy (POLYVOICE_PROXY_URL). For a dry run without the GPU node, point it at
General Compute (gemma-4-31B-it) instead; the bot code is identical, only the URL and model name change.
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

from pipecat_trainer import PolyvoiceObserver, polyvoice_llm   # <- polyvoice line 1: imports

load_dotenv(override=True)
HERE = Path(__file__).parent

transport_params = {
    "webrtc": lambda: TransportParams(audio_in_enabled=True, audio_out_enabled=True, vad_analyzer=SileroVADAnalyzer()),
    "eval": lambda: EvalTransportParams(audio_in_enabled=True, audio_out_enabled=True),
}


def _speech(runner_args: RunnerArguments):
    """Gradium in and out. Under `-t eval` the harness sends text turns and marks the LLM output skip_tts,
    so the sandbox needs neither."""
    if isinstance(runner_args, EvalRunnerArguments) or os.getenv("POLYVOICE_TEXT_ONLY") == "1":
        return None, None
    from pipecat.services.gradium.stt import GradiumSTTService
    from pipecat.services.gradium.tts import GradiumTTSService
    stt = GradiumSTTService(api_key=os.environ["GRADIUM_API_KEY"])
    tts_settings = GradiumTTSService.Settings()
    if os.getenv("GRADIUM_VOICE_ID"):
        tts_settings.voice = os.environ["GRADIUM_VOICE_ID"]
    tts = GradiumTTSService(api_key=os.environ["GRADIUM_API_KEY"], settings=tts_settings)
    return stt, tts


async def bot(runner_args: RunnerArguments):
    session_id = os.getenv("POLYVOICE_SESSION") or runner_args.session_id or str(uuid.uuid4())
    transport = await create_transport(runner_args, transport_params)
    stt, tts = _speech(runner_args)

    llm = polyvoice_llm(                                              # <- polyvoice line 2: the LLM slot
        session_id,
        base_url=os.environ["POLYVOICE_PROXY_URL"],
        model=os.getenv("POLYVOICE_MODEL", "polyvoice/live"),
        api_key=os.getenv("POLYVOICE_API_KEY", "polyvoice"),
        system_instruction=(HERE / "system.md").read_text(),
        temperature=float(os.getenv("POLYVOICE_TEMPERATURE", "0.7")),
    )
    context = LLMContext()
    user_agg, assistant_agg = LLMContextAggregatorPair(
        context, user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()))  # Smart Turn v3 is the default stop strategy

    stages = [transport.input(), stt, user_agg, llm, tts, transport.output(), assistant_agg]
    pipeline = Pipeline([s for s in stages if s is not None])

    traces = Path(os.getenv("POLYVOICE_TRACES_DIR", HERE / "traces"))
    observer = PolyvoiceObserver(session_id=session_id, path=traces / f"{session_id}.jsonl", llm=llm)  # <- polyvoice line 3: capture
    worker = PipelineWorker(pipeline, params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
                            conversation_id=session_id, observers=[observer],
                            idle_timeout_secs=runner_args.pipeline_idle_timeout_secs)
    observer.bind(worker)
    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)
    await runner.add_workers(worker)

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"session {session_id}: caller connected")
        context.add_message({"role": "developer", "content": "Answer the call: greet the caller briefly and ask how you can help."})
        await worker.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        await runner.cancel(reason="client-disconnected")

    await runner.run()


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
