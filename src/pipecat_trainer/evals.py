"""PipecatEvalsVerifier: K text-mode simulations per scenario, one bot process per rollout.

Relies on (pipecat >=1.11): EvalScenarioFile.load, EvalSession.from_scenario(persona_llm=, judge=),
EvalSessionParams(stop_bot=True), EvalSimulationResult fields; a single WebSocket client per bot process and no
context reset between runs; judge cache in-memory per EvalJudge instance (build one per rollout).
Text mode needs only the base `pipecat-ai` package; run the harness in its own venv.
"""
from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import json
import os
import socket
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from pipecat.evals.judge import EvalJudge
from pipecat.evals.results import EvalSimulationResult
from pipecat.evals.scenario import EvalScenarioFile, is_scenario_file
from pipecat.evals.session import EvalSession, EvalSessionParams
from pipecat.evals.simulation import EvalSimulationScenario
from pipecat.services.openai.llm import OpenAILLMService

BOT_CONNECT_TIMEOUT_S = 60.0
BOT_STOP_TIMEOUT_S = 10.0
ROLLOUT_SAFETY_TIMEOUT_S = 600.0


@dataclass
class LLMEndpoint:
    base_url: str
    api_key: str
    model: str
    temperature: float = 0.0
    seed: int | None = None

    def service(self) -> OpenAILLMService:
        settings = OpenAILLMService.Settings(model=self.model, temperature=self.temperature)
        if self.seed is not None:
            settings.seed = self.seed
        return OpenAILLMService(base_url=self.base_url, api_key=self.api_key, settings=settings)


@dataclass
class Conversation:
    scenario: str
    rollout: int
    session_id: str
    messages: list[dict]
    reward: float | None
    hint: str
    passed: bool
    succeeded: bool
    ended_by: str
    metrics: list[dict]
    tool_calls: list[dict]
    duration_ms: int
    error: str | None = None


def default_reward(r: EvalSimulationResult) -> float | None:
    if r.error is not None:
        return None  # not a verdict; drop from training
    scored = [m.score for m in r.metrics if m.score is not None]
    return float(r.succeeded) * (sum(scored) / len(scored) if scored else 1.0)


def default_hint(r: EvalSimulationResult) -> str:
    parts = [f"goal: {r.reason}"] if r.reason else []
    for m in r.metrics:
        for v in m.verdicts:
            if not v.passed and v.reason:
                parts.append(f"{m.name} turn {v.turn}: {v.reason}")
    return "; ".join(parts)


@dataclass
class PipecatEvalsVerifier:
    scenarios_dir: Path
    bot_path: Path
    policy_url: str
    judge: LLMEndpoint
    persona: LLMEndpoint
    k: int = 4
    concurrency: int = 4
    base_port: int = 7900
    host: str = "127.0.0.1"
    bot_python: str = sys.executable
    logs_dir: Path = Path("eval-runs")
    env: dict = field(default_factory=dict)
    reward_fn: Callable[[EvalSimulationResult], float | None] = default_reward
    hint_fn: Callable[[EvalSimulationResult], str] = default_hint

    def scenarios(self) -> list[EvalSimulationScenario]:
        out: list[EvalSimulationScenario] = []
        for p in sorted(self.scenarios_dir.rglob("*.y*ml")):
            if not is_scenario_file(p):
                continue  # !include fragments (judge_text.yaml etc.)
            out += [s for s in EvalScenarioFile.load(p) if isinstance(s, EvalSimulationScenario)]
        return out

    async def run(self, results_path: Path, scenarios: list[EvalSimulationScenario] | None = None) -> list[Conversation]:
        os.environ.setdefault("OMP_NUM_THREADS", str(max(1, (os.cpu_count() or 1) // self.concurrency)))
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        results_path.parent.mkdir(parents=True, exist_ok=True)
        jobs = [(s, i) for s in (scenarios or self.scenarios()) for i in range(self.k)]
        sem = asyncio.Semaphore(self.concurrency)

        async def one(idx: int, s: EvalSimulationScenario, i: int) -> Conversation:
            async with sem:
                c = await self._rollout(s, i, self.base_port + idx)
                with results_path.open("a") as f:
                    f.write(json.dumps(dataclasses.asdict(c)) + "\n")
                return c

        return list(await asyncio.gather(*(one(idx, s, i) for idx, (s, i) in enumerate(jobs))))

    async def _rollout(self, s: EvalSimulationScenario, i: int, port: int) -> Conversation:
        sid = f"{s.name.replace('/', '__')}-{i:03d}-{uuid.uuid4().hex[:8]}"
        log = self.logs_dir / f"{sid}.bot.log"
        if not _port_free(self.host, port):
            return Conversation(s.name, i, sid, [], None, "", False, False, "error", [], [], 0, f"port {port} in use")
        bot = await self._spawn_bot(port, sid, log)
        try:
            judge = EvalJudge(self.judge.service())  # fresh judge => fresh cache per rollout
            params = EvalSessionParams(connect_timeout_s=BOT_CONNECT_TIMEOUT_S, stop_bot=True)
            session = EvalSession.from_scenario(s, f"ws://{self.host}:{port}", params=params,
                                                persona_llm=self.persona.service(), judge=judge)
            r = await asyncio.wait_for(session.run(), timeout=ROLLOUT_SAFETY_TIMEOUT_S)
        except Exception as e:  # harness-side failure, incl. TimeoutError
            r = EvalSimulationResult(simulation_name=s.name, succeeded=False, error=f"{type(e).__name__}: {e}")
        finally:
            await _stop_bot(bot)
        (self.logs_dir / f"{sid}.eval.log").write_text("\n".join(r.debug_log) + "\n")
        calls = [{"name": e.get("name"), "args": e.get("args")} for e in r.events_seen if e.get("type") == "function_call"]
        return Conversation(s.name, i, sid, r.messages, self.reward_fn(r), self.hint_fn(r), r.passed, r.succeeded,
                            r.ended_by, [dataclasses.asdict(m) for m in r.metrics], calls, r.duration_ms, r.error)

    async def _spawn_bot(self, port: int, sid: str, log: Path) -> asyncio.subprocess.Process:
        env = {**os.environ, **self.env, "POLYVOICE_SESSION": sid, "POLYVOICE_POLICY_URL": self.policy_url}
        f = log.open("wb")  # closed when the process exits (fd inherited)
        return await asyncio.create_subprocess_exec(
            self.bot_python, str(Path(self.bot_path).resolve()), "-t", "eval", "--host", self.host, "--port", str(port),
            stdout=f, stderr=asyncio.subprocess.STDOUT, env=env, cwd=str(Path(self.bot_path).resolve().parent))


def _port_free(host: str, port: int) -> bool:
    with socket.socket() as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, port))
            return True
        except OSError:
            return False


async def _stop_bot(proc: asyncio.subprocess.Process) -> None:  # mirrors EvalSuite._stop_bot
    if proc.returncode is not None:
        return
    for step in (None, proc.terminate, proc.kill):
        if step:
            step()
        with contextlib.suppress(asyncio.TimeoutError, TimeoutError):
            await asyncio.wait_for(proc.wait(), timeout=BOT_STOP_TIMEOUT_S)
            return


# Subprocess / other-venv mode: judge and persona come from YAML `factory:` blocks that read env vars.
#   judge: {eval: {factory: pipecat_trainer.evals.judge_llm}}
#   simulator: {factory: pipecat_trainer.evals.persona_llm}
def judge_llm(cfg: dict) -> OpenAILLMService:
    return LLMEndpoint(os.environ["POLYVOICE_JUDGE_BASE_URL"], os.environ.get("POLYVOICE_JUDGE_API_KEY", "x"),
                       cfg.get("model") or os.environ["POLYVOICE_JUDGE_MODEL"]).service()


def persona_llm(cfg: dict) -> OpenAILLMService:
    return LLMEndpoint(os.environ["POLYVOICE_PERSONA_BASE_URL"], os.environ.get("POLYVOICE_PERSONA_API_KEY", "x"),
                       cfg.get("model") or os.environ["POLYVOICE_PERSONA_MODEL"], temperature=0.7).service()
