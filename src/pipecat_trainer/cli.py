"""`pipecat-trainer`: scenario conversion and sandbox rollouts from the command line."""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import click


@click.group()
def main():
    """Make any Pipecat bot trainable."""


@main.group()
def scenarios():
    """Scenario files (Pipecat simulated-scenario YAML is the canonical format)."""


@scenarios.command("convert")
@click.argument("src_json", type=click.Path(exists=True))
@click.argument("out_dir", type=click.Path())
@click.option("--metrics", type=click.Path(exists=True), help="YAML list of extra judge criteria added to every scenario.")
@click.option("--domain", default="a business", help="Who the caller is calling, for the persona text.")
@click.option("--max-turns", default=8, show_default=True)
def scenarios_convert(src_json, out_dir, metrics, domain, max_turns):
    """JSON list of {input, expected[], description} -> one Pipecat scenario YAML per record."""
    from pipecat_trainer.scenarios import convert_dir

    written = convert_dir(Path(src_json), Path(out_dir), metrics_file=Path(metrics) if metrics else None,
                          domain=domain, max_turns=max_turns)
    click.echo(f"{len(written)} scenario files in {out_dir}")


@main.command("rollout")
@click.option("--scenarios", "scenarios_dir", required=True, type=click.Path(exists=True), help="Directory of scenario YAML.")
@click.option("--bot", "bot_path", required=True, type=click.Path(exists=True), help="The bot (must support -t eval).")
@click.option("--policy-url", required=True, help="OpenAI-compatible base URL the bot's LLM answers from (the proxy).")
@click.option("--judge-url", required=True, help="OpenAI-compatible base URL for the judge (and the caller unless --persona-url).")
@click.option("--judge-model", required=True)
@click.option("--judge-key-env", default="JUDGE_API_KEY", show_default=True)
@click.option("--persona-url", default=None)
@click.option("--persona-model", default=None)
@click.option("--persona-key-env", default=None)
@click.option("-k", default=1, show_default=True, help="Rollouts per scenario.")
@click.option("--concurrency", default=4, show_default=True)
@click.option("--base-port", default=7900, show_default=True)
@click.option("--out", default="rollouts", show_default=True, help="Where results.jsonl and logs go.")
@click.option("--bot-python", default=sys.executable, help="Interpreter for the bot (its own venv is fine).")
@click.option("--env", "env_pairs", multiple=True, help="KEY=VALUE passed to every bot process.")
def rollout(scenarios_dir, bot_path, policy_url, judge_url, judge_model, judge_key_env, persona_url, persona_model,
            persona_key_env, k, concurrency, base_port, out, bot_python, env_pairs):
    """Run K simulated calls per scenario against the bot, judged; one bot process per call."""
    from pipecat_trainer.evals import LLMEndpoint, PipecatEvalsVerifier

    judge = LLMEndpoint(judge_url, os.environ.get(judge_key_env, "x"), judge_model, temperature=0.0)
    persona = LLMEndpoint(persona_url or judge_url, os.environ.get(persona_key_env or judge_key_env, "x"),
                          persona_model or judge_model, temperature=0.7)
    env = dict(p.split("=", 1) for p in env_pairs)
    env.setdefault("POLYVOICE_PROXY_URL", policy_url)
    out_dir = Path(out)
    v = PipecatEvalsVerifier(scenarios_dir=Path(scenarios_dir), bot_path=Path(bot_path), policy_url=policy_url,
                             judge=judge, persona=persona, k=k, concurrency=concurrency, base_port=base_port,
                             bot_python=bot_python, logs_dir=out_dir / "logs", env=env)
    convs = asyncio.run(v.run(out_dir / "results.jsonl"))
    scored = [c for c in convs if c.reward is not None]
    click.echo(f"{len(convs)} rollouts, {len(scored)} scored, mean reward "
               f"{(sum(c.reward for c in scored) / len(scored)) if scored else float('nan'):.3f}; results in {out_dir}/results.jsonl")
    for c in convs:
        click.echo(f"  {c.scenario:40s} rollout {c.rollout} reward={c.reward} ended_by={c.ended_by}"
                   + (f" error={c.error}" if c.error else ""))


@main.command("traces")
@click.argument("traces_dir", type=click.Path(exists=True))
def traces(traces_dir):
    """Summarise observer traces: one line per turn."""
    for p in sorted(Path(traces_dir).glob("*.jsonl")):
        for line in p.read_text().splitlines():
            r = json.loads(line)
            if r.get("type") == "turn":
                user = [m for m in r["messages"] if m.get("role") == "user"]
                click.echo(f"{p.stem[:8]} t{r['turn']:<3} ttfb={r.get('llm_ttfb_s') or '-':<6} "
                           f"{'INT ' if r.get('was_interrupted') else '    '}"
                           f"user: {(user[-1]['content'] if user else '')[:50]!r:54s} bot: {r.get('reply', '')[:60]!r}")
            else:
                click.echo(f"{p.stem[:8]} end  reason={r.get('end_reason')}")


@main.command("dashboard")
@click.option("--traces", "traces_dir", default="traces", show_default=True, type=click.Path())
@click.option("--results", default=None, type=click.Path(), help="A rollout results.jsonl to show alongside.")
@click.option("--runs", "runs_dir", default=None, type=click.Path(), help="A polyloop loop directory (<runs>/<loop>): cycles, ledger.")
@click.option("--names", "names_file", default=None, type=click.Path(), help="JSON {task id: scenario name} for readable rows.")
@click.option("--node-log", "node_log", default=None, type=click.Path(), help="The GPU run's log file (shown as a panel).")
@click.option("--port", default=7870, show_default=True)
def dashboard(traces_dir, results, runs_dir, names_file, node_log, port):
    """A local page with the whole flow: live calls, sandbox rollouts, the loop's cycle, the judged ledger."""
    from pipecat_trainer.dashboard import serve

    serve(Path(traces_dir), Path(results) if results else None, port, Path(runs_dir) if runs_dir else None,
          Path(names_file) if names_file else None, Path(node_log) if node_log else None)
