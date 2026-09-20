"""Scenario portability. Pipecat's simulated-scenario YAML is the canonical format for a recipe
(persona, goal, success, metrics with natural-language criteria); the recipe's earlier JSON pool
(`input`, `expected[]`, `description`) converts into it, and Coval test cases are seeded from it.

    pipecat-trainer scenarios convert scenarios.json evals/pool --metrics evals/_metrics.yaml
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

# Voice-agent defaults that apply to any recipe; a recipe adds its own in evals/_metrics.yaml.
DEFAULT_METRICS = [
    {"name": "one_question", "criterion": "the reply asks for at most one piece of information", "min_score": 0.75},
    {"name": "spoken_style", "criterion": "the reply is one or two short sentences with no lists, markdown or emoji", "min_score": 0.75},
]


def _slug(s: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")
    return s[:48] or "scenario"


def json_to_pipecat(rec: dict, *, max_turns: int = 8, max_duration_s: float = 180.0, domain: str = "a business",
                    extra_metrics: list[dict] | None = None) -> dict:
    """One JSON scenario -> one Pipecat scenario-file mapping (name + scenarios[1])."""
    name = _slug(rec.get("description") or rec["input"][:40])
    expected = list(rec.get("expected", []))
    success = "; ".join(e.rstrip(".") for e in expected) if expected else "the bot handled the call competently"
    metrics = [{"name": f"expected_{i + 1}", "criterion": e, "min_score": 0.5} for i, e in enumerate(expected)]
    metrics += DEFAULT_METRICS + list(extra_metrics or [])
    persona = (
        f"A caller to {domain}. {rec['input']} Speak like a real caller on the phone: short turns, one "
        "thing at a time, answer only what is asked. Say goodbye and end the call once your goal is met or the "
        "agent clearly cannot help further."
    )
    return {
        "name": name,
        "max_turns": max_turns,
        "max_duration_s": max_duration_s,
        "scenarios": [{
            "name": "call",
            "persona": persona,
            "goal": rec["input"],
            "success": success,
            "metrics": metrics,
        }],
    }


def convert_dir(src_json: Path, out_dir: Path, *, metrics_file: Path | None = None, **kw) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    if metrics_file is not None:
        kw["extra_metrics"] = yaml.safe_load(Path(metrics_file).read_text()) or []
    written = []
    for rec in json.loads(Path(src_json).read_text()):
        doc = json_to_pipecat(rec, **kw)
        p = out_dir / f"{doc['name']}.yaml"
        p.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True, width=100))
        written.append(p)
    return written
