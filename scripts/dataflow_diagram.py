"""Render docs/dataflow.svg: six categories of the stack in a loop, with what Pipecat provides at each.
Run: python scripts/dataflow_diagram.py"""
from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

W, H = 1180, 470
F = "font-family='Helvetica, Arial, sans-serif'"
INK, MUTE, LINE, LANE = "#1f2933", "#52606d", "#9aa5b1", "#8a94a6"
PIPECAT, OURS, INFRA = "#e8f1fb", "#fff4d6", "#f3eefc"
out: list[str] = []


def text(x, y, s, size=12, weight="400", fill=MUTE, anchor="start"):
    out.append(f"<text x='{x}' y='{y}' {F} font-size='{size}' font-weight='{weight}' fill='{fill}' text-anchor='{anchor}'>{escape(s)}</text>")


def box(x, y, w, h, title, what, pipecat, fill):
    out.append(f"<rect x='{x}' y='{y}' width='{w}' height='{h}' rx='12' fill='{fill}' stroke='{LINE}' stroke-width='1.2'/>")
    text(x + 14, y + 26, title, 15, "700", INK)
    for i, t in enumerate(what):
        text(x + 14, y + 48 + 16 * i, t, 12)
    if pipecat:
        out.append(f"<rect x='{x + 14}' y='{y + h - 34}' width='{w - 28}' height='22' rx='6' fill='white' stroke='{LINE}' stroke-width='0.8'/>")
        text(x + 22, y + h - 19, "Pipecat: " + pipecat, 11, "700", "#1c4e80")


def arrow(pts, label="", lx=None, ly=None, anchor="middle"):
    out.append(f"<polyline points='{' '.join(f'{x},{y}' for x, y in pts)}' fill='none' stroke='{INK}' stroke-width='1.6' marker-end='url(#a)'/>")
    if label:
        text(lx, ly, label, 11, anchor=anchor)


out.append(f"<svg xmlns='http://www.w3.org/2000/svg' width='{W}' height='{H}' viewBox='0 0 {W} {H}'>")
out.append("<defs><marker id='a' markerWidth='10' markerHeight='10' refX='9' refY='5' orient='auto'>"
           f"<path d='M0,0 L10,5 L0,10 z' fill='{INK}'/></marker></defs>")
out.append(f"<rect width='{W}' height='{H}' fill='white'/>")
text(24, 34, "The loop, by category of the stack, and what Pipecat provides at each step", 20, "700", INK)
for i, (c, l) in enumerate(((PIPECAT, "the voice agent (Pipecat)"), (OURS, "this package"), (INFRA, "the training stack (any TITO proxy, trainer, gate)"))):
    out.append(f"<rect x='{24 + 300 * i}' y='46' width='14' height='14' rx='3' fill='{c}' stroke='{LINE}'/>")
    text(44 + 300 * i, 57, l, 11)

BW, BH = 350, 118
# top row: agent → capture → sandbox+evals
box(24, 84, BW, BH, "1 · Voice agent", ["takes real calls: STT → turn detection → LLM → TTS", "the LLM is an OpenAI-compatible endpoint"], "the whole bot, unchanged", PIPECAT)
box(414, 84, BW, BH, "2 · Capture", ["TITO proxy records every turn token-exact by session;", "an observer records context, tools, interruptions, latency"], "LLM service class + observer API", OURS)
box(804, 84, BW, BH, "3 · Sandbox + evals  (the verifier)", ["scenarios run against the bot; a simulated caller talks,", "a judge scores each call and explains what went wrong"], "the eval harness (or a vendor like Coval)", OURS)
arrow([(374, 143), (414, 143)], "every LLM request", 394, 133)
arrow([(764, 143), (804, 143)], "session ids", 784, 133)

# bottom row: train ← gate ← promote (right to left), drawn left→right as promote ← gate ← train
box(804, 270, BW, BH, "4 · Trainer", ["rows = recorded turn + judge's explanation as the hint", "on-policy self-distillation by default; GRPO, SFT also"], None, INFRA)
box(414, 270, BW, BH, "5 · Gate", ["candidate vs incumbent on a frozen holdout set,", "paired, confidence interval, regression cap → receipt"], "the same eval harness replays the holdout", OURS)
box(24, 270, BW, BH, "6 · Promote", ["a passing receipt flips the adapter behind the endpoint;", "the bot restarts nothing and takes the next call"], "same endpoint, same bot", INFRA)
arrow([(979, 202), (979, 270)], "reward + reasons, per call", 990, 240, anchor="start")
arrow([(804, 329), (764, 329)], "candidate", 784, 319)
arrow([(414, 329), (374, 329)], "receipt", 394, 319)
arrow([(199, 270), (199, 202)], "next day's calls run on the winner (recursive)", 210, 240, anchor="start")

text(24, 424, "Simplest reading: Pipecat is the agent and the eval harness. This package is the two lines that capture calls and the driver that turns evals into", 11)
text(24, 440, "rollouts. Everything below the line is a standard training stack: a token-in/token-out proxy, a trainer that supports OPSD/GRPO/SFT, and a gate.", 11)
out.append("</svg>")
Path(__file__).resolve().parents[1].joinpath("docs", "dataflow.svg").write_text("\n".join(out))
print("wrote docs/dataflow.svg")
