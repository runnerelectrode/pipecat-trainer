"""Render docs/pipecat-parts.svg: which parts of Pipecat are used as-is, and what pipecat-trainer adds on each.
Run: python scripts/parts_diagram.py"""
from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

W, H = 1180, 600
F = "font-family='Helvetica, Arial, sans-serif'"
INK, MUTE, LINE, LANE = "#1f2933", "#52606d", "#9aa5b1", "#8a94a6"
PIPECAT, OURS = "#e8f1fb", "#fff4d6"
out: list[str] = []


def text(x, y, s, size=12, weight="400", fill=MUTE, anchor="start"):
    out.append(f"<text x='{x}' y='{y}' {F} font-size='{size}' font-weight='{weight}' fill='{fill}' text-anchor='{anchor}'>{escape(s)}</text>")


def box(x, y, w, h, title, lines, fill):
    out.append(f"<rect x='{x}' y='{y}' width='{w}' height='{h}' rx='10' fill='{fill}' stroke='{LINE}' stroke-width='1.2'/>")
    text(x + 12, y + 22, title, 13, "700", INK)
    for i, t in enumerate(lines):
        text(x + 12, y + 41 + 15 * i, t, 11)


def arrow(x1, y, x2, label=""):
    out.append(f"<line x1='{x1}' y1='{y}' x2='{x2}' y2='{y}' stroke='{INK}' stroke-width='1.4' marker-end='url(#a)'/>")
    if label:
        text((x1 + x2) / 2, y - 6, label, 10, anchor="middle")


out.append(f"<svg xmlns='http://www.w3.org/2000/svg' width='{W}' height='{H}' viewBox='0 0 {W} {H}'>")
out.append("<defs><marker id='a' markerWidth='10' markerHeight='10' refX='9' refY='5' orient='auto'>"
           f"<path d='M0,0 L10,5 L0,10 z' fill='{INK}'/></marker></defs>")
out.append(f"<rect width='{W}' height='{H}' fill='white'/>")
text(24, 34, "Which parts of Pipecat are used, and what pipecat-trainer adds on each", 20, "700", INK)
text(24, 54, "blue = Pipecat, used as-is (nothing forked, nothing patched) · yellow = pipecat-trainer, the part that learns", 12)

# column headers
out.append(f"<rect x='24' y='72' width='500' height='500' rx='12' fill='#f7f8fa' stroke='{LINE}' stroke-width='1' stroke-dasharray='6 4'/>")
text(36, 90, "PIPECAT  ·  used as-is", 11, "700", LANE)
out.append(f"<rect x='656' y='72' width='500' height='500' rx='12' fill='#f7f8fa' stroke='{LINE}' stroke-width='1' stroke-dasharray='6 4'/>")
text(668, 90, "PIPECAT-TRAINER  ·  what is added", 11, "700", LANE)

rows = [
    ("The bot pipeline", ["Gradium STT · Smart Turn · Silero VAD · Gradium TTS", "SmallWebRTC / Daily / Pipecat Cloud transports", "context aggregators · PipelineWorker · Flows · tools"],
     "unchanged", "Your bot stays your bot", ["two lines change: the LLM constructor and one observer", "everything upstream and downstream of the LLM is untouched"]),
    ("OpenAILLMService", ["the base class every hosted LLM vendor subclasses", "takes base_url + headers + settings"],
     "subclassed", "The LLM slot that learns", ["X-Session-Id + X-Turn-Type on every request (speculative turns never train)", "whole turns so the proxy records token-exact; answers from the promoted adapter", "promotion flips behind the URL: the bot never restarts"]),
    ("BaseObserver", ["watch every frame without touching the pipeline", "(Roark uses it for call analytics)"],
     "implemented", "The observer that writes training rows", ["exact context the model saw · reply · tool calls + results", "interrupted · TTFB · usage · end reason, keyed by session"]),
    ("Pipecat Evals", ["-t eval transport · scenario YAML (persona, goal, metrics)", "simulated caller · judge · one bot process per run", "built for pass/fail regression tests before a deploy"],
     "driven", "The rollout engine", ["K judged calls per scenario; the score is the reward", "the judge's per-turn reasons are the hindsight hint for OPSD", "scenario YAML is the single source: sandbox, Coval seed, frozen holdout"]),
    ("runner_args.session_id", ["per-call id on every transport; Pipecat Cloud sets it"],
     "reused", "The join key", ["header → observer record → proxy trace → judged conversation → hint"]),
]
y = 104
heights = [72, 84, 68, 84, 54]
for (lt, ll, verb, rt, rl), h in zip(rows, heights):
    box(36, y, 476, h, lt, ll, PIPECAT)
    box(668, y, 476, h, rt, rl, OURS)
    arrow(512, y + h / 2, 668, verb)
    y += h + 12

text(24, 590, "Not from Pipecat: the loop (polyloop: filter → train → gate → promote, receipts, recursion) and training/serving (rlcli on SkyRL: OPSD default; GRPO, SFT).", 11)
out.append("</svg>")
Path(__file__).resolve().parents[1].joinpath("docs", "pipecat-parts.svg").write_text("\n".join(out))
print("wrote docs/pipecat-parts.svg")
