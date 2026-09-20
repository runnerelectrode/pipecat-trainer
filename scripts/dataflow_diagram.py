"""Render docs/dataflow.svg: the data path from a call to a promoted adapter, colour-coded by who owns each
step (Pipecat / pipecat-trainer / verifier / polyloop / rlcli). Run: python scripts/dataflow_diagram.py"""
from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

W, H = 1180, 700
F = "font-family='Helvetica, Arial, sans-serif'"
INK, MUTE, LINE, LANE = "#1f2933", "#52606d", "#9aa5b1", "#8a94a6"
COL = {"pipecat": "#e8f1fb", "trainer": "#fff4d6", "verifier": "#fde8e8", "polyloop": "#f3eefc", "rlcli": "#e9f7ef"}
out: list[str] = []


def text(x, y, s, size=12, weight="400", fill=MUTE, anchor="start"):
    out.append(f"<text x='{x}' y='{y}' {F} font-size='{size}' font-weight='{weight}' fill='{fill}' text-anchor='{anchor}'>{escape(s)}</text>")


def box(x, y, w, h, title, lines, who, tag=None):
    out.append(f"<rect x='{x}' y='{y}' width='{w}' height='{h}' rx='10' fill='{COL[who]}' stroke='{LINE}' stroke-width='1.2'/>")
    text(x + 12, y + 21, title, 13, "700", INK)
    for i, t in enumerate(lines):
        text(x + 12, y + 40 + 15 * i, t, 11)
    if tag:
        text(x + w - 10, y + 16, tag, 10, "700", LANE, anchor="end")


def arrow(pts, label="", lx=None, ly=None, anchor="middle", dash=False):
    d = " stroke-dasharray='5 4'" if dash else ""
    out.append(f"<polyline points='{' '.join(f'{x},{y}' for x, y in pts)}' fill='none' stroke='{INK}' stroke-width='1.4'{d} marker-end='url(#a)'/>")
    if label:
        text(lx, ly, label, 10, anchor=anchor)


out.append(f"<svg xmlns='http://www.w3.org/2000/svg' width='{W}' height='{H}' viewBox='0 0 {W} {H}'>")
out.append("<defs><marker id='a' markerWidth='10' markerHeight='10' refX='9' refY='5' orient='auto'>"
           f"<path d='M0,0 L10,5 L0,10 z' fill='{INK}'/></marker></defs>")
out.append(f"<rect width='{W}' height='{H}' fill='white'/>")
text(24, 34, "Data flow: from a call to a promoted adapter, and who owns each step", 20, "700", INK)
# legend
lx = 24
for who, label in (("pipecat", "Pipecat, as-is"), ("trainer", "pipecat-trainer"), ("verifier", "verifier (judged calls)"), ("polyloop", "polyloop (the loop)"), ("rlcli", "rlcli on SkyRL (train / serve)")):
    out.append(f"<rect x='{lx}' y='46' width='14' height='14' rx='3' fill='{COL[who]}' stroke='{LINE}'/>")
    text(lx + 20, 57, label, 11)
    lx += 24 + 7 * len(label) + 20

# row 1: the call and the two capture channels
box(24, 84, 250, 92, "1 · The call", ["Gradium STT → Smart Turn → LLM → Gradium TTS", "WebRTC / Daily / Pipecat Cloud", "your pipeline, unchanged"], "pipecat", "PIPECAT")
box(314, 84, 270, 92, "2a · LLM slot  (TITO capture)", ["OpenAILLMService subclass", "X-Session-Id · X-Turn-Type · whole turns", "→ every request goes through the proxy"], "trainer", "PIPECAT-TRAINER")
box(624, 84, 250, 92, "3 · polyloop proxy  (TITO)", ["token-in / token-out: renders the messages,", "samples from rlcli, records prompt tokens +", "reply per turn, keyed by session"], "polyloop", "POLYLOOP")
box(914, 84, 242, 92, "4 · rlcli serve", ["SkyRL Tinker server", "PhoneLLM + LoRA slots", "sampler answers the call"], "rlcli", "RLCLI")
arrow([(274, 130), (314, 130)])
arrow([(584, 130), (624, 130)], "chat completions", 604, 120)
arrow([(874, 130), (914, 130)], "sample", 894, 120)

box(314, 200, 270, 78, "2b · Observer  (text capture)", ["BaseObserver subclass, inside the bot", "context · reply · tool results · interrupted ·", "TTFB · end reason  →  traces/*.jsonl"], "trainer", "PIPECAT-TRAINER")
arrow([(150, 176), (150, 239), (314, 239)], "every frame", 232, 232)
text(24, 300, "Two capture channels, same session id. The proxy's file is token-exact and is what training uses;", 11)
text(24, 316, "the observer's file is text-level, carries interruptions/tools/latency, and works for bots not behind the proxy.", 11)

# row 2: trace import → sandbox/verifier → ledger
box(24, 350, 250, 92, "5 · Trace import", ["polyloop reads the proxy traces:", "sessions grouped by X-Session-Id,", "held-out sessions excluded, speculative dropped"], "polyloop", "POLYLOOP")
box(314, 350, 270, 92, "6 · Sandbox + evals  = the verifier", ["Pipecat Evals: -t eval, scenario YAML, persona,", "judge; one bot per call (pipecat-trainer rollout)", "or Coval: persona + metrics via API"], "verifier", "PIPECAT EVALS / COVAL")
box(624, 350, 250, 92, "7 · Ledger", ["one judged conversation per row:", "session · task · reward · judge's explanation", "(polyvoice core; vendor-agnostic)"], "polyloop", "POLYVOICE")
box(914, 350, 242, 92, "8 · Rows", ["prompt (token-exact, from 3)", "+ hint (explanation, from 7)", "joined by session id"], "polyloop", "POLYLOOP")
arrow([(150, 176), (150, 350)], "", dash=True)
arrow([(274, 396), (314, 396)], "scenarios × K", 294, 386)
arrow([(584, 396), (624, 396)], "reward + reasons", 604, 386)
arrow([(874, 396), (914, 396)])
arrow([(749, 176), (749, 350)], "the bot's replies in every simulated call come from the model being trained", 760, 270, anchor="start")

# row 3: train → gate → promote
box(24, 500, 330, 100, "9 · OPSD trainer", ["student = PhoneLLM + LoRA samples fresh replies to the prompt", "teacher = same weights with the hint in context scores them", "loss closes the gap, KL penalty; GRPO / SFT also available"], "rlcli", "RLCLI ON SKYRL")
box(394, 500, 330, 100, "10 · Gate  (the verifier again)", ["frozen holdout scenarios, candidate vs incumbent,", "K repeats, paired per scenario, bootstrap CI,", "regression cap → receipt.json → promote or reject"], "verifier", "PIPECAT EVALS / COVAL")
box(764, 500, 392, 100, "11 · Promote", ["polyloop approve → live.json → proxy serves the new adapter", "the bot changes nothing; tomorrow's calls are captured under it", "and train the next candidate (recursive)"], "polyloop", "POLYLOOP")
arrow([(1035, 442), (1035, 470), (189, 470), (189, 500)], "rows", 620, 462)
arrow([(354, 550), (394, 550)], "candidate adapter", 374, 540)
arrow([(724, 550), (764, 550)], "receipt: pass", 744, 540)
arrow([(960, 600), (960, 640), (1140, 640), (1140, 130), (1156, 130)], "", dash=True)
text(1000, 656, "promoted adapter answers the next call →", 10)

text(24, 640, "Where Pipecat is used: 1 (the bot), 2a (its LLM service class), 2b (its observer API), 6 and 10 (its eval harness as the sandbox and, optionally, the gate).", 11)
text(24, 658, "Where the verifier sits: 6 (rollouts for training signal) and 10 (the gate). Coval and Pipecat Evals are interchangeable behind the same loop file.", 11)
text(24, 676, "Where TITO is: 3, the proxy; that file is the only source of training tokens. Where OPSD is: 9, rlcli's trainer on the SkyRL server.", 11)
out.append("</svg>")
Path(__file__).resolve().parents[1].joinpath("docs", "dataflow.svg").write_text("\n".join(out))
print("wrote docs/dataflow.svg")
