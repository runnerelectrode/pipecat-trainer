"""Render docs/architecture.svg (stdlib only). Run: python scripts/arch_diagram.py"""
from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

W, H = 1180, 640
F = "font-family='Helvetica, Arial, sans-serif'"
INK, MUTE, LINE, LANE = "#1f2933", "#52606d", "#9aa5b1", "#8a94a6"
FILL = {"bot": "#e8f1fb", "capture": "#fff4d6", "proxy": "#fff4d6", "server": "#e9f7ef", "loop": "#f3eefc",
        "sandbox": "#fde8e8", "store": "#f5f7fa", "lane": "#f7f8fa"}
out: list[str] = []


def text(x, y, s, size=12, weight="400", fill=MUTE, anchor="start"):
    out.append(f"<text x='{x}' y='{y}' {F} font-size='{size}' font-weight='{weight}' fill='{fill}' text-anchor='{anchor}'>{escape(s)}</text>")


def box(x, y, w, h, title, lines=(), kind="store", dashed=False):
    d = " stroke-dasharray='6 4'" if dashed else ""
    out.append(f"<rect x='{x}' y='{y}' width='{w}' height='{h}' rx='10' fill='{FILL[kind]}' stroke='{LINE}' stroke-width='1.2'{d}/>")
    text(x + 12, y + 22, title, 14, "700", INK)
    for i, t in enumerate(lines):
        text(x + 12, y + 42 + 16 * i, t)


def route(points, label="", dash=False, head=True, lx=None, ly=None, anchor="middle"):
    d = " stroke-dasharray='5 4'" if dash else ""
    m = " marker-end='url(#a)'" if head else ""
    out.append(f"<polyline points='{' '.join(f'{x},{y}' for x, y in points)}' fill='none' stroke='{INK}' stroke-width='1.4'{d}{m}/>")
    if label:
        text(lx, ly, label, 11, anchor=anchor)


out.append(f"<svg xmlns='http://www.w3.org/2000/svg' width='{W}' height='{H}' viewBox='0 0 {W} {H}'>")
out.append("<defs><marker id='a' markerWidth='10' markerHeight='10' refX='9' refY='5' orient='auto'>"
           f"<path d='M0,0 L10,5 L0,10 z' fill='{INK}'/></marker></defs>")
out.append(f"<rect width='{W}' height='{H}' fill='white'/>")
text(24, 34, "pipecat-trainer: recursive self-improvement for Pipecat bots", 20, "700", INK)
text(24, 54, "daytime: the bot takes calls and every turn is captured · nighttime: rollouts, training, a gate, a promotion · the winner takes tomorrow's calls and trains the next one", 12)

# lanes
out.append(f"<rect x='24' y='72' width='548' height='236' rx='12' fill='{FILL['lane']}' stroke='{LINE}' stroke-width='1' stroke-dasharray='6 4'/>")
text(36, 90, "DAYTIME  ·  production calls", 11, "700", LANE)
out.append(f"<rect x='604' y='72' width='552' height='236' rx='12' fill='{FILL['lane']}' stroke='{LINE}' stroke-width='1' stroke-dasharray='6 4'/>")
text(616, 90, "NIGHTTIME  ·  rollouts and the gate", 11, "700", LANE)

# daytime
text(40, 132, "browser / phone", 12, "700", INK)
route([(40, 140), (40, 160), (60, 160)], head=True)
box(60, 104, 300, 110, "Pipecat bot  (yours, unchanged)", ["STT → Smart Turn → [ LLM slot ] → TTS", "tools, Flows, transport as before", "line 1: polyvoice_llm(session_id, base_url=proxy)", "line 2: PolyvoiceObserver(session_id, path)"], "bot")
box(392, 104, 164, 110, "traces/*.jsonl", ["one record per turn:", "context, reply, tool calls,", "interrupted, TTFB, usage,", "end reason"], "capture")
route([(360, 190), (392, 190)], "observer", lx=376, ly=182)
text(60, 240, "every LLM request carries  X-Session-Id  (the join key)  and  X-Turn-Type: main | speculative", 11)
route([(210, 214), (210, 340)], "chat completions", lx=222, ly=290, anchor="start")

# nighttime
box(616, 104, 250, 96, "scenarios/pool  (Pipecat YAML)", ["persona · goal · success · metrics", "pipecat-trainer rollout: K calls each,", "one bot process per call, -t eval"], "sandbox")
box(890, 104, 250, 96, "scenarios/holdout  (frozen)", ["the loop never trains on it", "gate: candidate vs incumbent,", "paired per scenario, bootstrap CI"], "sandbox")
text(616, 226, "caller + judge = any OpenAI-compatible endpoint with tool calling (gemma on General Compute, Coval, …)", 11)
text(616, 244, "the bot's replies come from the model being trained; the judge's reasons become the hints", 11)
route([(741, 200), (741, 340)], "reward + hint", lx=753, ly=290, anchor="start")
route([(1015, 200), (1015, 340)], "receipt", lx=1027, ly=290, anchor="start")

# proxy + loop row
box(60, 340, 300, 96, "polyloop proxy  (OpenAI-compatible URL)", ["records every turn token-exact by session", "serves the live adapter; /admin/serve overrides", "promotion = flip here, no bot restart"], "proxy")
box(604, 340, 552, 96, "polyloop cycle", ["filter (rollouts under the incumbent) → train → gate → propose → promote", "train: OPSD on captured sessions with judge hints (default),", "GRPO / SFT / any algorithm SkyRL supports, through rlcli"], "loop")
route([(604, 388), (360, 388)], "serve(policy): which adapter answers", lx=482, ly=380)

# server
box(60, 470, 1096, 78, "rlcli serve  ·  SkyRL Tinker server on your GPU node (Modal, Lambda, own box)", ["trainer: Megatron LoRA (PhoneLLM alpha 1 by default; any open model)      sampler: vLLM with multi-adapter slots      adapters are small, per deployment, versioned"], "server")
route([(210, 436), (210, 470)], "sample", lx=222, ly=458, anchor="start")
route([(880, 436), (880, 470)], "train step / save adapter", lx=892, ly=458, anchor="start")

text(24, 588, "Tiers: 0 Pipecat Evals text mode (free) · 1 audio mode (Kokoro, Moonshine) · 2 a simulation vendor as the gate · 3 production via observer or proxy", 11)
text(24, 606, "Recursive: each cycle starts from the last cycle's promoted adapter and must beat it on the same frozen holdout. Nothing in the bot changes; the adapter flips behind the proxy URL.", 11)
out.append("</svg>")
Path(__file__).resolve().parents[1].joinpath("docs").mkdir(exist_ok=True)
Path(__file__).resolve().parents[1].joinpath("docs", "architecture.svg").write_text("\n".join(out))
print("wrote docs/architecture.svg")
