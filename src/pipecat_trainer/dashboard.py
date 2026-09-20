"""`pipecat-trainer dashboard`: one local page that shows the whole flow.

1. Live calls: one row per LLM turn from a traces directory (the observer's records).
2. Sandbox rollouts: judged simulations from a results.jsonl (reward, verdict, the judge's reasons).
3. The loop: the latest cycle in a polyloop runs directory: stage timeline, filter results, training
   metrics, the gate receipt and the promotion state.
4. Simulation environment: every judged conversation in the loop's ledger (Coval or the sandbox), with the
   judge's explanation, joined to the training hints.

Refreshes every 2 s. Stdlib only.
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>pipecat-trainer</title>
<style>
:root{--ink:#1f2933;--mute:#52606d;--line:#d9dee3;--ok:#1b7f4b;--bad:#b42318;--warn:#b7791f;--acc:#3b5bdb}
body{font:14px/1.45 -apple-system,Helvetica,Arial,sans-serif;color:var(--ink);margin:0;background:#fff}
header{padding:12px 20px;border-bottom:1px solid var(--line);display:flex;gap:24px;align-items:baseline;position:sticky;top:0;background:#fff;z-index:2}
h1{font-size:18px;margin:0}small{color:var(--mute)}
nav a{margin-right:14px;color:var(--acc);text-decoration:none;font-size:13px}
main{display:grid;grid-template-columns:1fr 1fr;gap:0}
section{padding:14px 20px;overflow:auto;border-bottom:1px solid var(--line)}section:nth-child(odd){border-right:1px solid var(--line)}
section.wide{grid-column:1/-1;border-right:0}
h2{font-size:13px;letter-spacing:.04em;text-transform:uppercase;color:var(--mute);margin:0 0 10px}
table{border-collapse:collapse;width:100%}td,th{padding:5px 8px;border-bottom:1px solid var(--line);vertical-align:top;text-align:left}
th{font-weight:600;color:var(--mute);font-size:12px}
.mono{font-family:ui-monospace,Menlo,monospace;font-size:12px}
.pill{display:inline-block;padding:1px 8px;border-radius:10px;font-size:12px;font-weight:600;white-space:nowrap}
.ok{background:#e6f4ea;color:var(--ok)}.bad{background:#fdecea;color:var(--bad)}.warn{background:#fff4d6;color:var(--warn)}.info{background:#e8edfb;color:var(--acc)}
.num{font-variant-numeric:tabular-nums;text-align:right}.dim{color:var(--mute)}
.stages{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:12px}
.stage{border:1px solid var(--line);border-radius:8px;padding:6px 10px;min-width:110px}
.stage b{display:block;font-size:12px}.stage span{font-size:12px;color:var(--mute)}
.stage.done{border-color:var(--ok)}.stage.running{border-color:var(--acc);background:#f4f6fe}.stage.failed{border-color:var(--bad)}
.receipt{display:grid;grid-template-columns:repeat(6,auto);gap:8px 22px;padding:10px 12px;border:1px solid var(--line);border-radius:10px;margin-bottom:12px;width:max-content}
.receipt div b{display:block;font-size:20px;font-variant-numeric:tabular-nums}.receipt div{font-size:12px;color:var(--mute)}
.bar{display:inline-block;height:8px;background:var(--acc);border-radius:4px;vertical-align:middle}
</style></head><body>
<header><h1>pipecat-trainer</h1><small id="meta">loading…</small><nav><a href="#calls">calls</a><a href="#rollouts">rollouts</a><a href="#loop">loop</a><a href="#node">node log</a><a href="#sims">simulation environment</a></nav></header>
<main>
<section id="calls"><h2>1 · Live calls · one row per LLM turn</h2><table><thead><tr><th>session</th><th>turn</th><th>caller said</th><th>bot said</th><th class="num">TTFB</th><th></th></tr></thead><tbody id="traces"></tbody></table></section>
<section id="rollouts"><h2>2 · Sandbox rollouts · Pipecat Evals, scenarios × K</h2><div id="summary" class="dim"></div><table><thead><tr><th>scenario</th><th>#</th><th class="num">reward</th><th>verdict</th><th>judge's reasons (the hint)</th></tr></thead><tbody id="rollouts"></tbody></table></section>
<section id="loop" class="wide"><h2>3 · The loop · <span id="cycle" class="mono"></span></h2><div class="stages" id="stages"></div><div id="receipt"></div>
<div style="display:grid;grid-template-columns:1fr 1fr;gap:24px">
<div><h2>filter · rollouts under the incumbent</h2><table><thead><tr><th>task</th><th>rewards</th><th class="num">mean</th></tr></thead><tbody id="filter"></tbody></table></div>
<div><h2>train · OPSD steps</h2><table><thead><tr><th>step</th><th class="num">reward</th><th class="num">teacher KL</th><th class="num">episodes</th><th>KL</th></tr></thead><tbody id="train"></tbody></table>
<h2 style="margin-top:14px">gate · paired on the frozen holdout</h2><table><thead><tr><th>scenario</th><th class="num">incumbent</th><th class="num">candidate</th><th class="num">Δ</th></tr></thead><tbody id="paired"></tbody></table></div>
</div></section>
<section id="sims" class="wide"><h2>4 · Simulation environment · every judged conversation in the ledger</h2><div id="simsum" class="dim"></div><table><thead><tr><th>when</th><th>stage</th><th>policy</th><th>scenario</th><th class="num">reward</th><th>judge's explanation (→ training hint)</th><th>session</th></tr></thead><tbody id="ledger"></tbody></table></section>
<section id="node" class="wide"><h2>5 · Node log · the GPU run (Modal)</h2><pre id="nodelog" class="mono" style="max-height:260px;overflow:auto;background:#f5f7fa;padding:10px;border-radius:8px;margin:0"></pre></section>
</main>
<script>
const esc=s=>String(s??'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
const f2=x=>x==null?'—':Number(x).toFixed(2);
async function tick(){
  const d=await (await fetch('/data')).json();
  document.getElementById('meta').textContent=`${d.turns.length} turns in ${d.sessions} sessions · ${d.rollouts.length} rollouts · ${d.loop?('cycle '+d.loop.cycle+' · '+d.loop.status):'no cycle yet'} · ${d.ledger.length} judged sims`;
  document.getElementById('traces').innerHTML=d.turns.slice(-60).reverse().map(t=>`<tr><td class="mono">${esc(t.session.slice(0,8))}</td><td class="num">${t.turn}</td><td>${esc(t.user)}</td><td>${esc(t.reply)}</td><td class="num">${t.ttfb==null?'—':t.ttfb.toFixed(2)+' s'}</td><td>${t.interrupted?'<span class="pill warn">interrupted</span>':''}${t.speculation?'<span class="pill warn">spec</span>':''}</td></tr>`).join('');
  const r=d.rollouts;const scored=r.filter(x=>x.reward!=null);
  document.getElementById('summary').textContent=scored.length?`mean reward ${(scored.reduce((a,x)=>a+x.reward,0)/scored.length).toFixed(3)} over ${scored.length} scored`:'';
  document.getElementById('rollouts').innerHTML=r.slice().reverse().map(x=>`<tr><td>${esc(x.scenario)}</td><td class="num">${x.rollout}</td><td class="num">${f2(x.reward)}</td><td>${x.error?'<span class="pill bad">error</span>':x.passed?'<span class="pill ok">pass</span>':'<span class="pill bad">fail</span>'}</td><td class="dim">${esc(x.error||x.hint)}</td></tr>`).join('');
  const L=d.loop;
  if(L){
    document.getElementById('cycle').textContent=L.loop+' / '+L.cycle+' · model '+(L.model||'');
    document.getElementById('stages').innerHTML=L.stages.map(s=>`<div class="stage ${s.state}"><b>${esc(s.name)}</b><span>${s.seconds!=null?s.seconds.toFixed(0)+' s':''} ${esc(s.note||'')}</span></div>`).join('');
    const p=L.paired;
    document.getElementById('receipt').innerHTML=p?`<div class="receipt"><div>incumbent<b>${f2(p.incumbent_mean)}</b></div><div>candidate<b>${f2(p.candidate_mean)}</b></div><div>paired Δ<b>${p.mean_delta>=0?'+':''}${f2(p.mean_delta)}</b></div><div>95% CI<b class="mono" style="font-size:15px">[${f2(p.delta_ci95[0])}, ${f2(p.delta_ci95[1])}]</b></div><div>wins / losses / ties<b>${p.wins} / ${p.losses} / ${p.ties}</b></div><div>decision<b><span class="pill ${L.decision==='promote'?'ok':L.decision?'bad':'info'}">${esc(L.decision||'pending')}</span></b><span>${esc(L.promote||'')}</span></div></div>`:'<div class="dim">gate not run yet</div>';
    document.getElementById('filter').innerHTML=L.filter.map(x=>`<tr><td>${esc(x.name)}</td><td class="mono">${x.rewards.map(f2).join(' ')}</td><td class="num">${f2(x.mean)} <span class="bar" style="width:${Math.round((x.mean||0)*60)}px"></span></td></tr>`).join('');
    document.getElementById('train').innerHTML=L.train.map(x=>`<tr><td class="num">${x.step}</td><td class="num">${f2(x.reward)}</td><td class="num">${x.kl==null?'—':x.kl.toFixed(4)}</td><td class="num">${x.episodes??''}</td><td><span class="bar" style="width:${Math.min(120,Math.round((x.kl||0)*400))}px"></span></td></tr>`).join('');
    document.getElementById('paired').innerHTML=L.holdout.map(x=>`<tr><td>${esc(x.name)}</td><td class="num">${f2(x.incumbent)}</td><td class="num">${f2(x.candidate)}</td><td class="num">${x.candidate==null||x.incumbent==null?'—':((x.candidate-x.incumbent)>=0?'+':'')+f2(x.candidate-x.incumbent)}</td></tr>`).join('');
  }
  document.getElementById('nodelog').textContent=d.node_log||'(no node log)';
  const led=d.ledger;const ls=led.filter(x=>x.reward!=null);
  document.getElementById('simsum').textContent=led.length?`${ls.length} scored · ${ls.filter(x=>x.reward<1).length} below 1.0 (these become hints) · ${new Set(led.map(x=>x.label)).size} stages`:'';
  document.getElementById('ledger').innerHTML=led.slice().reverse().slice(0,200).map(x=>`<tr><td class="mono dim">${esc((x.ts||'').slice(11,19))}</td><td><span class="pill ${x.label.startsWith('evaluate')?'info':'warn'}">${esc(x.label)}</span></td><td class="mono">${esc(x.policy)}</td><td>${esc(x.task_name)}</td><td class="num">${f2(x.reward)}</td><td class="dim">${esc(x.explanation||'')}</td><td class="mono dim">${esc((x.session||'').slice(0,10))}</td></tr>`).join('');
}
tick();setInterval(tick,2000);
</script></body></html>"""


def _jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out


def _turns(traces_dir: Path) -> tuple[list[dict], int]:
    turns, sessions = [], 0
    if not traces_dir.exists():
        return turns, sessions
    for p in sorted(traces_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime):
        sessions += 1
        for r in _jsonl(p):
            if r.get("type") != "turn":
                continue
            users = [m for m in r.get("messages", []) if m.get("role") == "user"]
            turns.append({"session": r["session_id"], "turn": r["turn"], "user": users[-1]["content"] if users else "",
                          "reply": r.get("reply", ""), "ttfb": r.get("llm_ttfb_s"), "interrupted": r.get("was_interrupted"),
                          "speculation": r.get("speculation")})
    return turns, sessions


def _rollouts(results: Path | None) -> list[dict]:
    if not results:
        return []
    return [{k: r.get(k) for k in ("scenario", "rollout", "reward", "passed", "hint", "error", "ended_by")} for r in _jsonl(results)]


STAGE_ORDER = ["snapshot", "preflight", "filter", "train", "evaluate", "gate", "promote"]


def _loop(runs_dir: Path | None, names: dict[str, str]) -> dict | None:
    """The latest cycle under <runs_dir>/cycles/<id>/ (a polyloop loop directory)."""
    if not runs_dir or not (runs_dir / "cycles").exists():
        return None
    cycles = sorted((runs_dir / "cycles").glob("*"))
    if not cycles:
        return None
    c = cycles[-1]
    events = _jsonl(c / "events.jsonl")
    state = json.loads((c / "state.json").read_text()) if (c / "state.json").exists() else {}
    receipt = json.loads((c / "receipt.json").read_text()) if (c / "receipt.json").exists() else {}
    stage_state, secs = {}, {}
    for e in events:  # in order: the latest event per stage wins, so a resumed stage shows as done
        st, kind = e.get("stage"), e.get("kind")
        if kind == "stage.start":
            stage_state[st] = "running"
        elif kind == "stage.done":
            stage_state[st] = "done"
        elif kind == "stage.failed":
            stage_state[st] = "failed"
        elif kind == "stage.seconds":
            secs[st] = e.get("seconds")
    summary = state.get("stage_summary") or {}
    stages = []
    for name in STAGE_ORDER:
        state_ = stage_state.get(name, "pending")
        note = ""
        s = summary.get(name) or {}
        if name == "filter" and s:
            note = f"{s.get('measured', '')} measured, {s.get('contested', '')} contested"
        elif name == "train" and s:
            note = f"→ {s.get('candidate', '')}"
        elif name == "gate" and s:
            note = s.get("decision", "")
        elif name == "promote" and s:
            note = s.get("action", "")
        stages.append({"name": name, "state": state_, "seconds": secs.get(name), "note": note})
    filt = []
    for r in _jsonl(c / "filter" / "results.jsonl"):
        rw = r.get("rewards") or []
        filt.append({"name": names.get(r.get("task"), r.get("task")), "rewards": rw, "mean": r.get("mean")})
    train = []
    for m in sorted((c / "train").glob("*/metrics.jsonl")) if (c / "train").exists() else []:
        for r in _jsonl(m):
            train.append({"step": r.get("step"), "reward": r.get("env/all/reward/total"), "kl": r.get("teacher_kl"),
                          "episodes": r.get("env/all/total_episodes")})
    inc, cand = state.get("eval_incumbent") or {}, state.get("eval_candidate") or {}
    holdout = [{"name": names.get(t, t), "incumbent": inc.get(t), "candidate": cand.get(t)} for t in sorted(set(inc) | set(cand))]
    return {"loop": state.get("loop") or receipt.get("loop"), "cycle": c.name, "status": state.get("status"),
            "model": state.get("model") or receipt.get("model"), "stages": stages,
            "paired": receipt.get("paired") or state.get("paired"), "decision": receipt.get("decision") or state.get("decision"),
            "promote": (summary.get("promote") or {}).get("hint"), "filter": filt, "train": train, "holdout": holdout}


def _ledger(runs_dir: Path | None, names: dict[str, str]) -> list[dict]:
    if not runs_dir:
        return []
    out = []
    for p in sorted(runs_dir.glob("*ledger*.jsonl")):
        for r in _jsonl(p):
            pol = r.get("policy_id") or ""
            out.append({"ts": r.get("ts"), "label": r.get("label") or "", "policy": "candidate" if pol not in ("base", "adhoc") and pol else pol,
                        "task_name": names.get(r.get("task"), r.get("task")), "reward": r.get("reward"),
                        "explanation": r.get("explanation"), "session": r.get("session"), "run_id": r.get("run_id")})
    return out


def _node_log(p: Path | None, n: int = 80) -> str:
    if not p or not p.exists():
        return ""
    lines = [l for l in p.read_text(errors="replace").splitlines() if l.startswith("[") or "Error" in l or "exit" in l]
    return "\n".join(lines[-n:])


def serve(traces_dir: Path, results: Path | None, port: int = 7870, runs_dir: Path | None = None, names_file: Path | None = None,
          node_log: Path | None = None) -> None:
    names = json.loads(names_file.read_text()) if names_file and names_file.exists() else {}

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):  # quiet
            pass

        def do_GET(self):
            if self.path.startswith("/data"):
                turns, sessions = _turns(traces_dir)
                body = json.dumps({"turns": turns, "sessions": sessions, "rollouts": _rollouts(results),
                                   "loop": _loop(runs_dir, names), "ledger": _ledger(runs_dir, names),
                                   "node_log": _node_log(node_log)}).encode()
                ctype = "application/json"
            else:
                body, ctype = PAGE.encode(), "text/html; charset=utf-8"
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    srv = ThreadingHTTPServer(("127.0.0.1", port), H)
    print(f"dashboard on http://localhost:{port}  traces={traces_dir} results={results} runs={runs_dir}", flush=True)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        pass
