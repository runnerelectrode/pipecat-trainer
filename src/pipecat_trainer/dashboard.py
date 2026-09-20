"""`pipecat-trainer dashboard`: one local page that shows calls becoming data.

Left: live traces (one row per LLM turn) from a traces directory. Right: judged rollouts from a
results.jsonl (reward, verdict, hints). Refreshes every 2 s. Stdlib only.
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>pipecat-trainer</title>
<style>
:root{--ink:#1f2933;--mute:#52606d;--line:#d9dee3;--ok:#1b7f4b;--bad:#b42318;--warn:#b7791f}
body{font:14px/1.45 -apple-system,Helvetica,Arial,sans-serif;color:var(--ink);margin:0;background:#fff}
header{padding:14px 20px;border-bottom:1px solid var(--line);display:flex;gap:24px;align-items:baseline}
h1{font-size:18px;margin:0}small{color:var(--mute)}
main{display:grid;grid-template-columns:1fr 1fr;gap:0;min-height:calc(100vh - 52px)}
section{padding:14px 20px;overflow:auto}section+section{border-left:1px solid var(--line)}
h2{font-size:13px;letter-spacing:.04em;text-transform:uppercase;color:var(--mute);margin:0 0 10px}
table{border-collapse:collapse;width:100%}td,th{padding:6px 8px;border-bottom:1px solid var(--line);vertical-align:top;text-align:left}
th{font-weight:600;color:var(--mute);font-size:12px}
.mono{font-family:ui-monospace,Menlo,monospace;font-size:12px}
.pill{display:inline-block;padding:1px 8px;border-radius:10px;font-size:12px;font-weight:600}
.ok{background:#e6f4ea;color:var(--ok)}.bad{background:#fdecea;color:var(--bad)}.warn{background:#fff4d6;color:var(--warn)}
.num{font-variant-numeric:tabular-nums}.dim{color:var(--mute)}
</style></head><body>
<header><h1>pipecat-trainer</h1><small id="meta">loading…</small></header>
<main>
<section><h2>Live calls · one row per LLM turn</h2><table><thead><tr><th>session</th><th>turn</th><th>caller said</th><th>bot said</th><th class="num">TTFB</th><th></th></tr></thead><tbody id="traces"></tbody></table></section>
<section><h2>Judged rollouts · scenarios × K</h2><div id="summary" class="dim"></div><table><thead><tr><th>scenario</th><th>#</th><th class="num">reward</th><th>verdict</th><th>judge's reasons (the hint)</th></tr></thead><tbody id="rollouts"></tbody></table></section>
</main>
<script>
const esc=s=>String(s??'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
async function tick(){
  const d=await (await fetch('/data')).json();
  document.getElementById('meta').textContent=`${d.traces_dir} · ${d.turns.length} turns in ${d.sessions} sessions · ${d.rollouts.length} rollouts`;
  document.getElementById('traces').innerHTML=d.turns.slice(-60).reverse().map(t=>`<tr><td class="mono">${esc(t.session.slice(0,8))}</td><td class="num">${t.turn}</td><td>${esc(t.user)}</td><td>${esc(t.reply)}</td><td class="num">${t.ttfb==null?'—':t.ttfb.toFixed(2)+' s'}</td><td>${t.interrupted?'<span class="pill warn">interrupted</span>':''}${t.speculation?'<span class="pill warn">spec</span>':''}</td></tr>`).join('');
  const r=d.rollouts;const scored=r.filter(x=>x.reward!=null);
  document.getElementById('summary').textContent=scored.length?`mean reward ${(scored.reduce((a,x)=>a+x.reward,0)/scored.length).toFixed(3)} over ${scored.length} scored`:'';
  document.getElementById('rollouts').innerHTML=r.slice().reverse().map(x=>`<tr><td>${esc(x.scenario)}</td><td class="num">${x.rollout}</td><td class="num">${x.reward==null?'—':x.reward.toFixed(2)}</td><td>${x.error?'<span class="pill bad">error</span>':x.passed?'<span class="pill ok">pass</span>':'<span class="pill bad">fail</span>'}</td><td class="dim">${esc(x.error||x.hint)}</td></tr>`).join('');
}
tick();setInterval(tick,2000);
</script></body></html>"""


def _turns(traces_dir: Path) -> tuple[list[dict], int]:
    turns, sessions = [], 0
    for p in sorted(traces_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime):
        sessions += 1
        for line in p.read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("type") != "turn":
                continue
            users = [m for m in r.get("messages", []) if m.get("role") == "user"]
            turns.append({"session": r["session_id"], "turn": r["turn"], "user": users[-1]["content"] if users else "",
                          "reply": r.get("reply", ""), "ttfb": r.get("llm_ttfb_s"), "interrupted": r.get("was_interrupted"),
                          "speculation": r.get("speculation")})
    return turns, sessions


def _rollouts(results: Path | None) -> list[dict]:
    if not results or not results.exists():
        return []
    out = []
    for line in results.read_text().splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        out.append({k: r.get(k) for k in ("scenario", "rollout", "reward", "passed", "hint", "error", "ended_by")})
    return out


def serve(traces_dir: Path, results: Path | None, port: int = 7870) -> None:
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):  # quiet
            pass

        def do_GET(self):
            if self.path.startswith("/data"):
                turns, sessions = _turns(traces_dir)
                body = json.dumps({"traces_dir": str(traces_dir), "turns": turns, "sessions": sessions,
                                   "rollouts": _rollouts(results)}).encode()
                ctype = "application/json"
            else:
                body, ctype = PAGE.encode(), "text/html; charset=utf-8"
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    srv = ThreadingHTTPServer(("127.0.0.1", port), H)
    print(f"dashboard on http://localhost:{port}  traces={traces_dir} results={results}", flush=True)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        pass
