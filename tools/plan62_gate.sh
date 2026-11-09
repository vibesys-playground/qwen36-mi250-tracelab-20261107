#!/bin/sh
# Plan 62 gate: the router's idle window at 10 s (the old default) vs 3 s (the new one),
# served on one MI250 node (4 x TP2 replicas, the server's default layout).
#
#   sh tools/plan62_gate.sh
#
# Server starts interleave 10 / 3 / 10 / 3 (two rounds per window). On each server two load
# passes run 12 s apart: the uniform replay-like shape of j241/j272 (48 sessions, 8 at a
# time, 6.3 rounds, 1800 first + 60 grow, 190 out, synthetic corpus), then a generic uneven
# shape (seeded draws, 0.5 s mean tool wait; see tools/serve_probe.py). Each pass prints
# its tok/s, TTFT and TPOT percentiles and the router's per-replica in-flight histogram.
#
# Result j290 (bbcb22f, means of two rounds, 10 s -> 3 s): uniform TPOT p90 3.97 -> 4.20 ms,
# share of replica time above 2 in flight 0.068 -> 0.072; uneven TPOT p90 4.68 -> 4.65 ms,
# share above 2 0.102 -> 0.092. The window alone does not remove the imbalance: a session
# that just ended counts as load under any window when its successor arrives at once.
W=ENGINE_SESSION_IDLE_SECONDS
timeout 3300 python -u tools/serve_probe.py --sessions 48 --concurrency 8 --rounds 6.3 \
  --first 1800 --grow 60 --max-new 190 --corpus synthetic --shapes uniform,uneven \
  --tool-wait 0.5 --seed 0 --settings $W=10,$W=3,$W=10,$W=3 2>&1 \
  | grep -v "^engine:" | python -u -c '
import json, sys
for line in sys.stdin:
    print(line, end="", flush=True)
    if not line.startswith("{\"event\": \"setting\""):
        continue
    d = json.loads(line)
    keys = ("setting", "shape", "wall_s", "requests", "output_tokens", "output_tok_per_s",
            "ttft_ms_p50", "ttft_ms_p90", "tpot_ms_p50", "tpot_ms_p90", "cached_prompt_tokens")
    print("SUMMARY", json.dumps({k: d.get(k) for k in keys} | {"router": d.get("router_pass")}),
          flush=True)
'
