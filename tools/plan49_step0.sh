#!/bin/sh
# Plan 49 step 0: migrate a session's cached state between replica processes on GPUs 0,1 and
# 2,3 (tools/plan49_migrate_probe.py), then feed the measured cost to the planner's simulation
# (tools/plan49_resim.py). Per-process timeouts; output streams as it runs.
#
#   sh tools/plan49_step0.sh [probe args...]
LOG=/tmp/plan49_probe.log
echo "=== probe"
HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0,1,2,3} timeout 2400 python -u tools/plan49_migrate_probe.py "$@" 2>&1 | tee "$LOG"
rc=$(grep -c '"kind": "summary"' "$LOG")
echo "=== resim"
ARGS=$(timeout 60 python - "$LOG" <<'PY'
import json, sys
s = None
for line in open(sys.argv[1]):
    if line.startswith("RESULT ") and '"kind": "summary"' in line:
        s = json.loads(line[7:])
if s is None:
    sys.exit(0)
out = []
for kind, stall_kind in (("host", "host"), ("host", "host_async"), ("ipc", "ipc")):
    for L in (4050, 8130):
        e2e = s.get(f"e2e_ms_{kind}_{L}")
        stall = s.get(f"stall_ms_{stall_kind}_8k")
        if e2e is None or stall is None:
            continue
        if stall_kind == "host_async":
            e2e = max(e2e, s.get("move_ms_host_async_8k", e2e))
        out += ["--cost-ms", str(e2e), "--stall-ms", str(stall), "--label", f"{stall_kind}@{L}"]
print(" ".join(out))
PY
)
echo "resim args: $ARGS"
timeout 300 python -u tools/plan49_resim.py .vibesys/tasks/mi2508x-tracelab/recorded_sessions.csv $ARGS
[ "$rc" -ge 1 ] || exit 1
grep -q '"identical_all": true' "$LOG" && grep -q '"bytes_equal_all": true' "$LOG" || exit 3
exit 0
