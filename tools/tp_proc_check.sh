#!/bin/sh
# Plan 30 stage 1 on one MI250 node: rank 1 in its own process (ENGINE_TP_PROC=1) against the
# threads (ENGINE_TP_PROC=0). Run from the project root.
#   1. prefill pass wall at 61/561/1,650 new tokens (3,600-token context), both settings, and
#      greedy tokens of 4 prompts (plain captured decode and the speculative path) compared
#      across the settings (tools/gil_probe.py, GCDs 0,1);
#   2. tools/spec_step_bench.py with ENGINE_TP_PROC=1: step times, speculative == plain on 8 prompts;
#   3. a served A/B/B/A smoke (tools/serve_probe.py, the whole node), then a check that no
#      rank-1 process outlived its server.
set -u
cd "$(dirname "$0")/.."
echo "=== 1. passes and parity: threads vs process"
python tools/gil_probe.py all --devices 0,1 --parts passes --procs 0,1 --modes both --reps 10 --parity 64
echo "=== 2. spec_step_bench ENGINE_TP_PROC=1"
ENGINE_TP=2 ENGINE_TP_PROC=1 HIP_VISIBLE_DEVICES=0,1 python tools/spec_step_bench.py --devices 0,1 --iters 20 --max-new 128
echo "=== 3. served smoke"
SHAPE="--corpus synthetic --sessions 16 --concurrency 8 --rounds 3 --first 1800 --grow 300 --max-new 128"
python tools/serve_probe.py $SHAPE --settings ENGINE_TP_PROC=0,ENGINE_TP_PROC=1
for f in /tmp/serve-probe-ENGINE_TP_PROC_1.log; do
  echo "--- $f (engine/error lines)"; grep -E "engine:|Error|error|Traceback|rank 1" "$f" | head -40
done
python tools/serve_probe.py $SHAPE --settings ENGINE_TP_PROC=1,ENGINE_TP_PROC=0
sleep 5
echo "=== 4. leftover processes"
# Only processes of this job count (the node may run other jobs): SLURM_JOB_ID in their env.
found=0
for p in $(ps -eo pid=,args= | grep -E "multiprocessing|engine.server" | grep -v grep | awk '{print $1}'); do
  env=$(tr '\0' '\n' < /proc/$p/environ 2>/dev/null | grep -E "^(SLURM_JOB_ID|ENGINE_TP_PROC)=" | tr '\n' ' ')
  case "$env" in *"SLURM_JOB_ID=${SLURM_JOB_ID:-none} "*) found=1; echo "LEFTOVER $(ps -o pid=,ppid=,etime=,args= -p $p) [$env]";; *) echo "other job: $p [$env]";; esac
done
[ $found = 0 ] && echo "RESULT orphans none"
