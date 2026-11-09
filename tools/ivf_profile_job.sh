#!/bin/bash
# profile-ivf job 1: served replay-like run, three settings (hist, hist+audit, IVF off).
timeout 9000 python -u tools/serve_probe.py --sessions 48 --concurrency 8 --rounds 6.3 --first 1800 --grow 60 --max-new 190 --corpus synthetic --settings "ENGINE_DRAFT_IVF_HIST=1+ENGINE_LOOP_TIMERS=1,ENGINE_DRAFT_IVF_HIST=1+ENGINE_DRAFT_IVF_AUDIT=1+ENGINE_LOOP_TIMERS=1,ENGINE_DRAFT_IVF=0+ENGINE_LOOP_TIMERS=1" 2>&1 | grep -v "^engine:"
echo "=== hist report"
python3 tools/ivf_hist_report.py
echo "=== engine lines"
grep -h "IVF drafts\|kv_pages" /tmp/serve-probe-*.log | head -20
