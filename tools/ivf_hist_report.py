"""Profile-only: summarize the last 'engine: ivf_hist' line of every device in serve-probe logs."""
import glob
import json
import sys

for path in sorted(glob.glob("/tmp/serve-probe-*.log")):
    last = {}
    for line in open(path, errors="replace"):
        if line.startswith("engine: ivf_hist {"):
            d = json.loads(line[len("engine: ivf_hist "):])
            last[d["dev"]] = d
    if not last:
        print(f"{path}: no ivf_hist lines")
        continue
    tot = [0] * 72
    st = [0] * 4
    for d in last.values():
        tot = [a + b for a, b in zip(tot, d["hist"])]
        st = [a + b for a, b in zip(st, d["stats"])]
    print(f"== {path} devices={len(last)} stats(rows,full rows,audit,unlike)={st}")
    for link in range(1, 8):
        for b, name in enumerate(("1slot", "2slot", "3+slot")):
            n, f, new = tot[((link - 1) * 3 + b) * 3: ((link - 1) * 3 + b) * 3 + 3]
            if n:
                print(json.dumps({"link": link, "slots": name, "calls": n, "full_rate": round(f / n, 4),
                                  "new_fallback_rate": round(new / n, 4)}))
    for b, name in enumerate(("1slot", "2slot", "3+slot")):
        link1 = tot[b * 3: b * 3 + 3]
        n = [tot[(l * 3 + b) * 3] for l in range(7)]
        f = [tot[(l * 3 + b) * 3 + 1] for l in range(7)]
        if n[0]:
            print(json.dumps({"slots": name, "steps": n[0], "mean_links": round(sum(n) / n[0], 3),
                              "mean_full_links_per_step": round(sum(f) / n[0], 3),
                              "step_any_fallback": round(tot[(0 * 3 + b) * 3 + 2] and sum(tot[(l * 3 + b) * 3 + 2] for l in range(7)) / n[0], 4)}))
sys.stdout.flush()
