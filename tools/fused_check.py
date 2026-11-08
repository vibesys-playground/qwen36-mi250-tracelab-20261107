"""Plan 11 stage 1: fused decode layers (``engine.fast.fused``) against the per-op decode path.

    python tools/fused_check.py [--layers 4] [--devices 0,1] [--barrier] [--rows 1,2,4,8]

Loads the checkpoint as the server does (``--layers N`` keeps the first N decoder layers, for
one MI210; ``--layers 0`` is the whole model; ``--devices`` sets HIP_VISIBLE_DEVICES before
torch loads, e.g. one 2-GCD replica). Then, all on ``FastModel``, toggling
``triton_kernels["fused_decode"]`` (everything else is the same object and weights):

1. Layer parity on real activations: 8 prompts of 300-1100 tokens prefilled into slots 0-7,
   then for R rows each layer is run per-op and fused from the same state on the per-op
   stack's hidden state: elements equal, max |diff|, and the conv/recurrent/KV state after.
2. Greedy parity: 4 prompts (1k-4k tokens), 64 free-running greedy tokens as one batch, per-op
   vs fused: identical tokens, max |logit diff|, top-1 agreement teacher-forced on per-op's
   tokens.
3. Time: one layer of each kind in a graph of 10 back-to-back calls, per-op vs fused, at each R;
   then the whole decode step (``FastModel.decode`` with its captured graphs) per-op vs fused.
4. Kernel count per fused layer (torch.profiler, eager).
``--barrier`` first runs the grid-barrier microbenchmark of ``tools/mega_stage0.py``.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--devices", default=None)
    ap.add_argument("--barrier", action="store_true")
    ap.add_argument("--rows", default="1,2,4,8")
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--slots", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=262144)
    args = ap.parse_args()
    if args.devices is not None:
        os.environ["HIP_VISIBLE_DEVICES"] = args.devices
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "tools"))
    import torch
    from torch.nn import functional as F

    import mega_stage0 as ms
    from engine.fast import kernels
    from engine.fast.model import FastModel
    from engine.generator import CHECKPOINT, Generator

    print("torch", torch.__version__, "hip", torch.version.hip, "gpus", torch.cuda.device_count(),
          [torch.cuda.get_device_properties(d).name for d in range(torch.cuda.device_count())])
    if args.barrier:
        try:
            ms.part_a(torch.device("cuda", 0), torch.cuda.get_device_properties(0).multi_processor_count)
        except Exception:  # noqa: BLE001
            traceback.print_exc()

    model, revision = CHECKPOINT["model"], CHECKPOINT["revision"]
    config = None
    if args.layers:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(model, revision=revision)
        for cfg in {id(c): c for c in (config, config.get_text_config())}.values():
            if hasattr(cfg, "layer_types"):
                cfg.layer_types = list(cfg.layer_types)[: args.layers]
                cfg.num_hidden_layers = args.layers
            if hasattr(cfg, "mtp_num_hidden_layers"):
                cfg.mtp_num_hidden_layers = 0
    t0 = time.perf_counter()
    gen = Generator.load(model, revision, config=config, max_model_len=args.max_len,
                         chunk_tokens=2048, prefix_cache_tokens=0, prefix_cache_entries=0)
    lm = gen.model
    fast = gen.fast
    if fast is None:  # ENGINE_FAST off or unavailable: build one
        fast = FastModel(lm, slots=args.slots, max_len=args.max_len)
    print(f"RESULT load_s={time.perf_counter() - t0:.1f} layers={len(fast.layers)} "
          f"fused={fast.fused is not None} fused_error={fast.fused_error} "
          f"graphs={fast.use_graphs} graph_error={fast.graph_error} "
          f"segments={[str(s.device) for s in fast.segments]}")
    for d in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(d)
        print(f"mem cuda:{d} used_GB={(total - free) / 1e9:.1f} of {total / 1e9:.1f}")
    if fast.fused is None:
        print("RESULT verdict no fused path")
        return 1
    rows_list = [int(r) for r in args.rows.split(",")]

    def toggle(flag: bool) -> None:
        fast.triton_kernels["fused_decode"] = flag

    def sync() -> None:
        for d in range(torch.cuda.device_count()):
            torch.cuda.synchronize(d)

    def prompts(lengths, skip=0):
        import transformers

        files = sorted(Path(transformers.__file__).parent.glob("models/*/modeling_*.py"))[skip:]
        out, index = [], 0
        for length in lengths:
            while True:
                ids = list(gen.tokenizer.encode(files[index].read_text()))
                index += 1
                if len(ids) >= length:
                    out.append(ids[:length])
                    break
        return out

    def prefill(slot, prompt):
        body = len(prompt) - 1
        pos = 0
        while pos < body:
            end = min((pos // 2048 + 1) * 2048, body)
            fast.prefill(slot, prompt[pos:end], pos)
            pos = end
        return fast.prefill(slot, prompt[body:], body)

    def save():
        return ({i: (c.clone(), r.clone()) for i, (c, r) in fast.gdn_state.items()},
                {i: (k.clone(), v.clone()) for i, (k, v) in fast.kv.items()})

    def load(saved):
        for i, (c, r) in saved[0].items():
            fast.gdn_state[i][0].copy_(c)
            fast.gdn_state[i][1].copy_(r)
        for i, (k, v) in saved[1].items():
            fast.kv[i][0].copy_(k)
            fast.kv[i][1].copy_(v)

    torch.set_grad_enabled(False)
    # ------------------------------------------------------------ 1. layer parity
    # Parity runs eagerly: a captured graph replays whichever path it was captured with.
    graphs_wanted = fast.use_graphs
    fast.use_graphs = False
    lp = prompts([300 + 100 * k for k in range(8)], skip=40)
    first = []
    with torch.inference_mode():
        toggle(False)
        for slot, p in enumerate(lp):
            fast.reset(slot)
            first.append(int(prefill(slot, p).argmax()))
    small_state = None
    if args.layers:  # the full model's state is too large to clone per row count
        small_state = save()
    try:
        for R in rows_list:
            if R > 8:
                continue
            slots = list(range(R))
            positions = [len(lp[s]) for s in slots]
            for s, p in zip(slots, positions):
                fast._ensure_pages(s, p + 1)
            with torch.inference_mode():
                x = F.embedding(torch.tensor(first[:R], device=fast.embed.device), fast.embed)
                worst = []
                for i in range(len(fast.layers)):
                    dev = fast.layers[i].input_layernorm.weight.device
                    x = x.to(dev)
                    meta = fast.devices[dev]
                    st = torch.tensor(slots, device=dev)
                    pt = torch.tensor(positions, device=dev)
                    keep = None
                    if i in fast.gdn_state:
                        keep = (fast.gdn_state[i][0].clone(), fast.gdn_state[i][1].clone())
                    elif i in fast.kv:
                        keep = None  # KV writes go to (page, offset) of these positions only
                    toggle(False)
                    ref = fast.layer_decode(i, x, st, pt, meta).clone()
                    ref_state = None
                    if keep is not None:
                        ref_state = (fast.gdn_state[i][0].clone(), fast.gdn_state[i][1].clone())
                        fast.gdn_state[i][0].copy_(keep[0])
                        fast.gdn_state[i][1].copy_(keep[1])
                    else:
                        k_pool, v_pool = fast.kv[i]
                        pages = meta.page_table[st, pt // 64].long()
                        ref_kv = (k_pool[pages, :, pt % 64].clone(), v_pool[pages, :, pt % 64].clone())
                        k_pool[pages, :, pt % 64] = 0
                        v_pool[pages, :, pt % 64] = 0
                    toggle(True)
                    got = fast.layer_decode(i, x, st, pt, meta).clone()
                    sync()
                    if ref_state is not None:
                        sd = [float((a.float() - b.float()).abs().max()) for a, b in
                              zip(ref_state, fast.gdn_state[i])]
                        se = [bool(torch.equal(a[st], b[st])) for a, b in zip(ref_state, fast.gdn_state[i])]
                        # Continue from the per-op state.
                        fast.gdn_state[i][0].copy_(ref_state[0])
                        fast.gdn_state[i][1].copy_(ref_state[1])
                        state = f"conv_max={sd[0]:.4g} rec_max={sd[1]:.4g} conv_equal={se[0]} rec_equal={se[1]}"
                    else:
                        k_pool, v_pool = fast.kv[i]
                        gk, gv = k_pool[pages, :, pt % 64], v_pool[pages, :, pt % 64]
                        state = (f"k_max={float((gk.float() - ref_kv[0].float()).abs().max()):.4g} "
                                 f"v_max={float((gv.float() - ref_kv[1].float()).abs().max()):.4g} "
                                 f"k_equal={bool(torch.equal(gk, ref_kv[0]))} v_equal={bool(torch.equal(gv, ref_kv[1]))}")
                        k_pool[pages, :, pt % 64] = ref_kv[0]
                        v_pool[pages, :, pt % 64] = ref_kv[1]
                    diff = (got.float() - ref.float()).abs()
                    eq = float((got == ref).float().mean())
                    worst.append(float(diff.max()))
                    if i < 4 or i >= len(fast.layers) - 2 or float(diff.max()) > 0.25:
                        print(f"RESULT layer_parity R={R} layer={i} {fast.layer_types[i]} "
                              f"max_abs={float(diff.max()):.4g} equal_frac={eq:.5f} "
                              f"ref_absmax={float(ref.float().abs().max()):.3g} {state}")
                    x = ref
                print(f"RESULT layer_parity_summary R={R} worst_max_abs={max(worst):.4g} "
                      f"mean_max_abs={sum(worst) / len(worst):.4g}")
            if small_state is not None:
                load(small_state)
            else:  # the per-op stack advanced the state one token: restart from fresh prefills
                with torch.inference_mode():
                    toggle(False)
                    for slot, p in enumerate(lp):
                        fast.reset(slot)
                        prefill(slot, p)
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        print("RESULT layer_parity error")

    # ------------------------------------------------------------ 2. greedy parity
    gp = prompts([1024, 2048, 3072, 4000])
    rows = list(range(len(gp)))
    runs = {}
    try:
        for flag in (False, True):
            toggle(flag)
            with torch.inference_mode():
                current = []
                for slot, p in enumerate(gp):
                    fast.reset(slot)
                    current.append(int(prefill(slot, p).argmax()))
                tokens, logits = [], []
                for step in range(args.steps):
                    out = fast.decode(current, rows, [len(p) + step for p in gp]).float()
                    logits.append(out.cpu())
                    current = [int(t) for t in out.argmax(-1).tolist()]
                    tokens.append(current)
                runs[flag] = (tokens, logits)
        a, b = runs[False], runs[True]
        same_prefix = []
        for r in rows:
            n = 0
            while n < args.steps and a[0][n][r] == b[0][n][r]:
                n += 1
            same_prefix.append(n)
        diffs = [float((x - y).abs().max()) for x, y, n in zip(a[1], b[1], range(min(same_prefix) + 1))]
        print(f"RESULT greedy_parity identical_prefix={same_prefix} of {args.steps} "
              f"max_logit_diff_while_identical={max(diffs):.4f} first4={[round(d, 4) for d in diffs[:4]]}")
        # Teacher-forced: fused fed per-op's tokens.
        toggle(True)
        agree = total = 0
        tf_max = 0.0
        with torch.inference_mode():
            current = []
            for slot, p in enumerate(gp):
                fast.reset(slot)
                current.append(int(prefill(slot, p).argmax()))
            for step in range(args.steps - 1):
                out = fast.decode(current, rows, [len(p) + step for p in gp]).float().cpu()
                ref = a[1][step]
                tf_max = max(tf_max, float((out - ref).abs().max()))
                for r in rows:
                    total += 1
                    agree += int(int(out[r].argmax()) == int(ref[r].argmax()))
                current = a[0][step]
        print(f"RESULT teacher_forced top1_pct={100.0 * agree / total:.2f} ({agree}/{total}) "
              f"max_logit_diff={tf_max:.4f}")
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        print("RESULT greedy_parity error")

    # ------------------------------------------------------------ 3. time
    dev0 = torch.device("cuda", 0)
    try:
        tp = prompts([2048] * 8, skip=80)
        with torch.inference_mode():
            toggle(False)
            for slot, p in enumerate(tp):
                fast.reset(slot)
                prefill(slot, p)
        kinds = {}
        for i, t in enumerate(fast.layer_types):
            kinds.setdefault(t, i)
        for kind, i in kinds.items():
            dev = fast.layers[i].input_layernorm.weight.device
            meta = fast.devices[dev]
            with torch.cuda.device(dev):
                for R in rows_list:
                    x = torch.randn(R, fast.embed.shape[1]).to(dev, torch.bfloat16)
                    st = torch.arange(R, device=dev)
                    pt = torch.full((R,), 2048, device=dev)
                    res = {}
                    for flag in (False, True):
                        toggle(flag)

                        def call():
                            return fast.layer_decode(i, x, st, pt, meta)

                        with torch.inference_mode():
                            res[flag] = ms.graph_ms(call)
                    print(f"RESULT layer_ms kind={kind} R={R} per_op={res[False]:.4f} fused={res[True]:.4f} "
                          f"speedup={res[False] / res[True]:.2f}x")
                    if R >= 4:  # tile variants of the fused path
                        from engine.fast import fused as fused_mod

                        base = dict(fused_mod.TILES)
                        toggle(True)
                        for variant in ({"bn": 16, "bk4": 512, "bk8": 256},
                                        {"bn": 4, "bk4": 2048, "bk8": 1024},
                                        {"moe_bn": 16}, {"moe_bn": 4}):
                            fused_mod.TILES.update(variant)
                            try:
                                with torch.inference_mode():
                                    t = ms.graph_ms(lambda: fast.layer_decode(i, x, st, pt, meta))
                                print(f"RESULT layer_tiles kind={kind} R={R} tiles={fused_mod.TILES} ms={t:.4f}")
                            except Exception as exc:  # noqa: BLE001
                                print(f"RESULT layer_tiles kind={kind} R={R} tiles={variant} error={exc}")
                            fused_mod.TILES.clear()
                            fused_mod.TILES.update(base)
        # The whole decode step through FastModel.decode (captured graphs per bucket).
        pos = [len(p) for p in tp]
        fast.use_graphs = graphs_wanted
        kept = []  # graphs of an earlier capture stay alive: freeing their pool mid-run asserts
        for flag in (False, True):
            toggle(flag)
            kept.append([(dict(seg.graphs), dict(seg.inputs), dict(seg.outputs)) for seg in fast.segments])
            fast.__dict__["_pools"] = {}  # a fresh private pool per capture
            for seg in fast.segments:
                seg.graphs.clear()
                seg.inputs.clear()
                seg.outputs.clear()
            fast.graph_error = None
            t0 = time.perf_counter()
            fast.capture_all()
            sync()
            print(f"RESULT capture fused={flag} s={time.perf_counter() - t0:.1f} error={fast.graph_error}")
            for R in rows_list:

                def step(R=R):
                    with torch.inference_mode():
                        fast.decode([100] * R, list(range(R)), [pos[s] for s in range(R)])
                    for s in range(R):
                        pos[s] += 1

                for _ in range(3):
                    step()
                sync()
                n = 20
                t0 = time.perf_counter()
                for _ in range(n):
                    step()
                sync()
                ms_step = (time.perf_counter() - t0) / n * 1e3
                print(f"RESULT step_ms fused={flag} R={R} ms={ms_step:.2f} layers={len(fast.layers)}")
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        print("RESULT time error")

    # ------------------------------------------------------------ 4. kernels per fused layer
    try:
        from torch.profiler import ProfilerActivity, profile

        toggle(True)
        for kind, i in kinds.items():
            dev = fast.layers[i].input_layernorm.weight.device
            meta = fast.devices[dev]
            with torch.cuda.device(dev):
                x = torch.randn(2, fast.embed.shape[1]).to(dev, torch.bfloat16)
                st = torch.arange(2, device=dev)
                pt = torch.full((2,), 2048, device=dev)
                with torch.inference_mode():
                    fast.layer_decode(i, x, st, pt, meta)
                    sync()
                    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                        for _ in range(10):
                            fast.layer_decode(i, x, st, pt, meta)
                        sync()
            kev = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
            per: dict[str, list[float]] = {}
            for e in kev:
                per.setdefault(e.name, []).append(e.time_range.elapsed_us())
            print(f"RESULT fused_profile kind={kind} R=2 kernels_per_layer={len(kev) / 10:.1f} "
                  f"busy_us_per_layer={sum(e.time_range.elapsed_us() for e in kev) / 10:.1f}")
            for name, ts in sorted(per.items(), key=lambda kv: -sum(kv[1])):
                print(f"  kernel[{kind}] n/layer={len(ts) / 10:.1f} avg_us={sum(ts) / len(ts):.1f} {name[:90]}")
    except Exception:  # noqa: BLE001
        traceback.print_exc()
    return 0


if __name__ == "__main__":
    sys.exit(main())
