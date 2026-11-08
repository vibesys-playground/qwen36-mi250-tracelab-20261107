"""``ENGINE_LOG_JIT=1``: log every Triton compile with its duration and whether the server was
already serving (plan 35: are kernels JIT-compiled on live requests?). Off by default; a
diagnostic, it changes no kernel.

Each compile prints ``engine-jit: compile pid=<pid> serving=<bool> s=<seconds> fn=<name>``.
A compile served from Triton's disk cache shows as a short one.
"""

from __future__ import annotations

import os
import time

SERVING = False
_installed: list[str] = []


def mark_serving() -> None:
    global SERVING
    SERVING = True
    if _installed:
        print(f"engine-jit: serving pid={os.getpid()} hooks={_installed}", flush=True)


def _name(args, kwargs) -> str:
    src = args[0] if args else kwargs.get("src")
    fn = getattr(src, "fn", None)
    name = getattr(fn, "__name__", None) or getattr(src, "name", None) or type(src).__name__
    return str(name)


def install() -> None:
    if os.environ.get("ENGINE_LOG_JIT") != "1" or _installed:
        return
    try:
        import triton
        import triton.compiler as tc
        import triton.compiler.compiler as tcc
        import triton.runtime.jit as tj
    except Exception as exc:  # noqa: BLE001
        print(f"engine-jit: not installed ({type(exc).__name__}: {exc})", flush=True)
        return
    orig = tcc.compile

    def compile(*args, **kwargs):  # noqa: A001
        t0 = time.perf_counter()
        try:
            return orig(*args, **kwargs)
        finally:
            print(
                f"engine-jit: compile pid={os.getpid()} serving={SERVING} "
                f"s={time.perf_counter() - t0:.3f} fn={_name(args, kwargs)}",
                flush=True,
            )

    for mod in (tcc, tc, triton, tj):
        if getattr(mod, "compile", None) is orig:
            mod.compile = compile
            _installed.append(mod.__name__ + ".compile")
    # JITFunction may keep its own reference to compile
    jf = getattr(tj, "JITFunction", None)
    if jf is not None and getattr(jf, "compile", None) is orig:
        jf.compile = staticmethod(compile)
        _installed.append("JITFunction.compile")
    # Triton's own post-compile hook (3.4+), counted separately: no duration
    runtime = getattr(getattr(triton, "knobs", None), "runtime", None)
    if runtime is not None and getattr(runtime, "jit_post_compile_hook", "absent") is None:

        def hook(*args, **kwargs):
            fn = kwargs.get("fn")
            name = getattr(getattr(fn, "fn", fn), "__name__", None) or getattr(fn, "name", "?")
            print(f"engine-jit: hook pid={os.getpid()} serving={SERVING} fn={name}", flush=True)
            return False

        runtime.jit_post_compile_hook = hook
        _installed.append("knobs.runtime.jit_post_compile_hook")
    print(f"engine-jit: installed pid={os.getpid()} hooks={_installed}", flush=True)
