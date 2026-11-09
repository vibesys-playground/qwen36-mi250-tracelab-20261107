"""Plan 38 tool: log every Triton JIT compile of this process (tools/live_compile_probe.py).

Active only with ``TRITON_COMPILE_LOG`` set (a file path): on startup it wraps
``triton.runtime.jit.JITFunction._do_compile`` (called on an in-memory cache miss, whether the
binary then comes from the disk cache or a fresh compile) to append one JSON line per call:
wall time, pid, kernel name, and seconds spent. Changes nothing else.
"""

import os

_LOG = os.environ.get("TRITON_COMPILE_LOG")
if _LOG:
    try:
        import json
        import time

        from triton.runtime import jit as _jit

        _orig = _jit.JITFunction._do_compile

        def _do_compile(self, *args, **kwargs):
            t0 = time.time()
            try:
                return _orig(self, *args, **kwargs)
            finally:
                try:
                    with open(_LOG, "a") as f:
                        f.write(json.dumps({"t": t0, "pid": os.getpid(), "name": getattr(self, "__name__", "?"),
                                            "s": round(time.time() - t0, 3)}) + "\n")
                except Exception:  # noqa: BLE001
                    pass

        _jit.JITFunction._do_compile = _do_compile
    except Exception as exc:  # noqa: BLE001
        import sys

        print(f"compile_hook: not installed: {exc!r}", file=sys.stderr, flush=True)
