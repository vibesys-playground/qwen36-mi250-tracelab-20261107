"""OpenAI-compatible HTTP server for the baseline engine.

    python -m engine.server [--host 127.0.0.1] [--port 8000]

Serves ``GET /health``, ``GET /v1/models``, ``POST /v1/completions`` and
``POST /v1/chat/completions``, streaming (SSE) or not. Connection threads parse
requests and write responses; one worker thread owns the model. It repeats one
loop: prefill the newly arrived requests, then pick one token for every running
request and feed all of them back in a single batched decode step. On the fast
path the prefills of all waiting requests share passes (``PackedPrefill``), one
pass per loop. The port opens only after the weights are loaded, so a readiness
probe on ``/health`` passes once the server can answer.

With more than one replica in its layout (by default one per two visible GPUs,
so four 2-GCD replicas on an eight-GCD MI250 node; ``--replicas``,
``--gpus-per-replica`` and ``--replica-devices`` or their ``ENGINE_*`` variables
change it), this process instead becomes the front of ``engine.router``: it
starts each replica as this same server on its own devices and routes every
request to one of them, keeping a session on one replica. With one GPU, none,
or ``--replicas 1``, it serves in-process as described above.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import sys
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from engine import (
    cpu_place,
    experts,
    looptimers,
    pair_host,
    protocol,
    router,
    speculative,
    split_prefill,
)
from engine.generator import CHECKPOINT, Generation, Generator, PackedPrefill, Step
from engine.protocol import (
    ChatRequest,
    CompletionRequest,
    Envelope,
    RequestError,
    Sampling,
    Usage,
)
from engine.text import Detokenizer, StopMatcher

_DONE = object()


@dataclass(frozen=True)
class AwaitDecode:
    """Yielded by a job after it has handled one token and needs the next decode step."""

    generation: Generation


@dataclass(frozen=True)
class Prefill:
    """Yielded by a job for its prompt; the worker sends back the request's ``Generation``."""

    prompt: tuple[int, ...]
    sampling: Sampling


@dataclass
class Job:
    """A request handed to the worker; the worker answers on ``events``.

    ``run`` yields a ``Prefill`` first (the worker sends the ``Generation``
    back), then the request's response events, and an ``AwaitDecode`` after
    each token that is not its last.
    """

    run: Callable[[], Iterator[dict[str, Any] | AwaitDecode]]
    events: queue.Queue = field(default_factory=queue.Queue)
    cancelled: threading.Event = field(default_factory=threading.Event)


def _internal(exc: Exception) -> RequestError:
    return RequestError(f"internal error: {type(exc).__name__}: {exc}", 500)


class Worker:
    """The only thread that touches the model: prefills new jobs, then decodes all running jobs together.

    On the fast path new jobs' prefills are packed (``PackedPrefill``): every
    loop admits the jobs that arrived, runs one pass over the next prefill
    chunk of each waiting prompt, hands the prompts that finished to decoding,
    and runs one decode step for every running job. Otherwise each new job is
    prefilled alone (``Generator.start``) before the decode step, as before.
    """

    def __init__(self, generator: Generator, gate: Any = None, rank: int = 0) -> None:
        self._generator = generator
        self._jobs: queue.Queue[Job] = queue.Queue()
        # A split-prefill host's pair (``engine.split_prefill``): each loop iteration is a turn,
        # between which another pair may borrow this one; ``rank`` is its first rank core.
        self._gate = gate if gate is not None else split_prefill.NoGate()
        self._rank = rank
        threading.Thread(target=self._loop, name="engine-worker", daemon=True).start()

    def submit(self, job: Job) -> None:
        self._jobs.put(job)

    @staticmethod
    def _advance(
        job: Job, events: Iterator[Any], send: Generation | None = None
    ) -> AwaitDecode | Prefill | None:
        """Run ``job`` (resumed with ``send``) up to its next decode step or its prefill;
        None once it has ended."""
        try:
            event = events.send(send)  # type: ignore[attr-defined]
            while True:
                if job.cancelled.is_set():
                    break
                if isinstance(event, (AwaitDecode, Prefill)):
                    return event
                if looptimers.ENABLED:
                    t = looptimers.now()
                    job.events.put(event)
                    looptimers.add("emit", looptimers.now() - t)
                else:
                    job.events.put(event)
                event = next(events)
        except StopIteration:
            pass
        except RequestError as exc:
            job.events.put(exc)
        except Exception as exc:  # report to the client; the worker keeps serving
            job.events.put(_internal(exc))
        job.events.put(_DONE)
        return None

    @classmethod
    def _advance_all(
        cls, job: Job, events: Iterator[Any], send: Generation | None = None
    ) -> AwaitDecode | Prefill | None:
        """``_advance``, repeated while the request's next token is already picked (a verify
        step of ``engine.speculative`` emits two), so it waits only for a real decode step."""
        want = cls._advance(job, events, send)
        while isinstance(want, AwaitDecode) and want.generation.has_queued():
            want = cls._advance(job, events)
        return want

    @staticmethod
    def _fail(job: Job, events: Iterator[Any], exc: Exception) -> None:
        job.events.put(exc if isinstance(exc, RequestError) else _internal(exc))
        job.events.put(_DONE)
        events.close()  # type: ignore[attr-defined]

    def _loop(self) -> None:
        cpu_place.pin_rank(self._rank)  # this thread drives rank 0 (and the whole replica on one GPU)
        gate = self._gate
        g = self._generator
        packer = PackedPrefill(g) if g.packing else None
        prefilling: dict[int, tuple[Job, Iterator[Any]]] = {}
        running: list[tuple[Job, Iterator[Any]]] = []
        timed = looptimers.ENABLED
        T = looptimers
        prev_batch: int | None = None
        while True:
            if timed:
                T.begin()
                t_mark = T.now()
                arrivals = 0
                prefilled = False
            block = not running and not prefilling
            ready: list[tuple[Job, Iterator[Any], Generation]] = []
            serial: list[tuple[Job, Iterator[Any], Prefill]] = []
            gate.leave()  # between turns: another pair may borrow this one now
            first = self._jobs.get() if block else None  # idle: wait for a job outside a turn
            gate.enter()
            while True:
                try:
                    job = first if first is not None else self._jobs.get(block=False)
                except queue.Empty:
                    break
                first = None
                if timed:
                    arrivals += 1
                events = job.run()
                want = self._advance(job, events)
                if not isinstance(want, Prefill):
                    if want is not None:  # a job that decodes before its prefill: not ours
                        self._fail(job, events, RuntimeError("job decoded before its prefill"))
                    continue
                if packer is None:
                    serial.append((job, events, want))
                else:
                    prefilling[packer.add(want.prompt, want.sampling)] = (job, events)
            if speculative.pending(g) and (
                serial or prefilling or ready_pending(packer)
                or any(job.cancelled.is_set() for job, _ in running)
            ):
                # Plan 42: the batch is about to change; finish the step launched ahead first.
                speculative.drain(g)
            if timed:
                t = T.now()
                T.add("poll", t - t_mark)
                t_mark = t
            if packer is not None:
                for ticket, exc in packer.failed():
                    self._fail(*prefilling.pop(ticket), exc)
                if packer.busy and gate.may_prefill():
                    for ticket, result in packer.step():
                        job, events = prefilling.pop(ticket)
                        if isinstance(result, Exception):
                            self._fail(job, events, result)
                        else:
                            ready.append((job, events, result))
                    if timed:
                        prefilled = True
            if timed:
                t = T.now()
                T.add("prefill", t - t_mark)
                t_mark = t
            waiting = []
            for job, events in running:
                want = self._advance_all(job, events)
                if isinstance(want, AwaitDecode):
                    waiting.append((job, events, want.generation))
            if timed:
                t = T.now()
                T.add("adv", t - t_mark)
                t_mark = t
            for job, events, generation in ready:
                want = self._advance_all(job, events, generation)
                if isinstance(want, AwaitDecode):
                    waiting.append((job, events, want.generation))
            if timed:
                t = T.now()
                T.add("adv_new", t - t_mark)
                t_mark = t
            # Unpacked: each new job is prefilled and emits its first token before the next
            # job's prefill, after the running jobs' tokens (the order the base engine used).
            for job, events, prefill in serial:
                try:
                    generation = g.start(prefill.prompt, prefill.sampling)
                except Exception as exc:  # noqa: BLE001 - this request only
                    self._fail(job, events, exc)
                    continue
                want = self._advance_all(job, events, generation)
                if isinstance(want, AwaitDecode):
                    waiting.append((job, events, want.generation))
            running = [(job, events) for job, events, _ in waiting]
            if not waiting:
                speculative.drain(g)  # every request of a step launched ahead has left
                continue
            try:
                # Run ahead only with nothing else to start: no job queued, no prefill pending
                # (asked again right before the launch, which may wait for the step's end).
                quiet = not serial and not prefilling and not ready_pending(packer)

                def ahead_ok(quiet: bool = quiet) -> bool:
                    return quiet and self._jobs.empty()

                if timed:
                    batch = [generation for _, _, generation in waiting]
                    T.add("pre", T.now() - t_mark)
                    g.decode(batch, ahead_ok)
                    T.finish(prev_batch, arrivals, prefilled, T.cur.get("_tokens", 0))
                    prev_batch = len(batch)
                else:
                    g.decode([generation for _, _, generation in waiting], ahead_ok)
            except Exception as exc:  # the step failed for every request in it
                g._ahead = None
                for job, _, _ in waiting:
                    job.events.put(_internal(exc))
                    job.events.put(_DONE)
                running = []


def ready_pending(packer: PackedPrefill | None) -> bool:
    """Whether the packed prefill has work queued."""
    return packer is not None and bool(packer.busy)


class _Output:
    """Splits generated tokens into reasoning and content text, with stop strings applied."""

    def __init__(self, generator: Generator, stops: tuple[str, ...], thinking: bool) -> None:
        self._generator = generator
        self._stops = stops
        self.in_reasoning = thinking
        self._detok = Detokenizer(generator.tokenizer)
        self._matcher = StopMatcher(stops)
        self.content: list[str] = []
        self.reasoning: list[str] = []

    @property
    def stopped(self) -> bool:
        return self._matcher.stopped

    def push(self, step: Step) -> tuple[str, str]:
        """(reasoning delta, content delta) for one step."""
        if not step.emit:
            return "", ""
        if self.in_reasoning and step.token_id == self._generator.think_end_id:
            tail = self._detok.flush()
            self.reasoning.append(tail)
            self.in_reasoning = False
            self._detok = Detokenizer(self._generator.tokenizer)
            return tail, ""
        if looptimers.ENABLED:
            looptimers.cur["_tokens"] = looptimers.cur.get("_tokens", 0) + 1
            t = looptimers.now()
            text = self._detok.push(step.token_id)
            t1 = looptimers.now()
            looptimers.add("detok", t1 - t)
            if self.in_reasoning:
                self.reasoning.append(text)
                return text, ""
            out = self._matcher.push(text)
            looptimers.add("stop", looptimers.now() - t1)
            self.content.append(out)
            return "", out
        text = self._detok.push(step.token_id)
        if self.in_reasoning:
            self.reasoning.append(text)
            return text, ""
        out = self._matcher.push(text)
        self.content.append(out)
        return "", out

    def flush(self) -> tuple[str, str]:
        tail = self._detok.flush()
        if self.in_reasoning:
            self.reasoning.append(tail)
            return tail, ""
        out = self._matcher.push(tail) + self._matcher.flush()
        self.content.append(out)
        return "", out


def _run(
    generation: Generation, output: _Output, usage: Usage
) -> Iterator[tuple[Step, str, str, protocol.FinishReason | None]]:
    """Yield (step, reasoning delta, content delta, finish) per token until the request ends.

    The caller yields ``AwaitDecode`` after each unfinished step, so the worker
    has decoded the token before this generator resumes.
    """
    while True:
        step = generation.next_step()
        usage.completion_tokens += int(step.emit)
        reasoning, content = output.push(step)
        finish = "stop" if output.stopped else step.finish
        if finish is not None:
            generation.retire()
            r_tail, c_tail = ("", "") if output.stopped else output.flush()
            yield step, reasoning + r_tail, content + c_tail, finish
            return
        yield step, reasoning, content, None


def completion_job(
    generator: Generator, model: str, request: CompletionRequest
) -> Iterator[dict[str, Any] | AwaitDecode]:
    """Events for ``/v1/completions``: chunks when streaming, else one response."""
    prompt = (
        generator.encode_text(request.prompt) if isinstance(request.prompt, str) else request.prompt
    )
    generation = yield Prefill(prompt, request.sampling)
    env = Envelope(model=model, kind="text_completion")
    usage = Usage(prompt_tokens=len(prompt), cached_tokens=generation.cached_tokens)
    output = _Output(generator, request.sampling.stop, thinking=False)
    token_ids: list[int] = []
    finish: protocol.FinishReason = "length"
    for step, _, text, finish_now in _run(generation, output, usage):
        if step.emit:
            token_ids.append(step.token_id)
        if request.stream:
            if looptimers.ENABLED:
                t = looptimers.now()
            chunk = protocol.completion_chunk(
                env,
                text,
                step.token_id if step.emit else None,
                finish_now,
                request.return_token_ids,
            )
            if looptimers.ENABLED:
                looptimers.add("chunk", looptimers.now() - t)
            yield chunk
        finish = finish_now or finish
        if finish_now is None:
            yield AwaitDecode(generation)
    if request.stream:
        if request.include_usage:
            yield protocol.usage_chunk(env, usage)
    else:
        yield protocol.completion_response(
            env, "".join(output.content), token_ids, finish, usage, request.return_token_ids
        )


def chat_job(
    generator: Generator, model: str, request: ChatRequest
) -> Iterator[dict[str, Any] | AwaitDecode]:
    """Events for ``/v1/chat/completions``; reasoning streams as ``reasoning_content``."""
    prompt = generator.encode_chat(request)
    generation = yield Prefill(prompt, request.sampling)
    env = Envelope(model=model, kind="chat.completion")
    usage = Usage(prompt_tokens=len(prompt), cached_tokens=generation.cached_tokens)
    output = _Output(generator, request.sampling.stop, thinking=request.thinking)
    token_ids: list[int] = []
    finish: protocol.FinishReason = "length"
    if request.stream:
        yield protocol.chat_chunk(
            env, {"role": "assistant", "content": ""}, None, None, request.return_token_ids
        )
    for step, reasoning, content, finish_now in _run(generation, output, usage):
        if step.emit:
            token_ids.append(step.token_id)
        if request.stream:
            delta: dict[str, Any] = {"content": content}
            if reasoning:
                delta = {
                    "reasoning_content": reasoning,
                    **({"content": content} if content else {}),
                }
            if looptimers.ENABLED:
                t = looptimers.now()
            chunk = protocol.chat_chunk(
                env,
                delta,
                step.token_id if step.emit else None,
                finish_now,
                request.return_token_ids,
            )
            if looptimers.ENABLED:
                looptimers.add("chunk", looptimers.now() - t)
            yield chunk
        finish = finish_now or finish
        if finish_now is None:
            yield AwaitDecode(generation)
    if request.stream:
        if request.include_usage:
            yield protocol.usage_chunk(env, usage)
    else:
        reasoning_text = "".join(output.reasoning) if request.thinking else None
        yield protocol.chat_response(
            env,
            "".join(output.content),
            reasoning_text,
            token_ids,
            finish,
            usage,
            request.return_token_ids,
        )


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server: _Server

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _send_json(self, status: int, body: dict[str, Any]) -> None:
        payload = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:
        if self.path == "/health":
            self._send_json(
                200, {"status": "ok", **self.server.generator.stats(), **looptimers.summary()}
            )
        elif self.path == "/v1/models":
            card = {
                "id": self.server.model_name,
                "object": "model",
                "owned_by": "vibesys",
                "max_model_len": self.server.generator.max_model_len,
            }
            self._send_json(200, {"object": "list", "data": [card]})
        else:
            self._send_json(404, protocol.error_body(f"no route {self.path}", 404))

    def do_POST(self) -> None:
        routes = {
            "/v1/completions": (protocol.parse_completion, completion_job),
            "/v1/chat/completions": (protocol.parse_chat, chat_job),
        }
        if self.path not in routes:
            self._send_json(404, protocol.error_body(f"no route {self.path}", 404))
            return
        parse, job_of = routes[self.path]
        try:
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"null")
            protocol.check_model(body if isinstance(body, dict) else {}, self.server.model_name)
            request = parse(body)
        except json.JSONDecodeError:
            self._send_json(400, protocol.error_body("request body is not JSON", 400))
            return
        except RequestError as exc:
            self._send_json(exc.status, protocol.error_body(str(exc), exc.status))
            return
        generator, model = self.server.generator, self.server.model_name
        job = Job(run=lambda: job_of(generator, model, request))
        self.server.worker.submit(job)
        self._respond(job, stream=request.stream)

    def _respond(self, job: Job, *, stream: bool) -> None:
        first = job.events.get()
        if isinstance(first, RequestError):
            self._send_json(first.status, protocol.error_body(str(first), first.status))
            self._drain(job)
            return
        if not stream:
            self._send_json(200, first)
            self._drain(job)
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        event = first
        try:
            while event is not _DONE:
                if isinstance(event, RequestError):
                    payload = protocol.error_body(str(event), event.status)
                else:
                    payload = event
                self.wfile.write(b"data: " + json.dumps(payload).encode() + b"\n\n")
                self.wfile.flush()
                event = job.events.get()
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except OSError:  # the client went away: stop generating for it
            job.cancelled.set()
            self._drain(job)

    @staticmethod
    def _drain(job: Job) -> None:
        while job.events.get() is not _DONE:
            pass


class _Server(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        address: tuple[str, int],
        generator: Generator,
        model_name: str,
        gate: Any = None,
        rank: int = 0,
    ) -> None:
        super().__init__(address, _Handler)
        self.generator = generator
        self.model_name = model_name
        self.worker = Worker(generator, gate, rank)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--model", default=CHECKPOINT["model"], help="Hugging Face id or local directory"
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="checkpoint revision (default: the pinned one for the default model)",
    )
    parser.add_argument(
        "--served-model-name", default=None, help="name clients send as `model` (default: --model)"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-model-len", type=int, default=262144)
    parser.add_argument("--prefill-chunk-tokens", type=int, default=2048)
    parser.add_argument(
        "--prefix-cache-tokens",
        type=int,
        default=4194304,
        help="tokens of snapshots kept (counted per entry, though fast-path entries share "
        "full KV pages)",
    )
    parser.add_argument(
        "--prefix-cache-entries",
        type=int,
        default=128,
        help="snapshots kept per replica (0 disables the cache). A request stores up to two "
        "snapshots and one session tail, so 8 sessions on one replica hold 24 per round; 128 "
        "keeps about five rounds of them. On the fast path an entry's own memory is its Gated "
        "DeltaNet states (about 65 MB), so 128 entries stay under --prefix-cache-gib",
    )
    parser.add_argument(
        "--prefix-cache-gib",
        type=float,
        default=16.0,
        help="GiB of snapshot tensors kept by this process, summed over its devices; with "
        "--replicas it is each replica's budget over that replica's GPUs",
    )
    parser.add_argument(
        "--experts-implementation",
        choices=("eager", "grouped_mm", "batched_mm", experts.NAME),
        default=None,
        help=f"Transformers MoE experts implementation (default: {experts.NAME} on ROCm, Transformers' own elsewhere)",
    )
    parser.add_argument(
        "--replicas",
        type=int,
        default=router.env_int("ENGINE_REPLICAS"),
        help="data-parallel model replicas, each its own process behind a session-affinity "
        "router (env ENGINE_REPLICAS; default: one per --gpus-per-replica visible GPUs, "
        "one on the CPU)",
    )
    parser.add_argument(
        "--gpus-per-replica",
        type=int,
        default=router.env_int("ENGINE_GPUS_PER_REPLICA"),
        help="GPUs each replica shards its weights over (env ENGINE_GPUS_PER_REPLICA; "
        "default 2, or all GPUs / --replicas when that is given)",
    )
    parser.add_argument(
        "--replica-devices",
        default=os.environ.get("ENGINE_REPLICA_DEVICES"),
        help="explicit device groups such as '0,1;2,3;4,5;6,7' (env ENGINE_REPLICA_DEVICES)",
    )
    if argv is None:
        argv = sys.argv[1:]
    args = parser.parse_args(argv)
    import torch

    revision = args.revision
    if revision is None and args.model == CHECKPOINT["model"]:
        revision = CHECKPOINT["revision"]
    pair_ports = pair_host.ports()
    if pair_ports:  # two replicas in this process (split prefill; see engine.pair_host)
        pair_host.serve(args, revision, pair_ports)
        return
    gpus = router.visible_gpus()
    layout = router.resolve_layout(
        gpus, args.replicas, args.gpus_per_replica, args.replica_devices or None
    )
    if len(layout) > 1 or (gpus and len(layout[0]) < gpus):
        router.serve(argv, args.host, args.port, layout, hip=bool(torch.version.hip))
        return
    generator = Generator.load(
        args.model,
        revision,
        max_model_len=args.max_model_len,
        chunk_tokens=args.prefill_chunk_tokens,
        prefix_cache_tokens=args.prefix_cache_tokens,
        prefix_cache_entries=args.prefix_cache_entries,
        prefix_cache_bytes=int(args.prefix_cache_gib * 2**30),
        experts_impl=args.experts_implementation,
    )
    server = _Server((args.host, args.port), generator, args.served_model_name or args.model)
    if gpus:
        per_gpu = ", ".join(
            f"{d}: {torch.cuda.memory_allocated(d) / 2**30:.1f} GiB" for d in range(gpus)
        )
        print(f"engine: weights loaded, allocated per GPU {per_gpu}", flush=True)
    print(f"engine: serving {args.model} on http://{args.host}:{args.port}", flush=True)
    try:
        from engine.fast import jit_log

        jit_log.mark_serving()
    except ImportError:
        pass
    server.serve_forever()


if __name__ == "__main__":
    main()
