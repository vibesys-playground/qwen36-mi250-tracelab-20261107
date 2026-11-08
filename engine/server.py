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

With ``ENGINE_ZERO_SYNC`` on (the default; "0" turns it off) the worker thread
does no text work for requests without stop strings: a job yields a ``Render``
per token, and the connection thread that streams the response detokenizes the
token and builds its chunk, while the worker runs the next decode step. Such a
request's end depends on token ids alone (length, end of sequence), which the
worker still decides; the text and chunks are the same as when the worker
renders them. Requests with stop strings, whose end depends on the text, are
rendered on the worker as before.

With ``ENGINE_LOOKAHEAD`` on (the default; "0" turns it off) a fast-path decode
step whose requests are all greedy is launched without waiting
(``Generator.launch``) and collected at the top of the next loop, before new
jobs are admitted, so a prefill still runs right after the step in flight.
When the collected step leaves every one of its requests needing another step
after the tokens it picked (known from the token ids: length, end of sequence;
never for requests with stop strings) and no request joins, the next step is
launched from those picks at once, and the jobs take their tokens while it
runs (one-step lookahead). The batch, the inputs and the order of steps are the
ones the synchronous loop runs, so outputs and session tails do not change.
"""

from __future__ import annotations

import argparse
import functools
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

from engine import experts, protocol, router
from engine.generator import CHECKPOINT, Generation, Generator, Launched, PackedPrefill, Step
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


@dataclass(frozen=True)
class Render:
    """Yielded by a job in place of a response event: the connection thread calls ``make``
    in order, and sends what it returns (None: nothing to send for this token)."""

    make: Callable[[], dict[str, Any] | None]


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
    With ``ENGINE_LOOKAHEAD`` the decode step may be in flight across loops
    (see the module doc).
    """

    def __init__(self, generator: Generator) -> None:
        self._generator = generator
        self._jobs: queue.Queue[Job] = queue.Queue()
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
        g = self._generator
        packer = PackedPrefill(g) if g.packing else None
        prefilling: dict[int, tuple[Job, Iterator[Any]]] = {}
        running: list[tuple[Job, Iterator[Any], Generation]] = []
        # A decode step launched without waiting (``Generator.launch``), collected next loop.
        launched: Launched | None = None
        while True:
            # The step in flight ends first, so a job arriving meanwhile is prefilled right
            # after it, as when the step ran synchronously.
            collected: Launched | None = None
            if launched is not None:
                collected, launched = launched, None
                try:
                    g.collect(collected)
                except Exception as exc:  # the step failed for every request in it
                    for job, _, _ in running:
                        job.events.put(_internal(exc))
                        job.events.put(_DONE)
                    running, collected = [], None
            block = not running and not prefilling
            ready: list[tuple[Job, Iterator[Any], Generation]] = []
            serial: list[tuple[Job, Iterator[Any], Prefill]] = []
            while True:
                try:
                    job = self._jobs.get(block=block)
                except queue.Empty:
                    break
                block = False
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
            if packer is not None:
                for ticket, exc in packer.failed():
                    self._fail(*prefilling.pop(ticket), exc)
                if packer.busy:
                    for ticket, result in packer.step():
                        job, events = prefilling.pop(ticket)
                        if isinstance(result, Exception):
                            self._fail(job, events, result)
                        else:
                            ready.append((job, events, result))
            # One-step lookahead: when the step just collected leaves every one of its
            # requests needing another step (judged on the token ids it picked) and no
            # request joins, the next step is launched from those picks now, and the
            # tokens are handed to the jobs (text, chunks, events) while it runs.
            ahead = (
                collected is not None
                and not ready
                and not serial
                and [gen for _, _, gen in running] == collected.generations
                and all(gen.runs_past_picked() for gen in collected.generations)
                and g.launchable(collected.generations)
            )
            if ahead:
                try:
                    launched = g.launch(collected.generations, ahead=True)
                except Exception as exc:  # the step failed for every request in it
                    for job, _, _ in running:
                        job.events.put(_internal(exc))
                        job.events.put(_DONE)
                    running, ahead = [], False
            waiting = []
            for job, events, _ in running:
                want = self._advance_all(job, events)
                if isinstance(want, AwaitDecode):
                    waiting.append((job, events, want.generation))
            for job, events, generation in ready:
                want = self._advance_all(job, events, generation)
                if isinstance(want, AwaitDecode):
                    waiting.append((job, events, want.generation))
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
            running = waiting
            if ahead or not waiting:
                # Ahead: the launched step holds exactly these requests (a job cancelled
                # meanwhile rides along; the next loop sees the batch changed).
                continue
            try:
                launched = g.launch([generation for _, _, generation in waiting])
            except Exception as exc:  # the step failed for every request in it
                for job, _, _ in waiting:
                    job.events.put(_internal(exc))
                    job.events.put(_DONE)
                running = []


class _Output:
    """Splits generated tokens into reasoning and content text, with stop strings applied."""

    def __init__(
        self, generator: Generator, stops: tuple[str, ...], thinking: bool, tokenizer: Any = None
    ) -> None:
        self._generator = generator
        self._tokenizer = tokenizer if tokenizer is not None else generator.tokenizer
        self._stops = stops
        self.in_reasoning = thinking
        self._detok = Detokenizer(self._tokenizer)
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
            self._detok = Detokenizer(self._tokenizer)
            return tail, ""
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


def _deferred(generator: Generator, sampling: Sampling) -> bool:
    """Whether a request's text is rendered on its connection thread (see the module doc)."""
    return generator.zero_sync and not sampling.stop


def _output(generator: Generator, sampling: Sampling, thinking: bool) -> _Output:
    tokenizer = generator.text_tokenizer if _deferred(generator, sampling) else None
    return _Output(generator, sampling.stop, thinking, tokenizer)


def _render(
    output: _Output, step: Step, finish: protocol.FinishReason | None
) -> tuple[str, str]:
    """(reasoning delta, content delta) of one step of a request without stop strings."""
    reasoning, content = output.push(step)
    if finish is not None:
        r_tail, c_tail = output.flush()
        return reasoning + r_tail, content + c_tail
    return reasoning, content


def _run(
    generation: Generation, output: _Output, usage: Usage, defer: bool = False
) -> Iterator[tuple[Step, str, str, protocol.FinishReason | None]]:
    """Yield (step, reasoning delta, content delta, finish) per token until the request ends.

    The caller yields ``AwaitDecode`` after each unfinished step, so the worker
    has decoded the token before this generator resumes. With ``defer`` the
    text is not rendered here (both deltas are empty; see ``_render``).
    """
    while True:
        step = generation.next_step()
        usage.completion_tokens += int(step.emit)
        if defer:
            if step.finish is not None:
                generation.retire()
                yield step, "", "", step.finish
                return
            yield step, "", "", None
            continue
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
) -> Iterator[dict[str, Any] | AwaitDecode | Render]:
    """Events for ``/v1/completions``: chunks when streaming, else one response."""
    prompt = (
        generator.encode_text(request.prompt) if isinstance(request.prompt, str) else request.prompt
    )
    generation = yield Prefill(prompt, request.sampling)
    env = Envelope(model=model, kind="text_completion")
    usage = Usage(prompt_tokens=len(prompt), cached_tokens=generation.cached_tokens)
    output = _output(generator, request.sampling, thinking=False)
    defer = _deferred(generator, request.sampling)
    token_ids: list[int] = []
    finish: protocol.FinishReason = "length"

    def chunk(step: Step, text: str, finish_now: protocol.FinishReason | None) -> dict[str, Any]:
        return protocol.completion_chunk(
            env, text, step.token_id if step.emit else None, finish_now, request.return_token_ids
        )

    def render(step: Step, finish_now: protocol.FinishReason | None) -> dict[str, Any] | None:
        _, text = _render(output, step, finish_now)
        return chunk(step, text, finish_now) if request.stream else None

    for step, _, text, finish_now in _run(generation, output, usage, defer):
        if step.emit:
            token_ids.append(step.token_id)
        if defer:
            yield Render(functools.partial(render, step, finish_now))
        elif request.stream:
            yield chunk(step, text, finish_now)
        finish = finish_now or finish
        if finish_now is None:
            yield AwaitDecode(generation)
    if request.stream:
        if request.include_usage:
            yield protocol.usage_chunk(env, usage)
        return

    def response(finish: protocol.FinishReason) -> dict[str, Any]:
        return protocol.completion_response(
            env, "".join(output.content), token_ids, finish, usage, request.return_token_ids
        )

    yield Render(functools.partial(response, finish)) if defer else response(finish)


def chat_job(
    generator: Generator, model: str, request: ChatRequest
) -> Iterator[dict[str, Any] | AwaitDecode | Render]:
    """Events for ``/v1/chat/completions``; reasoning streams as ``reasoning_content``."""
    prompt = generator.encode_chat(request)
    generation = yield Prefill(prompt, request.sampling)
    env = Envelope(model=model, kind="chat.completion")
    usage = Usage(prompt_tokens=len(prompt), cached_tokens=generation.cached_tokens)
    output = _output(generator, request.sampling, thinking=request.thinking)
    defer = _deferred(generator, request.sampling)
    token_ids: list[int] = []
    finish: protocol.FinishReason = "length"
    if request.stream:
        yield protocol.chat_chunk(
            env, {"role": "assistant", "content": ""}, None, None, request.return_token_ids
        )

    def chunk(
        step: Step, reasoning: str, content: str, finish_now: protocol.FinishReason | None
    ) -> dict[str, Any]:
        delta: dict[str, Any] = {"content": content}
        if reasoning:
            delta = {"reasoning_content": reasoning, **({"content": content} if content else {})}
        return protocol.chat_chunk(
            env, delta, step.token_id if step.emit else None, finish_now, request.return_token_ids
        )

    def render(step: Step, finish_now: protocol.FinishReason | None) -> dict[str, Any] | None:
        reasoning, content = _render(output, step, finish_now)
        return chunk(step, reasoning, content, finish_now) if request.stream else None

    for step, reasoning, content, finish_now in _run(generation, output, usage, defer):
        if step.emit:
            token_ids.append(step.token_id)
        if defer:
            yield Render(functools.partial(render, step, finish_now))
        elif request.stream:
            yield chunk(step, reasoning, content, finish_now)
        finish = finish_now or finish
        if finish_now is None:
            yield AwaitDecode(generation)
    if request.stream:
        if request.include_usage:
            yield protocol.usage_chunk(env, usage)
        return

    def response(finish: protocol.FinishReason) -> dict[str, Any]:
        reasoning_text = "".join(output.reasoning) if request.thinking else None
        return protocol.chat_response(
            env,
            "".join(output.content),
            reasoning_text,
            token_ids,
            finish,
            usage,
            request.return_token_ids,
        )

    yield Render(functools.partial(response, finish)) if defer else response(finish)


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
            self._send_json(200, {"status": "ok", **self.server.generator.stats()})
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

    @staticmethod
    def _next(job: Job) -> Any:
        """The job's next event, with ``Render`` events made here (on this connection thread)."""
        while True:
            event = job.events.get()
            if not isinstance(event, Render):
                return event
            try:
                made = event.make()
            except Exception as exc:  # noqa: BLE001 - reported to this request's client
                return _internal(exc)
            if made is not None:
                return made

    def _respond(self, job: Job, *, stream: bool) -> None:
        first = self._next(job)
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
                event = self._next(job)
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

    def __init__(self, address: tuple[str, int], generator: Generator, model_name: str) -> None:
        super().__init__(address, _Handler)
        self.generator = generator
        self.model_name = model_name
        if generator.zero_sync:
            # Connection threads render text while the worker waits on the device; hand the
            # GIL back to the worker sooner than the default 5 ms once its wait ends.
            sys.setswitchinterval(0.0005)
        self.worker = Worker(generator)


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

    gpus = router.visible_gpus()
    layout = router.resolve_layout(
        gpus, args.replicas, args.gpus_per_replica, args.replica_devices or None
    )
    if len(layout) > 1 or (gpus and len(layout[0]) < gpus):
        router.serve(argv, args.host, args.port, layout, hip=bool(torch.version.hip))
        return
    revision = args.revision
    if revision is None and args.model == CHECKPOINT["model"]:
        revision = CHECKPOINT["revision"]
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
    server.serve_forever()


if __name__ == "__main__":
    main()
