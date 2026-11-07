# The task image (schema 0.2): the judge. It runs the service checker and the TraceLab replay
# against the candidate's server at VIBESYS_SERVICE_URL; it has no GPU and no candidate code.
# The candidate's server runs in its own image, built from the candidate's commit
# (vibesys.input.toml, [candidate]).
#
# It holds what the earlier [evaluator] (vibesys-evaluator-request-factory 0.1.0) supplied:
# Request Factory's session_runner, built at the evaluator's pinned revision, and its
# adapter.py, unchanged. checker.py, replay.py and recorded_sessions.csv are byte-for-byte
# copies of .vibesys/tasks/default/accuracy_checker/checker.py and
# .vibesys/shared/tracelab/{replay.py,recorded_sessions.csv}, kept in this directory so that
# they are part of the task's version and of this image, never read from a candidate's checkout.

# session_runner from Request Factory at the evaluator's pinned revision. req-frontend links
# OpenSSL through openssl-sys, which locates it with pkg-config, and builds C++ with cc-rs.
FROM rust:1.90-slim-bookworm@sha256:64232e656c058f4468e8d024e990acff04f0fd5a5c0a88a574dc37773d7325c9 AS request-factory
ARG REQUEST_FACTORY_REV=89dce4a64ae12e7084fbe464ddda882fa0a4c482
RUN apt-get update \
 && apt-get install -y --no-install-recommends git ca-certificates pkg-config libssl-dev g++ \
 && rm -rf /var/lib/apt/lists/*
RUN git init -q /src \
 && git -C /src fetch -q --depth 1 https://github.com/uw-syfi/request-factory "${REQUEST_FACTORY_REV}" \
 && git -C /src checkout -q FETCH_HEAD \
 && test "$(git -C /src rev-parse HEAD)" = "${REQUEST_FACTORY_REV}" \
 && cargo install --locked --path /src --bin session_runner --root /out \
 && rm -rf /src /usr/local/cargo/registry

FROM python:3.12.12-slim-bookworm@sha256:593bd06efe90efa80dc4eee3948be7c0fde4134606dd40d8dd8dbcade98e669c
RUN apt-get update \
 && apt-get install -y --no-install-recommends libssl3 ca-certificates \
 && rm -rf /var/lib/apt/lists/*
# The checker's one dependency, at the project's pin (requirements-agentx.txt).
RUN pip install --no-cache-dir httpx==0.28.1
COPY --from=request-factory /out/bin/session_runner /usr/local/bin/session_runner
COPY . /task
# A job runs as the invoking user, whose HOME may not exist; the image's files are read-only
# under Apptainer.
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HOME=/tmp
RUN session_runner --help >/dev/null && python -I -c "import httpx"
