# Backpressure — a beginner's guide to this module

> **Read this first.** This document explains *what* backpressure is, *why* this
> folder exists, and *how* the pieces fit together — in plain paragraphs, from
> the mental model down to the implementation details. If you have never touched
> a backpressure system before, start at the top and read straight through. By
> the end you should be able to reason about it well enough to change it safely.

---

## 1. The problem, in one picture

Imagine a small restaurant kitchen with three cooks. On a normal evening, orders
come in at a pace the cooks can handle, meals go out, everyone is happy. Now
imagine a coach party of two hundred people walks in and every one of them orders
at once. The waiter has two choices. He can **accept every order and shout them
all into the kitchen** — in which case the cooks are buried, tickets pile up,
*every* meal comes out late (including the regulars who ordered before the rush),
and half the food is cold or wrong by the time it's plated. Or the waiter can
walk up to the back of the line and say, politely and immediately, *"I'm sorry,
the kitchen is full right now — please come back in ten minutes."*

That second choice is **backpressure**. It is the discipline of **saying "no"
quickly when the system is already overloaded**, instead of accepting work you
cannot actually do. The counter-intuitive insight — and the reason beginners
find it strange — is that **rejecting some requests fast is kinder to everyone
than accepting all requests slowly.** A fast, honest `503 "try again in 10
seconds"` lets a client retry sanely. A request accepted into an overwhelmed
system just times out later, having consumed resources and helped no one.

This module is the waiter at the back of the line for our token-allocation
service.

---

## 2. Where it sits in the request path

Our hot endpoint is `POST /acquire` — a client calls it to reserve token capacity
before making an LLM call. That endpoint is protected by a series of cheap
guard checks that run **before** any expensive work happens. Backpressure is one
of those guards. Conceptually the order is:

```
request → validate input → rate-limit check → BACKPRESSURE CHECK → do the real work
                                                    │
                                            (this module)
```

The key property is that the backpressure check is **fast and cheap**. It reads
a handful of already-collected health signals and makes a yes/no decision in
microseconds. It never does slow work of its own — the whole point is to protect
the slow work from happening when the system can't afford it. If backpressure
says "reject," the request is turned away with a `503` and never reaches the
token-allocation logic at all.

> **A note on naming.** Backpressure is **Layer 1** in the resilience stack
> (Layer 0 = rate limiter, Layer 2 = circuit breaker), per `SYSTEM_DESIGN.md`.
> An earlier endpoint comment mislabeled it "Layer 2"; that has been corrected so
> every file now agrees. See [PROPOSED_DESIGN.md](./PROPOSED_DESIGN.md) for the
> reasoning.

---

## 3. The mental model: three gauges and a rule

The cleanest way to hold this module in your head is as a **dashboard of gauges
plus a decision rule**.

There are **three gauges** (we call them *probes* in the code). Each one measures
exactly one thing about the health of the system, and nothing else:

1. **Queue depth** — how many token-allocation jobs are currently waiting in line
   to be processed. A long line means the workers are falling behind.
2. **Database pool utilization** — what percentage of our database connections are
   currently in use. Near 100% means new requests will have to wait for a free
   connection.
3. **Database circuit-breaker state** — whether the "circuit breaker" protecting
   the database has *tripped*. A circuit breaker is a separate safety device that
   flips to "open" after repeated database failures, and while it's open we know
   the database is effectively down.

Then there is **one rule**: *if any gauge is in the red, reject the request; the
first gauge to read red decides the response.* That's the entire logic. Everything
else in this folder is machinery to read those three gauges accurately, apply the
rule cleanly, and translate the outcome into an HTTP response.

Hold onto that image — **three gauges, one rule** — and the rest of the code is
just the honest implementation of it.

---

## 4. The two flows (this is the part that confuses people)

Here is the single most important structural fact about this folder: **there are
two completely separate flows living in it, and they only ever meet at one shared
Redis key.** Beginners get lost because they assume it's all one call chain. It
isn't.

### Flow A — DECIDE (runs on every request, inside the API process)

This is the flow you care about first. When a request hits `/acquire`, the API
reads the three gauges, applies the rule, and either lets the request through or
raises a `503`. This flow **reads** the queue-depth number.

```
/acquire request
      │
      ▼
read gauge 1 (queue depth)  ─► in the red? ─► yes ─► raise 503, done
      │ no
      ▼
read gauge 2 (DB pool)      ─► in the red? ─► yes ─► raise 503, done
      │ no
      ▼
read gauge 3 (breaker)      ─► in the red? ─► yes ─► raise 503, done
      │ no
      ▼
allow request through
```

### Flow B — PUBLISH (runs on a timer, inside a background worker)

Here's the subtlety. The API process **cannot directly measure** how many jobs
are waiting in the RabbitMQ queue — that measurement lives in a different process
(the background worker), and doing it on every request would be far too slow
anyway. So a background task runs every few seconds, measures the real queue
length once, and **writes** that number into a Redis key with a short expiry.
Flow A then just reads that pre-computed number instantly.

```
every ~5 seconds (background worker)
      │
      ▼
measure RabbitMQ queue length
      │
      ▼
write the number into Redis key "token_alloc:queue_depth" (expires shortly after)
```

**The two flows never call each other.** Flow B leaves a note on the fridge;
Flow A reads the note. The note is the Redis key. This indirection is deliberate:
it keeps the per-request check O(microseconds) instead of O(a-network-round-trip-
to-RabbitMQ). Once you see that the queue-depth gauge is *read* in Flow A but
*written* in Flow B, the whole folder stops feeling tangled.

---

## 5. Walking the code, in the order it runs

Now let's map the mental model onto the actual files. I'll follow Flow A from the
outside in, because that's the flow a request actually travels.

**The front door — [`dependency.py`](./dependency.py).** FastAPI endpoints declare
their guards with `Depends(...)`. The `/acquire` endpoint declares
`Depends(backpressure_dependency)`, so *this function is the real entry point of
the entire module.* It is deliberately tiny: it asks the evaluator for a decision,
then hands that decision to the HTTP translator. That's it. It carries a `Request`
parameter it doesn't use yet — that's a reserved seat for future features like
letting privileged callers bypass backpressure, without having to change the
function's signature later.

**The brain — [`evaluator.py`](./evaluator.py).** This is where the *rule* lives.
It calls the three probes **in a fixed priority order** and returns the moment one
of them reads red. The order is not arbitrary and it matters: queue depth is
checked first because it's the broadest early-warning signal, then DB pool
saturation, then the circuit breaker as the most severe "database is actually
down" signal. Whichever fires first determines the `reason` and the `Retry-After`
the client receives. Crucially, the evaluator returns a **plain data object**, not
an HTTP response — it decides *what is true*, not *how to phrase it to a client*.
That separation is what lets you unit-test the decision logic without spinning up
a web server.

**The three gauges — the probes.** These live together in the
[`probes/`](./probes/) sub-package, so the folder tree mirrors the model. Each
probe is a small module whose only job is to read one signal and return a clean
number (or `None` if it can't read it):

- [`probes/queue_depth.py`](./probes/queue_depth.py) reads the queue-depth
  number that Flow B parked in Redis. It also contains the small helper that
  *estimates* how long the client should wait (more on that math in §7).
- [`probes/db_pool.py`](./probes/db_pool.py) asks the database
  connection pool how many connections are checked out versus its total size, and
  returns that as a percentage.
- [`probes/circuit_state.py`](./probes/circuit_state.py) reads a read-only snapshot of
  the database circuit breaker — is it open, how long until it retries, etc. It is
  strictly a *reader*; it never trips or resets the breaker.

**The translator — [`http_response.py`](./http_response.py).**
Once the evaluator has produced a decision, this module turns it into the actual
web response. If the decision says "don't reject," it does nothing and the request
proceeds. If it says "reject," it raises a `503 Service Unavailable` carrying a
`Retry-After` header, a machine-readable reason code, and a human-friendly message.
This is the *only* file in the module that knows anything about HTTP. Swap out the
web framework tomorrow and this is the single file you'd rewrite.

**The shared vocabulary — [`constants.py`](./constants.py).** The Redis key name and
the reason-code strings ("`queue_depth_exceeded`", etc.) are defined here once, so
that Flow A (which reads the key) and Flow B (which writes it) can never disagree
about the spelling. Small file, real purpose.

**Flow B's worker — [`publisher.py`](./publisher.py).** This
is the background half. It connects to RabbitMQ, measures the work-queue length,
and writes it into the shared Redis key with a short time-to-live. The TTL is
deliberately a small multiple of the publish interval, so that if the publisher
*stops* running, the number simply **expires and disappears** rather than going
stale — and the queue-depth gauge then reads "unknown" instead of lying with an
old value. It's invoked on a schedule from the token-maintenance task runner, not
from any web request.

**The typed decision itself — `BackpressureDecision`.** This lives one folder up in
`app/models/resilience_models.py`, but it's the contract that ties the evaluator
to the translator. It's a small validated record: *should we reject? why? how many
seconds should the client wait? what were the observed numbers?* It even enforces
an invariant — if `should_reject_request` is true, a `retry_after_seconds` value
**must** be present — so a malformed "reject with no retry guidance" decision is
impossible to construct by accident.

---

## 6. The design principle that governs everything: fail *open*

This is the single most important behavioral rule in the module, and it's worth
its own section because it's easy to get backwards.

Every probe is wrapped so that **if it cannot read its signal, it returns "I don't
know" (`None`) rather than raising an error.** And the evaluator treats "I don't
know" as **"don't block the request."** In other words, when the backpressure
system itself is broken — Redis is unreachable, the pool object is missing,
something throws — the system **lets requests through**, not turns them away.

Why? Because backpressure is a *protective* mechanism, not a *critical-path*
one. If a bug or an outage in the protection layer could reject legitimate traffic,
then the safety device becomes the outage. We would rather occasionally fail to
protect the system during a rare double-fault than have the protector itself take
the service down. So the guiding rule is: **when in doubt, allow.** You'll see this
everywhere — `try/except` blocks that log and `return None`, the malformed-payload
path that returns `None`, the missing-pool path that returns `None`. That is not
sloppiness; it is the fail-open contract, applied consistently.

The one thing that is *not* forgiving is the decision contract itself: a
"reject" decision with no `Retry-After` is treated as a programming error and
rejected loudly, because a client told to retry with no idea *when* is a bug we
want to catch in tests, not in production.

---

## 7. How the "come back in N seconds" number is computed

When we reject a request, we don't just say "no" — we tell the client *when* to
try again, via the `Retry-After` header. For the queue-depth case this number is
**estimated**, and the estimate is a nice small piece of arithmetic worth
understanding because it shows how the config knobs connect to real behavior.

The idea: we consider the queue "healthy" up to a *safe depth*, which is a
fraction of the maximum. Anything above that safe depth is *excess* that has to
drain off before the system is comfortable again. We assume the workers drain the
queue at a roughly constant rate, so the wait time is simply *excess ÷ drain
rate*, clamped to a sane range.

Using the real default config values:

| Knob | Default | Meaning |
|---|---|---|
| `bp_max_queue_depth` | `10000` | Above this, we reject outright |
| `bp_queue_safe_depth_ratio` | `0.8` | "Healthy" ceiling = 80% of max = **8000** |
| `bp_drain_rate_per_second` | `400` | Assumed jobs drained per second |
| `bp_retry_after_cap_seconds` | `60` | Never tell a client to wait longer than this |

Worked example — suppose the queue depth is **12000**:

```
safe_depth   = 10000 × 0.8      = 8000
excess       = 12000 − 8000     = 4000
drain_seconds= 4000 ÷ 400       = 10
Retry-After  = clamp(10, 1..60) = 10   →  "come back in 10 seconds"
```

So a client hitting a saturated queue is told to wait ten seconds — long enough
for the backlog to meaningfully drain, short enough to be reasonable, and never
more than the 60-second cap. The DB-pool and circuit-breaker rejections use fixed
`Retry-After` values instead (`bp_db_pool_retry_after_seconds = 5`, and the
breaker's own recovery timeout of `30s`) because those conditions clear on their
own timers rather than by draining a measurable backlog.

---

## 8. The configuration knobs, in one place

All of these live in `app/core/config/yaml/resiliency.yaml` and can be overridden
per-environment via `.env`. You tune behavior here without touching code.

| Setting | Default | What it controls |
|---|---|---|
| `bp_max_queue_depth` | `10000` | Queue length above which we reject |
| `bp_db_pool_saturation_pct` | `90` | DB pool % at/above which we reject |
| `bp_drain_rate_per_second` | `400` | Assumed drain rate for the Retry-After estimate |
| `bp_retry_after_cap_seconds` | `60` | Upper bound on any queue Retry-After |
| `bp_queue_safe_depth_ratio` | `0.8` | Fraction of max considered "healthy" |
| `bp_db_pool_retry_after_seconds` | `5` | Fixed Retry-After for pool saturation |
| `bp_queue_depth_publish_interval_secs` | `5` | How often Flow B republishes the depth |

---

## 9. Why the code is split into small files (and a `probes/` sub-package)

A beginner's natural reaction to this folder is "why so many files for something
you described in three gauges and one rule?" The answer is **separation of
concerns**, and each split earns its place:

- **Reading a signal** (the [`probes/`](./probes/) sub-package) is separated from
  **deciding** (the evaluator) so you can test each gauge in isolation and swap a
  signal source without touching the rule. The three gauges live together under
  `probes/` so the directory tree itself teaches the "read → decide → translate"
  flow.
- **Deciding** (the evaluator) is separated from **presenting** (the HTTP translator)
  so the decision logic has zero knowledge of web frameworks and can be tested as
  pure data-in/data-out.
- **The request-time flow** is separated from **the background publish flow** because
  they run in different processes, on different triggers, and share only a key.
- **The wiring** (the dependency) is a thin seam so the framework integration is one
  small, obvious file.

The reward for this discipline is that *every file has exactly one reason to
change*. Change how we measure the pool? Only `probes/db_pool.py` moves. Change the
`503` body? Only the translator moves. Change the ordering of checks? Only the
evaluator moves. That is the payoff, and it's why the structure is worth keeping
even though it looks like "a lot of files" at first glance.

This folder was cleaned up to match that model: a dead compatibility class and a
type-checker shim that used to obscure the picture have been removed, and the
probes were grouped into `probes/`. [PROPOSED_DESIGN.md](./PROPOSED_DESIGN.md)
records exactly what changed and why.

---

## 10. A thirty-second recap

- Backpressure = **say "no" fast when the system is already overloaded.** A quick,
  honest `503` beats a slow, doomed acceptance.
- The model is **three gauges** (queue depth, DB pool, DB breaker) and **one rule**
  (first gauge in the red wins → `503`).
- There are **two flows**: one *reads* the gauges on every request (Flow A), one
  *writes* the queue-depth gauge on a timer (Flow B). They meet only at a Redis key.
- The evaluator produces a **plain decision object**; the translator turns it into
  HTTP. Reading, deciding, and presenting are three separate jobs.
- The prime directive is **fail open**: if the protector can't read a signal, it
  lets the request through rather than risk becoming the outage.
- The Retry-After for queue rejection is **excess ÷ drain rate, clamped** — real
  arithmetic driven by real config knobs.

Once those five bullets feel obvious, you understand this module well enough to
change it. Welcome aboard.
