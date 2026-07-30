# Circuit Breaker — a beginner's guide to this module

> **Read this first.** This explains *what* a circuit breaker is, *why* this folder
> exists, *how* the pieces work, *where* the breakers are actually used, what every
> configuration value means, and whether you can reuse this in another project. It
> is written in plain paragraphs for someone who knows the term "circuit breaker"
> only vaguely. Read top to bottom once and it will stick.

---

## 1. The problem, by analogy

Your house has a circuit breaker in the electrical panel. When something goes
wrong — a short circuit, too much current — the breaker **trips** and cuts the
power to that line. It seems destructive ("why would I want my power cut?"), but
it prevents a far worse outcome: an overheated wire starting a fire. Once things
are safe, you flip it back on.

A **software circuit breaker** does exactly the same thing for calls to a
dependency your service relies on — here, PostgreSQL, Redis, and RabbitMQ. When a
dependency starts failing badly, the breaker **trips** and your service stops
calling it for a while. That sounds destructive too, until you see the fire it
prevents.

### The fire: cascading failure

Imagine Redis goes down and each call to it hangs for a 5-second timeout before
failing. Without a breaker:

- Every incoming request tries Redis → every request blocks for 5 seconds.
- Those 5-second waits pile up: threads, connections, and memory are all held
  hostage waiting on a dependency that is never going to answer.
- Your service runs out of threads/connections and becomes unresponsive **for
  everything** — even requests that had nothing to do with Redis.

One sick dependency just took down the entire service. That is a *cascading
failure*, and it is one of the most common ways distributed systems fall over.

### The fix

With a breaker in front of Redis: after a handful of consecutive failures, the
breaker **opens** and every subsequent call **fails instantly** (microseconds, no
5-second wait, no thread held) with a `CircuitBreakerError`. Your service stays
responsive, the caller falls back to a safe alternative, and Redis gets breathing
room to recover instead of being hammered while it's down. That is the whole point:
**fail fast and cheap instead of slow and expensive, and give the sick dependency
room to heal.**

---

## 2. The three states (the heart of it)

A circuit breaker is a tiny state machine with three states. Understanding these
three states *is* understanding circuit breakers — everything else is detail.

```
                    failures reach the threshold
        ┌────────┐  ───────────────────────────▶  ┌────────┐
        │ CLOSED │                                 │  OPEN  │
        │ normal │  ◀───────────────────────────   │ tripped│
        └────────┘        probe call succeeds      └────────┘
             ▲                                          │
             │                                          │ after `timeout_duration`
             │ probe succeeds                           ▼  seconds have passed
             │                                    ┌───────────┐
             └────────────────────────────────    │ HALF_OPEN │
                        probe call fails ─────────▶│  testing  │
                        (back to OPEN)             └───────────┘
```

- **CLOSED — normal operation.** Calls flow straight through to the dependency.
  The breaker quietly *counts consecutive failures*. Every success resets the
  count to zero. If the count reaches the configured **threshold** (`fail_max`),
  the breaker trips to OPEN.

- **OPEN — tripped.** Calls are **rejected immediately** with a
  `CircuitBreakerError`; the dependency is not even contacted. The breaker stays
  OPEN for `timeout_duration` seconds (the "recovery timeout"). This is the state
  that saves you from the cascading failure.

- **HALF_OPEN — cautiously testing recovery.** Once the recovery timeout elapses,
  the breaker lets **one** call through as a probe. If that probe **succeeds**, the
  dependency looks healthy again → back to CLOSED (fully recovered). If the probe
  **fails**, the dependency is still sick → straight back to OPEN for another
  timeout. This prevents a "thundering herd" from slamming a still-fragile
  dependency the instant the timeout expires.

That is the complete behavior. The `aiobreaker` library implements this machine;
this folder configures and wires it.

---

## 3. What THIS module actually is (and is not)

A crucial clarification that removes most of the confusion: **this folder does not
implement the state machine above.** That algorithm lives in the third-party
`aiobreaker` library. What this folder provides is everything *around* the library:

- **A factory** that builds three named breakers — `postgres`, `redis`,
  `rabbitmq` — each tuned with its own threshold and timeout.
- **A storage decision** for each breaker: where its CLOSED/OPEN state is kept
  (in local memory vs. in Redis, shared across all server replicas).
- **A singleton registry** so everyone in the app shares the *same* breaker
  instance per dependency (if two places used different `redis` breaker objects,
  they'd count failures separately and neither would trip correctly).
- **A logging listener** that records every state change, failure, and success.

So the mental model is: *"a small, well-organized factory that hands out three
pre-configured, shared circuit breakers, and remembers their state in the right
place."*

---

## 4. The files, one by one

- [`breaker_state.py`](./breaker_state.py) — a small `CircuitBreakerState` enum
  (`CLOSED` / `OPEN` / `HALF_OPEN`) using *our* normalized string values
  (lowercase, hyphenated: `"half-open"`). Its only job is to insulate the rest of
  our code from aiobreaker's own enum, so callers compare against stable values we
  control.

- [`breaker_listener.py`](./breaker_listener.py) — a `CircuitBreakerListener` that
  aiobreaker calls on every event. It emits a structured log line on each **state
  transition** (e.g. `CLOSED -> OPEN`), each **failure** (with the running
  `count/threshold`), and each **success**. This is your observability window — in
  production these logs are how you *see* a breaker trip. One shared, stateless
  instance is attached to every breaker.

- [`breaker_storage.py`](./breaker_storage.py) — decides **where each breaker's
  state is stored**, and owns a synchronous Redis client for that purpose. The
  `build_breaker_storage(name)` factory returns either in-memory storage (for
  `postgres`) or Redis-backed storage (for `redis`/`rabbitmq`). Section 6 explains
  why they differ. It also builds/closes the singleton Redis client used for that
  storage.

- [`breaker_registry.py`](./breaker_registry.py) — the **factory + cache**. Its
  `create_circuit_breaker(...)` builds a breaker once and caches it (thread-safe,
  via double-checked locking); the three `get_db/redis/rmq_circuit_breaker()`
  helpers call it with each dependency's configured threshold and timeout;
  `get_circuit_breaker_states()` reports every breaker's current state for health
  endpoints.

- [`__init__.py`](./__init__.py) — the public front door that re-exports the
  symbols other packages import (`get_db_circuit_breaker`, `CircuitBreakerState`,
  etc.).

---

## 5. Where the breakers are actually *consumed* (the part you were missing)

The factory hands out breakers, but the **protection** only happens where real
calls get wrapped. There are exactly three such sites, and they all use the **same
one-line idiom**:

```python
result = await breaker.call_async(the_real_operation, *args)
```

`call_async` is the gate. If the breaker is CLOSED or HALF_OPEN, it runs
`the_real_operation` and records the outcome (success resets the count; an
exception increments it). If the breaker is OPEN, it **skips the operation
entirely** and raises `aiobreaker.CircuitBreakerError`. So every caller follows the
same shape: *try through the breaker; catch `CircuitBreakerError`; fall back.*

| Breaker | Wrapped at | What it protects | On trip (OPEN), the caller… |
|---|---|---|---|
| `postgres` (DB) | [`services/token_acquisition_service.py`](../../services/token_acquisition_service.py) | the synchronous PostgreSQL fallback path | logs and returns a degraded result instead of hanging on a dead DB |
| `redis` | [`redis_token_counter/counter_service.py`](../redis_token_counter/counter_service.py) | the Redis Lua fast-path (reserve/release/get counters) | short-circuits the fast path and falls back |
| `rabbitmq` | [`token_queue/publisher.py`](../token_queue/publisher.py) | publishing allocation messages to RabbitMQ | skips the publish and takes the fallback route |

There is also one **reader** that does *not* wrap a call: the backpressure Layer‑1
probe [`backpressure/probes/circuit_state.py`](../backpressure/probes/circuit_state.py)
just *reads* the DB breaker's state to decide whether to shed load early. Reading
state and wrapping calls are different jobs — only the three sites above actually
drive the breaker.

---

## 6. The three breakers and what their configuration means

All three are created from the same factory but tuned differently. The values live
in `app/core/config/yaml/resiliency.yaml` (overridable per environment via `.env`).

| Breaker | Threshold (`fail_max`) | Recovery (`timeout_duration`) | State storage |
|---|---|---|---|
| `postgres` | `cb_db_failure_threshold` = **5** | `cb_db_recovery_timeout` = **30s** | **in-memory** (local) |
| `redis` | `cb_redis_failure_threshold` = **3** | `cb_redis_recovery_timeout` = **10s** | **Redis** (shared) |
| `rabbitmq` | `cb_rmq_failure_threshold` = **3** | `cb_rmq_recovery_timeout` = **15s** | **Redis** (shared) |

**What the two numbers mean:**

- **Threshold (`fail_max`)** — how many *consecutive* failures trip the breaker.
  Higher = more tolerant before cutting the dependency off. PostgreSQL gets **5**
  (it's the durable source of truth and the last-resort fallback, so we tolerate a
  few blips before cutting it). Redis and RabbitMQ get **3** (they front fast,
  optional paths — trip sooner, protect faster).

- **Recovery timeout (`timeout_duration`)** — how long the breaker stays OPEN
  before it allows a HALF_OPEN probe. Shorter = retries the dependency sooner.
  Redis is **10s** (fast path, we want it back quickly); RabbitMQ **15s**;
  PostgreSQL **30s** (give a struggling database real time to recover rather than
  poking it every few seconds).

**Why storage differs (this is the clever part):**

- The **`postgres` breaker uses in-memory storage** so that DB protection **does
  not depend on Redis being up**. If it stored its state in Redis, a Redis outage
  would break your *database* breaker too — coupling two failures together. Keeping
  it local means the DB breaker keeps working no matter what Redis is doing. The
  trade-off: each server replica has its own independent view of the DB breaker.

- The **`redis` and `rabbitmq` breakers use Redis-backed storage**, so their
  OPEN/CLOSED state is **shared across all API replicas**. If one replica trips the
  `redis` breaker, every other replica instantly sees it as OPEN too — the whole
  fleet stops hammering a sick Redis together, rather than each replica having to
  discover the outage on its own.

---

## 7. The design decision that surprises people: this fails **CLOSED**

If you also read the backpressure guide, note the deliberate contrast. Backpressure
**fails *open*** — "when unsure, let the request through." The circuit breaker does
the **opposite**: it **fails *closed*** — "when unsure, assume the dependency is
down and block."

You can see this in two places:

1. The Redis-backed breakers are created with
   `fallback_circuit_state = OPEN`. If the breaker *cannot even read its own state
   from Redis*, it assumes **OPEN** (blocked).
2. In the registry, if building the storage throws for any reason, the fallback is
   a local breaker that starts **OPEN**.

Why the opposite choice? Because the two mechanisms guard against opposite risks.
Backpressure decides *whether to admit load*, and wrongly rejecting a healthy
request is a real cost, so it leans toward admitting. A circuit breaker guards
*against calling something that may be broken*; if it can't tell whether the
dependency is healthy, the safe assumption is "it might be on fire — don't touch
it." Assume-the-worst is correct for protection, assume-the-best is correct for
admission. Holding both ideas at once is a sign you actually understand resilience
design.

---

## 8. A full walk-through of one trip

Putting it all together, here is the life of the `redis` breaker during an outage:

1. **Healthy (CLOSED).** `counter_service` reserves tokens via
   `redis_breaker.call_async(reserve_op, ...)`. Calls succeed; the failure count
   stays at 0. The listener logs quiet `success` lines.
2. **Redis starts failing.** Each failed call increments the count. The listener
   logs `Failure recorded (1/3)`, `(2/3)`, `(3/3)`.
3. **Trip (→ OPEN).** On the 3rd consecutive failure the breaker opens. The
   listener logs `State transition: CLOSED -> OPEN`. Because storage is Redis-backed,
   **every replica** now sees `redis` as OPEN.
4. **Protected.** For the next 10 seconds, every `call_async` returns instantly
   with `CircuitBreakerError`. `counter_service` catches it and takes its fallback
   path — no threads hang waiting on Redis.
5. **Probe (→ HALF_OPEN).** After 10 seconds, the next call is allowed through as a
   test.
6. **Recover or re-trip.** If that probe succeeds → `OPEN -> HALF_OPEN -> CLOSED`,
   normal service resumes. If it fails → back to OPEN for another 10 seconds.

---

## 9. Can I reuse this in another project?

**Yes — the pattern is very portable, and worth carrying with you.** What you'd
lift is not really "these files" so much as the *shape*:

- A **factory + singleton registry** so each dependency has exactly one shared
  breaker.
- A **per-dependency storage choice** (local vs. shared) driven by whether you want
  fleet-wide state.
- A **listener** for observability.
- The **consumption idiom** at every call site: `await breaker.call_async(op, ...)`
  wrapped in `try/except CircuitBreakerError → fallback`. This idiom is the actually
  reusable heart of it.

The **seams to swap** when porting:
- **Config source** — the `cb_*` settings (thresholds/timeouts). Point these at
  your own config.
- **The synchronous Redis client** in `breaker_storage.py` — repoint host/port/db,
  or drop Redis storage entirely and go all in-memory if you don't run multiple
  replicas.
- **The library** — this is built on `aiobreaker` (async). For sync code the
  equivalent is `pybreaker`; the same three-state model and the same call-wrapping
  idiom apply.

A good rule for *whether* to add a breaker in a new project: put one in front of
any **remote/network dependency whose slow failure could stall your service** (a
database, cache, queue, or third-party API). Don't bother wrapping pure local
computation — there's nothing to trip.

---

## 10. Thirty-second recap

- A circuit breaker **stops calling a failing dependency** so one sick dependency
  can't cascade into taking down your whole service.
- **Three states:** CLOSED (normal, counting failures) → OPEN (tripped, fail fast)
  → HALF_OPEN (one probe) → CLOSED or back to OPEN.
- Two knobs: **threshold** (how many failures trip it) and **recovery timeout**
  (how long it stays open before probing).
- This folder is the **factory/config/storage/logging** around the `aiobreaker`
  library; the real protection happens where code does
  `await breaker.call_async(...)` and catches `CircuitBreakerError`.
- **Three breakers:** `postgres` (in-memory, tolerant) and `redis`/`rabbitmq`
  (Redis-shared, trip sooner).
- It deliberately **fails closed** (assume-the-worst) — the mirror image of
  backpressure's fail-open.
- The pattern is **reusable**: keep the factory-registry shape and the
  `call_async` + fallback idiom; swap the config, storage, and library as needed.
