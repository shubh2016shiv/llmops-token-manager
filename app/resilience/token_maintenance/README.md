# Token Maintenance — what this module is for, and what it should become

> **Read this first.** This explains, in plain terms, *why* this module exists,
> *what benefit each piece gives*, *what is currently unnecessary*, and a
> **proposed (not final)** redesign with a clear flow. For the general production
> concepts referenced here (multiple replicas, schedule persistence, job
> monitoring), see the companion [PRODUCTION_PATTERNS.md](./PRODUCTION_PATTERNS.md).

---

## 1. The one-sentence purpose

Everything else in the token manager runs **when a request arrives**. This module
is the opposite: it runs **on a timer, in the background, with no request**, doing
the quiet housekeeping that keeps the fast path honest and the database tidy.

Think of a shop. The registers (the API) serve customers all day. But someone also
has to, every so often, **recount the till against the sales log** (reconciliation),
**refresh the "how busy are we" sign** (queue-depth publishing), and **throw out
yesterday's expired tickets** (cleanup). No customer asks for these; they just have
to happen periodically or the shop slowly falls into disorder. That background
"someone" is this module.

---

## 2. What it actually does — three periodic jobs

| Job | How often | What it does | Is it load-bearing? |
|---|---|---|---|
| **Queue-depth publish** | ~5s | Measures the RabbitMQ work-queue length and writes it to Redis | **Yes** — backpressure depends on it |
| **Reconciliation** | ~60s | Repairs drift between Redis token counters and the PostgreSQL source of truth | **Yes** — keeps token accounting correct |
| **Cleanup** | periodic | Deletes expired token-allocation rows from PostgreSQL | Nice-to-have — housekeeping |

### Job ① Queue-depth publish — *keeps backpressure alive*

The backpressure guard (Layer 1) rejects requests when the work queue is too deep.
But it can't measure the RabbitMQ queue on every request — too slow. So this job
measures it **once every ~5 seconds** and parks the number in Redis, where the
per-request check reads it instantly. **If this job stops, backpressure's
queue-depth gauge silently goes blind** — it reads "unknown" and stops shedding
load on queue depth. That is the benefit: it is the heartbeat that makes a whole
resilience layer functional. (The measuring code itself lives in
[`backpressure/publisher.py`](../backpressure/publisher.py); this module's only
job is to *call it on a schedule*.)

### Job ② Reconciliation — *keeps token counts correct*

The token manager keeps a **fast counter in Redis** (so it can reserve tokens in
~1ms) and the **authoritative record in PostgreSQL** (durable truth). These two
drift apart over time: a process crashes between the Redis update and the PG write,
an async persist fails, a Redis key expires. Left alone, the drift compounds until
the counters are simply wrong — and "wrong" here means either **falsely rejecting**
requests (Redis thinks more is allocated than really is) or **over-allocating**
past the real limit (Redis thinks less). Reconciliation sweeps every active
deployment every ~60s, compares Redis to PG, and corrects Redis toward the truth.
It also buckets the drift it finds into a histogram and logs large drifts, so you
can *see* whether the system is healthy. That visibility is a second real benefit.

This is a textbook **reconciliation loop** (a self-healing, eventual-consistency
pattern) — explained generally in the companion doc.

### Job ③ Cleanup — *keeps the database bounded*

Expired allocations pile up in PostgreSQL forever unless something deletes them.
This job periodically removes them. It is the least critical of the three — it
could even be pushed down into the database itself — but it stops unbounded table
growth.

---

## 3. What is currently *unnecessary* (and why the module feels broken)

Here is the honest state of the code today, and it explains why nothing seems to
"fit": **the module is dormant, half-disabled scaffolding built around Celery, and
Celery has already been pulled out.**

- **Celery is the heavyweight in the room.** Celery is a *distributed task queue* —
  a broker + a fleet of worker processes + a separate "beat" scheduler + result
  backends + serialization. That is the right tool for *fanning out lots of
  discrete, on-demand, possibly long jobs to workers you scale independently*. What
  we have is **three tiny, periodic, internal heartbeats**. That is cron-shaped, not
  queue-shaped. Celery here is ceremony — a whole subsystem to run ~150 lines of
  periodic code.

- **The code already senses this.** In [`tasks.py`](./tasks.py) every
  `@celery_app.task(...)` decorator is **commented out** and the `celery_app` import
  is disabled. [`healthcheck.py`](./healthcheck.py) hard-returns *"Celery runtime is
  disabled."* So none of these jobs actually run right now.

- **It references settings that no longer exist.**
  [`schedule_registry.py`](./schedule_registry.py) and
  [`reconciliation.py`](./reconciliation.py) read
  `settings.celery_token_maintenance_queue_name`,
  `settings.celery_reconcile_interval_secs`, and
  `settings.celery_cleanup_interval_secs` — all of which were **renamed** to
  non-`celery_` names (e.g. `reconcile_interval_secs`) and no longer exist. That is
  the `AttributeError` breaking the maintenance tests.

So the parts that are unnecessary are specifically the **Celery machinery**:
`schedule_registry.py` (Celery beat metadata + task routes), the Celery task
wrappers in `tasks.py`, the Celery-runtime probe in `healthcheck.py`, and the lazy
`__getattr__` in [`__init__.py`](./__init__.py) (that indirection exists only to
avoid triggering the Celery runtime during import). The **jobs themselves** —
reconciliation, queue-depth publish, cleanup — are keepers.

---

## 4. The proposed redesign (not final — a direction)

The token manager is already a long-running **FastAPI / asyncio** service. That
service is itself the perfect place to run three small periodic loops — no broker,
no worker fleet, no separate scheduler process. The pattern is: **start the loops
when the app starts, stop them when it stops.**

### Proposed flow

```
FastAPI app startup (lifespan)
        │
        ▼
   scheduler.start()  ──creates 3 asyncio background loops──┐
        │                                                    │
        ▼                                                    ▼
  ┌───────────────────────────────────────────────────────────────────┐
  │ every ~5s   → publish_queue_depth_snapshot()   (feeds backpressure)│
  │ every ~60s  → reconcile_async()                (guarded by a lock) │
  │ every ~Ns   → cleanup_expired_allocations()    (housekeeping)      │
  └───────────────────────────────────────────────────────────────────┘
        │
        ▼
FastAPI app shutdown (lifespan)
        │
        ▼
   scheduler.stop()  ──cancels the loops cleanly──
```

Each loop is just: *do the work → sleep the interval → repeat*, with every
iteration wrapped so a failure is logged and the loop keeps ticking.

### Why the single Redis lock is enough (no Celery needed for correctness)

If you run several copies of the service (see PRODUCTION_PATTERNS §1), you don't
want all of them reconciling at once. The code **already solves this** with a Redis
lock: `SET reconcile_lock "1" NX EX 45` — the first replica to grab it runs; the
others see it's taken and skip this tick ([reconciliation.py:63](./reconciliation.py#L63)).
That is all the coordination reconciliation needs. The queue-depth publisher is
"last write wins" with a short TTL, so overlapping publishers are harmless. So we
get multi-replica safety **without** Celery's dedicated-scheduler guarantee.

### Proposed file shape

```
token_maintenance/
  __init__.py          # plain small exports (drop the lazy __getattr__)
  scheduler.py         # NEW: starts/stops the periodic asyncio loops — the heart
  reconciliation.py    # KEEP the logic; fix the dead `celery_*` settings
  cleanup.py           # the expired-allocation deletion (lifted out of tasks.py)
  health.py            # "is the scheduler running / when did each job last run?"
  # schedule_registry.py   ← REMOVE (Celery beat metadata)
  # tasks.py               ← REMOVE (Celery task wrappers; loops call async fns directly)
  # healthcheck.py         ← FOLD into health.py (no Celery runtime to probe)
```

Note two simplifications this unlocks:
- The Celery wrappers in `tasks.py` exist only because Celery workers are
  **synchronous**, so they wrap the async functions in `asyncio.run(...)`. In-process
  loops are already async, so they call `reconcile_async()` / etc. **directly** — the
  wrappers vanish.
- The lazy-import `__getattr__` in `__init__.py` exists to avoid importing the
  Celery runtime too early. No Celery → a plain, ordinary `__init__.py`.

---

## 5. The benefit ledger (what we keep, what we drop)

**Keep (real benefit):**
- **Reconciliation** → correct token accounting; drift visibility. *This is the
  crown jewel of the module.*
- **Queue-depth publish** → keeps backpressure's queue gauge working.
- **Cleanup** → bounded database growth.
- **The Redis reconciliation lock** → multi-replica safety, already built.

**Drop (cost without matching benefit *at current scale*):**
- Celery broker/worker/beat machinery, `schedule_registry.py`, the Celery task
  wrappers, the Celery-runtime health probe, and the lazy `__init__`.

**Fix:**
- The dead `celery_*` settings → point at the real names
  (`reconcile_interval_secs`, `reconcile_drift_warning_threshold`), and add a
  cleanup-interval setting (currently missing entirely).

---

## 6. When this decision should be revisited

Dropping Celery is right **for now**, not forever. Bring back a real scheduler /
worker (Celery, or lighter, APScheduler) if any of these become true — each is
explained in [PRODUCTION_PATTERNS.md](./PRODUCTION_PATTERNS.md):

- Reconciliation grows heavy (minutes, millions of rows) and you want it **off the
  API process** so it doesn't compete with request handling.
- You need **retries, schedule persistence across restarts, or a monitoring
  dashboard** for the jobs.
- Jobs become **on-demand** (triggered by events/requests), not just periodic.

Until then: three asyncio loops in the service you already run. Simpler, fewer
moving parts, nothing to operate.

---

## 7. Thirty-second recap

- This module is the **background housekeeper**: it runs on a timer, not per
  request.
- **Three jobs:** queue-depth publish (feeds backpressure), reconciliation (keeps
  Redis/PG token counts correct — the crown jewel), cleanup (bounded DB).
- **Scheduling is needed**; **Celery is not** — these are small periodic heartbeats,
  not a distributed work queue. Celery was already disabled, leaving broken
  scaffolding and dead settings.
- **Redesign:** run the three jobs as **in-process asyncio loops** started in
  FastAPI's lifespan, using the **Redis lock you already have** for multi-replica
  safety. Delete the Celery machinery; keep and fix the job logic.
