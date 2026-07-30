# Production patterns for background jobs — a from-scratch guide

> **Why this doc exists.** The [README](./README.md) mentions ideas — *multiple
> replicas*, *distributed locks*, *schedule persistence*, *job monitoring*,
> *idempotency*, *reconciliation* — that you meet the moment you run periodic work
> in production, but that don't come up when you're learning to code features. This
> is a beginner-level tour of those patterns, made concrete with this module, then
> generalized so you can carry them to any project.

---

## 1. Why "one running copy" stops being true: replicas

When you run a service on your laptop, there is **one process**. Everything —
including any "every 60 seconds do X" loop — happens exactly once, because there is
exactly one of you.

In production you almost never run one copy. You run **several identical copies of
the same service** (call them *replicas* or *instances*) behind a **load balancer**
that spreads incoming requests across them:

```
                 ┌─────────────┐
   requests ───▶ │ load        │ ───▶ replica A  (full copy of the service)
                 │ balancer    │ ───▶ replica B  (full copy of the service)
                 └─────────────┘ ───▶ replica C  (full copy of the service)
```

You do this for two reasons:
- **Throughput** — three copies handle roughly three times the requests.
- **Availability** — if replica B crashes, A and C keep serving; no downtime.

This is called **horizontal scaling** ("add more copies") as opposed to **vertical
scaling** ("make one copy bigger"). It is the default shape of almost every real
web backend.

### The catch this creates for background jobs

Here's the problem that surprises everyone the first time. Your "every 60 seconds,
reconcile the counters" loop is **part of the service** — so it exists in **every
replica**. With three replicas, your once-a-minute job now tries to run **three
times a minute, simultaneously**, once from each copy. For a read-only job that's
just wasteful. For a job that **writes** (like reconciliation correcting counters),
three copies stomping on each other at once can corrupt the very data they're
meant to fix.

So the core question of running periodic work at scale is:

> **"I have N identical copies of my service. How do I make sure this job runs
> once, not N times?"**

That single question is what "multi-replica safety" means. There are three common
answers.

---

## 2. Three ways to make a job run "once, not N times"

### Answer A — a distributed lock ("first one grabs it wins")

Before doing the work, every replica tries to grab a shared **lock**. Only one can
hold it; that one does the work, the rest skip this round. The lock lives somewhere
all replicas can see — here, Redis.

This is **exactly what this module already does** for reconciliation:

```python
lock_acquired = await redis.set("token:lock:reconcile", "1", nx=True, ex=45)
if not lock_acquired:
    return  # someone else is already reconciling this round; skip.
```

The magic is in two Redis flags:
- **`nx` = "set only if Not eXists."** Redis guarantees that when three replicas
  fire this at once, **exactly one** succeeds in creating the key; the other two get
  "already exists" and back off. Redis does the arbitration for you, atomically.
- **`ex=45` = "expire after 45 seconds."** This is the safety net. Imagine the
  replica holding the lock **crashes** before releasing it. Without an expiry, the
  lock would be stuck "held" forever and the job would **never run again**. The TTL
  guarantees the lock auto-releases, so a later run can proceed. (This module sets
  the TTL a bit *below* the run interval — `interval - 15s` — so the lock is always
  gone before the next scheduled tick.)

A distributed lock is the **lightest** option: no extra infrastructure beyond the
Redis you already have. Its trade-off: it's "best effort, per tick" — if a run is
skipped, the next tick just tries again. That's perfect for *idempotent, periodic
repair* work (see §4), which is exactly what reconciliation is.

### Answer B — leader election ("one replica is boss")

The replicas agree among themselves that **one of them is the leader**, and only the
leader runs scheduled jobs. If the leader dies, the others notice and elect a new
one. This needs a coordination system (etcd, ZooKeeper, Kubernetes leases, a
database advisory lock). It's more robust than a per-tick lock but heavier to
operate. You reach for it when you have *many* scheduled jobs and want one clear
owner rather than a lock per job.

### Answer C — a dedicated scheduler/worker (the Celery model)

Run the scheduled work **outside** the request-serving replicas entirely: a single
**scheduler** process ("Celery beat") decides *when*, and a pool of **worker**
processes do the work. The API replicas do zero background work. This is the most
powerful and the most infrastructure — a broker, workers, a scheduler, all to
operate and monitor.

**Which does this module need?** Only Answer A. Three small periodic jobs, one of
which already writes safely under a Redis lock, do not justify B or C. The
[README](./README.md) argues this in context.

| | Distributed lock (A) | Leader election (B) | Dedicated scheduler (C) |
|---|---|---|---|
| Extra infrastructure | none (reuses Redis) | a coordinator | broker + workers + beat |
| Complexity | low | medium | high |
| Best for | a few idempotent periodic jobs | many jobs, one owner | heavy/on-demand/independently-scaled work |
| This module | **✅ fits** | overkill | overkill |

---

## 3. Where the job "lives" — and what a restart does

### In-process vs out-of-process

- **In-process** — the loop runs *inside* your API service (e.g. an asyncio task
  started at startup). Zero extra moving parts; the job shares the service's memory
  and lifecycle. This is what the README proposes.
- **Out-of-process** — the job runs in a separate program (a Celery worker, a cron
  container, a Kubernetes CronJob). More isolation (a heavy job can't slow your API),
  more to run.

### "Schedule persistence across restarts" — what that phrase means

Every scheduler holds two kinds of information: **the schedule** ("run reconcile
every 60s") and **the run history** ("last reconcile was at 12:00:00"). The question
is: **when the process restarts (deploy, crash, scale-down), is that remembered?**

- An **in-memory scheduler** (a plain asyncio loop) forgets everything on restart.
  It just starts ticking fresh. If the process was down from 12:00 to 12:05, the
  runs that *would* have happened in those 5 minutes simply **didn't** — they are
  not "made up." For periodic self-healing work, that's completely fine: the next
  tick repairs whatever accumulated. **This is the normal, acceptable case for this
  module.**
- A **persistent scheduler** stores schedule + history in a database, so after a
  restart it knows "I missed the 12:01, 12:02, 12:03 runs" and can decide what to do
  about them. That decision is called **misfire handling**:
  - *skip* the missed runs (just resume on the next tick), or
  - *catch up* by running the missed ones (sometimes "coalesce" many missed runs
    into a single catch-up run).

You only care about persistence/misfire when a **missed run has real consequences** —
e.g. "email every customer their monthly invoice on the 1st"; skipping that because
you deployed at midnight is a real problem. For "recount the counters every minute,"
a missed minute is a non-event. **Rule of thumb: periodic *repair* jobs don't need
persistence; scheduled *events with deadlines* do.**

---

## 4. Idempotency — the property that makes all of this safe

**Idempotent** = running it **once has the same end-state as running it twice (or
three times, or a half-finished time).** Turning a light switch to "ON" is
idempotent (flip it twice, still on). "Add $10 to the balance" is **not** (twice
adds $20).

Why it's the keystone here: every safe-at-scale strategy above is really
"best-effort, might run more than once or get interrupted." A lock can expire mid-run
and let a second run start; a crash can leave a job half-done. If your job is
idempotent, none of that can corrupt anything — worst case you did harmless
redundant work. If it isn't, you need much heavier machinery (exactly-once
semantics, transactions, dedup keys).

This is why reconciliation is written to **set Redis toward the PG truth** ("the
allocated count *should be* 500") rather than to **apply a delta** ("add 30"). The
first is idempotent — run it five times, the answer is still 500. The second would
be catastrophic under retries. **Designing jobs to be idempotent is what lets you
use the cheap coordination (a Redis lock) instead of expensive coordination.**

Closely related vocabulary you'll see:
- **At-least-once** — the system guarantees a job runs, but *maybe more than once*
  (so make it idempotent). This is the common, practical default.
- **At-most-once** — runs zero or one time, never duplicated, but might be missed.
- **Exactly-once** — runs precisely once. Very hard in distributed systems; usually
  faked with at-least-once + idempotency/dedup. When someone promises "exactly once,"
  they almost always mean this combination.

---

## 5. Reconciliation — the pattern behind the crown-jewel job

Reconciliation is a named pattern, worth knowing beyond this codebase. The setup:
you keep the **same fact in two places** for different reasons —

- **Redis**: a fast, in-memory copy so you can reserve tokens in ~1ms.
- **PostgreSQL**: the slow, durable **source of truth**.

Keeping two copies always creates **drift**: crashes, partial failures, expiries,
and races leave the fast copy slightly wrong. Instead of trying to make the two
*perfectly* consistent on every single write (very expensive, and still not
bulletproof), you accept **eventual consistency**: let them drift a little, and run
a periodic **reconciliation loop** that reads the source of truth and repairs the
fast copy toward it.

```
   every 60s:
     for each deployment:
        truth   = read from PostgreSQL          (authoritative)
        cached  = read from Redis               (fast copy)
        drift   = | truth - cached |
        record drift in a histogram, warn if large
        set Redis  ← truth                       (idempotent repair)
```

Two things this buys you beyond correctness:
- **Self-healing** — the system tolerates transient failures because the next sweep
  cleans them up. You don't need every write path to be perfect.
- **Observability** — by bucketing the drift it finds (0, 1–10, 11–100, …) and
  logging large drifts, reconciliation becomes a **health signal**: growing drift
  means something upstream is misbehaving. That histogram in
  [`reconciliation.py`](./reconciliation.py) is not decoration; it's a gauge of
  system health.

This "source of truth + fast cache + periodic repair" shape appears everywhere:
cache invalidation, search-index rebuilds, materialized views, inventory counts,
account balances. Recognize it once and you'll see it constantly.

---

## 6. Monitoring background jobs — "Flower-style" observability

A web request that fails is *visible*: someone gets an error, it shows in request
logs, dashboards spike. A **background job that fails is invisible** — no user is
watching, so a broken reconciliation loop can be silently dead for days while the
data quietly rots. Therefore background work needs its **own** observability, and
you must build it deliberately. The three things worth exposing per job:

1. **Liveness** — *is the loop even running?* (last-heartbeat timestamp)
2. **Recency & outcome** — *when did it last run, how long did it take, did it
   succeed or fail?*
3. **Domain signal** — the meaningful number the job produces — here, the **drift
   histogram** and the counts of corrections applied.

**"Flower"** is simply Celery's web dashboard that shows exactly this for Celery
tasks: which ran, when, how long, pass/fail, retries. When someone says "Flower-style
monitoring," they mean *a view into your background jobs' health*, whether or not
you use Celery or Flower specifically. Without Celery you get the same visibility by:
- emitting **structured logs** with fields you can search/alert on (this module
  already logs a per-run summary),
- exposing **metrics** (e.g. Prometheus counters/gauges: last-run time, duration,
  success/failure, drift),
- and surfacing **last-run timestamps on a health endpoint** so a human or an uptime
  check can confirm the loops are alive.

The principle to remember: **if a background job breaking wouldn't page anyone, it
will eventually break and no one will know.** Give every scheduled job a way to be
seen.

---

## 7. Putting it together for THIS module

| Pattern | How it shows up here |
|---|---|
| Multiple replicas | Several copies of the token manager run behind a load balancer |
| Run-once coordination | Redis `SET NX EX` lock around reconciliation (Answer A) |
| In-process scheduling | Proposed: asyncio loops in the FastAPI service, no Celery |
| Restart behavior | In-memory schedule; missed ticks are fine (periodic repair) |
| Idempotency | Reconciliation writes toward truth (set-to-value), not deltas |
| Reconciliation pattern | Redis (fast) ↔ PostgreSQL (truth), periodic drift repair |
| Monitoring | Per-run summary logs + drift histogram; add metrics + a health endpoint |

None of these require Celery. They require **one Redis lock, idempotent jobs, and a
little observability** — all of which are either already here or cheap to add.

---

## 8. Glossary (quick reference)

- **Replica / instance** — one running copy of your service; production runs many.
- **Horizontal scaling** — adding more copies (vs. **vertical** = a bigger copy).
- **Load balancer** — spreads incoming requests across replicas.
- **Distributed lock** — a shared flag (in Redis, etc.) that lets only one replica
  proceed; `SET key val NX EX ttl` is the Redis idiom.
- **TTL (time-to-live)** — an auto-expiry on a key; here it stops a crashed holder
  from freezing a lock forever.
- **Leader election** — the replicas agree on one "leader" that runs scheduled work.
- **Idempotent** — running it again doesn't change the end result.
- **At-least-once / at-most-once / exactly-once** — delivery/execution guarantees;
  the practical default is at-least-once **+ idempotency**.
- **Eventual consistency** — copies of data are allowed to differ briefly and are
  reconciled over time, rather than kept identical on every write.
- **Reconciliation loop** — a periodic job that repairs a fast copy toward the
  source of truth.
- **Misfire handling** — what a persistent scheduler does about runs it missed while
  down (skip / catch-up / coalesce).
- **Beat / worker (Celery)** — *beat* = the scheduler that decides *when*; *worker* =
  the process that does the job.
- **Flower** — Celery's dashboard; shorthand for "a view into background-job health."
