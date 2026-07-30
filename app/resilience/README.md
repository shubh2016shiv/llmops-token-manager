# The Resilience Layer — how this service stays fast *and* survives failure

> **What this document is.** The token manager is built from several resilience
> components (rate limiting, backpressure, circuit breakers, a Redis fast path, a
> durable queue, and background maintenance). Each has its own README. **This
> document is the map that ties them all together** — how they fit, how they
> interact, and why the whole thing holds up under load and failure.
>
> **Who it's for.** Someone with **zero system-design background.** Every term is
> explained where it first appears. Read top to bottom, one section at a time.
>
> **How it's organized.** It's built up in numbered sections, each self-contained.
> Read Section 1 first; the rest can be read in order or dipped into.

---

## Table of contents

1. **Start here** — what problem this whole layer solves (no jargon)
2. **The grand picture** — the master diagram of the entire system
3. **The mental model** — "layers" and "two worlds"
4. **Layer 0 — Rate Limiting** — fairness between callers
5. **Layer 1 — Backpressure** — refuse work when saturated
6. **Layer 2 — Circuit Breakers** — stop calling broken dependencies
7. **Layer 3 — Redis Fast Path** — reserve capacity in ~1ms
8. **Layer 4 — Queue Absorption** — persist durably, off the request
9. **The background world** — reconciliation & maintenance
10. **How the layers fail together** — the resilience matrix
11. **Cross-cutting patterns** — the ideas that repeat everywhere
12. **Production considerations** — what makes it safe to actually run
13. **End-to-end walkthrough** — one request through every layer
14. **Glossary** — every term in one place

Per-component deep dives live next to their code:
[backpressure/](./backpressure/README.md) ·
[circuit_breaker/](./circuit_breaker/README.md) ·
[redis_token_counter/](./redis_token_counter/README.md) ·
[token_queue/](./token_queue/README.md) ·
[token_maintenance/](./token_maintenance/README.md) ·
[token_maintenance/PRODUCTION_PATTERNS.md](./token_maintenance/PRODUCTION_PATTERNS.md)

---

## 1. Start here — what problem does this whole layer solve?

### What this service actually is

The token manager is a **ledger**. Its one job: when some other service wants to
use an LLM, it asks *"can I spend N tokens on this model right now?"*, and the token
manager answers **yes/no** and keeps count so nobody exceeds the paid capacity.
Think of it as the **bouncer + tab-keeper** for a very busy, very expensive bar: it
decides who gets in, tracks everyone's tab, and must never let the bar oversell
drinks it can't serve.

Two things about that job make it hard:

1. **It must be blisteringly fast.** It sits on the *hot path* — every single LLM
   call in the whole platform waits on its answer. If it takes 200ms, everything
   downstream is 200ms slower. The target is **~1 millisecond**.
2. **It must never fall over.** It depends on other systems — a database
   (PostgreSQL), a cache (Redis), a message broker (RabbitMQ). Those *will*
   occasionally be slow or down. When they are, the token manager still has to
   behave sensibly instead of collapsing and taking the whole platform with it.

Fast **and** correct **and** unfailing, while standing on top of things that fail.
That tension is the entire reason the resilience layer exists.

### Why "just write to the database" doesn't work

The obvious design is: on each request, read the current count from the database,
check the limit, write the new count. Simple — and wrong on both counts:

- **Too slow.** A database round-trip on every LLM call, under heavy traffic, is far
  more than 1ms and buckles under load.
- **Too fragile.** If the database has a bad minute, *every* request fails, and the
  outage cascades outward to every service that was waiting on an answer.

So instead of one simple-but-fragile step, the token manager uses a **chain of
purpose-built resilience components**, each solving one slice of "fast and correct
and unfailing." The rest of this document is that chain.

### The one idea that unifies everything: graceful degradation

If you remember nothing else, remember this. A resilient system is not one that
*never* has problems — that's impossible. It's one that, when a piece breaks,
**gets a little worse instead of completely broken.** A dependency goes down and the
service gets *slower*, or *sheds some load*, or *falls back to a backup path* — but
it keeps answering. That "bend, don't snap" behavior is called **graceful
degradation**, and every component you're about to meet is one specific way of
bending instead of snapping.

Here's the same idea as a picture — the difference between a fragile system and this
one when a dependency (say, Redis) has a bad moment:

```
   FRAGILE design                        RESILIENT design (this service)
   ─────────────                         ───────────────────────────────
   request → Redis (down) → 💥 error     request → Redis (down)
            every request fails                   │ circuit breaker fails fast
            threads pile up                       ▼
            whole service hangs           falls back to the database path
            outage spreads outward        → slower, but still answers ✔
```

Same failure, two completely different outcomes. The left column is a fragile
service. The right column — *degrade, don't die* — is what the next 13 sections
build, piece by piece.

---

## 2. The grand picture — the whole system on one diagram

Before we zoom into any single component, here is the entire thing at once. Don't
worry about understanding every box yet — the goal is just to see the **shape** and
learn the **names**. We'll unpack each part in its own section.

First, the three outside systems the token manager leans on (its **dependencies**):

- **Redis** — a *cache*: an extremely fast, in-memory store. Holds the live token
  counters and a few shared signals. Fast but volatile (data can expire/vanish).
- **PostgreSQL** — the *database*: the slow but durable **source of truth**. The
  real ledger lives here.
- **RabbitMQ** — a *message broker*: a durable "outbox" that reliably hands work
  from one process to another, surviving crashes.

Now the whole system. Read it top to bottom:

```
 LEGEND   ▶ a guard (can reject the request)     ═▶ hot path (runs per request)
          ──▶ async / background flow            [STORE] an outside dependency

 ┌──────────────────────────────────────────────────────────────────────────┐
 │  CLIENT (another service)  ──  POST /acquire "spend N tokens on model X"   │
 └───────────────────────────────────┬──────────────────────────────────────┘
                                      ║
                                      ▼   THE REQUEST GAUNTLET
                                          (cheapest checks first, real work last)
 ┌──────────────────────────────────────────────────────────────────────────┐
 │  Layer 0 · Rate Limiter    ▶ is THIS caller sending too fast? ....... 429  │
 │  Layer 1 · Backpressure    ▶ is the whole SYSTEM saturated? ......... 503  │
 │  Layer 3 · Redis Fast Path   reserve N tokens atomically (~1ms) ──────────┼──▶ [REDIS]
 │  Layer 4 · Queue Publish     hand off the durable DB write ───────────────┼──▶ [RABBITMQ]
 └───────────────────────────────────┬──────────────────────────────────────┘
                                      ║
                                      ▼
                         201 Created ═══════════════════════▶ back to CLIENT (fast!)

     Layer 2 · CIRCUIT BREAKERS wrap every [STORE] call above (and below), so a
     sick dependency fails fast in ~0ms instead of hanging the request.

 ── ASYNC (off the request) ────────────────────────────────────────────────────
   [RABBITMQ] ──▶ QUEUE CONSUMER (its own process) ──▶ write ──▶ [POSTGRES]
                     └─ on failure: retry queues ──▶ DLQ ──▶ release Redis tokens

 ── BACKGROUND (on timers) ──────────────────────────────────────────────────────
   token_maintenance ──▶ reconcile [REDIS] ↔ [POSTGRES]  (every ~60s)
                    ├──▶ publish queue depth ──▶ [REDIS]  (feeds Layer 1)
                    └──▶ cleanup expired rows ──▶ [POSTGRES]
```

### The cast of characters

Every component, its layer number, its one-line job, and where its code lives:

| Layer | Component | Its one job | Lives in |
|---|---|---|---|
| **0** | Rate Limiter | Fairness: cap how fast each caller can ask | `app/core/rate_limiter.py` |
| **1** | Backpressure | Shed load fast (503) when the system is saturated | [`backpressure/`](./backpressure/README.md) |
| **2** | Circuit Breakers | Stop calling a dependency that's clearly broken | [`circuit_breaker/`](./circuit_breaker/README.md) |
| **3** | Redis Fast Path | Reserve tokens atomically in ~1ms | [`redis_token_counter/`](./redis_token_counter/README.md) |
| **4** | Queue Absorption | Persist durably, off the request, with retries | [`token_queue/`](./token_queue/README.md) |
| **—** | Token Maintenance | Background reconcile / cleanup / signal-publishing | [`token_maintenance/`](./token_maintenance/README.md) |

### Two things to notice already

1. **The layers are ordered cheapest-to-most-expensive.** Rate limiting is a tiny
   arithmetic check; backpressure reads a few gauges; only *then* do we do real work
   in Redis and RabbitMQ. We reject bad requests as early and cheaply as possible —
   why spend effort on a request you're going to refuse anyway?
2. **The client's answer comes back *before* the database is written.** Look again:
   `201 Created` is returned right after the Redis reserve and the queue publish. The
   actual PostgreSQL write happens **later, asynchronously**, in the consumer. That
   split is the secret to the ~1ms speed — and Section 8 explains how it stays safe.

That's the whole machine. The next section gives you the two mental models that make
it easy to hold in your head.

---

## 3. The mental model — two ideas that make it all fit

The grand diagram has a lot in it. But you only need **two mental models** to hold
the whole thing comfortably in your head. Everything else is detail hanging off
these two.

### Idea 1: The layers are a *gauntlet of guards*

Picture a nightclub with a series of checkpoints on the way in. At each checkpoint,
a guard either **waves you through** or **turns you away** — and the cheap, quick
checks come first so troublemakers are stopped before they waste anyone's time.

That's exactly what the "layers" are. Each layer is a guard with one question:

```
   request
     │
     ▼
   ┌─────────────────────────────────────────────────────────┐
   │ Layer 0  "Are YOU personally sending too fast?"          │ ── yes ─▶ 429, done
   ├─────────────────────────────────────────────────────────┤
   │ Layer 1  "Is the whole system too overloaded right now?" │ ── yes ─▶ 503, done
   ├─────────────────────────────────────────────────────────┤
   │ Layer 2  "Is the dependency I need actually alive?"      │ ── no  ─▶ fail fast / fall back
   ├─────────────────────────────────────────────────────────┤
   │ Layer 3  "Reserve the tokens" (the real work begins)     │
   ├─────────────────────────────────────────────────────────┤
   │ Layer 4  "Save it durably, without slowing the reply"    │
   └─────────────────────────────────────────────────────────┘
     │
     ▼
   success
```

Two properties of a gauntlet, both deliberate:

- **Cheap guards first.** Rejecting at Layer 0 costs almost nothing; doing real work
  at Layer 3–4 costs the most. Ordering them cheap→expensive means a request that
  will be refused is refused *early*, before it consumes the expensive resources.
- **Any guard can stop the line.** The moment one guard says "no," the request is
  turned away and the later, costlier guards never run. This is why an overloaded
  system stays *responsive*: it spends its scarce capacity saying a fast "no" rather
  than a slow, doomed "yes."

> The numbering is a bit uneven (Layer 2, circuit breakers, isn't a single
> checkpoint — it wraps the dependency calls that happen *during* Layers 3–4). Don't
> over-index on the exact numbers; the **order of concerns** is what matters:
> *who are you → is the system ok → is the dependency ok → do the work → save it.*

### Idea 2: There are *two worlds*, and they meet at the data stores

This is the single most clarifying fact about the codebase, and the thing that
confuses newcomers most. There are **two completely separate kinds of code** here:

```
   THE REQUEST WORLD  (foreground)          THE BACKGROUND WORLD
   ─────────────────────────────           ─────────────────────
   • runs ON a request                      • runs ON A TIMER (no request)
   • must be FAST (~1ms)                     • can be slow (~seconds)
   • the gauntlet above                      • reconcile, cleanup, publish signals
   • Layers 0–4                              • token_maintenance
             │                                        │
             └──────────►  [ REDIS ]  ◄───────────────┘
             └──────────►  [POSTGRES] ◄───────────────┘
                     (they only ever meet at the shared data stores)
```

- The **request world** is the gauntlet: it runs per request, must be fast, and does
  the actual reserving and hand-off.
- The **background world** (`token_maintenance`) runs on timers with no request in
  sight. Its job is to keep the fast world *honest*: it repairs drift between Redis
  and PostgreSQL, refreshes the "how busy are we" signal that Layer 1 reads, and
  deletes old rows.

**They never call each other.** The background world doesn't handle requests, and the
request world doesn't run maintenance. They communicate *only* by leaving data in
the shared stores — one writes a value to Redis, the other reads it later. (You saw
this exact pattern inside backpressure: a background job *publishes* the queue depth,
and the request path *reads* it. It's the shape of the whole service, not just one
module.)

### Why these two ideas are enough

With just these, you can place any piece of code:

- *Does it run on a request?* → it's in the **request world**, and it's one of the
  **gauntlet layers**.
- *Does it run on a timer?* → it's the **background world**.
- *Is it deciding whether to let a request proceed?* → it's a **guard** (Layer 0/1/2).
- *Is it doing the actual token work?* → Layer 3 (reserve) or Layer 4 (persist).

Hold onto **gauntlet** and **two worlds**. The next six sections walk the gauntlet
one guard at a time, then visit the background world — and each one will slot neatly
into this frame.

---

## 4. Layer 0 — Rate Limiting (the fairness guard)

*Code: `app/core/redis_rate_limiter/` · mounted on `/acquire` as a FastAPI
dependency. It technically lives outside the `resilience/` folder, but it's the
first guard in the gauntlet, so we cover it here.*

### The problem it solves: the noisy neighbour

Many different services call the token manager. Now imagine **one** of them has a
bug — a runaway retry loop firing thousands of requests per second. Without
protection, that one misbehaving caller floods the service, eats all its capacity,
and **every other well-behaved caller gets slow or fails**. One bad neighbour ruins
the whole building. That's the problem rate limiting exists to prevent.

### What rate limiting *is* (plain version)

A rate limiter caps **how many requests a given caller may make per unit of time**
— e.g. "at most 500 requests per minute." Go over, and you're told **HTTP 429 Too
Many Requests** ("slow down, try again shortly") instead of being served. That's it:
counting requests and saying "enough" past a threshold.

### The mechanism: separate buckets per caller

The clever part is *how* it counts. It doesn't count all requests together — it
sorts them into **buckets**, and each bucket has its own independent budget. For the
`/acquire` endpoint the bucket key is **`X-Service-Id` + client IP** — meaning *each
calling microservice gets its own private 500/minute allowance*:

```
   ms-gateway  (normal) ──▶┐
                           │   ┌────────────────────────────────────────┐
                           ├──▶│  Rate Limiter (Layer 0)                 │
   ms-pipeline (runaway) ──▶┘   │  count per bucket in a moving 60s window │
                               └────────────────────────────────────────┘
                                  bucket "ms-gateway:IP"   →  12 / 500  ✔ allow
                                  bucket "ms-pipeline:IP"  → 501 / 500  ✗ 429
```

Because each service has its **own** bucket, the runaway `ms-pipeline` hits its own
429 wall while `ms-gateway` sails through completely unaffected. The blast radius of
one misbehaving caller is contained to *itself*. This "isolate callers so one can't
sink the others" idea is a core resilience pattern (sometimes called a **bulkhead**,
after the watertight compartments that stop one flooded section from sinking a ship).

### One subtlety worth knowing: "moving window"

There are two common ways to count "per minute":

- **Fixed window** — reset the counter to zero every minute on the clock. Simple, but
  it has a flaw: a caller can fire its whole budget at 11:59:59 *and again* at
  12:00:00 — double the intended rate in one moment, right at the boundary.
- **Moving window** (what this uses) — always look back over the *last* 60 seconds,
  continuously. No boundary to exploit; the limit is honoured at every instant.

There's also a small security detail: to identify the real caller behind a load
balancer, it reads the client IP from a **trusted position** in the
`X-Forwarded-For` header, so an attacker can't spoof a fresh IP on every request to
mint unlimited buckets. (Details in the code's own comments — it's well documented.)

### Production considerations baked in

- **Distributed.** The counts live in **Redis**, not in one server's memory. So if
  the token manager runs as several replicas (it does — see Section 12), they all
  share one honest count. A caller can't get 500 *per replica* by spreading requests
  around.
- **Fails open.** If Redis (the counter store) is unreachable, the limiter **lets
  the request through** rather than blocking everyone. A broken *limiter* must not
  become an outage — and the later guards (backpressure, circuit breakers) still
  protect the system if that fail-open lets a surge in. (You'll see "fail open"
  again and again; Section 11 makes it a first-class idea.)
- **Generous by default.** 500/min per service is set high on purpose: real traffic
  never notices it, but a runaway loop is caught. It's a *safety net*, not a
  throttle on normal use.

### Where it sits in the gauntlet

Layer 0 is deliberately **first and cheapest** — a bit of arithmetic against a Redis
counter. It answers *"is this specific caller behaving?"* before any of the more
expensive guards even look at the request. If you're going to refuse a flood, refuse
it here, at the door, for almost no cost.

---

## 5. Layer 1 — Backpressure (the whole-system health guard)

*Code: [`backpressure/`](./backpressure/README.md) · mounted on `/acquire` right
after the rate limiter.*

### First, the word itself

**"Backpressure"** is borrowed from plumbing. If you keep pumping water into a pipe
faster than it can drain, pressure builds up and pushes *back* against the pump —
that push-back is backpressure, and it's the system's way of signalling *"stop
sending, I can't take more."* In software it means the same thing: when a service is
receiving work faster than it can finish, it needs a way to push back on the sender
and say *"not right now."* This layer is that push-back.

### How it differs from Layer 0

Layer 0 (rate limiting) asks about **one caller**: *"are YOU sending too fast?"*
Layer 1 asks about the **whole system**: *"regardless of who's asking, am I — the
service — currently too overloaded to safely take on more work?"* A caller can be
perfectly well-behaved and still get turned away here, simply because the system as
a whole is struggling. Different question, different guard.

### What "overloaded" means: three health gauges

Backpressure doesn't guess. It reads **three specific health signals** (the code
calls them *probes* — small readers that each measure one thing). Think of them as
three dashboard gauges:

```
   ┌──────────────────────────────────────────────────────────────────────┐
   │  Gauge 1 · Queue depth        how many jobs are waiting to be done?     │
   │  Gauge 2 · DB pool usage      how full is the database connection pool?  │
   │  Gauge 3 · DB breaker state   is the database circuit breaker tripped?   │
   └──────────────────────────────────────────────────────────────────────┘
              first gauge in the RED zone  ─────▶  reject with 503
              all gauges green             ─────▶  let the request through
```

Two of those need a plain-language definition:

- **Queue depth** — later work (the durable database write) is handed to a waiting
  line called a queue (Layer 4). "Queue depth" is simply **how long that line is**.
  A long line means the workers are falling behind — an early warning of overload.
- **Database connection pool** — the service keeps a small, fixed set of open
  connections to PostgreSQL and reuses them (opening a new one per request would be
  slow). That reusable set is the **pool**. If nearly all connections are already in
  use ("pool usage near 100%"), new work will have to wait for one to free up — a
  sign of saturation. (Gauge 3, the circuit breaker, is Layer 2 — the next section.)

### The rule and the response

The rule is dead simple: **check the three gauges in order; the first one in the red
wins.** If any gauge says "saturated," the request is rejected immediately with:

- **HTTP 503 Service Unavailable** — the standard "I'm healthy but temporarily
  overloaded, not broken" reply.
- A **`Retry-After` header** — a number telling the caller *when* to try again (e.g.
  "retry after 10 seconds"). This is the polite, useful part: instead of a bare "no,"
  the client gets a concrete "come back in N seconds," so it can back off sanely
  instead of hammering. (For the queue-depth case that number is *estimated* from how
  fast the queue is draining; the backpressure README shows the arithmetic.)

This behaviour — deliberately refusing some work so the rest stays fast — is the
resilience pattern called **load shedding**: when you can't do everything, shed
(drop) some load quickly so you don't fail at *everything* slowly.

### The two-worlds connection (a callback to Section 3)

Remember "two worlds"? Backpressure is where they visibly touch. The request path
can't measure the RabbitMQ queue on every call — too slow. So the **background
world** (`token_maintenance`) measures the queue length every few seconds and writes
the number into Redis; the request path just **reads** that pre-computed number
instantly. Backpressure's queue gauge is fed by a background job. That's the two
worlds meeting at a shared store, exactly as Section 3 described.

### Production considerations baked in

- **Fails open.** If a gauge can't be read (Redis down, pool object missing), the
  probe returns *"I don't know,"* and the evaluator treats "I don't know" as **"don't
  block."** A broken *protector* must never reject legitimate traffic. When unsure,
  **allow.** (This is the exact opposite choice from circuit breakers in Section 6 —
  and Section 11 explains why the opposite is correct there.)
- **Everything is tunable.** The thresholds (max queue length, pool-saturation
  percent, retry-after values) all live in configuration, so operators can tighten or
  loosen the pressure per environment without touching code.
- **Cheap.** Like Layer 0, it's just reading a few numbers and comparing — no
  expensive work — so it can run on every request without slowing things down.

### Where it sits in the gauntlet

Layer 1 is the **second guard**: after "is this caller ok?" comes "is the system
ok?" Only if *both* pass do we proceed to the real, expensive work — reserving
tokens (Layer 3) and handing off the write (Layer 4). And crucially, it protects the
very resources the later layers are about to use.

---

## 6. Layer 2 — Circuit Breakers (stop calling what's already broken)

*Code: [`circuit_breaker/`](./circuit_breaker/README.md) · not a single checkpoint —
it wraps the dependency calls that happen during Layers 3–4.*

### The analogy in the name

Your home has a circuit breaker in its electrical panel. When something goes wrong —
a short circuit, too much current — it **trips** and cuts power to that line. Cutting
your own power sounds bad, until you realise what it prevents: an overheating wire
starting a fire. Once things are safe, you flip it back on.

A **software circuit breaker** does the same for calls to a dependency (PostgreSQL,
Redis, RabbitMQ). When a dependency starts failing badly, the breaker **trips** and
the service *stops calling it* for a while. Same trade: a small, deliberate cut to
prevent a much bigger fire.

### The fire it prevents: cascading failure

Here's the disaster scenario, step by step. Say Redis goes down, and each call to it
hangs for a **timeout** — a fixed wait (say 5 seconds) before the call gives up.

- A "**thread**" is one worker that handles one request at a time; a service has a
  limited number of them.
- With no breaker, every incoming request tries Redis and **blocks for 5 seconds**.
- Those 5-second waits pile up: all the threads get stuck waiting on a dependency
  that's never going to answer.
- The service runs out of free threads and becomes **unresponsive to everything** —
  even requests that had nothing to do with Redis.

One sick dependency just took down the entire service. When that failure then spreads
to *other* services waiting on this one, it's called a **cascading failure** — a
chain reaction of collapse. It's one of the most common ways distributed systems die.

The breaker stops it: after a few failures it trips, and further calls **fail
instantly** (microseconds, no 5-second hang) instead of piling up. The service stays
responsive, and the sick dependency gets breathing room to recover.

### The three states (this is the whole idea)

A circuit breaker is a tiny machine with three states:

```
                  too many failures in a row
       ┌────────┐ ─────────────────────────▶ ┌────────┐
       │ CLOSED │                             │  OPEN  │
       │ normal │ ◀───────────────────────── │ tripped│
       └────────┘   the test call succeeded  └────────┘
            ▲                                     │ after a wait ("recovery timeout")
            │ test call succeeds                  ▼
            │                               ┌───────────┐
            └──────────────────────────────│ HALF_OPEN │  let ONE test call through
                     test call fails ──────▶│  testing  │
                     (back to OPEN)         └───────────┘
```

- **CLOSED** = normal. Calls flow through; the breaker quietly counts consecutive
  failures. ("Closed" means the circuit is complete and electricity — i.e. traffic —
  flows. It's the *healthy* state, which trips people up at first.)
- **OPEN** = tripped. Calls are refused instantly without touching the dependency.
  This is the state that saves you. It stays open for a set **recovery timeout**.
- **HALF_OPEN** = cautiously testing. After the timeout, the breaker lets **one** call
  through as a probe. Succeeds → back to CLOSED (recovered). Fails → back to OPEN for
  another wait. This stops a herd of requests from slamming a still-fragile dependency
  the instant the timer expires.

### Three breakers, tuned differently

The service has **three independent breakers**, one per dependency, each tuned for
its role:

| Breaker | Trips after | Stays open for | Where its state lives |
|---|---|---|---|
| PostgreSQL | 5 failures | 30 seconds | local memory |
| Redis | 3 failures | 10 seconds | shared in Redis |
| RabbitMQ | 3 failures | 15 seconds | shared in Redis |

Two design choices worth understanding:

- **The database breaker keeps its state in *local memory*, on purpose.** If it
  stored its state in Redis, then a *Redis* outage would also break the *database*
  breaker — chaining two unrelated failures together. Keeping it local means DB
  protection works no matter what Redis is doing.
- **The Redis and RabbitMQ breakers share their state *through Redis*,** so that all
  replicas of the service agree. If one replica trips the Redis breaker, every
  replica instantly sees it as open and they *all* stop hammering the sick dependency
  together — instead of each having to rediscover the outage on its own.

### How the rest of the code uses it

Everywhere the service makes a risky call, it wraps it in one line — conceptually:

```
   result = breaker.run( the_real_dependency_call )
       • breaker CLOSED/HALF_OPEN → run the call, record success/failure
       • breaker OPEN             → skip it, raise "CircuitBreakerError" immediately
```

Callers catch that `CircuitBreakerError` and **fall back** — e.g. the Redis fast path
falling back to the database, or the queue publish falling back to a synchronous
write. The breaker turns "hang forever on a dead dependency" into "fail instantly and
take the backup route." (And its *state* is exactly what Backpressure reads as Gauge
3 — so a tripped DB breaker also makes Layer 1 start shedding load. The layers feed
each other.)

### The deliberately-opposite choice: fail **closed**

Backpressure fails **open** ("when unsure, allow"). Circuit breakers do the reverse —
they fail **closed** ("when unsure, block"): if a breaker literally can't read its own
state, it assumes the dependency is unhealthy and stays OPEN. Why opposite? Because
they guard opposite risks. Backpressure decides *whether to admit load* — wrongly
rejecting a healthy request has a real cost, so it leans toward admitting. A breaker
guards *against calling something that might be on fire* — if it can't tell, the safe
bet is "don't touch it." **Assume-the-best for admission; assume-the-worst for
protection.** Holding both at once is a sign you genuinely understand the design.

### Where it sits in the gauntlet

Layer 2 isn't a single door you pass through; it's a **guard wrapped around each
dependency call** in Layers 3 and 4. It's what lets those layers degrade gracefully:
when Redis or RabbitMQ or PostgreSQL misbehaves, the breaker makes the call fail fast
so the request can take a backup path instead of hanging.

---

## 7. Layer 3 — Redis Fast Path (reserve capacity in ~1 millisecond)

*Code: [`redis_token_counter/`](./redis_token_counter/README.md) · the first layer
that does real work instead of just guarding.*

### What just changed

Layers 0, 1, and 2 were all **guards** — they only decide *whether* a request may
proceed. Layer 3 is the first one that actually *does the job*: it reserves the
tokens. This is the heart of the whole service, and it has to be both **fast** (~1
millisecond — a thousandth of a second) and **correct** (never hand out capacity that
isn't there).

### Why not just use the database?

The source of truth for token counts is PostgreSQL. But reading from and writing to a
database takes several milliseconds and struggles under heavy traffic. Doing that on
every single LLM call would be far too slow. So instead, the live count is kept in
**Redis** — an **in-memory** store (it keeps data in RAM, not on disk), which makes it
extraordinarily fast. Reserving against Redis is the "fast path." (Keeping Redis and
PostgreSQL in agreement is a separate job — that's reconciliation, Section 9.)

### The hard part: the race condition

Here's the naive way to reserve tokens, and why it's subtly broken. To reserve 10
tokens:

```
   1. READ the current count      (say it's 90, and the limit is 100)
   2. CHECK 90 + 10 ≤ 100         (yes, there's room)
   3. WRITE the new count = 100
```

Now imagine **two requests do this at the exact same instant.** Both perform step 1
and read `90` (neither has written yet). Both perform step 2 and see room. Both write.
The count ends at **110 — over the limit.** You just sold capacity you don't have.

This is a **race condition**: two operations "race" and the result depends on their
exact timing. The specific flavour here has a name — **TOCTOU**, "Time Of Check To
Time Of Use" — the dangerous gap between *checking* a value (step 2) and *using* it
(step 3), during which another request can slip in. Under real concurrency this isn't
rare; it happens constantly. It's the single hardest thing about a shared counter.

### The fix: make the three steps *un-interruptible* (atomic + Lua)

The problem is that another request can sneak in *between* the steps. So the fix is to
make the three steps into **one indivisible step** that nothing can interrupt. The
technical word is **atomic** — meaning "all-or-nothing, with no gap in the middle"
(from the Greek for "indivisible").

Redis provides atomicity through **Lua** — a tiny scripting language. Redis can run a
small Lua script **server-side, as a single atomic unit**: while a script runs, Redis
does *nothing else*. So if we put "read, check, write" *inside one Lua script*, those
three steps become inseparable. There is no gap for a second request to exploit. The
race is gone.

```
   ┌─────────────────────────────────────────────┐
   │  ONE atomic Lua script (nothing runs between) │
   │    read count → check vs limit → write        │
   └─────────────────────────────────────────────┘
   result:  1 = ALLOCATED   (room existed; tokens claimed)
            0 = EXHAUSTED   (would exceed the limit; refused)
           -1 = COUNTER_MISS (no count in Redis; use the database instead)
```

That is the whole reason this module uses Lua — **not** for speed tricks, but for
**correctness under concurrency**. "Check then act" is only safe when check and act
can't be split apart, and an atomic script is how you make them inseparable.

### The safety net: "fail through" to the database

What if Redis is slow, down, or its circuit breaker (Layer 2) is open? The fast path
doesn't error out — it returns that **`COUNTER_MISS`** result, which the caller reads
as *"the fast path couldn't answer — go use the slower PostgreSQL path instead."* This
is called **failing through**: Redis trouble makes the request *slower* (it takes the
database route), but never *fails*. Same theme as always — degrade, don't die.

```
   reserve in Redis ─┬─ ALLOCATED / EXHAUSTED ─▶ fast answer (~1ms)
                     └─ COUNTER_MISS / error   ─▶ fall through to PostgreSQL (slower, still correct)
```

### Production considerations baked in

- **Correct under concurrency** — the atomic-Lua design is the entire point; without
  it, high traffic would silently oversell capacity.
- **Fails through, not down** — Redis problems degrade to the DB path, guarded by the
  Redis circuit breaker so the service isn't stuck waiting on a dead Redis.
- **Self-expiring** — the Redis counters carry a **TTL** (time-to-live, an automatic
  expiry). If they ever stop being refreshed, they vanish rather than serving a stale,
  wrong number — and a missing counter safely triggers the database fallback.

### Where it sits in the gauntlet

Layer 3 is the **first layer that produces a result** rather than a yes/no. Once the
tokens are reserved here, one thing remains: recording that reservation *durably*, so
it survives a crash — without making the client wait for the slow database write.
That's Layer 4.

---

## 8. Layer 4 — Queue Absorption (save it durably, without slowing the reply)

*Code: [`token_queue/`](./token_queue/README.md) · the last hot-path layer, plus a
separate consumer process that does the actual database write.*

### The problem: durable *and* fast, which seem to conflict

Layer 3 reserved the tokens in Redis. But Redis is a cache — fast, yet volatile
(data can expire or be lost). The reservation must also be written to **PostgreSQL**,
the durable source of truth, so it survives forever. The catch: a database write
takes several milliseconds, and we just spent all that effort getting the response
down to ~1ms. If we make the client wait for the database write, we throw the speed
away.

So we need two things that seem to fight each other: the write must be **durable**
(never lost) *and* the reply must be **fast** (not wait for it). The resolution is to
**decouple** them — do the write *after* replying, but do it so reliably that "after"
is safe.

### The mechanism: a durable outbox

Think of an **outbox** on a desk. You write a note ("please file this in the
database"), drop it in the outbox, and immediately get on with your day. A separate
mail clerk empties the outbox and files each note. You didn't wait for the filing;
the note won't be lost because it's sitting safely in the outbox until the clerk gets
to it.

That outbox is **RabbitMQ**, a **message broker** — software specialized in reliably
passing "messages" (small packets of work) from one program to another. The vocabulary:

- **Publish** = drop a message into the broker.
- **Consume** = pick a message up to process it.
- A **queue** = a named waiting-line inside the broker where messages sit.

So Layer 4's hot-path job is tiny: **publish** the write request to RabbitMQ and
return `201 Created` to the client *right away*. A separate **consumer** process —
running independently, not on the request — picks up the message and does the actual
PostgreSQL write in the background.

```
   /acquire ─ reserve in Redis (Layer 3) ─ PUBLISH write to RabbitMQ ─▶ 201 to client (fast!)
                                                    │
                                                    ▼   (later, off the request)
                         CONSUMER process ── reads the message ── writes to PostgreSQL
```

### Why a real queue, and not just a background task?

The key word is **durability across a crash**. Once the message is in RabbitMQ, it is
saved (RabbitMQ writes it to disk and — see below — replicates it). If the API server
crashes the instant after replying, the pending write is *already safe in the queue*
and gets processed when things recover. A simple in-memory background task would
**lose that write on a crash**. That durability is the entire justification for
running a message broker here. (Contrast: the periodic maintenance jobs in Section 9
*don't* need this, which is why they're plain timers, not a queue.)

The guarantee this gives is called **at-least-once delivery**: the system promises
every message gets processed *at least once* (possibly more, if a retry happens). The
consumer only tells RabbitMQ *"done, you can delete this"* — an **acknowledgement**,
or **"ack"** — *after* the database write succeeds. If the consumer crashes
mid-write without acking, RabbitMQ simply hands the message to another consumer. No
write is silently dropped.

### When the write fails: the retry ladder

Databases have bad moments (a brief timeout, a blip). The consumer doesn't panic or
drop the work — it climbs a ladder:

```
   write fails
      │
      ├─ retries left?  ── yes ──▶ send to a DELAYED retry queue (wait, then try again)
      │
      └─ retries used up ──▶ send to the DEAD-LETTER queue (give up gracefully, alert a human)
```

**The clever part — delayed retry with no scheduler.** How do you make a message
"wait 30 seconds, then try again" without writing any timer code? RabbitMQ does it for
you with two built-in features:

- **TTL (time-to-live)** — a message can be set to *expire* after N seconds.
- **Dead-lettering** — when a message expires (or is rejected), RabbitMQ can
  automatically *re-route* it somewhere else.

So the service creates "parking-lot" retry queues: a failed message is published to
the `retry.30s` queue, which has **no consumer** — the message just *sits* there. After
its 30-second TTL expires, RabbitMQ dead-letters it, and the routing is configured to
send it **right back to the main work queue** for another attempt. The delay is just a
message expiring; the retry is just RabbitMQ's re-routing. Elegant, and completely
scheduler-free.

### The last resort: the dead-letter queue (DLQ)

If every retry stage is used up, the message goes to the **dead-letter queue (DLQ)** —
the "needs a human" tray for messages that can't be processed. Its handler does two
important things:

1. **Compensate.** The allocation failed to persist for good, so the tokens reserved
   back in Layer 3 must be *given back* — otherwise Redis would keep counting capacity
   that was never really used (a slow leak). It calls the fast path's `release_tokens`.
   Undoing an earlier action because a later step failed is called a **compensating
   action**.
2. **Alert loudly** so a human can investigate, with the full failed message logged.

This closes the loop: even a total failure leaves the system **consistent** (no leaked
capacity) and **observable** (someone is told).

### The breaker and the fallback (tying back to Layer 2)

Every publish goes through the **RabbitMQ circuit breaker**. If RabbitMQ itself is
down, the breaker is open, the publish fails fast, and the caller falls back to
writing to PostgreSQL **synchronously** on the request. Slower for those requests, but
nothing is lost — the fast path degrades to the safe path, exactly as in every other
layer.

### Production considerations baked in

- **Durable, at-least-once, never-lost.** Messages are persistent and stored on
  **quorum queues** — a RabbitMQ queue type that keeps copies on multiple broker
  nodes, so one node dying loses nothing. Combined with ack-on-success and
  requeue-on-trouble, no write vanishes.
- **Robust retries** without blocking the API (the TTL parking-lot trick).
- **Self-healing consistency** via the DLQ's compensating token-release.
- **Independent scaling** — the consumer runs as its own process (or a pool of them),
  so you can scale writing separately from serving requests.

### Where it sits in the gauntlet

Layer 4 is the **final hot-path step**: reserve (Layer 3), hand off the durable write
(Layer 4), reply. Everything after that — the actual database write, retries, DLQ —
happens **off the request**, in the background, which is the whole reason `/acquire`
can answer in about a millisecond.

---

## 9. The background world — keeping the fast layers honest

*Code: [`token_maintenance/`](./token_maintenance/README.md) · plus the deep-dive on
the patterns here in
[`token_maintenance/PRODUCTION_PATTERNS.md`](./token_maintenance/PRODUCTION_PATTERNS.md).*

### We're leaving the request path now

Everything so far ran **on a request** — a client asked, the gauntlet answered. This
section is the *other* world from Section 3: code that runs **on a timer**, with no
request in sight. Its whole purpose is to keep the fast request-path honest and tidy
in the background. There are three such jobs.

### Job 1 — Reconciliation (the crown jewel)

Remember the setup: the live token count is kept in **Redis** (fast) *and* in
**PostgreSQL** (the durable truth). Keeping the same number in two places always
creates a problem — over time the two copies **drift** apart. A process crashes
between the Redis update and the database write; an async persist fails; a Redis key
expires. Each little mishap leaves Redis slightly wrong. Left alone, that error
compounds until the counts are simply incorrect — which means either **falsely
rejecting** requests (Redis thinks more is used than really is) or **overselling**
(Redis thinks less).

**Reconciliation** is the periodic repair job that fixes this. Every ~60 seconds it
reads the true count from PostgreSQL and corrects Redis toward it. The design accepts
that the two stores will disagree *briefly* and simply repairs the difference on a
schedule — a strategy called **eventual consistency** ("they don't have to match at
every instant, just converge over time"). This "two copies + a periodic repair loop"
is a classic, reusable pattern: a **reconciliation loop**.

```
   every ~60s:
     for each deployment:
        truth  = read PostgreSQL   (authoritative)
        cached = read Redis        (fast copy)
        if they differ → correct Redis toward the truth
        record the size of the drift (a health signal)
```

Two production-grade details make this safe:

- **It's idempotent.** "**Idempotent**" means *running it again produces the same
  end result* — like a light switch set to "on" (flip it twice, still on), as opposed
  to "add $10" (twice adds $20). Reconciliation *sets Redis toward the true value*
  rather than *applying a difference*, so running it twice, or a half-finished time,
  still lands on the right answer. That safety is what lets it use the lightweight
  coordination below instead of heavy machinery.
- **It's observable.** It sorts the drift it finds into size buckets (0, 1–10,
  11–100, …) and logs large drifts. A growing drift is an early warning that
  something upstream is misbehaving — so the repair job doubles as a **health gauge**.

### Job 2 — Queue-depth publishing (feeding Layer 1)

You've already met this one from the other side. Backpressure (Layer 1) needs to know
how long the RabbitMQ work queue is, but it can't measure the broker on every request
— too slow. So this background job measures the queue length every few seconds and
writes the number into Redis, where Layer 1 reads it instantly. **This job is the
heartbeat that keeps a whole resilience layer's gauge alive** — if it stopped, Layer
1's queue gauge would go blind.

### Job 3 — Cleanup (keeping the database bounded)

Expired allocation rows would pile up in PostgreSQL forever. This job periodically
deletes them so the table stays a sensible size. It's the least critical of the three
— pure housekeeping — but unbounded growth is a slow-motion problem.

### The production challenge unique to background jobs: "run once, not N times"

Here's a subtlety that only appears in the background world. The service runs as
several identical copies (**replicas**) for capacity and availability. But a periodic
job lives *inside every replica* — so "reconcile every 60 seconds" would try to run
**three times at once** with three replicas, and three copies writing to Redis
simultaneously could corrupt the very data they're fixing.

The fix is a **distributed lock**: before reconciling, each replica tries to grab a
shared flag in Redis; only the one that gets it runs, the others skip this round. The
flag also auto-expires, so if the holder crashes the lock doesn't get stuck forever.
It's a lightweight way to guarantee "only one runs at a time" without any heavy
coordination system — and it works *because* the job is idempotent (a skipped or
double run does no harm). (The companion PRODUCTION_PATTERNS doc explains replicas,
locks, and this whole topic from scratch.)

### A note on what this module is *not*

This module used to run its jobs through **Celery** — a heavyweight "distributed task
queue" system (a broker + worker processes + a scheduler). That was **overkill**:
Celery is built for fanning out lots of on-demand jobs to a worker fleet, whereas
these are just three small periodic heartbeats. The right-sized tool is a plain
**in-process timer loop** inside the service, using the Redis lock above for
multi-replica safety. The distinction is worth remembering: *use a queue to hand off
durable work (Layer 4); don't use one just to run a timer.* Same reasoning that made
Layer 4's queue clearly justified makes Celery here clearly not.

### Where it sits: the second world

This is the entirety of the background world from Section 3. It never handles a
request and never calls the request path; it communicates only by leaving values in
the shared stores — repairing Redis against PostgreSQL, and dropping the queue-depth
number where Layer 1 will read it. Quiet, periodic, and essential to keeping the fast
world correct.

---

## 10. How the layers fail together — the real production question

We've met the components one at a time. But the question that actually matters in
production is: **when something breaks, what happens across the *whole* stack?** This
is where the design pays off, and where you see the layers were never really separate
— they back each other up.

### The governing principle: defense in depth

The stack is built on **defense in depth** — a term borrowed from castle design: not
one wall, but *many* walls, so that breaching one doesn't lose the castle. Here it
means **no single failure is catastrophic, because another layer catches it.** Every
layer has a fallback, and the fallbacks overlap. Let's prove it by knocking out each
dependency in turn.

### When a dependency goes down

Recall the three outside systems: **Redis** (fast cache), **PostgreSQL** (durable
truth), **RabbitMQ** (durable outbox). Here's what happens to the *whole* service when
each one fails:

**① Redis goes down**

| Layer affected | What happens |
|---|---|
| Layer 0 (rate limiter) | Fails **open** — requests still pass (limiting is briefly off) |
| Layer 1 (backpressure) | Loses its queue-depth gauge (fails open on it); the DB-pool and breaker gauges still work |
| Layer 3 (fast path) | Redis breaker trips → reserve **fails through** to the synchronous PostgreSQL path |
| Background reconciliation | Pauses (it needs Redis) — drift repair waits, no immediate harm |

> **Net effect:** the service gets **slower** (it's now doing token math directly
> against PostgreSQL) but keeps serving correctly. Degraded, not down.

**② PostgreSQL goes down**

| Layer affected | What happens |
|---|---|
| Layer 3 (fast path) | Still reserves from **Redis** in ~1ms — Redis doesn't need PostgreSQL to answer |
| Layer 4 (async write) | The consumer can't write; messages wait safely in the **retry queues** and drain when PostgreSQL returns — **nothing is lost** |
| Layer 2 (DB breaker) | Trips after a few failures |
| Layer 1 (backpressure) | Sees the tripped DB breaker (Gauge 3) → starts **shedding new requests with 503 + Retry-After** |

> **Net effect:** the service **deliberately refuses new work** with a polite "try
> again soon" (rather than accepting allocations it can't durably record), while
> already-queued writes sit safely and flush when the database recovers. Controlled
> degradation, **zero data loss**, no collapse.

**③ RabbitMQ goes down**

| Layer affected | What happens |
|---|---|
| Layer 4 (publish) | RMQ breaker trips → publish fails fast → caller falls back to a **synchronous** PostgreSQL write on the request |
| Layer 1 (backpressure) | Loses its queue-depth gauge (fails open on it); other gauges still protect |

> **Net effect:** requests get **slower** (they now write to the database inline) but
> stay durable and correct. Degraded, not down.

Notice the pattern in every row: a dependency dies, a **circuit breaker** trips, and
some layer takes a **fallback** path. The service always ends up *slower* or *shedding
some load* — never *broken*.

### When a resilience layer *itself* misbehaves ("who guards the guards?")

A fair question: what if one of the protective layers has a bug or fails? The design
is careful here too — each layer fails into a *safe* state:

| If this fails… | The result |
|---|---|
| Rate limiter (Layer 0) | Fails open — all requests pass; backpressure + breakers still protect the system |
| Backpressure (Layer 1) | No load-shedding, so queues may grow — but the API keeps serving, and breakers still guard each dependency |
| Circuit breakers (Layer 2) | Default to a safe state; individual calls just succeed or fail on their own merits |
| Redis fast path (Layer 3) | Falls through to the database path |
| Queue absorption (Layer 4) | Falls back to a synchronous database write |

No single layer failing takes the service down — the layer below (or beside) it
absorbs the loss.

### The punchline: it takes *multiple* simultaneous failures to break

Add it all up and you get the real strength of the design: **any one thing failing
only degrades the service.** To actually take it *down*, you'd need several failures
at once — for example, **both Redis *and* PostgreSQL** unreachable at the same time,
so there's no fast path *and* no fallback. Single failures bend the system; only a
rare, compound failure could snap it. That is exactly what "resilient" means in
practice — and it's the direct payoff of every fallback and breaker in the previous
nine sections.

---

## 11. Cross-cutting patterns — the reusable ideas underneath it all

If you re-read the previous sections, the *same handful of ideas* keep showing up in
different costumes. Those recurring ideas are **patterns** — named, reusable solutions
to problems that appear over and over. Learn them once here and you'll recognise them
everywhere, in this codebase and far beyond it. This section is the toolbox.

### 1. Graceful degradation (the umbrella idea)

**What it is:** when something breaks, get *worse*, not *broken*. Every layer, on
failure, takes a slower-but-working path instead of collapsing.
**Where you saw it:** all of Section 10 — Redis down → DB path; RabbitMQ down →
synchronous write; PostgreSQL down → shed load politely.
**The takeaway:** the goal of resilience is never "no failures" (impossible); it's
"failures that *bend* the system instead of *snapping* it."

### 2. Fail-fast + fallback

**What it is:** when a dependency is unhealthy, *stop trying immediately* (fail fast,
in microseconds) and take a pre-planned backup route (fall back), rather than hanging.
**Where you saw it:** circuit breakers (Layer 2) turning a 5-second hang into an
instant `CircuitBreakerError`, and callers catching it to take the DB path.
**The takeaway:** a fast, honest failure you can react to beats a slow one that ties
up resources.

### 3. Fail-open vs fail-closed (choose your direction of safety)

**What it is:** when a component is *unsure* what to do, it must default *somewhere* —
either **open** (allow / let traffic through) or **closed** (block / stop traffic).
Which default is "safe" depends on what you're guarding.
**Where you saw it:** backpressure fails **open** ("if I can't read a gauge, don't
block a legitimate request"); circuit breakers fail **closed** ("if I can't tell
whether the dependency is alive, assume it's on fire and don't touch it").
**The takeaway:** *assume-the-best for admission decisions; assume-the-worst for
protection decisions.* Always ask "which way does this fail, and is that the safe
way?"

### 4. Atomicity for "check-then-act"

**What it is:** an **atomic** operation happens all-at-once with no interruptible gap
in the middle. Any logic shaped like "read a value, decide, write it back" is unsafe
under concurrency *unless* those steps are atomic.
**Where you saw it:** the Redis fast path (Layer 3) wrapping "read count → check limit
→ write" in one atomic Lua script to defeat the race condition.
**The takeaway:** the moment two things can touch the same data at the same time,
reach for an atomic operation — it's the difference between a correct counter and one
that silently oversells.

### 5. Idempotency (safe to run again)

**What it is:** an **idempotent** action produces the same end result whether it runs
once or many times (set-to-a-value, not add-a-delta).
**Where you saw it:** reconciliation *sets* Redis toward the true value, so re-running
it is harmless; that safety is exactly what lets it use a lightweight lock and
at-least-once delivery.
**The takeaway:** design operations to be idempotent and retries become *free* — you
can re-run on any doubt without fear.

### 6. At-least-once delivery + acknowledgement

**What it is:** a durable handoff guarantees a message is processed *at least once*.
The receiver **acknowledges** ("ack") only *after* it succeeds; if it dies first, the
message is redelivered. (Pairs naturally with idempotency, since "at least once" can
mean "more than once.")
**Where you saw it:** Layer 4's RabbitMQ queue — ack-on-success, requeue-on-crash, so
no database write is ever silently lost.
**The takeaway:** for work that *must not be lost*, use a durable queue and ack only
on real success — never before.

### 7. Bulkhead (isolate callers so one can't sink the rest)

**What it is:** partition a shared resource so one bad actor's failure is contained.
Named after a ship's watertight compartments — one flooded section doesn't sink the
vessel.
**Where you saw it:** rate limiting (Layer 0) giving each calling service its *own*
budget bucket, so a runaway caller hits only its own wall.
**The takeaway:** whenever many users share one resource, ask "can one of them ruin it
for everyone?" — and compartmentalise if so.

### 8. Cost-ordered guards (reject early and cheap)

**What it is:** run your checks cheapest-first, so a request that will be refused is
refused before it consumes expensive resources.
**Where you saw it:** the gauntlet order — a cheap rate-limit check, then a cheap
health check, *then* the expensive Redis/RabbitMQ work.
**The takeaway:** put your cheapest "no" first; never spend on a request you're going
to reject anyway.

### 9. Distributed coordination (singletons & locks done safely)

**What it is:** two techniques for sharing state safely across concurrency. A
**singleton** = exactly one shared instance of a thing per process (e.g. one Redis
connection pool), built with **double-checked locking** (a check-lock-check dance so
two threads can't both build it). A **distributed lock** = a shared flag (in Redis)
so only one *replica* runs a job at a time.
**Where you saw it:** the shared service registries and circuit-breaker registry
(singletons), and reconciliation's Redis lock (run-once across replicas).
**The takeaway:** shared state under concurrency always needs coordination — within a
process (a lock/singleton) and across processes (a distributed lock).

### 10. TTL / self-expiring data (stale data cleans itself up)

**What it is:** a **TTL** (time-to-live) is an automatic expiry stamped on cached
data. If it stops being refreshed, it vanishes on its own.
**Where you saw it:** the Redis token counters and the queue-depth signal — both
expire, so a stopped publisher or a dead process leaves *"unknown"* (which fails
safely) rather than a stale, wrong value that lingers forever.
**The takeaway:** give cached/derived data a TTL so "nobody updated this" degrades to
"we don't know" instead of "here's an old lie."

### The meta-lesson

None of these are exotic. They're a small, learnable vocabulary — and *this whole
service is just these ten ideas, combined thoughtfully.* Recognise them, and any
resilient system (including your next one) stops looking like magic and starts looking
like a known set of moves.

---

## 12. Production considerations — what it takes to actually run this

Correct code is only half of "production-ready." The other half is *operating* it:
running it at scale, seeing what it's doing, tuning it, and starting/stopping it
cleanly. This section covers those operational realities in plain terms.

### Running many copies at once (horizontal scaling)

In production the token manager doesn't run as one process — it runs as several
identical copies, called **replicas**, sitting behind a **load balancer** (a traffic
cop that spreads incoming requests across them). You do this for two reasons: **more
throughput** (three copies handle roughly three times the traffic) and **availability**
(if one copy crashes, the others keep serving — no downtime). Adding more copies is
called **horizontal scaling** ("scale out"), as opposed to making one copy bigger
("scale up").

The whole design is built to make this safe:

- **Requests carry no local state.** Each request's truth lives in the *shared* stores
  (Redis, PostgreSQL), not in one server's memory. So any replica can handle any
  request identically — a property called being **stateless**. That's what makes it
  safe to spread requests freely across replicas.
- **Shared signals live in Redis**, so all replicas agree: the rate-limit counts, the
  distributed-state circuit breakers, the queue-depth gauge. One replica tripping the
  Redis breaker means *all* of them see it.
- **Background jobs use a distributed lock** (Section 9) so "run every 60 seconds"
  doesn't become "run N times at once" across N replicas.

Without these three, running multiple copies would double-count, disagree, and corrupt
data. With them, you can add replicas freely.

### Seeing what it's doing (observability)

A background failure is *invisible* — no user is watching a timer job, so it can be
silently broken for days. The fix is **observability**: deliberately making the
system's internal state *visible*. This service builds it in several ways:

- **Structured logs.** Every significant event is logged with named fields (not just
  prose), so you can search and alert on them: circuit-breaker state changes
  (`CLOSED -> OPEN`), each backpressure rejection with its reason, reconciliation's
  drift summary, DLQ "needs a human" alerts. "Structured" means machine-searchable,
  not just human-readable.
- **Health endpoints.** The service exposes HTTP endpoints that report whether it and
  its dependencies are healthy — used both by humans and by the orchestrator (see
  below). These are the **readiness** and **liveness** checks: *liveness* = "is the
  process alive?", *readiness* = "is it ready to take traffic?"
- **Meaningful signals, not just up/down.** The reconciliation drift histogram and the
  queue-depth number are *health gauges* in their own right — a growing drift or a
  growing queue is an early warning long before anything actually fails.

What an operator would **watch** in a dashboard: the rate of 429s (callers being
throttled), the rate of 503s (load being shed), each circuit breaker's state, the
queue depth over time, and the reconciliation drift. Those five tell you almost
everything about the system's health. (In a fuller setup you'd feed these into
**metrics** — numeric time-series — and define **SLOs**, "Service Level Objectives":
explicit targets like "99.9% of requests succeed," with alerts when you're burning
through your error budget. The hooks for that are the structured logs already here.)

### Tuning it without touching code

Every threshold in the whole stack lives in **configuration**, not hard-coded:
rate-limit budgets, backpressure thresholds and retry-after values, circuit-breaker
failure counts and recovery timeouts, TTLs, retry schedules, reconciliation interval.
Defaults live in YAML files, and any of them can be overridden per environment via
environment variables (`.env`) — so you can run tighter limits in one environment and
looser in another, or tune under load, **without a code change or redeploy**. That
separation of *policy* (config) from *mechanism* (code) is itself a production virtue.

### Starting and stopping cleanly (lifecycle)

- **On startup**, the service declares its RabbitMQ queues (idempotently — safe to run
  every boot) and runs its dependency health checks. Note a deliberate choice from
  Section 10: a failing *token-maintenance* check does **not** block startup (the
  service can serve requests without the background jobs), whereas a truly required
  dependency would.
- **On shutdown**, it closes its connection pools and clients cleanly, so it doesn't
  leak resources or leave half-open connections. The queue consumer, running as its
  own process, drains and stops on an interrupt signal.

### Earning trust: how you'd *verify* all this

A crucial production truth: **passing unit tests does not prove a resilience layer
works under stress** — stress behaviour is emergent and only appears with real load
and real failures. Confidence comes from a *ladder* of testing: unit tests (is the
logic right?) → integration tests (do the real dependencies wire up?) → **load tests**
(does it degrade gracefully when driven past its limits?) → **fault injection / chaos**
(kill Redis and *watch* the breaker trip and the fallback engage) → **production
observability** (keep watching, continuously). The last two rungs — deliberately
breaking things and watching the system bend — are what actually turn "I think it
works" into "I've seen it work."

### The bottom line

Production-readiness here isn't one feature — it's the sum of: **scales out safely**
(stateless + shared state + locks), **is observable** (structured logs + health checks
+ meaningful gauges), **is tunable** (config-driven), **starts and stops cleanly**, and
**is verifiable** (the testing ladder). Those are the operational counterparts to all
the correctness work in Sections 4–11.

---

## 13. End-to-end walkthrough — one request through the whole machine

Time to make it concrete. We'll follow a *single* request all the way through, first
when everything is healthy, then when a dependency is down — so you can watch the
layers cooperate as one system instead of ten separate parts.

**Our example request:** the `ms-gateway` service calls
`POST /acquire` asking to reserve **1,000 tokens** on model **gpt-4o**, before it
sends an LLM request.

### Walkthrough A — the happy path (everything healthy)

```
 t=0.0ms   CLIENT: POST /acquire  {service: ms-gateway, model: gpt-4o, tokens: 1000}
              │
 t=0.1ms   Layer 0 · Rate limit      bucket "ms-gateway:IP" = 12/500  → ✔ allow
              │
 t=0.2ms   Layer 1 · Backpressure    queue depth ok · DB pool 40% · breaker CLOSED → ✔ allow
              │
 t=0.6ms   Layer 3 · Redis fast path  atomic Lua: 8000+1000 ≤ 100000?  → ✔ ALLOCATED
              │                        (Redis breaker CLOSED — normal)
 t=1.0ms   Layer 4 · Publish          send "persist this" to RabbitMQ  → ✔ published
              │                        (RabbitMQ breaker CLOSED — normal)
 t=1.2ms   RESPONSE: 201 Created  ───────────────────────────────────▶ back to ms-gateway
              ╎
              ╎  (the client is DONE at ~1.2ms — everything below is off the request)
              ▼
 t≈50ms    CONSUMER: reads the message → writes the row to PostgreSQL → ack ✔
 t≈60s     RECONCILE: checks Redis vs PostgreSQL for gpt-4o → they match → drift 0 ✔
```

Read what just happened: five layers cooperated, the client got a correct answer in
**~1.2 milliseconds**, and the durable database write happened **50ms later in the
background** without the client ever waiting for it. That gap between "replied at
1.2ms" and "persisted at 50ms" — safely bridged by the queue — is the entire reason
the service is fast. And a minute later, reconciliation quietly confirms Redis and
PostgreSQL still agree.

### Walkthrough B — the same request, but RabbitMQ is down

Now the message broker is having an outage. Watch where the trace **diverges** — and
where it stays exactly the same:

```
 t=0.0ms   CLIENT: POST /acquire  {ms-gateway, gpt-4o, 1000}         (identical)
 t=0.1ms   Layer 0 · Rate limit    → ✔ allow                          (identical)
 t=0.2ms   Layer 1 · Backpressure  → ✔ allow                          (identical)
 t=0.6ms   Layer 3 · Redis reserve → ✔ ALLOCATED                      (identical)
              │
 t=0.7ms   Layer 4 · Publish        RabbitMQ breaker is OPEN
              │                      → publish FAILS FAST (no hanging) ◀── the divergence
              │
 t=8ms     FALLBACK: write the row to PostgreSQL SYNCHRONOUSLY, right now, on the request
              │
 t=8.2ms   RESPONSE: 201 Created  ───────────────────────────────────▶ back to ms-gateway
```

The client **still gets its 201** — the reservation is still correct and still
durably saved. The only difference is the request took **~8ms instead of ~1.2ms**,
because it had to write to the database inline instead of handing off to the (dead)
queue. That is graceful degradation in one picture: **slower, but working.** The
circuit breaker made the dead broker fail *instantly* instead of hanging for a
timeout, and the synchronous-write fallback kept the request correct.

### Walkthrough C — the worst single failure, briefly (PostgreSQL down)

What if instead *PostgreSQL* — the durable truth — is down for a sustained period?
Then there's no safe place to record new allocations, so the honest answer is to stop
accepting them. After a few failed writes trip the DB circuit breaker, **Layer 1
(backpressure) sees the open DB breaker and starts replying `503 Service Unavailable`
+ `Retry-After: 30`.** The client is politely told "come back in 30 seconds" and backs
off; meanwhile any writes already in the queue wait safely and flush the moment
PostgreSQL recovers — **nothing is lost.** The service refuses *cleanly* rather than
collapsing.

### The whole point, in one line

Same request, three different world-states, three sensible outcomes:

| World state | Outcome | Client experience |
|---|---|---|
| Healthy | 201 in ~1.2ms | fast success |
| RabbitMQ down | 201 in ~8ms | slower success |
| PostgreSQL down | 503 + Retry-After | polite "try again in 30s" |

**Never** a hang, never a crash, never a lost write, never oversold capacity. That is
the resilience layer doing its job — and it's the sum of every pattern in this
document working together on one request.

---

## 14. Glossary — every term in one place

An alphabetical reference for every technical term used in this document. Skim it, or
jump back here whenever a word stops meaning something.

- **Acknowledgement (ack)** — a consumer telling the message broker "I've finished
  this message, you may delete it." Sent only *after* the work succeeds, so a crash
  before the ack causes redelivery instead of loss.
- **At-least-once delivery** — a durable-queue guarantee that every message is
  processed *at least* once (possibly more, if a retry happens). Pairs with
  idempotency.
- **Atomic / atomicity** — an operation that happens all-at-once with no interruptible
  gap in the middle ("indivisible"). Makes "read-check-write" safe under concurrency.
- **Backpressure** — pushing back on senders when a system is receiving work faster
  than it can process, by refusing new work (here, with a 503). From plumbing.
- **Bucket** — a named group of requests that share one rate-limit budget (e.g. per
  calling service). See *rate limiting*.
- **Bulkhead** — isolating a shared resource so one bad actor can't sink the rest;
  named after a ship's watertight compartments. Rate-limit buckets are a bulkhead.
- **Cache** — a fast, usually in-memory store of data for quick access (here, Redis).
  Fast but volatile; the durable copy lives elsewhere.
- **Cascading failure** — a chain reaction where one failure (a slow dependency)
  causes another (threads pile up), collapsing the whole service and spreading outward.
- **Circuit breaker** — a device that "trips" to stop calls to a failing dependency,
  preventing cascading failure. Has three states: **CLOSED** (normal), **OPEN**
  (tripped, failing fast), **HALF_OPEN** (testing recovery with one probe call).
- **Compensating action** — undoing an earlier action because a later step failed
  (here, releasing reserved Redis tokens when the durable write terminally fails).
- **Connection pool** — a small, fixed set of reusable open connections to a
  dependency (e.g. PostgreSQL), shared instead of opening a new one per request.
- **Consumer** — a process that reads and processes messages from a queue.
- **Dead-letter queue (DLQ)** — the "needs a human" queue where messages that can't be
  processed after all retries end up, for alerting and manual review.
- **Dead-lettering** — RabbitMQ automatically re-routing a message when it expires or
  is rejected; used here to build delayed retries.
- **Defense in depth** — using many overlapping layers of protection so breaching one
  doesn't lose everything. From castle design.
- **Dependency** — an outside system this service relies on: Redis, PostgreSQL,
  RabbitMQ.
- **Distributed lock** — a shared flag (in Redis) that lets only one *replica* run a
  job at a time, so a periodic job doesn't run N times across N copies.
- **Drift** — the gradual disagreement between two copies of the same data (Redis vs
  PostgreSQL token counts) as small mishaps accumulate. Repaired by reconciliation.
- **Eventual consistency** — accepting that two data copies may differ *briefly* and
  will *converge* over time (via reconciliation), rather than matching at every instant.
- **Fail fast** — failing immediately (microseconds) when a dependency is known-bad,
  instead of hanging on a timeout.
- **Fail open / fail closed** — which way a component defaults when unsure: **open** =
  allow/let through (backpressure); **closed** = block/stop (circuit breakers).
- **Fallback** — a pre-planned backup path taken when the primary one fails (e.g. Redis
  fast path → database path).
- **Fixed vs moving window** — two ways to count "per minute": *fixed* resets on the
  clock (exploitable at the boundary); *moving* always looks back over the last 60
  seconds (no boundary to exploit). Rate limiting uses moving.
- **Graceful degradation** — getting *worse* rather than *broken* when something fails;
  the umbrella goal of the whole resilience layer.
- **Health check (liveness / readiness)** — endpoints reporting service health.
  *Liveness* = "is the process alive?"; *readiness* = "is it ready for traffic?"
- **Horizontal scaling** — adding more copies (replicas) of a service to handle more
  load, versus *vertical scaling* (making one copy bigger).
- **Hot path** — the code that runs on every request and must be fast (here, the
  `/acquire` gauntlet). Contrast with background work.
- **Idempotent** — an operation that yields the same result run once or many times
  (set-to-a-value, not add-a-delta). Makes retries safe.
- **In-memory** — data kept in RAM rather than on disk; very fast but lost if the
  process stops (unless persisted). Redis is in-memory.
- **Load balancer** — infrastructure that spreads incoming requests across replicas.
- **Load shedding** — deliberately dropping/refusing some work quickly so the rest
  stays fast, instead of failing at everything slowly. Backpressure's 503 is this.
- **Lua** — a small scripting language Redis can run server-side *atomically*; used to
  make token reservation race-free.
- **Message broker** — software that reliably passes messages between programs (here,
  RabbitMQ). Provides the durable "outbox."
- **Metrics** — numeric time-series (e.g. requests/sec, error rate) used to watch and
  alert on system health.
- **Observability** — deliberately making a system's internal state visible (via logs,
  health checks, metrics) so you can tell what it's doing, especially when it breaks.
- **Publish / consume** — *publish* = drop a message into the broker; *consume* = pick
  one up to process it.
- **Queue** — a named waiting-line inside a broker where messages sit until consumed.
- **Queue depth** — how many messages are waiting in a queue; a backpressure gauge and
  an overload early-warning.
- **Quorum queue** — a RabbitMQ queue type that keeps replicated copies across broker
  nodes, so one node failing loses no messages.
- **Race condition** — a bug where the outcome depends on the exact timing of
  concurrent operations (see *TOCTOU*).
- **Rate limiting** — capping how many requests a caller may make per unit time;
  returns *429* when exceeded.
- **Recovery timeout** — how long a circuit breaker stays OPEN before testing the
  dependency again with a HALF_OPEN probe.
- **Reconciliation (loop)** — a periodic job that repairs a fast copy of data (Redis)
  toward the durable source of truth (PostgreSQL), correcting drift.
- **Replica** — one running copy of the service; production runs several for
  throughput and availability.
- **Retry-After** — an HTTP header giving the client a concrete number of seconds to
  wait before retrying, sent with a 503.
- **Singleton** — exactly one shared instance of something per process (e.g. one
  connection pool), typically built with **double-checked locking** (check → lock →
  check again) so two threads can't both create it.
- **SLO (Service Level Objective)** — an explicit reliability target (e.g. "99.9% of
  requests succeed") that alerts are built around.
- **Source of truth** — the authoritative, durable copy of the data (PostgreSQL);
  other copies (Redis) are reconciled toward it.
- **Stateless** — holding no per-request state locally, so any replica can handle any
  request identically. What makes horizontal scaling safe.
- **Structured logging** — logging events with named, machine-searchable fields (not
  just prose) so you can search and alert on them.
- **Thread** — one worker that handles one request at a time; a service has a limited
  number, which is why hanging on a dead dependency is dangerous.
- **Throughput** — how much work a system handles per unit time (e.g. requests/sec).
- **TOCTOU** — "Time Of Check To Time Of Use": the dangerous gap between checking a
  value and using it, during which another operation can change it. A race condition.
- **TTL (time-to-live)** — an automatic expiry on cached data, so stale data cleans
  itself up (becomes "unknown," which fails safely) instead of lingering wrong.
- **429 / 503** — HTTP status codes. **429 Too Many Requests** = rate-limited (slow
  down). **503 Service Unavailable** = healthy but temporarily overloaded (retry soon).

---

### That's the whole map

You've now walked the entire resilience layer: the problem it solves (Section 1), the
grand picture (2), the two mental models (3), each layer in turn (4–8), the background
world (9), how they fail together (10), the reusable patterns (11), production
realities (12), a full request trace (13), and this glossary (14).

**Where to go from here:** each component has its own deep-dive README next to its
code, and each of those has step-by-step comments in the source itself:
[backpressure/](./backpressure/README.md) ·
[circuit_breaker/](./circuit_breaker/README.md) ·
[redis_token_counter/](./redis_token_counter/README.md) ·
[token_queue/](./token_queue/README.md) ·
[token_maintenance/](./token_maintenance/README.md) ·
[token_maintenance/PRODUCTION_PATTERNS.md](./token_maintenance/PRODUCTION_PATTERNS.md).

If this document did its job, none of those should feel like magic anymore — just a
known set of moves, combined thoughtfully.

*Document complete — 14 of 14 sections.*
