# Redis Token Counter — a beginner's guide

> **Read this first.** This is the **fast path** (Layer 3) of the token manager —
> the part that reserves token capacity in ~1 millisecond. It's confusing at first
> because the real algorithm lives in **Lua scripts that run inside Redis**, which
> is not something most application code does. This guide explains the *why*, the
> *flow*, and every Lua script in plain terms, from scratch.

---

## 1. The job in one sentence

When a client calls `/acquire`, we must answer *instantly*: **"is there enough
token capacity left for this deployment, and if so, claim it."** This module does
exactly that against Redis, atomically and fast, and gets out of the way.

The authoritative record lives in PostgreSQL, but PostgreSQL is too slow to touch
on every request. So we keep a **fast counter in Redis** and reserve against it.
(The slower "keep Redis honest vs Postgres" job is reconciliation — see the
`token_maintenance` module.)

---

## 2. The problem that makes this hard: the race condition

Here's the naive way to reserve tokens, and why it's broken:

```
1. read the current counter        (say it's 90, limit is 100)
2. check 90 + 10 <= 100  → yes
3. write counter = 100
```

Now imagine **two requests do this at the same moment**. Both read `90` in step 1
(before either has written). Both check `90 + 10 <= 100` → both say yes. Both write.
The counter ends at **110 — over the limit.** You just handed out capacity you
didn't have.

This is the classic **TOCTOU race** (Time-Of-Check to Time-Of-Use): the gap
between *checking* a value and *using* it is a window where another request can
sneak in. Under real concurrency it happens constantly, and it's the single
hardest thing about a shared counter.

---

## 3. The solution: run the whole thing atomically *inside* Redis (Lua)

Redis can execute a small **Lua script server-side as one atomic unit** — while a
script runs, Redis runs **nothing else**. So if we put "read, check, write" *into
one Lua script*, those three steps become a single indivisible operation. There is
no gap for another request to slip into. The race is gone.

That is the entire reason this module uses Lua. Not for speed tricks — for
**atomicity**. "Check-then-act" is only safe when "check" and "act" can't be split
apart, and a Lua script is how Redis lets you make them inseparable.

> **Mental model:** the Lua scripts in `lua_script_definitions.py` are the actual
> algorithm. The Python around them (`counter_service.py`) just *sends* the right
> script to Redis with the right arguments and interprets the number that comes
> back. Read the Lua to understand the logic; read the Python to understand the
> plumbing.

---

## 4. The keys: two per deployment

Each deployment (a model + endpoint pair) gets **two Redis keys**, built in
`_build_counter_keys()`:

```
token:counter:{model}:{endpoint_hash}   → how many tokens are currently allocated
token:limit:{model}:{endpoint_hash}     → the maximum capacity
```

- The endpoint URL is **hashed** (`sha256`, first 16 chars) so the key stays short
  and safe regardless of how long or messy the URL is.
- The model name is sanitized (`/` and `:` → `_`) so it can't break the key format.
- Both keys carry a **TTL** (`settings.redis_token_counter_ttl_secs`). If the fast
  path stops being reconciled/seeded, the keys simply expire and the counter
  reports "missing" rather than serving stale numbers — the same fail-safe idea you
  saw with the queue-depth key.

Reserving needs both keys (you can't check a limit you don't have). Releasing only
touches the counter.

---

## 5. The four operations (the Lua algorithms, in plain terms)

There are exactly four scripts. Each returns a small integer code, and those codes
map **one-to-one** onto the Python enums in `counter_results.py` — that's the clean
contract between Lua and Python.

### ① RESERVE — the crown jewel (`LUA_RESERVE_TOKENS`)

*"Claim N tokens if there's room."* Runs atomically:

```
counter = GET counter_key
if counter is missing → return -1        # COUNTER_MISS: not seeded; caller falls back to DB
limit   = GET limit_key
if limit   is missing → return -1        # COUNTER_MISS
if counter + requested > limit → return 0   # EXHAUSTED: no capacity left
INCRBY counter_key by requested          # claim it
return 1                                  # ALLOCATED
```

The three return codes are the three things that can happen, and they line up with
`TokenReservationResult`:

| Lua returns | Enum | Meaning | What the caller does |
|---|---|---|---|
| `1` | `ALLOCATED` | tokens reserved | proceed — success |
| `0` | `EXHAUSTED` | limit would be exceeded | capacity full → client waits/retries |
| `-1` | `COUNTER_MISS` | counter not in Redis | **fall through to the PostgreSQL path** |

Because the whole thing is one atomic script, the "check `> limit`" and the
"`INCRBY`" can never be split — the race from §2 is impossible.

### ② RELEASE — give tokens back (`LUA_RELEASE_TOKENS`)

*"Return N tokens, but never go below zero."*

```
current = GET counter_key (or 0 if missing)
new     = max(0, current - release)      # clamp: a counter can never be negative
SET counter_key = new
return new
```

The `max(0, …)` is a defensive guard: even if releases and reservations get out of
sync, the counter can't underflow into a nonsensical negative balance.

### ③ SEED — warm a cold counter (`LUA_SEED_COUNTER`)

*"Initialize this deployment's counter and limit from a database snapshot."*

```
SET counter_key = allocated  (with TTL)
SET limit_key   = max_limit  (with TTL)
return 1
```

Used at startup (or first touch) to load the fast counter from PostgreSQL truth so
`reserve` has something to reserve against. Before a counter is seeded, `reserve`
returns `COUNTER_MISS` and requests fall through to the DB.

### ④ RECONCILE — repair drift (`LUA_RECONCILE_COUNTER`)

*"Correct the Redis counter toward the PostgreSQL truth."* This is the script the
`token_maintenance` reconciliation job calls every ~60s. It handles four cases and
returns which one happened (mapping to `CounterReconciliationResult`):

```
both keys missing        → seed both from DB          → return 3  (INITIALIZED_MISSING)
counter missing only     → seed both from DB          → return 2  (RESEEDED_PARTIAL)
otherwise:
    delta = db_allocated - redis_allocated
    if delta ≠ 0 → INCRBY counter by delta (clamp <0 to 0)   # move Redis to the DB value
    refresh the counter's TTL
    if limit missing → set it                        → return 2  (RESEEDED_PARTIAL)
    if limit changed → update it, else refresh TTL
    if anything changed → return 1 (DELTA_APPLIED) else return 0 (UNCHANGED)
```

**One subtlety worth understanding:** it uses `INCRBY delta` rather than
`SET db_allocated`. Because the script is atomic, `redis_allocated` read at the top
is still the value when `INCRBY` runs, so the counter lands on exactly
`db_allocated` either way — but the *delta framing* makes the "nothing to do" case
(`delta == 0`) explicit and lets it return `UNCHANGED`, which is what feeds the
reconciliation job's drift histogram.

---

## 6. The flow: how a reservation actually travels

Follow one `/acquire` reservation through the code:

```
caller (token acquisition service)
    │
    ▼
RedisTokenCounterService.reserve_tokens(model, endpoint, count)   [counter_service.py]
    │   1. wrap the call in the Redis circuit breaker (fail fast if Redis is down)
    ▼
_reserve_tokens_raw(...)
    │   2. build the two keys
    ▼
_execute_lua_script("reserve", keys, args)
    │   3. run the RESERVE Lua script atomically in Redis
    ▼
Redis executes LUA_RESERVE_TOKENS  →  returns 1 / 0 / -1
    │
    ▼
TokenReservationResult(...)  →  ALLOCATED / EXHAUSTED / COUNTER_MISS
```

Two protective layers wrap the raw Lua call, and they're the key to why this is
*resilient*, not just fast:

- **The circuit breaker** (`_call_with_redis_circuit_breaker`) — if Redis is
  failing, the breaker opens and calls fail *instantly* instead of hanging. (This
  is the `redis` breaker from the `circuit_breaker` module.)
- **Fail-through** — `reserve_tokens`/`release_tokens`/`get_counter` catch *both*
  `CircuitBreakerError` and any other exception and return the safe "miss" value
  (`COUNTER_MISS` / `None`). The caller reads that as *"the fast path couldn't
  answer — go use the slower PostgreSQL path."* Redis trouble degrades to a slower
  correct answer; it never hard-fails the request.

Note: `reserve`, `release`, and `get_counter` are breaker-wrapped (they're on the
hot request path). `seed` and `reconcile` are **not** — seeding is startup work and
reconciliation already runs under its own lock in the maintenance job.

---

## 7. Two implementation details that look odd (but are correct)

- **The NOSCRIPT retry** (`_execute_lua_script`). Redis remembers scripts by a hash
  (you register once, then invoke by hash — fast). If Redis restarts, it forgets
  the script and returns a `NOSCRIPT` error. The code catches that *once*,
  re-registers the script, and retries. It's a small self-healing step, not a bug.

- **The `Protocol` classes** (`AsyncRedisClientProtocol`, etc.). These describe the
  slice of the Redis/breaker interface this service uses, so tests can pass in
  lightweight fakes without subclassing the real Redis client. They're type-checker
  helpers — read past them; the logic is in the methods.

---

## 8. The supporting cast (the other files)

- [`lua_script_definitions.py`](./lua_script_definitions.py) — the four Lua scripts.
  **This is the algorithm.**
- [`counter_service.py`](./counter_service.py) — `RedisTokenCounterService`: builds
  keys, registers/runs scripts, wraps calls in the breaker, does fail-through.
- [`counter_results.py`](./counter_results.py) — the result enums whose integer
  values **match the Lua return codes** exactly.
- [`service_registry.py`](./service_registry.py) — creates one shared service per
  process (thread-safe singleton) and closes it on shutdown.
- [`__init__.py`](./__init__.py) — the public surface callers import from.

---

## 9. Can I reuse this? (production perspective)

Yes — this is a genuinely reusable **distributed quota / rate-limiter primitive**,
and the ideas travel:

- **Atomic check-then-act via Lua** is the correct way to build *any* Redis counter
  with a limit (rate limits, seat counts, inventory, credits). The moment you have
  "read a value, decide, write it" under concurrency, reach for a Lua script.
- **Fail-through + circuit breaker** is what makes it production-safe: a cache/quota
  layer must degrade to the source of truth, never take the request down with it.
- **The Lua-return-code ↔ enum contract** keeps the boundary clean and typed.

Seams to swap when porting: the key format (`_build_counter_keys`), the TTL and
connection settings, and the breaker (or drop it if you don't need one). The Lua
scripts themselves are portable almost as-is.

---

## 10. Thirty-second recap

- This is the **~1ms fast path**: reserve token capacity in Redis instead of the
  slow database.
- A shared counter has a **race condition** (check-then-act); the fix is running the
  whole check-and-update as **one atomic Lua script** inside Redis.
- **Four scripts:** `reserve` (claim if room), `release` (give back, clamp at 0),
  `seed` (warm from DB), `reconcile` (repair drift toward DB truth). Their return
  codes map 1:1 to the result enums.
- Every hot-path call is wrapped in the **Redis circuit breaker** and **fails
  through** to PostgreSQL — Redis trouble means "slower," never "broken."
- The pattern is a reusable, production-grade distributed quota primitive.
