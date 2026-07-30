"""
Redis Lua script definitions — THE algorithm of the fast path.

Each string below is a small program that runs INSIDE Redis, atomically: while a
script runs, Redis executes nothing else. That atomicity is the whole point — it
lets "read a value, decide, write it" happen as one indivisible step, which is the
only safe way to update a shared counter under concurrency (see README.md §2–3).

The Python side (`counter_service.py`) just sends these to Redis and reads back the
integer each one returns. Those integers map 1:1 onto the enums in
`counter_results.py`:

    reserve   -> TokenReservationResult:    1 ALLOCATED | 0 EXHAUSTED | -1 COUNTER_MISS
    reconcile -> CounterReconciliationResult: 3 INITIALIZED_MISSING | 2 RESEEDED_PARTIAL
                                              | 1 DELTA_APPLIED | 0 UNCHANGED

Conventions inside the scripts:
    KEYS[1] = counter key (current allocated tokens)
    KEYS[2] = limit key   (max capacity)
    ARGV[]  = numeric arguments passed from Python
    GET on a missing key returns Lua `false` (not nil) — that's our "not seeded" signal.

Author: Engineering Team
"""

# RESERVE — atomically claim ARGV[1] tokens if there is room.
LUA_RESERVE_TOKENS = """
-- Step 1: read current allocated count. Missing key => counter not seeded yet.
local counter_val = redis.call('GET', KEYS[1])
if counter_val == false then
    return -1
end
-- Step 2: read the configured limit. Missing => also treat as "not seeded".
local limit_val = redis.call('GET', KEYS[2])
if limit_val == false then
    return -1
end
-- Step 3: would granting this request push us over the limit? If so, reject.
local current = tonumber(counter_val)
local limit = tonumber(limit_val)
local requested = tonumber(ARGV[1])
if current + requested > limit then
    return 0
end
-- Step 4: room exists -> claim the tokens (INCRBY) and report success.
redis.call('INCRBY', KEYS[1], requested)
return 1
"""

# RELEASE — give ARGV[1] tokens back, never dropping below zero.
LUA_RELEASE_TOKENS = """
-- Step 1: read current value (default to 0 if the key is gone).
local current = tonumber(redis.call('GET', KEYS[1]) or '0')
-- Step 2: subtract, clamped at 0 so the counter can never go negative.
local release = tonumber(ARGV[1])
local new_val = math.max(0, current - release)
-- Step 3: write the new value back and return it.
redis.call('SET', KEYS[1], new_val)
return new_val
"""

# SEED — initialize counter + limit from a DB snapshot, both with a TTL.
LUA_SEED_COUNTER = """
local allocated = tonumber(ARGV[1])
local max_limit = tonumber(ARGV[2])
local ttl_seconds = tonumber(ARGV[3])
-- Set both keys with an expiry so a stale/un-refreshed counter self-clears.
redis.call('SET', KEYS[1], allocated, 'EX', ttl_seconds)
redis.call('SET', KEYS[2], max_limit, 'EX', ttl_seconds)
return 1
"""

# RECONCILE — correct the Redis counter/limit toward the DB truth (drift repair).
LUA_RECONCILE_COUNTER = """
local counter_val = redis.call('GET', KEYS[1])
local limit_val = redis.call('GET', KEYS[2])
local db_allocated = tonumber(ARGV[1])
local db_limit = tonumber(ARGV[2])
local ttl_seconds = tonumber(ARGV[3])

-- Case A: neither key exists -> cold start, seed both from the DB. (return 3)
if counter_val == false and limit_val == false then
    redis.call('SET', KEYS[1], db_allocated, 'EX', ttl_seconds)
    redis.call('SET', KEYS[2], db_limit, 'EX', ttl_seconds)
    return 3
end

-- Case B: counter missing (limit present) -> reseed both from the DB. (return 2)
if counter_val == false then
    redis.call('SET', KEYS[1], db_allocated, 'EX', ttl_seconds)
    redis.call('SET', KEYS[2], db_limit, 'EX', ttl_seconds)
    return 2
end

-- Case C: counter exists -> nudge it toward the DB value by the difference.
-- Because this whole script is atomic, `redis_allocated` is unchanged between the
-- GET above and the INCRBY below, so INCRBY(delta) lands exactly on db_allocated.
-- The delta framing lets us detect the "no change" case (delta == 0 -> UNCHANGED).
local redis_allocated = tonumber(counter_val)
local delta = db_allocated - redis_allocated
local new_allocated = redis_allocated

if delta ~= 0 then
    new_allocated = tonumber(redis.call('INCRBY', KEYS[1], delta))
    -- Safety clamp: a correction must never leave the counter negative.
    if new_allocated < 0 then
        redis.call('SET', KEYS[1], 0)
    end
end
redis.call('EXPIRE', KEYS[1], ttl_seconds)

-- The limit can be missing even when the counter exists -> set it. (return 2)
if limit_val == false then
    redis.call('SET', KEYS[2], db_limit, 'EX', ttl_seconds)
    return 2
end

-- Otherwise: update the limit if the DB changed it, else just refresh its TTL.
local redis_limit = tonumber(limit_val)
local limit_changed = redis_limit ~= db_limit
if limit_changed then
    redis.call('SET', KEYS[2], db_limit, 'EX', ttl_seconds)
else
    redis.call('EXPIRE', KEYS[2], ttl_seconds)
end

-- Report whether anything actually changed (feeds the reconciliation drift stats).
if delta ~= 0 or limit_changed then
    return 1
end
return 0
"""
