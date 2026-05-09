"""
Redis Lua script definitions - atomic script sources for counter operations.

Architecture:
-------------
    ┌──────────────────────────────┐     ┌──────────────────────────────┐
    │ counter_service.py           │────▶│ lua_script_definitions.py    │
    │ registers and executes       │     │ Lua source constants         │
    └──────────────────────────────┘     └──────────────────────────────┘

Dependencies:
    - Redis Lua runtime - executes scripts atomically server-side

Author: Engineering Team
Last Updated: 2026-05-09
"""

LUA_RESERVE_TOKENS = """
local counter_val = redis.call('GET', KEYS[1])
if counter_val == false then
    return -1
end
local limit_val = redis.call('GET', KEYS[2])
if limit_val == false then
    return -1
end
local current = tonumber(counter_val)
local limit = tonumber(limit_val)
local requested = tonumber(ARGV[1])
if current + requested > limit then
    return 0
end
redis.call('INCRBY', KEYS[1], requested)
return 1
"""

LUA_RELEASE_TOKENS = """
local current = tonumber(redis.call('GET', KEYS[1]) or '0')
local release = tonumber(ARGV[1])
local new_val = math.max(0, current - release)
redis.call('SET', KEYS[1], new_val)
return new_val
"""

LUA_SEED_COUNTER = """
local allocated = tonumber(ARGV[1])
local max_limit = tonumber(ARGV[2])
local ttl_seconds = tonumber(ARGV[3])
redis.call('SET', KEYS[1], allocated, 'EX', ttl_seconds)
redis.call('SET', KEYS[2], max_limit, 'EX', ttl_seconds)
return 1
"""

LUA_RECONCILE_COUNTER = """
local counter_val = redis.call('GET', KEYS[1])
local limit_val = redis.call('GET', KEYS[2])
local db_allocated = tonumber(ARGV[1])
local db_limit = tonumber(ARGV[2])
local ttl_seconds = tonumber(ARGV[3])

if counter_val == false and limit_val == false then
    redis.call('SET', KEYS[1], db_allocated, 'EX', ttl_seconds)
    redis.call('SET', KEYS[2], db_limit, 'EX', ttl_seconds)
    return 3
end

if counter_val == false then
    redis.call('SET', KEYS[1], db_allocated, 'EX', ttl_seconds)
    redis.call('SET', KEYS[2], db_limit, 'EX', ttl_seconds)
    return 2
end

local redis_allocated = tonumber(counter_val)
local delta = db_allocated - redis_allocated
local new_allocated = redis_allocated

if delta ~= 0 then
    new_allocated = tonumber(redis.call('INCRBY', KEYS[1], delta))
    if new_allocated < 0 then
        redis.call('SET', KEYS[1], 0)
    end
end
redis.call('EXPIRE', KEYS[1], ttl_seconds)

if limit_val == false then
    redis.call('SET', KEYS[2], db_limit, 'EX', ttl_seconds)
    return 2
end

local redis_limit = tonumber(limit_val)
local limit_changed = redis_limit ~= db_limit
if limit_changed then
    redis.call('SET', KEYS[2], db_limit, 'EX', ttl_seconds)
else
    redis.call('EXPIRE', KEYS[2], ttl_seconds)
end

if delta ~= 0 or limit_changed then
    return 1
end
return 0
"""
