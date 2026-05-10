# PgBouncer — Connection Pooler for LLM Token Manager

## Table of Contents

1. [What Is PgBouncer and Why Does This System Need It](#1-what-is-pgbouncer-and-why-does-this-system-need-it)
2. [The Problem: PostgreSQL Connection Overhead](#2-the-problem-postgresql-connection-overhead)
3. [Transaction Mode vs Session Mode](#3-transaction-mode-vs-session-mode)
4. [Architecture: Where PgBouncer Lives](#4-architecture-where-pgbouncer-lives)
5. [Connection Flow: From App to Database](#5-connection-flow-from-app-to-database)
6. [Configuration Reference (pgbouncer.ini)](#6-configuration-reference-pgbouncerini)
7. [Authentication (userlist.txt)](#7-authentication-userlisttxt)
8. [How It Is Integrated](#8-how-it-is-integrated)
9. [Failure Mode and Recovery](#9-failure-mode-and-recovery)
10. [Monitoring and Admin Interface](#10-monitoring-and-admin-interface)
11. [Transaction Mode Incompatibilities](#11-transaction-mode-incompatibilities)
12. [Sizing Rationale](#12-sizing-rationale)

---

## 1. What Is PgBouncer and Why Does This System Need It

PgBouncer is a lightweight connection pooler that sits between the application and
PostgreSQL. It accepts thousands of short-lived connections from the application and
maps them onto a small, stable set of real PostgreSQL backend connections.

PostgreSQL creates a **OS process per connection**. At the default
`max_connections = 100`, each idle connection consumes ~5–10 MB of RAM and a
background process slot — even if the connection does nothing. With Celery workers,
FastAPI's thread pool, and health checkers all opening connections simultaneously,
connection count balloons fast.

The system has 4 Celery worker types with concurrency 4–10 each. Without pooling:

```
  FastAPI:                  pool_size=20 + max_overflow=10  =  30 connections
  Celery worker:            concurrency=10                  =  10 connections
  Token maintenance worker: concurrency=4                   =   4 connections
  Celery beat:              ~                               =   2 connections
  Token queue consumer:     concurrency=8                   =   8 connections
  Health checks + misc:                                     =   5 connections
                                                          ─────────────────
  Total to PostgreSQL:                                      ~ 59 connections
```

Each of those is held **permanently**, even when idle. That is 300–600 MB in
PostgreSQL process overhead just for connection management — before a single query runs.

With PgBouncer in **transaction mode**, a connection to PostgreSQL is borrowed only
for the duration of one transaction (an INSERT, an UPDATE, etc.), then returned to
the pool. A Celery worker that is sleeping between tasks holds **zero** PostgreSQL
connections.

```
  WITHOUT PgBouncer:
  ─────────────────
  FastAPI ──────────────► PG connection 1  (held always, even idle)
  FastAPI ──────────────► PG connection 2
  FastAPI ──────────────► PG connection 3
  ... 59 connections total, all open permanently.

  WITH PgBouncer (transaction mode):
  ───────────────────────────────────
  FastAPI ──►┐
  Celery  ──►┤  PgBouncer  ──► PG connection 1  (borrowed per-txn)
  Worker  ──►┤   (pool=25) ──► PG connection 2
  ...         ──►┘             PG connection 3
                                ... 25 connections total, reused.

  When a Celery worker is idle between tasks, it holds ZERO PG connections.
  PostgreSQL sees a small, stable pool of 25 backends instead of 59.
```

---

## 2. The Problem: PostgreSQL Connection Overhead

PostgreSQL's architecture is a **process-per-connection** model. This is different
from thread-per-connection servers (like MySQL's thread pool). Each new connection
forks a new backend process. The implications:

```
  POSTGRESQL PROCESS MODEL:
  ─────────────────────────

  postgres (master)
   ├── postgres: checkpointer
   ├── postgres: background writer
   ├── postgres: walwriter
   ├── postgres: autovacuum launcher
   ├── postgres: stats collector
   ├── postgres: <backend 1> (your FastAPI connection)
   ├── postgres: <backend 2> (your Celery connection)
   ├── postgres: <backend N> (another connection)
   └── ...

  Each backend:  ~5–10 MB RSS (resident set size)
  59 connections × 7.5 MB average = ~440 MB just for connection overhead
  This is BEFORE a single query executes.

  LOCK CONTENTION CEILING:
  At 4,000 RPS burst with 20ms/query average:
    30 direct connections ÷ 0.02s = 1,500 queries/sec (theoretical)
    Real-world with lock contention: 300–500 queries/sec sustained
    4,000 RPS burst → 3,500 RPS overflow → connection queue backs up → timeouts in ~2s
```

PgBouncer in transaction mode raises this ceiling significantly:

```
  CAPACITY TABLE (from system design):
  ─────────────────────────────────────────────────────────────────────────────
  COMPONENT           WITHOUT PGBOUNCER     WITH PGBOUNCER        GAIN
  ─────────────────── ──────────────────    ──────────────────    ───────────
  PostgreSQL backends 59 persistent         25 pooled backends    ~2.4× fewer
  Idle memory (RSS)   ~440 MB               ~190 MB               ~2.3× less
  Write throughput    300–500 writes/sec    2,000+ writes/sec     4–6× more
  Connection setup    per request (~5ms)    amortised (~0ms)      eliminate
```

---

## 3. Transaction Mode vs Session Mode

PgBouncer has three pool modes. This system uses **transaction mode**.

```
  POOL MODE COMPARISON:
  ─────────────────────────────────────────────────────────────────────────────
  MODE          CONNECTION LIFETIME          BEST FOR            LIMITS
  ──────────    ─────────────────────────    ────────────────    ───────────────
  session       Entire client session        Legacy apps         No pooling benefit
                A → PG from connect to       (one connection     for async apps
                disconnect.                  per session).

  transaction   One transaction.             FastAPI/asyncpg,    Cannot use:
  (THIS SYSTEM) Released after COMMIT or     Celery workers,       SET session_var
                ROLLBACK. Reused by next     high-write OLTP.      LISTEN/NOTIFY
                client transaction.                                advisory locks
                                                                   prepared stmts

  statement     One SQL statement.           Very simple         Cannot use:
                Released after each stmt.    read-only apps.       multi-stmt txns
                                                                   any txn control
  ─────────────────────────────────────────────────────────────────────────────
```

**Why transaction mode is correct for this system:**

Celery workers execute discrete INSERT tasks and then wait for the next message.
The waiting time between tasks is orders of magnitude longer than the INSERT itself.
In session mode, the idle wait holds a PostgreSQL connection doing nothing. In
transaction mode, the connection is returned to the pool the instant the INSERT
commits. The next task borrows a connection only when its INSERT actually runs.

---

## 4. Architecture: Where PgBouncer Lives

PgBouncer is **Layer 5** in the resilience stack, sitting between Celery workers and
PostgreSQL. It is a transparent proxy — the application thinks it is talking to
PostgreSQL on port 6432; PgBouncer speaks the PostgreSQL wire protocol and forwards
to PostgreSQL on port 5432 (internal Docker network only).

```
  ╔═════════════════════════════════════════════════════════════════════════╗
  ║                     LAYERED RESILIENCE ARCHITECTURE                     ║
  ╠═════════════════════════════════════════════════════════════════════════╣
  ║                                                                         ║
  ║   Microservices 1–4  ──────────────────────────────────────────────►   ║
  ║                                                                         ║
  ║   L0: Rate Limiter (Redis sliding window)     ~1 ms    429             ║
  ║   L1: Backpressure (AIMD, queue depth)        ~1 ms    503             ║
  ║   L2: Circuit Breakers (DB / Redis / RMQ)     ~1 ms    503             ║
  ║   L3: Redis Fast Path (Lua atomic INCRBY)     ~0.2 ms  201 ← HOT PATH ║
  ║   L4: RabbitMQ Queue (burst absorption)       ~2 ms    async           ║
  ║                                                                         ║
  ║   ┌────────────────────────────────────────────────────────────────┐   ║
  ║   │  L5: CELERY WORKERS  →  PgBouncer  →  PostgreSQL               │   ║
  ║   │                                                                 │   ║
  ║   │   Celery picks up RabbitMQ message                             │   ║
  ║   │   ↓                                                            │   ║
  ║   │   Worker borrows a PgBouncer connection     ←── PGBOUNCER ──►  │   ║
  ║   │   ↓                                          transaction mode   │   ║
  ║   │   INSERT token_allocations ...               pool_size = 25     │   ║
  ║   │   ↓                                                            │   ║
  ║   │   COMMIT → connection returned to pool                         │   ║
  ║   │   Worker idle → holds ZERO PG connections                      │   ║
  ║   └────────────────────────────────────────────────────────────────┘   ║
  ║                                                                         ║
  ║   L6: Periodic Reconciliation (Redis ↔ PostgreSQL drift correction)    ║
  ╚═════════════════════════════════════════════════════════════════════════╝
```

**Development topology (Docker Compose):**

```
  ┌────────────────────────────────────────────────────────────────────┐
  │                       DOCKER HOST (laptop)                          │
  │                                                                     │
  │  ┌──────────────────────────────────────────────────────────────┐  │
  │  │                   token_manager_network (bridge)              │  │
  │  │                                                               │  │
  │  │  ┌─────────────┐       ┌──────────────┐                      │  │
  │  │  │ PostgreSQL   │       │  PgBouncer   │                      │  │
  │  │  │ llm_postgres │◄──────│ llm_pgbouncer│                      │  │
  │  │  │ :5432        │       │ :6432        │                      │  │
  │  │  │ (NOT exposed │       │ pool=25 txn  │                      │  │
  │  │  │  externally) │       │              │                      │  │
  │  │  └─────────────┘       └──────┬───────┘                      │  │
  │  │   ↑ internal only             │                               │  │
  │  │   accessed only by PgBouncer  │ pgbouncer:6432                │  │
  │  │                               │ (internal DNS)                │  │
  │  │           ┌───────────────────┼────────────────────────┐     │  │
  │  │           │                   │                         │     │  │
  │  │  ┌────────┴───────┐  ┌────────┴───────┐  ┌────────────┴──┐  │  │
  │  │  │ celery_worker   │  │ celery_beat    │  │ token_queue_  │  │  │
  │  │  │ :6432 (pg)      │  │ :6432 (pg)     │  │ consumer      │  │  │
  │  │  └────────────────┘  └────────────────┘  └───────────────┘  │  │
  │  │                                                               │  │
  │  └──────────────────────────────────────────────────────────────┘  │
  │                                                                     │
  │  HOST PORTS EXPOSED:                                                │
  │    localhost:5433  →  PostgreSQL (pg_isready, psql, migrations)     │
  │    localhost:6432  →  PgBouncer  (app traffic, pgbouncer admin)     │
  │    localhost:6379  →  Redis                                         │
  │    localhost:5672  →  RabbitMQ AMQP                                 │
  │    localhost:15672 →  RabbitMQ Management UI                        │
  └────────────────────────────────────────────────────────────────────┘
```

**Port separation is intentional:**

| Port | Container | Who connects | Purpose |
|------|-----------|-------------|---------|
| `5432` (internal) | PostgreSQL | PgBouncer only | PostgreSQL wire protocol |
| `5433` (host) | PostgreSQL | Developers, migration tools | Direct admin access |
| `6432` (host + internal) | PgBouncer | All app processes | Pooled app traffic |

PostgreSQL on `5432` is **not exposed to the host** — all application traffic must go
through PgBouncer. This enforces the pooling invariant: no code path can accidentally
bypass the pooler.

---

## 5. Connection Flow: From App to Database

```
  HAPPY PATH (token allocation persistence):
  ─────────────────────────────────────────────────────────────────────────────

  RabbitMQ           Celery Worker        PgBouncer            PostgreSQL
     │                    │                   │                     │
     │  DELIVER MSG        │                   │                     │
     │  (req_abc123)       │                   │                     │
     │───────────────────► │                   │                     │
     │                     │                   │                     │
     │                     │  connect(pgbouncer:6432)               │
     │                     │──────────────────► │                     │
     │                     │                   │  borrow PG conn 3   │
     │                     │                   │────────────────────► │
     │                     │                   │                     │
     │                     │  INSERT INTO token_allocations          │
     │                     │──────────────────►│──────────────────► │
     │                     │                   │                     │  execute
     │                     │                   │                     │  INSERT
     │                     │                   │  ◄── row inserted ──│
     │                     │                   │                     │
     │                     │  COMMIT            │                     │
     │                     │──────────────────► │                     │
     │                     │                   │  COMMIT ───────────► │
     │                     │                   │                     │
     │                     │                   │  return PG conn 3   │
     │                     │                   │◄────────────────────│
     │                     │                   │  (back to pool)      │
     │                     │                   │                     │
     │                     │  ◄── OK ──────────│                     │
     │                     │                   │                     │
     │  ACK message        │                   │                     │
     │◄───────────────────│                   │                     │
     │                     │                   │                     │
     │                     │  [IDLE — waiting for next message]      │
     │                     │  holds ZERO PG connections              │
     │                     │                   │                     │

  TIMING:
    Transaction duration:   ~15–30 ms (includes network to PgBouncer + INSERT)
    Connection borrow time: ~0 ms     (connection already open in pool)
    Idle between tasks:     0 PG connections held (transaction mode!)

  POOL UTILIZATION (at 400 writes/sec steady state):
    Active transactions at any instant: 400 × 0.02s = 8 concurrent transactions
    PgBouncer pool_size = 25 → 17 connections spare for bursts
```

---

## 6. Configuration Reference (pgbouncer.ini)

```ini
[databases]
; "mydb" is the alias the application connects to.
; PgBouncer resolves it to host=postgres:5432 dbname=mydb (Docker internal network).
mydb = host=postgres port=5432 dbname=mydb
```

The database **alias** (`mydb`) is what the application specifies in its connection
string. PgBouncer translates it to the real PostgreSQL host/port/dbname. The
application never knows the real PostgreSQL address — it only knows PgBouncer's address.

### Pool Settings

| Setting | Value | Explanation |
|---------|-------|-------------|
| `pool_mode` | `transaction` | Connection released after each transaction. Correct for async OLTP workloads. |
| `default_pool_size` | `25` | Max real PostgreSQL connections per (database + user) pair. Sized to leave headroom below PostgreSQL `max_connections = 100`. |
| `max_client_conn` | `500` | Max simultaneous app-side connections (FastAPI + all Celery workers). Should exceed total app concurrency. |
| `min_pool_size` | `5` | Keep 5 connections warm even when idle. Eliminates cold-start latency on first burst. |
| `reserve_pool_size` | `5` | Additional connections above `default_pool_size` for brief overload spikes. |
| `reserve_pool_timeout` | `5` | Seconds a client waits for a reserve connection before receiving an error. |

### Lifecycle Settings

| Setting | Value | Explanation |
|---------|-------|-------------|
| `server_idle_timeout` | `60` | Closes idle PostgreSQL backend connections after 60s of no use. Prevents connection accumulation. |
| `client_idle_timeout` | `60` | Closes idle client (app-side) connections after 60s. |
| `server_lifetime` | `3600` | Recycles backend connections after 1 hour. Prevents memory leaks in long-running PostgreSQL sessions. |
| `server_check_delay` | `30` | PgBouncer pings idle backends every 30s with `SELECT 1` to detect silently failed connections. |

### Async App Compatibility

```ini
ignore_startup_parameters = extra_float_digits,statement_timeout,lock_timeout
```

When asyncpg (the Python PostgreSQL driver) connects, it sends several session
parameters in the startup packet. PgBouncer's transaction mode does not support
session-scoped parameters. `ignore_startup_parameters` tells PgBouncer to silently
drop these parameters instead of returning an error.

**Without this setting:** asyncpg connections fail at handshake with
`"unsupported startup parameter: statement_timeout"`.

### Authentication

```ini
auth_type = plain
auth_file = /etc/pgbouncer/userlist.txt
```

`plain` is correct for development because `userlist.txt` stores passwords in
plaintext. For production, use `scram-sha-256` and store SCRAM verifiers in
`userlist.txt` (generated with `pg_dumpall -g` or `psql -c "\password"`).

---

## 7. Authentication (userlist.txt)

```
"myuser"          "mypassword"
"pgbouncer_admin" "pgbouncer_admin_password"
```

PgBouncer uses this file to authenticate **client connections** (from Celery, FastAPI)
and to establish **server connections** to PostgreSQL (on their behalf). The credentials
must match what PostgreSQL expects in its `pg_hba.conf`.

**Two users are needed:**

| User | Purpose |
|------|---------|
| `myuser` | Application database user. All app queries run as this user. |
| `pgbouncer_admin` | PgBouncer admin console access (`psql -d pgbouncer`). Allows running `SHOW STATS`, `SHOW POOLS`, `RELOAD`, etc. |

**For production**, replace plain passwords with SCRAM verifiers:

```bash
# Generate a SCRAM-SHA-256 verifier for 'mypassword':
psql -c "SELECT concat('\"myuser\" \"', rolpassword, '\"') FROM pg_authid WHERE rolname = 'myuser';"
# Copy the output (starts with SCRAM-SHA-256$...) into userlist.txt
```

And update `pgbouncer.ini`:
```ini
auth_type = scram-sha-256
```

---

## 8. How It Is Integrated

### 8.1 docker-compose.yml

```yaml
pgbouncer:
  image: edoburu/pgbouncer:1.22.1
  container_name: llm_pgbouncer
  depends_on:
    postgres:
      condition: service_healthy    # wait for PG to be ready before accepting connections
  ports:
    - "6432:6432"                   # host:6432 → container:6432 (PgBouncer listen port)
  volumes:
    - ./pgbouncer/pgbouncer.ini:/etc/pgbouncer/pgbouncer.ini:ro
    - ./pgbouncer/userlist.txt:/etc/pgbouncer/userlist.txt:ro
  healthcheck:
    test: ["CMD-SHELL", "pg_isready -h 127.0.0.1 -p 6432 -U myuser || exit 1"]
```

**Key details:**

- **Mounted ini file takes precedence** over the image's env-var-based config generation.
  When `/etc/pgbouncer/pgbouncer.ini` exists in the container, the edoburu image uses it
  directly without regenerating from environment variables.
- **Port mapping is `6432:6432`** — the left side is the host port (what your laptop sees),
  the right side is the container port (what `listen_port = 6432` in the ini sets).
- **All worker services** use `DATABASE_HOST: pgbouncer` and `DATABASE_PORT: "6432"` so
  they connect to PgBouncer on the internal Docker network, never to PostgreSQL directly.

### 8.2 app/core/config.py — PgBouncer Settings

```python
pgbouncer_enabled: bool = Field(default=True)
pgbouncer_host: str = Field(default="localhost")
pgbouncer_port: int = Field(default=6432)

@property
def effective_database_url(self) -> str:
    if self.pgbouncer_enabled:
        return (
            f"postgresql+asyncpg://{self.database_user}:{self.database_password}"
            f"@{self.pgbouncer_host}:{self.pgbouncer_port}/{self.database_name}"
        )
    return self.database_url
```

The `effective_database_url` property returns the PgBouncer URL when enabled,
or the direct PostgreSQL URL when disabled (e.g., for migrations that need session
mode). Set `PGBOUNCER_ENABLED=false` in `.env` to connect directly (useful for
schema migrations that use advisory locks or `LISTEN/NOTIFY`).

### 8.3 app/core/database.py — Connection Routing

```python
async def initialize(self, config=None):
    if config is None:
        if settings.pgbouncer_enabled:
            host = settings.pgbouncer_host   # localhost (local dev) or pgbouncer (Docker)
            port = settings.pgbouncer_port   # 6432
        else:
            host = settings.database_host    # localhost (local dev) or postgres (Docker)
            port = settings.database_port    # 5432 (direct PG)
        ...
    db_url = f"postgresql+asyncpg://{user}:{password}@{host}:{port}/{dbname}"
    self._engine = create_async_engine(db_url, pool_size=..., max_overflow=...)
```

When `pgbouncer_enabled=True`, all SQLAlchemy connections are routed through
PgBouncer. The SQLAlchemy pool itself is a **second layer** of pooling on top of
PgBouncer — see the sizing note below on why this matters.

### 8.4 Environment Variable Flow

```
  LOCAL DEVELOPMENT (FastAPI / Celery running on host, not in Docker):
  ──────────────────────────────────────────────────────────────────────
  .env:
    PGBOUNCER_ENABLED=true
    PGBOUNCER_HOST=localhost
    PGBOUNCER_PORT=6432

  database.py reads settings.pgbouncer_host/port → connects to localhost:6432
  PgBouncer (in Docker) listens on host port 6432 → forwards to postgres:5432

  DOCKER (worker services running inside Docker):
  ────────────────────────────────────────────────
  docker-compose.yml overrides:
    DATABASE_HOST=pgbouncer       # Docker service name resolves to container IP
    DATABASE_PORT=6432            # PgBouncer's listen port inside container
    PGBOUNCER_HOST=pgbouncer
    PGBOUNCER_PORT=6432
    PGBOUNCER_ENABLED=true

  database.py reads settings.pgbouncer_host → "pgbouncer" (Docker DNS)
                  settings.pgbouncer_port → 6432
  Connects to pgbouncer:6432 on the internal Docker bridge network.
```

---

## 9. Failure Mode and Recovery

```
  FAILURE              DETECTION              IMPACT                 RECOVERY
  ──────────────────   ──────────────────────  ─────────────────────  ─────────────────────────
  PgBouncer crash      Pool timeout errors in  Celery persistence     Container auto-restarts
                       Celery workers (~5s).   fails. Queue backs up  in ~2s (restart policy:
                       DB circuit breaker      in RabbitMQ.           unless-stopped).
                       may not trip (errors    API unaffected         Celery retries messages
                       are pool timeouts,      (Redis fast path).     via exponential backoff.
                       not DB errors).                                No data loss (messages
                                                                      durable in RabbitMQ).

  PgBouncer pool full  `query_wait_timeout`    Celery workers block   Transient: pool drains
  (all 25 used + 5     (10s) expires.          for up to 10s, then   as transactions commit.
  reserve connections  SQLAlchemy pool_timeout  receive timeout.       Structural: scale up
  exhausted)           (30s) expires.          DB circuit breaker     Celery concurrency or
                                               may trip after 5       increase pool_size in
                                               consecutive timeouts.  pgbouncer.ini.

  PostgreSQL down      PgBouncer gets          PgBouncer queues new   PgBouncer reconnects
                       connection errors.      connections until      automatically when PG
                       server_check_delay=30   pool_timeout. Then     recovers. No restart
                       detects dead backends.  rejects with error.    needed.
                       DB circuit breaker
                       trips after 5 failures.
```

**The key insight:** PgBouncer failure does **not** affect the hot path. The Redis
Lua fast path (Layer 3) operates independently. Callers still get 201 ACQUIRED
responses from Redis at ~0.2 ms. Only the async persistence (Celery → PgBouncer →
PostgreSQL) is affected. RabbitMQ queues accumulate during the outage and drain
automatically when PgBouncer recovers.

---

## 10. Monitoring and Admin Interface

PgBouncer exposes an admin console as a virtual database named `pgbouncer`.

```bash
# Connect to the admin console (from host):
psql -h localhost -p 6432 -U pgbouncer_admin pgbouncer

# From inside a Docker container:
docker exec -it llm_pgbouncer psql -h 127.0.0.1 -p 6432 -U pgbouncer_admin pgbouncer
```

**Key admin commands:**

```sql
-- Pool utilization: how many connections in use vs idle vs waiting
SHOW POOLS;

-- Aggregated statistics per database (transactions/sec, bytes/sec, wait times)
SHOW STATS;

-- Live stats per second (refreshes every stats_period = 60s)
SHOW STATS_TOTALS;

-- Individual client connections (FastAPI, Celery workers)
SHOW CLIENTS;

-- Individual server (PostgreSQL backend) connections
SHOW SERVERS;

-- Current configuration (from pgbouncer.ini)
SHOW CONFIG;

-- Reload configuration without restart (pick up pgbouncer.ini changes)
RELOAD;

-- Gracefully close idle server connections
RECONNECT;
```

**What to look for in SHOW POOLS:**

```
 database | user   | cl_active | cl_waiting | sv_active | sv_idle | sv_used | maxwait
----------+--------+-----------+------------+-----------+---------+---------+--------
 mydb     | myuser |         8 |          0 |         8 |        17|       0 |       0
```

| Field | Meaning | Alert if |
|-------|---------|----------|
| `cl_active` | App connections actively executing a query | — |
| `cl_waiting` | App connections waiting for a free pool slot | > 0 for sustained periods |
| `sv_active` | PostgreSQL connections executing a query | approaches `pool_size` |
| `sv_idle` | PostgreSQL connections open but idle | — |
| `maxwait` | Seconds the longest-waiting client has waited | > 1s |

If `cl_waiting > 0` sustained, either `default_pool_size` is too small or
PostgreSQL query latency is too high.

---

## 11. Transaction Mode Incompatibilities

Transaction mode releases the PostgreSQL connection after every transaction. This
breaks features that are scoped to a session, not a transaction.

```
  INCOMPATIBLE WITH TRANSACTION MODE:
  ────────────────────────────────────────────────────────────────────────────
  Feature               Why it breaks                 Workaround
  ─────────────────     ──────────────────────────    ────────────────────────
  SET session_var       SET is session-scoped.        Use ALTER ROLE ... SET
                        PgBouncer switches connections  or set in postgresql.conf.
                        between transactions — the SET
                        is lost on the next txn.

  LISTEN/NOTIFY         LISTEN registers on the        Use a separate connection
                        session. The connection is     with session mode (separate
                        returned to pool after txn;    pool or direct PG connection).
                        LISTEN registration is lost.

  Advisory locks        pg_try_advisory_lock() is      Use table-based locking
  (session-scoped)      session-scoped. Connection     (SELECT FOR UPDATE) or
                        rotation releases the lock.    app-level locking (Redis).

  Named prepared        Prepared statements are        Disable prepared statements
  statements            session-scoped. asyncpg        in asyncpg: pass
                        uses them by default.          prepared_statement_cache_size=0.
                        PgBouncer drops them on        OR use pgbouncer.ini:
                        connection switch.             max_prepared_statements=0.

  Multi-statement txns  NOT affected. A transaction    —
  via BEGIN/COMMIT      is held for its full duration.
                        PgBouncer only releases the
                        connection AFTER the COMMIT.
  ────────────────────────────────────────────────────────────────────────────
```

**This system avoids all of these patterns:**

- No `SET` session variables (using `ignore_startup_parameters` to drop asyncpg's startup params)
- No `LISTEN/NOTIFY` (circuit breaker state in Redis; queue via RabbitMQ)
- No session-scoped advisory locks (row-level locks via `SELECT FOR UPDATE` where needed)
- Prepared statements: asyncpg's prepared statement cache works within a single connection.
  Since PgBouncer transaction mode can switch connections, prepared statements may reference
  the wrong connection. The `ignore_startup_parameters` setting sidesteps the worst of this,
  but if you see `prepared statement does not exist` errors, set
  `prepared_statement_cache_size=0` in asyncpg's connection arguments.

---

## 12. Sizing Rationale

### Why pool_size = 25?

The PostgreSQL container is configured with the default `max_connections = 100`.
Subtract reserved connections:

```
  PostgreSQL max_connections: 100
  Reserved for superuser:      3   (pg_hba.conf reserve)
  Reserved for autovacuum:     3
  Reserved for pg_dump/admin: 2
  Available for PgBouncer:    92
                             ────
  PgBouncer default_pool_size: 25 (well within limit, leaves 67 for growth)
```

25 is also consistent with the Little's Law calculation: at 400 writes/sec and
20ms per write, you need `400 × 0.02 = 8` concurrent connections at peak. 25
gives 3× headroom for burst absorption during queue drain.

### SQLAlchemy Pool + PgBouncer: Avoid Double-Pooling

SQLAlchemy maintains its own connection pool (`pool_size=20, max_overflow=10`).
When PgBouncer is in front, this creates **two layers of pooling**:

```
  FastAPI (asyncpg) → SQLAlchemy Pool → PgBouncer Pool → PostgreSQL
                       20 connections    25 connections    real backends
```

This is redundant but not harmful in practice: SQLAlchemy connections are long-lived
and just sit in PgBouncer's client pool waiting. The real benefit of transaction mode
is on the *server* (PostgreSQL) side of PgBouncer — the server connections are
released per-transaction regardless of what happens on the client side.

For maximum efficiency (especially in Celery workers where connections are truly
short-lived), you could reduce SQLAlchemy's pool to `pool_size=1` and use
`NullPool` — but the current configuration works correctly and the overhead is
acceptable at the target scale.
