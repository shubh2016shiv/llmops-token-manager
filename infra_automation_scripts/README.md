# Infrastructure Automation Scripts

Convenience scripts for managing the LLM Token Manager infrastructure
(PostgreSQL, Redis, RabbitMQ, Celery Worker) via Docker Compose.

---

## Quick Start

| OS | Command |
|----|---------|
| **Linux / macOS** | `./infra_automation_scripts/start.sh` |
| **Windows** | `infra_automation_scripts\start.bat` |

Both scripts will:
1. Detect Docker and Docker Compose (v2 preferred, v1 fallback)
2. Create a `.env` file with defaults if one doesn't exist
3. Stop any previously running containers
4. Build the Celery worker image and start all services
5. Run health checks on every service
6. **Print a professional summary table** with network, IP address, ports, and health status

---

## Scripts

| Script | Purpose |
|--------|---------|
| `start.sh` / `start.bat` | Build images, start all services, show health + summary table |
| `stop.sh` / `stop.bat` | Graceful shutdown with pre/post summary tables and cleanup options |

---

## Docker Compose Profiles

The `docker-compose.yml` uses profiles for selective startup:

| Profile | Services Started |
|---------|-----------------|
| `infra` | PostgreSQL, Redis, RabbitMQ only |
| `worker` | PostgreSQL, Redis, RabbitMQ + Celery Worker |
| `all` | Everything (worker + FastAPI when activated) |

The start/stop scripts use `--profile worker` by default.

**Manual profile commands:**

```bash
# Infrastructure only (no worker — useful when running FastAPI locally)
docker compose --profile infra up -d

# Infrastructure + worker (default)
docker compose --profile worker up -d --build

# Everything including FastAPI (when ready)
docker compose --profile all up -d --build
```

---

## Summary Table (Automatic Output)

Every `start` or `stop` action prints a table like this:

```
  ╔══════════════════╤══════════════════════════╤═════════════════╤══════════════════════════════╤════════════════╗
  ║ Service          │ Network                  │ IP Address      │ Ports (Host→Container)       │ Status         ║
  ╠══════════════════╪══════════════════════════╪═════════════════╪══════════════════════════════╪════════════════╣
  ║ PostgreSQL       │ token_manager_network    │ 172.18.0.2      │ 5432->5433                   │ healthy        ║
  ║ Redis            │ token_manager_network    │ 172.18.0.3      │ 6379->6379                   │ healthy        ║
  ║ RabbitMQ         │ token_manager_network    │ 172.18.0.4      │ 5672->5672, 15672->15672     │ healthy        ║
  ║ Celery Worker    │ token_manager_network    │ 172.18.0.5      │ —                            │ healthy        ║
  ╚══════════════════╧══════════════════════════╧═════════════════╧══════════════════════════════╧════════════════╝
```

This gives you immediate visibility into what's running, where, and on which ports — no need to run `docker ps` or `docker inspect` separately.

---

## Shutdown & Cleanup

Run `stop.sh` / `stop.bat` for a graceful shutdown. After stopping, you'll be presented with cleanup options:

| Option | Effect |
|--------|--------|
| **1 (default)** | Keep all data volumes (safe for dev — data persists across restarts) |
| **2** | Remove containers, keep volumes |
| **3** | **DESTRUCTIVE** — Remove everything including database data (requires typing `DELETE` to confirm) |
| **4** | Skip cleanup entirely |

Both `start` and `stop` scripts show a **pre- and post-action summary table** so you always know the exact state of the infrastructure.

---

## Daily Workflow

```bash
# Morning — bring everything up
./infra_automation_scripts/start.sh

# During development — check logs
docker compose --profile worker logs -f celery_worker

# End of day — graceful shutdown (preserves data)
./infra_automation_scripts/stop.sh     # Choose option 1
```

---

## Troubleshooting

| Symptom | Check |
|---------|-------|
| "No running services detected" | Docker Desktop might not be running. Start it and re-run `start.sh`. |
| Celery worker stays "starting" | The worker takes ~20s to initialise. Wait for the next healthcheck interval. |
| PostgreSQL healthcheck fails | Run `docker compose --profile worker logs postgres` to see error details. |
| Port already in use | Change the host port in `.env` (e.g., `POSTGRES_HOST_PORT=5434`). |

---

## Environment Variables

All configurable values are read from `.env` (copy from `.env.example`). The most common ones:

| Variable | Default | Purpose |
|----------|---------|---------|
| `POSTGRES_HOST_PORT` | `5433` | Host port for PostgreSQL |
| `REDIS_HOST_PORT` | `6379` | Host port for Redis |
| `RABBITMQ_AMQP_PORT` | `5672` | Host port for RabbitMQ AMQP |
| `RABBITMQ_MGMT_PORT` | `15672` | Host port for RabbitMQ Management UI |
| `CELERY_WORKER_CONCURRENCY` | `10` | Number of Celery worker processes |
| `LOG_LEVEL` | `INFO` | Log level for Celery worker |
