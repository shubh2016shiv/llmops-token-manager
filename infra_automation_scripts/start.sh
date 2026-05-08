#!/bin/bash

# ============================================================
# LLM Token Manager - Infrastructure Start Script
# ============================================================
# Starts all infrastructure services and prints a professional
# summary table of network details, IPs, ports, and health.
# ============================================================

set -e

print_infra_summary() {
    local CONTAINERS=("llm_postgres" "llm_redis" "llm_rabbitmq" "llm_celery_worker")
    local DISPLAY_NAMES=("PostgreSQL" "Redis" "RabbitMQ" "Celery Worker")

    echo ""
    echo "  +------------------+--------------------------+-----------------+------------------------------+----------------+"
    printf "  | %-16s | %-24s | %-15s | %-28s | %-14s |\n" "Service" "Network" "IP Address" "Ports (Host->Container)" "Status"
    echo "  +------------------+--------------------------+-----------------+------------------------------+----------------+"

    for i in "${!CONTAINERS[@]}"; do
        local container="${CONTAINERS[$i]}"
        local display_name="${DISPLAY_NAMES[$i]}"
        local net ip ports status

        net=$(docker inspect --format '{{range $k,$v := .NetworkSettings.Networks}}{{$k}}{{end}}' "$container" 2>/dev/null || true)
        ip=$(docker inspect --format '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' "$container" 2>/dev/null || true)
        ports=$(docker inspect --format '{{range $k,$v := .NetworkSettings.Ports}}{{if $v}}{{$k}}->{{(index $v 0).HostPort}}, {{end}}{{end}}' "$container" 2>/dev/null || true)
        status=$(docker inspect --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}no healthcheck{{end}}' "$container" 2>/dev/null || true)

        if [ -z "$net" ]; then net="-"; fi
        if [ -z "$ip" ]; then ip="-"; fi
        if [ -z "$ports" ]; then
            ports="-"
        else
            ports="${ports%, }"
            ports="${ports//\/tcp/}"
        fi
        if [ -z "$status" ]; then status="not running"; fi

        printf "  | %-16s | %-24s | %-15s | %-28s | %-14s |\n" "$display_name" "$net" "$ip" "$ports" "$status"
    done

    echo "  +------------------+--------------------------+-----------------+------------------------------+----------------+"
    echo ""
}

echo ""
echo "  ============================================================"
echo "    LLM Token Manager - Infrastructure Start"
echo "  ============================================================"
echo ""

if ! command -v docker >/dev/null 2>&1; then
    echo "  ERROR: Docker is not installed."
    echo "  Visit: https://docs.docker.com/get-docker/"
    exit 1
fi

if ! command -v docker-compose >/dev/null 2>&1 && ! docker compose version >/dev/null 2>&1; then
    echo "  ERROR: Docker Compose is not available."
    echo "  Visit: https://docs.docker.com/compose/install/"
    exit 1
fi

# Normalise compose command (docker-compose v1 -> docker compose v2 fallback)
if docker compose version >/dev/null 2>&1; then
    COMPOSE_CMD="docker compose"
else
    COMPOSE_CMD="docker-compose"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.." || { echo "  ERROR: Failed to change to project root"; exit 1; }

if [ ! -f docker-compose.yml ]; then
    echo "  ERROR: docker-compose.yml not found in project root."
    exit 1
fi

if [ ! -f .env ]; then
    if [ -f .env.example ]; then
        echo "  INFO: .env file not found - creating from .env.example..."
        cp .env.example .env
    else
        echo "  INFO: .env file not found - creating one with defaults..."
        echo "# LLM Token Manager Environment Variables" > .env
    fi
fi

echo "  Stopping any previously running containers..."
$COMPOSE_CMD --profile worker down --remove-orphans >/dev/null 2>&1 || true

echo ""
echo "  Building images and starting services..."
$COMPOSE_CMD --profile worker up -d --build

echo ""
echo "  Waiting for services to initialise..."
sleep 8

echo ""
echo "  -- Health Checks -------------------------------------------"
echo ""

DB_USER="${DATABASE_USER:-myuser}"
DB_NAME="${DATABASE_NAME:-mydb}"
REDIS_PASSWORD_VALUE="$(grep -E '^REDIS_PASSWORD=' .env 2>/dev/null | head -n 1 | cut -d= -f2-)"
if [ -z "$REDIS_PASSWORD_VALUE" ]; then
    REDIS_PASSWORD_VALUE="redis_password"
fi
CHECK_OK="[OK] healthy"
CHECK_FAIL="[FAIL] unavailable"

echo -n "  PostgreSQL ... "
$COMPOSE_CMD exec -T postgres pg_isready -U "$DB_USER" -d "$DB_NAME" -q 2>/dev/null \
    && echo "$CHECK_OK" || echo "$CHECK_FAIL"

echo -n "  Redis      ... "
$COMPOSE_CMD exec -T redis redis-cli --no-auth-warning -a "$REDIS_PASSWORD_VALUE" ping 2>/dev/null | grep -q "PONG" \
    && echo "$CHECK_OK" || echo "$CHECK_FAIL"

echo -n "  RabbitMQ   ... "
$COMPOSE_CMD exec -T rabbitmq rabbitmq-diagnostics -q ping 2>/dev/null \
    && echo "$CHECK_OK" || echo "$CHECK_FAIL"

echo -n "  Celery     ... "
CELERY_STATUS=$(docker inspect --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}no healthcheck{{end}}' llm_celery_worker 2>/dev/null || echo "unknown")
echo "$CELERY_STATUS"

print_infra_summary

echo "  -- Access Points -------------------------------------------"
echo ""
echo "  PostgreSQL   :  localhost:${POSTGRES_HOST_PORT:-5433}   (user: ${POSTGRES_USER:-myuser} / db: ${POSTGRES_DB:-mydb})"
echo "  Redis        :  localhost:${REDIS_HOST_PORT:-6379}   (password: ${REDIS_PASSWORD_VALUE})"
echo "  RabbitMQ     :  localhost:${RABBITMQ_AMQP_PORT:-5672}   (AMQP)"
echo "  RabbitMQ UI  :  http://localhost:${RABBITMQ_MGMT_PORT:-15672}  (${RABBITMQ_DEFAULT_USER:-rmq_user})"
echo ""
echo "  -- Quick Commands ------------------------------------------"
echo ""
echo "  $COMPOSE_CMD --profile worker logs -f"
echo "  $COMPOSE_CMD --profile worker logs -f celery_worker"
echo "  $COMPOSE_CMD --profile worker ps"
echo "  ./infra_automation_scripts/stop.sh"
echo "  $COMPOSE_CMD --profile infra up -d"
echo ""
echo "  ============================================================"
echo "  All services are running in the background."
echo "  ============================================================"
echo ""
