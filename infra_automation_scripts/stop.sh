#!/bin/bash

# ============================================================
# LLM Token Manager - Infrastructure Stop Script
# ============================================================
# Gracefully stops infrastructure, shows current state before
# and after shutdown in a professional table format.
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
        if [ -z "$status" ]; then status="stopped"; fi

        printf "  | %-16s | %-24s | %-15s | %-28s | %-14s |\n" "$display_name" "$net" "$ip" "$ports" "$status"
    done

    echo "  +------------------+--------------------------+-----------------+------------------------------+----------------+"
    echo ""
}

echo ""
echo "  ============================================================"
echo "    LLM Token Manager - Infrastructure Stop"
echo "  ============================================================"
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.." || { echo "  ERROR: Failed to change to project root"; exit 1; }

if [ ! -f docker-compose.yml ]; then
    echo "  ERROR: docker-compose.yml not found in project root."
    exit 1
fi

if ! docker info >/dev/null 2>&1; then
    echo "  ERROR: Docker is not running or not accessible."
    exit 1
fi

# Normalise compose command
if docker compose version >/dev/null 2>&1; then
    COMPOSE_CMD="docker compose"
else
    COMPOSE_CMD="docker-compose"
fi

echo "  -- Current Infrastructure State -----------------------------"
RUNNING=$($COMPOSE_CMD --profile worker ps --services --filter "status=running" 2>/dev/null || true)

if [ -z "$RUNNING" ]; then
    echo ""
    echo "  INFO: No running services detected. Infrastructure is already down."
    echo ""
    echo "  -- Post-Shutdown Summary -----------------------------------"
    print_infra_summary
    echo "  ============================================================"
    echo ""
    exit 0
fi

print_infra_summary

echo "  -- Graceful Shutdown ----------------------------------------"
echo ""
echo "  This will stop:"
echo "    * PostgreSQL    (llm_postgres)"
echo "    * Redis         (llm_redis)"
echo "    * RabbitMQ      (llm_rabbitmq)"
echo "    * Celery Worker (llm_celery_worker)"
echo ""
read -r -p "  Proceed? (y/N): " CONFIRM

if [[ ! "$CONFIRM" =~ ^[Yy]$ ]]; then
    echo "  Cancelled by user."
    echo "  ============================================================"
    echo ""
    exit 0
fi

echo ""
echo "  Step 1/4 -- Stopping services gracefully (timeout: 30s)..."
if ! $COMPOSE_CMD --profile worker stop --timeout 30; then
    echo "  WARNING: Some services did not stop gracefully -- forcing..."
    $COMPOSE_CMD --profile worker kill
fi

echo "  Step 2/4 -- Waiting for services to fully stop..."
sleep 5

echo "  Step 3/4 -- Removing containers..."
$COMPOSE_CMD --profile worker rm -f || echo "  WARNING: Some containers could not be removed cleanly."

echo "  Step 4/4 -- Checking for orphaned containers..."
ORPHANED=$(docker ps -a --filter "name=llm_" --format "{{.Names}}" 2>/dev/null || true)
if [ -n "$ORPHANED" ]; then
    while IFS= read -r cname; do
        if [ -n "$cname" ]; then
            echo "  Removed orphaned: $cname"
            docker stop "$cname" >/dev/null 2>&1 || true
            docker rm "$cname" >/dev/null 2>&1 || true
        fi
    done <<< "$ORPHANED"
else
    echo "  None found."
fi

echo ""
echo "  -- Cleanup Options ------------------------------------------"
echo ""
echo "    1  Keep all data (volumes preserved) -- RECOMMENDED"
echo "    2  Remove containers only (keep volumes)"
echo "    3  Remove everything INCLUDING data volumes -- DESTRUCTIVE"
echo "    4  Skip cleanup"
echo ""
read -r -p "  Choice (1-4) [default: 1]: " CLEANUP_LEVEL
CLEANUP_LEVEL=${CLEANUP_LEVEL:-1}

case $CLEANUP_LEVEL in
    1)
        echo "  -> Volumes preserved."
        ;;
    2)
        echo "  -> Removing containers..."
        $COMPOSE_CMD --profile worker down
        ;;
    3)
        echo "  WARNING: DESTRUCTIVE - This will permanently delete all data!"
        read -r -p "  Type 'DELETE' to confirm: " CONFIRM_DEL
        if [ "$CONFIRM_DEL" = "DELETE" ]; then
            echo "  -> Removing everything including volumes..."
            $COMPOSE_CMD --profile worker down -v --remove-orphans
            echo "  OK: All data volumes have been permanently deleted."
        else
            echo "  Cancelled."
        fi
        ;;
    4)
        echo "  -> Cleanup skipped."
        ;;
    *)
        echo "  -> Invalid choice - cleanup skipped."
        ;;
esac

echo ""
echo "  -- Post-Shutdown Summary ------------------------------------"
print_infra_summary

echo "  -- Restart ---------------------------------------------------"
echo ""
echo "  ./infra_automation_scripts/start.sh"
echo "  $COMPOSE_CMD --profile worker down -v"
echo ""
echo "  ============================================================"
echo ""
