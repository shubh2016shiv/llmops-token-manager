#!/bin/bash

# ============================================================
# LLM Token Manager - Infrastructure Quick Start Script
# ============================================================
# This script helps you get started quickly with the infrastructure services.
# ============================================================

set -e

echo "============================================================"
echo "  LLM Token Manager - Infrastructure Quick Start"
echo "============================================================"
echo ""

if ! command -v docker &> /dev/null; then
    echo "Docker is not installed. Please install Docker first."
    echo "Visit: https://docs.docker.com/get-docker/"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "Docker Compose is not installed. Please install Docker Compose first."
    echo "Visit: https://docs.docker.com/compose/install/"
    exit 1
fi

echo "Docker and Docker Compose are installed"
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.." || { echo "Failed to change to project root"; exit 1; }

if [ ! -f docker-compose.yml ]; then
    echo "docker-compose.yml not found in project root directory."
    exit 1
fi

if [ ! -f .env ]; then
    echo "NOTE: .env file not found. Creating empty .env file..."
    echo "# LLM Token Manager Environment Variables" > .env
    echo "Created empty .env file"
    echo ""
fi

echo "============================================================"
echo "  Starting Infrastructure Services"
echo "============================================================"
echo ""

echo "Stopping any existing containers..."
docker-compose down

echo ""
echo "Building and starting services..."
docker-compose up -d --build

echo ""
echo "Waiting for services to be ready..."
sleep 10

echo ""
echo "============================================================"
echo "  Service Status"
echo "============================================================"
docker-compose ps

echo ""
echo "============================================================"
echo "  Testing Infrastructure Health"
echo "============================================================"
echo ""

echo "Checking PostgreSQL..."
if docker-compose exec -T postgres pg_isready -U myuser -d mydb; then
    echo "PostgreSQL is healthy!"
else
    echo "WARNING: PostgreSQL may not be ready yet."
fi

echo ""
echo "Checking Redis..."
if docker-compose exec -T redis redis-cli ping | grep -q "PONG"; then
    echo "Redis is healthy!"
else
    echo "WARNING: Redis may not be ready yet."
fi

echo ""
echo "Checking RabbitMQ..."
if docker-compose exec -T rabbitmq rabbitmqctl status > /dev/null 2>&1; then
    echo "RabbitMQ is healthy!"
else
    echo "WARNING: RabbitMQ may not be ready yet."
fi

echo ""
echo "Checking Celery worker..."
CELERY_HEALTH=$(docker inspect --format "{{if .State.Health}}{{.State.Health.Status}}{{else}}no-healthcheck{{end}}" llm_celery_worker 2>/dev/null || true)
if [ "$CELERY_HEALTH" = "healthy" ]; then
    echo "Celery worker is healthy!"
elif [ "$CELERY_HEALTH" = "starting" ]; then
    echo "WARNING: Celery worker health check is still starting."
else
    echo "WARNING: Celery worker may not be ready yet."
fi

echo ""
echo "============================================================"
echo "  Infrastructure Deployment Complete!"
echo "============================================================"
echo ""
echo "Access points:"
echo "  - PostgreSQL:          localhost:5433 (user: myuser, password: mypassword, db: mydb)"
echo "  - Redis:               localhost:6379"
echo "  - RabbitMQ AMQP:       localhost:5672"
echo "  - RabbitMQ Dashboard:  http://localhost:15672 (login: rmq_user / rmq_password)"
echo "  - Celery Worker:       docker inspect llm_celery_worker --format \"{{.State.Health.Status}}\""
echo ""
echo "Useful commands:"
echo "  - View logs:          docker-compose logs -f"
echo "  - View Celery logs:   docker-compose logs -f celery_worker"
echo "  - Check Celery state: docker inspect llm_celery_worker --format \"{{.State.Health.Status}}\""
echo "  - Stop services:      docker-compose down"
echo "  - Check health:       python check_infra_service_health.py"
echo ""
echo "Next steps:"
echo "  1. Connect to PostgreSQL with your preferred client"
echo "  2. Connect to Redis with your preferred client"
echo "  3. Access RabbitMQ management UI at http://localhost:15672"
echo "  4. Verify Celery worker health if you plan to queue async jobs"
echo ""
echo "============================================================"
