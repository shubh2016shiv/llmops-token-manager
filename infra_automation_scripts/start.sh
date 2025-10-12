#!/bin/bash

# ============================================================
# LLM Token Manager - Infrastructure Quick Start Script
# ============================================================
# This script helps you get started quickly with the infrastructure services.
# ============================================================

set -e  # Exit on error

echo "============================================================"
echo "  LLM Token Manager - Infrastructure Quick Start"
echo "============================================================"
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    echo "   Visit: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    echo "   Visit: https://docs.docker.com/compose/install/"
    exit 1
fi

echo "✓ Docker and Docker Compose are installed"
echo ""

# Determine script directory and change to project root (parent)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.." || { echo "❌ Failed to change to project root"; exit 1; }

# Check if docker-compose.yml exists in project root
if [ ! -f docker-compose.yml ]; then
    echo "❌ docker-compose.yml not found in project root directory."
    exit 1
fi

# Check if .env file exists (optional)
if [ ! -f .env ]; then
    echo "NOTE: .env file not found. Creating empty .env file..."
    echo "# LLM Token Manager Environment Variables" > .env
    echo "✓ Created empty .env file"
    echo ""
fi

echo "============================================================"
echo "  Starting Infrastructure Services"
echo "============================================================"
echo ""

# Stop any existing containers
echo "Stopping any existing containers..."
docker-compose down

# Build and start services
echo ""
echo "Building and starting services..."
docker-compose up -d --build

# Wait for services to be healthy
echo ""
echo "Waiting for services to be ready..."
sleep 10

# Check service status
echo ""
echo "============================================================"
echo "  Service Status"
echo "============================================================"
docker-compose ps

# Test health of services
echo ""
echo "============================================================"
echo "  Testing Infrastructure Health"
echo "============================================================"
echo ""

# Check PostgreSQL
echo "Checking PostgreSQL..."
if docker-compose exec -T postgres pg_isready -U myuser; then
    echo "✓ PostgreSQL is healthy!"
else
    echo "⚠️ WARNING: PostgreSQL may not be ready yet."
fi

# Check Redis
echo ""
echo "Checking Redis..."
if docker-compose exec -T redis redis-cli ping | grep -q "PONG"; then
    echo "✓ Redis is healthy!"
else
    echo "⚠️ WARNING: Redis may not be ready yet."
fi

# Check RabbitMQ
echo ""
echo "Checking RabbitMQ..."
if docker-compose exec -T rabbitmq rabbitmqctl status > /dev/null 2>&1; then
    echo "✓ RabbitMQ is healthy!"
else
    echo "⚠️ WARNING: RabbitMQ may not be ready yet."
fi

echo ""
echo "============================================================"
echo "  Infrastructure Deployment Complete!"
echo "============================================================"
echo ""
echo "Access points:"
echo "  • PostgreSQL:          localhost:5432 (user: myuser, password: mypassword, db: mydb)"
echo "  • Redis:               localhost:6379"
echo "  • RabbitMQ AMQP:       localhost:5672"
echo "  • RabbitMQ Dashboard:  http://localhost:15672 (login: rmq_user / rmq_password)"
echo ""
echo "Useful commands:"
echo "  • View logs:          docker-compose logs -f"
echo "  • Stop services:      docker-compose down"
echo "  • Check health:       python check_infra_service_health.py"
echo ""
echo "Next steps:"
echo "  1. Connect to PostgreSQL with your preferred client"
echo "  2. Connect to Redis with your preferred client"
echo "  3. Access RabbitMQ management UI at http://localhost:15672"
echo ""
echo "============================================================"