#!/bin/bash

# ============================================================
# LLM Token Manager - Infrastructure Graceful Stop Script
# ============================================================
# This script provides a robust way to gracefully stop all
# infrastructure services with proper cleanup options.
# ============================================================

set -e  # Exit on error

echo "============================================================"
echo "  LLM Token Manager - Infrastructure Graceful Stop"
echo "============================================================"
echo ""

# Determine script directory and change to project root (parent)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.." || { echo "❌ Failed to change to project root"; exit 1; }

# Check if docker-compose.yml exists in project root
if [ ! -f docker-compose.yml ]; then
    echo "❌ docker-compose.yml not found in project root directory."
    echo "   Please run this script from the project root or ensure docker-compose.yml exists."
    exit 1
fi

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running or not accessible."
    echo "   Please start Docker and try again."
    exit 1
fi

echo "============================================================"
echo "  Checking Current Infrastructure Status"
echo "============================================================"
echo ""

# Check if any containers are running
RUNNING_SERVICES=$(docker-compose ps --services --filter "status=running" 2>/dev/null || true)

if [ -z "$RUNNING_SERVICES" ]; then
    echo "ℹ️  No running containers found. Infrastructure may already be stopped."
    echo ""
    show_final_status
    exit 0
fi

echo "Current running services:"
docker-compose ps --filter "status=running"
echo ""

# Function to show final status
show_final_status() {
    echo "============================================================"
    echo "  Final Status Check"
    echo "============================================================"
    echo ""

    echo "Checking final infrastructure status..."
    docker-compose ps

    echo ""
    echo "Checking for any remaining llm_* containers..."
    REMAINING_CONTAINERS=$(docker ps -a --filter "name=llm_" --format "{{.Names}}" 2>/dev/null || true)
    if [ -n "$REMAINING_CONTAINERS" ]; then
        echo "⚠️  WARNING: Found remaining containers:"
        echo "$REMAINING_CONTAINERS"
    else
        echo "✓ No remaining llm_* containers found"
    fi

    echo ""
    echo "============================================================"
    echo "  Infrastructure Shutdown Complete!"
    echo "============================================================"
    echo ""
    echo "Summary:"
    echo "  • All services have been stopped"
    echo "  • Containers have been removed"
    if [ "$CLEANUP_LEVEL" = "3" ]; then
        echo "  • All data volumes have been removed"
    else
        echo "  • Data volumes have been preserved"
    fi
    echo ""
    echo "To restart the infrastructure:"
    echo "  • Run: ./infra_automation_scripts/start.sh"
    echo ""
    echo "To completely clean up (remove volumes):"
    echo "  • Run: docker-compose down -v"
    echo ""
    echo "============================================================"
}

# Function to handle cleanup
perform_cleanup() {
    case $CLEANUP_LEVEL in
        1)
            echo "✓ Keeping all data volumes (recommended for development)"
            ;;
        2)
            echo "Removing containers only..."
            docker-compose down
            ;;
        3)
            echo "⚠️  WARNING: This will remove ALL data including databases!"
            read -p "Are you absolutely sure? Type 'DELETE' to confirm: " CONFIRM_DESTRUCTIVE
            if [ "$CONFIRM_DESTRUCTIVE" = "DELETE" ]; then
                echo "Removing everything including data volumes..."
                docker-compose down -v --remove-orphans
                echo ""
                echo "⚠️  WARNING: All data has been permanently deleted!"
            else
                echo "Destructive cleanup cancelled."
            fi
            ;;
        4)
            echo "Skipping cleanup..."
            ;;
        *)
            echo "Invalid choice. Skipping cleanup..."
            ;;
    esac
}

# Ask user for confirmation
echo "============================================================"
echo "  Graceful Shutdown Confirmation"
echo "============================================================"
echo ""
echo "This will gracefully stop all infrastructure services:"
echo "  • PostgreSQL (llm_postgres)"
echo "  • Redis (llm_redis)"
echo "  • RabbitMQ (llm_rabbitmq)"
echo ""
read -p "Are you sure you want to stop all services? (y/N): " CONFIRM

if [[ ! "$CONFIRM" =~ ^[Yy]$ ]]; then
    echo "Operation cancelled by user."
    exit 0
fi

echo ""
echo "============================================================"
echo "  Performing Graceful Shutdown"
echo "============================================================"
echo ""

# Step 1: Stop services gracefully with timeout
echo "Step 1: Stopping services gracefully..."
if ! docker-compose stop --timeout 30; then
    echo "⚠️  WARNING: Some services may not have stopped gracefully."
    echo "Proceeding with force stop..."
    docker-compose kill
fi

echo ""
echo "Step 2: Waiting for services to fully stop..."
sleep 5

# Step 3: Remove containers
echo "Step 3: Removing containers..."
if ! docker-compose rm -f; then
    echo "⚠️  WARNING: Some containers may not have been removed cleanly."
fi

echo ""
echo "Step 4: Checking for orphaned containers..."
ORPHANED_CONTAINERS=$(docker ps -a --filter "name=llm_" --format "{{.Names}}" 2>/dev/null || true)
if [ -n "$ORPHANED_CONTAINERS" ]; then
    echo "Found orphaned containers:"
    echo "$ORPHANED_CONTAINERS"
    echo "Removing orphaned containers..."
    echo "$ORPHANED_CONTAINERS" | xargs -r docker stop
    echo "$ORPHANED_CONTAINERS" | xargs -r docker rm
    echo "✓ Removed orphaned containers"
fi

echo ""
echo "============================================================"
echo "  Cleanup Options"
echo "============================================================"
echo ""
echo "Choose cleanup level:"
echo "  1. Keep all data (volumes preserved) - RECOMMENDED"
echo "  2. Remove containers only (keep volumes)"
echo "  3. Remove everything including data volumes - DESTRUCTIVE"
echo "  4. Skip cleanup"
echo ""
read -p "Enter choice (1-4) [default: 1]: " CLEANUP_LEVEL

# Set default if empty
CLEANUP_LEVEL=${CLEANUP_LEVEL:-1}

# Perform cleanup
perform_cleanup

# Show final status
show_final_status
