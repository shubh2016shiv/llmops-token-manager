#!/usr/bin/env python3
"""
Infrastructure Service Health Checker
------------------------------------
Checks the health of all infrastructure services defined in docker-compose.yml:
- PostgreSQL
- Redis
- RabbitMQ

Provides clear feedback on service status and connection details.
"""

import socket
import subprocess
import sys
import time
from typing import Dict

import psycopg2
import redis
import requests
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Service connection details
SERVICES = {
    "PostgreSQL": {
        "host": "localhost",
        "port": 5432,
        "user": "myuser",
        "password": "mypassword",
        "database": "mydb",
    },
    "Redis": {"host": "localhost", "port": 6379},
    "RabbitMQ": {
        "host": "localhost",
        "amqp_port": 5672,
        "mgmt_port": 15672,
        "user": "rmq_user",
        "password": "rmq_password",
    },
}

# Initialize Rich console for pretty output
console = Console()


def check_docker_running() -> bool:
    """Check if Docker is running."""
    try:
        subprocess.run(
            ["docker", "info"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            timeout=5,
        )
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        return False


def check_containers_running() -> Dict[str, bool]:
    """Check if the required containers are running."""
    containers = {
        "PostgreSQL": "llm_postgres",
        "Redis": "llm_redis",
        "RabbitMQ": "llm_rabbitmq",
    }

    results = {}

    try:
        # Get list of running containers
        output = subprocess.run(
            ["docker", "ps", "--format", "{{.Names}}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            timeout=5,
            text=True,
        ).stdout

        running_containers = output.strip().split("\n")

        # Check if our containers are in the list
        for service, container in containers.items():
            results[service] = container in running_containers

        return results
    except (subprocess.SubprocessError, FileNotFoundError):
        # If docker command fails, assume no containers are running
        return {service: False for service in containers.keys()}


def check_port_open(host: str, port: int, timeout: int = 2) -> bool:
    """Check if a port is open and accepting connections."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except (socket.error, socket.timeout, ConnectionRefusedError):
        return False


def check_postgresql() -> tuple[bool, str]:
    """Check PostgreSQL connection and basic functionality."""
    pg_config = SERVICES["PostgreSQL"]

    # First check if port is open
    if not check_port_open(pg_config["host"], pg_config["port"]):
        return False, "Port is closed"

    # Try to connect and run a simple query
    try:
        conn = psycopg2.connect(
            host=pg_config["host"],
            port=pg_config["port"],
            user=pg_config["user"],
            password=pg_config["password"],
            database=pg_config["database"],
            connect_timeout=5,
        )

        cursor = conn.cursor()
        cursor.execute("SELECT version();")
        version = cursor.fetchone()[0]
        cursor.close()
        conn.close()

        return True, f"Connected successfully: {version.split(',')[0]}"
    except psycopg2.Error as e:
        return False, f"Connection error: {e}"


def check_redis() -> tuple[bool, str]:
    """Check Redis connection and basic functionality."""
    redis_config = SERVICES["Redis"]

    # First check if port is open
    if not check_port_open(redis_config["host"], redis_config["port"]):
        return False, "Port is closed"

    # Try to connect and ping
    try:
        r = redis.Redis(
            host=redis_config["host"],
            port=redis_config["port"],
            socket_timeout=5,
            socket_connect_timeout=5,
        )
        if r.ping():
            info = r.info()
            version = info.get("redis_version", "Unknown")
            return True, f"Connected successfully: Redis v{version}"
        else:
            return False, "Ping failed"
    except redis.RedisError as e:
        return False, f"Connection error: {e}"


def check_rabbitmq() -> tuple[bool, str]:
    """Check RabbitMQ connection and management API."""
    rmq_config = SERVICES["RabbitMQ"]

    # First check if AMQP port is open
    if not check_port_open(rmq_config["host"], rmq_config["amqp_port"]):
        return False, "AMQP port is closed"

    # Then check if management port is open
    if not check_port_open(rmq_config["host"], rmq_config["mgmt_port"]):
        return False, "Management port is closed"

    # Try to connect to management API
    try:
        url = f"http://{rmq_config['host']}:{rmq_config['mgmt_port']}/api/overview"
        response = requests.get(url, auth=(rmq_config["user"], rmq_config["password"]), timeout=5)

        if response.status_code == 200:
            data = response.json()
            version = data.get("rabbitmq_version", "Unknown")
            return True, f"Connected successfully: RabbitMQ v{version}"
        else:
            return False, f"API returned status code: {response.status_code}"
    except requests.RequestException as e:
        return False, f"Connection error: {e}"


def check_docker_compose_file() -> bool:
    """Check if docker-compose.yml exists in current directory."""
    try:
        with open("docker-compose.yml", "r") as f:
            content = f.read()
            return "postgres:" in content and "redis:" in content and "rabbitmq:" in content
    except FileNotFoundError:
        return False


def start_services() -> bool:
    """Start services using docker-compose if needed."""
    try:
        console.print("[yellow]Starting infrastructure services with docker-compose...[/]")
        subprocess.run(
            ["docker-compose", "up", "-d"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            timeout=60,
        )

        # Give services time to initialize
        console.print("[yellow]Waiting for services to initialize (30 seconds)...[/]")
        time.sleep(30)
        return True
    except subprocess.SubprocessError as e:
        console.print(f"[red]Failed to start services: {e}[/]")
        return False


def main():
    """Main function to check all services."""
    console.print(
        Panel.fit(
            "[bold blue]Infrastructure Service Health Checker[/]",
            subtitle="[italic]LLM Token Manager[/]",
        )
    )

    # Check if Docker is running
    if not check_docker_running():
        console.print("[bold red]Error:[/] Docker is not running. " "Please start Docker and try again.")
        sys.exit(1)

    # Check if containers are running
    container_status = check_containers_running()
    all_running = all(container_status.values())

    # If containers aren't running, check if docker-compose file exists and offer to start
    if not all_running:
        if check_docker_compose_file():
            console.print("[yellow]Not all required containers are running.[/]")

            # List containers that are not running
            not_running = [svc for svc, running in container_status.items() if not running]
            console.print(f"[yellow]Missing services: {', '.join(not_running)}[/]")

            response = input("Do you want to start the services with docker-compose? (y/n): ").strip().lower()
            if response == "y":
                if not start_services():
                    console.print("[bold red]Failed to start services. Exiting.[/]")
                    sys.exit(1)
            else:
                console.print("[yellow]Continuing with health check of available services...[/]")
        else:
            console.print("[yellow]docker-compose.yml not found in current directory.[/]")
            console.print("[yellow]Continuing with health check of available services...[/]")

    # Create results table
    table = Table(title="Service Health Status")
    table.add_column("Service", style="cyan")
    table.add_column("Status", style="bold")
    table.add_column("Details", style="green")
    table.add_column("Connection Info", style="blue")

    # Check each service
    all_healthy = True

    # PostgreSQL
    pg_running = container_status.get("PostgreSQL", False)
    if pg_running:
        pg_healthy, pg_details = check_postgresql()
        all_healthy = all_healthy and pg_healthy

        pg_conn = (
            f"psycopg2.connect(host='{SERVICES['PostgreSQL']['host']}', "
            f"port={SERVICES['PostgreSQL']['port']}, "
            f"user='{SERVICES['PostgreSQL']['user']}', "
            f"password='****', "
            f"database='{SERVICES['PostgreSQL']['database']}')"
        )

        table.add_row(
            "PostgreSQL",
            "[bold green]✓ HEALTHY[/]" if pg_healthy else "[bold red]✗ UNHEALTHY[/]",
            pg_details,
            pg_conn,
        )
    else:
        all_healthy = False
        table.add_row("PostgreSQL", "[bold red]✗ NOT RUNNING[/]", "Container not found", "N/A")

    # Redis
    redis_running = container_status.get("Redis", False)
    if redis_running:
        redis_healthy, redis_details = check_redis()
        all_healthy = all_healthy and redis_healthy

        redis_conn = f"redis.Redis(host='{SERVICES['Redis']['host']}', " f"port={SERVICES['Redis']['port']})"

        table.add_row(
            "Redis",
            "[bold green]✓ HEALTHY[/]" if redis_healthy else "[bold red]✗ UNHEALTHY[/]",
            redis_details,
            redis_conn,
        )
    else:
        all_healthy = False
        table.add_row("Redis", "[bold red]✗ NOT RUNNING[/]", "Container not found", "N/A")

    # RabbitMQ
    rmq_running = container_status.get("RabbitMQ", False)
    if rmq_running:
        rmq_healthy, rmq_details = check_rabbitmq()
        all_healthy = all_healthy and rmq_healthy

        rmq_conn = (
            f"AMQP: amqp://{SERVICES['RabbitMQ']['user']}:****@"
            f"{SERVICES['RabbitMQ']['host']}:{SERVICES['RabbitMQ']['amqp_port']}/\n"
            f"Management: http://{SERVICES['RabbitMQ']['host']}:"
            f"{SERVICES['RabbitMQ']['mgmt_port']}"
        )

        table.add_row(
            "RabbitMQ",
            "[bold green]✓ HEALTHY[/]" if rmq_healthy else "[bold red]✗ UNHEALTHY[/]",
            rmq_details,
            rmq_conn,
        )
    else:
        all_healthy = False
        table.add_row("RabbitMQ", "[bold red]✗ NOT RUNNING[/]", "Container not found", "N/A")

    # Print results
    console.print(table)

    # Print overall status
    if all_healthy:
        console.print("\n[bold green]✓ ALL SERVICES ARE HEALTHY![/]")
        console.print("[green]Your infrastructure is ready for development.[/]")
        sys.exit(0)
    else:
        console.print("\n[bold red]✗ SOME SERVICES ARE UNHEALTHY OR NOT RUNNING[/]")
        console.print("[yellow]Please check the details above and fix any issues.[/]")
        sys.exit(1)


if __name__ == "__main__":
    main()
