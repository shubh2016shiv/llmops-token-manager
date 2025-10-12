@echo off
REM ============================================================
REM LLM Token Manager - Infrastructure Quick Start Script (Windows)
REM ============================================================

echo ============================================================
echo   LLM Token Manager - Infrastructure Quick Start
echo ============================================================
echo.

REM Check if Docker is installed
docker --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Docker is not installed. Please install Docker Desktop first.
    echo Visit: https://www.docker.com/products/docker-desktop
    pause
    exit /b 1
)

REM Check if Docker Compose is installed
docker-compose --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Docker Compose is not installed.
    pause
    exit /b 1
)

echo Docker and Docker Compose are installed
echo.

REM Determine script directory and project root
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%.."

REM Check if docker-compose.yml exists in root directory
if not exist docker-compose.yml (
    echo ERROR: docker-compose.yml not found in project root directory.
    pause
    exit /b 1
)

REM Check if .env file exists (optional)
if not exist .env (
    echo NOTE: .env file not found. Creating empty .env file...
    echo # LLM Token Manager Environment Variables > .env
    echo Created empty .env file
    echo.
)

echo ============================================================
echo   Starting Infrastructure Services
echo ============================================================
echo.

REM Stop any existing containers
echo Stopping any existing containers...
docker-compose down

REM Build and start services
echo.
echo Building and starting services...
docker-compose up -d --build

REM Wait for services to be healthy
echo.
echo Waiting for services to be ready...
timeout /t 10 /nobreak >nul

REM Check service status
echo.
echo ============================================================
echo   Service Status
echo ============================================================
docker-compose ps

echo.
echo ============================================================
echo   Testing Infrastructure Health
echo ============================================================
echo.

REM Check PostgreSQL
echo Checking PostgreSQL...
docker-compose exec -T postgres pg_isready -U myuser
if errorlevel 1 (
    echo WARNING: PostgreSQL may not be ready yet.
) else (
    echo PostgreSQL is healthy!
)

REM Check Redis
echo.
echo Checking Redis...
docker-compose exec -T redis redis-cli ping
if errorlevel 1 (
    echo WARNING: Redis may not be ready yet.
) else (
    echo Redis is healthy!
)

REM Check RabbitMQ
echo.
echo Checking RabbitMQ...
docker-compose exec -T rabbitmq rabbitmqctl status > nul 2>&1
if errorlevel 1 (
    echo WARNING: RabbitMQ may not be ready yet.
) else (
    echo RabbitMQ is healthy!
)

echo.
echo ============================================================
echo   Infrastructure Deployment Complete!
echo ============================================================
echo.
echo Access points:
echo   - PostgreSQL:          localhost:5432 (user: myuser, password: mypassword, db: mydb)
echo   - Redis:               localhost:6379
echo   - RabbitMQ AMQP:       localhost:5672
echo   - RabbitMQ Dashboard:  http://localhost:15672 (login: rmq_user / rmq_password)
echo.
echo Useful commands:
echo   - View logs:          docker-compose logs -f
echo   - Stop services:      docker-compose down
echo   - Check health:       python check_infra_service_health.py
echo.
echo Next steps:
echo   1. Connect to PostgreSQL with your preferred client
echo   2. Connect to Redis with your preferred client
echo   3. Access RabbitMQ management UI at http://localhost:15672
echo.
echo ============================================================
echo.
pause