@echo off
REM ============================================================
REM LLM Token Manager - Infrastructure Graceful Stop Script (Windows)
REM ============================================================

echo ============================================================
echo   LLM Token Manager - Infrastructure Graceful Stop
echo ============================================================
echo.

REM Determine script directory and project root
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%.."

REM Check if docker-compose.yml exists in root directory
if not exist docker-compose.yml (
    echo ERROR: docker-compose.yml not found in project root directory.
    echo Please run this script from the project root or ensure docker-compose.yml exists.
    pause
    exit /b 1
)

REM Check if Docker is running
docker info >nul 2>&1
if errorlevel 1 (
    echo ERROR: Docker is not running or not accessible.
    echo Please start Docker Desktop and try again.
    pause
    exit /b 1
)

echo ============================================================
echo   Checking Current Infrastructure Status
echo ============================================================
echo.

REM Check if any containers are running
docker-compose ps --services --filter "status=running" >nul 2>&1
if errorlevel 1 (
    echo No running containers found. Infrastructure may already be stopped.
    echo.
    goto :show_final_status
)

echo Current running services:
docker-compose ps --filter "status=running"
echo.

REM Ask user for confirmation
echo ============================================================
echo   Graceful Shutdown Confirmation
echo ============================================================
echo.
echo This will gracefully stop all infrastructure services:
echo   - PostgreSQL (llm_postgres)
echo   - Redis (llm_redis)
echo   - RabbitMQ (llm_rabbitmq)
echo.
set /p CONFIRM="Are you sure you want to stop all services? (y/N): "
if /i not "%CONFIRM%"=="y" (
    echo Operation cancelled by user.
    pause
    exit /b 0
)

echo.
echo ============================================================
echo   Performing Graceful Shutdown
echo ============================================================
echo.

REM Step 1: Stop services gracefully with timeout
echo Step 1: Stopping services gracefully...
docker-compose stop --timeout 30
if errorlevel 1 (
    echo WARNING: Some services may not have stopped gracefully.
    echo Proceeding with force stop...
    docker-compose kill
)

echo.
echo Step 2: Waiting for services to fully stop...
timeout /t 5 /nobreak >nul

REM Step 3: Remove containers
echo Step 3: Removing containers...
docker-compose rm -f
if errorlevel 1 (
    echo WARNING: Some containers may not have been removed cleanly.
)

echo.
echo Step 4: Checking for orphaned containers...
for /f "tokens=*" %%i in ('docker ps -a --filter "name=llm_" --format "{{.Names}}"') do (
    echo Found orphaned container: %%i
    docker stop %%i >nul 2>&1
    docker rm %%i >nul 2>&1
    echo Removed orphaned container: %%i
)

echo.
echo ============================================================
echo   Cleanup Options
echo ============================================================
echo.
echo Choose cleanup level:
echo   1. Keep all data (volumes preserved) - RECOMMENDED
echo   2. Remove containers only (keep volumes)
echo   3. Remove everything including data volumes - DESTRUCTIVE
echo   4. Skip cleanup
echo.
set /p CLEANUP_LEVEL="Enter choice (1-4) [default: 1]: "

if "%CLEANUP_LEVEL%"=="" set CLEANUP_LEVEL=1

if "%CLEANUP_LEVEL%"=="1" (
    echo Keeping all data volumes (recommended for development)
    goto :show_final_status
) else if "%CLEANUP_LEVEL%"=="2" (
    echo Removing containers only...
    docker-compose down
) else if "%CLEANUP_LEVEL%"=="3" (
    echo WARNING: This will remove ALL data including databases!
    set /p CONFIRM_DESTRUCTIVE="Are you absolutely sure? Type 'DELETE' to confirm: "
    if not "%CONFIRM_DESTRUCTIVE%"=="DELETE" (
        echo Destructive cleanup cancelled.
        goto :show_final_status
    )
    echo Removing everything including data volumes...
    docker-compose down -v --remove-orphans
    echo.
    echo WARNING: All data has been permanently deleted!
) else if "%CLEANUP_LEVEL%"=="4" (
    echo Skipping cleanup...
) else (
    echo Invalid choice. Skipping cleanup...
)

:show_final_status
echo.
echo ============================================================
echo   Final Status Check
echo ============================================================
echo.

REM Check final status
echo Checking final infrastructure status...
docker-compose ps

echo.
echo Checking for any remaining llm_* containers...
for /f "tokens=*" %%i in ('docker ps -a --filter "name=llm_" --format "{{.Names}}" 2^>nul') do (
    echo WARNING: Found remaining container: %%i
)

echo.
echo ============================================================
echo   Infrastructure Shutdown Complete!
echo ============================================================
echo.
echo Summary:
echo   - All services have been stopped
echo   - Containers have been removed
if "%CLEANUP_LEVEL%"=="3" (
    echo   - All data volumes have been removed
) else (
    echo   - Data volumes have been preserved
)
echo.
echo To restart the infrastructure:
echo   - Run: start.bat
echo.
echo To completely clean up (remove volumes):
echo   - Run: docker-compose down -v
echo.
echo ============================================================
echo.
pause
