@echo off
REM ============================================================
REM LLM Token Manager - Infrastructure Stop Script (Windows)
REM ============================================================
REM Gracefully stops infrastructure, shows current state before
REM and after shutdown in a professional table format.
REM ============================================================

setlocal enabledelayedexpansion

echo.
echo   ============================================================
echo     LLM Token Manager - Infrastructure Stop
echo   ============================================================
echo.

REM --- Navigate to project root -------------------------------------
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%.."

if not exist docker-compose.yml (
    echo   ERROR: docker-compose.yml not found in project root.
    pause
    exit /b 1
)

docker info >nul 2>&1
if errorlevel 1 (
    echo   ERROR: Docker is not running or not accessible.
    pause
    exit /b 1
)

REM --- Normalise compose command (v2 preferred) ---------------------
set "COMPOSE_CMD=docker-compose"
docker compose version >nul 2>&1
if not errorlevel 1 set "COMPOSE_CMD=docker compose"

REM --- Pre-shutdown snapshot ----------------------------------------
echo   -- Current Infrastructure State ---------------------------

REM Check if anything is running
%COMPOSE_CMD% --profile worker ps --services --filter "status=running" 2>nul | findstr /r "." >nul
if errorlevel 1 (
    echo.
    echo   INFO: No running services detected. Infrastructure is already down.
    echo.
    echo   -- Post-Shutdown Summary ---------------------------------
    call :print_infra_summary
    echo   All containers are stopped. You can safely close this window.
    echo   ============================================================
    echo.
    pause
    exit /b 0
)

call :print_infra_summary

REM --- Confirmation -------------------------------------------------
echo   -- Graceful Shutdown --------------------------------------
echo.
echo   This will stop:
echo     * PostgreSQL    (llm_postgres^)
echo     * Redis         (llm_redis^)
echo     * RabbitMQ      (llm_rabbitmq^)
echo     * Celery Worker (llm_celery_worker^)
echo.
set /p CONFIRM="  Proceed? (y/N): "
if /i not "%CONFIRM%"=="y" (
    echo   Cancelled by user.
    echo   ============================================================
    echo.
    pause
    exit /b 0
)

REM --- Shutdown steps -----------------------------------------------
echo.
echo   Step 1/4 -- Stopping services gracefully (timeout: 30s^)...
%COMPOSE_CMD% --profile worker stop --timeout 30
if errorlevel 1 (
    echo   WARNING: Some services did not stop gracefully -- forcing...
    %COMPOSE_CMD% --profile worker kill
)

echo   Step 2/4 -- Waiting for services to fully stop...
timeout /t 5 /nobreak >nul

echo   Step 3/4 -- Removing containers...
%COMPOSE_CMD% --profile worker rm -f 2>nul || echo   WARNING: Some containers could not be removed cleanly.

echo   Step 4/4 -- Checking for orphaned containers...
for /f "usebackq tokens=*" %%i in (`docker ps -a --filter "name=llm_" --format "{{.Names}}" 2^>nul`) do (
    echo   Removed orphaned: %%i
    docker stop %%i >nul 2>&1
    docker rm %%i >nul 2>&1
)

REM --- Cleanup options ----------------------------------------------
echo.
echo   -- Cleanup Options ----------------------------------------
echo.
echo     1  Keep all data (volumes preserved^) -- RECOMMENDED
echo     2  Remove containers only (keep volumes^)
echo     3  Remove everything INCLUDING data volumes -- DESTRUCTIVE
echo     4  Skip cleanup
echo.
set /p CLEANUP_LEVEL="  Choice (1-4) [default: 1]: "
if "%CLEANUP_LEVEL%"=="" set CLEANUP_LEVEL=1

if "%CLEANUP_LEVEL%"=="1" (
    echo   -> Volumes preserved.
) else if "%CLEANUP_LEVEL%"=="2" (
    echo   -> Removing containers...
    %COMPOSE_CMD% --profile worker down
) else if "%CLEANUP_LEVEL%"=="3" (
    echo   WARNING: DESTRUCTIVE -- This will permanently delete all data!
    set /p CONFIRM_DEL="  Type 'DELETE' to confirm: "
    if "!CONFIRM_DEL!"=="DELETE" (
        echo   -> Removing everything including volumes...
        %COMPOSE_CMD% --profile worker down -v --remove-orphans
        echo   OK: All data volumes have been permanently deleted.
    ) else (
        echo   Cancelled.
    )
) else if "%CLEANUP_LEVEL%"=="4" (
    echo   -> Cleanup skipped.
) else (
    echo   -> Invalid choice -- cleanup skipped.
)

REM --- Post-shutdown summary ----------------------------------------
echo.
echo   -- Post-Shutdown Summary ----------------------------------
call :print_infra_summary

echo   -- Restart -----------------------------------------------
echo.
echo   infra_automation_scripts\start.bat         # restart everything
echo   %COMPOSE_CMD% --profile worker down -v     # full cleanup (volumes^)
echo.
echo   All containers are stopped. You can safely close this window.
echo   ============================================================
echo.
pause
exit /b 0

REM ==================================================================
REM  print_infra_summary - table of all services (ASCII-safe)
REM ==================================================================
:print_infra_summary
echo.
echo   +------------------+--------------------------+-----------------+------------------------------+----------------+
echo   ^| Service          ^| Network                  ^| IP Address      ^| Ports (Host-^>Container^)     ^| Status         ^|
echo   +------------------+--------------------------+-----------------+------------------------------+----------------+
call :print_row "PostgreSQL"      llm_postgres
call :print_row "Redis"           llm_redis
call :print_row "RabbitMQ"        llm_rabbitmq
call :print_row "Celery Worker"   llm_celery_worker
echo   +------------------+--------------------------+-----------------+------------------------------+----------------+
echo.
exit /b

REM ==================================================================
REM  print_row - prints one row of the summary table
REM ==================================================================
:print_row
set "DISPLAY_NAME=%~1"
set "CONTAINER=%~2"

REM Get network name
for /f "usebackq delims=" %%i in (`docker inspect --format "{{range $k,$v := .NetworkSettings.Networks}}{{$k}}{{end}}" %CONTAINER% 2^>nul`) do set "NET=%%i"
if "%NET%"=="" set "NET=-"

REM Get IP address
for /f "usebackq delims=" %%i in (`docker inspect --format "{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}" %CONTAINER% 2^>nul`) do set "IP=%%i"
if "%IP%"=="" set "IP=-"

REM Get port mappings (strip protocol suffix like /tcp for brevity)
set "PORT_STR=-"
for /f "usebackq delims=" %%i in (`docker inspect --format "{{range $k,$v := .NetworkSettings.Ports}}{{if $v}}{{$k}}-^>{{(index $v 0).HostPort}}, {{end}}{{end}}" %CONTAINER% 2^>nul`) do (
    set "PORTS=%%i"
    if not "!PORTS!"=="" (
        set "PORT_STR=!PORTS:~0,-2!"
        set "PORT_STR=!PORT_STR:/tcp=!"
    )
)

REM Get health status
for /f "usebackq delims=" %%i in (`docker inspect --format "{{if .State.Health}}{{.State.Health.Status}}{{else}}no healthcheck{{end}}" %CONTAINER% 2^>nul`) do set "STATUS=%%i"
if "%STATUS%"=="" set "STATUS=stopped"

REM Build and print the row
set "ROW=  ^| %DISPLAY_NAME%"
call :pad DISPLAY_NAME 16
set "ROW=%ROW%^| %NET%"
call :pad NET 24
set "ROW=%ROW%^| %IP%"
call :pad IP 15
set "ROW=%ROW%^| %PORT_STR%"
call :pad PORT_STR 28
set "ROW=%ROW%^| %STATUS%"
call :pad STATUS 14
echo %ROW%^|
exit /b

REM ==================================================================
REM  pad - appends spaces to a variable until it reaches target width
REM ==================================================================
:pad
set "VAR=!%1!"
set "WIDTH=%~2"
:Loop
if "!VAR!"=="" set "VAR= "
set "LEN=0"
for /l %%A in (1,1,%WIDTH%) do if "!VAR:~%%A,1!" neq "" set /a LEN=%%A+1
if !LEN! lss %WIDTH% (
    set "VAR=!VAR! "
    goto :Loop
)
set "%1=!VAR!"
exit /b
