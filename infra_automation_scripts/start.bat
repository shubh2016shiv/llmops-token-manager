@echo off
REM ============================================================
REM LLM Token Manager - Infrastructure Start Script (Windows)
REM ============================================================
REM Starts all infrastructure services and prints a professional
REM summary table of network details, IPs, ports, and health.
REM ============================================================

setlocal enabledelayedexpansion

echo.
echo   ============================================================
echo     LLM Token Manager - Infrastructure Start
echo   ============================================================
echo.

REM --- Check Docker -------------------------------------------------
docker --version >nul 2>&1
if errorlevel 1 (
    echo   ERROR: Docker is not installed.
    echo   Visit: https://www.docker.com/products/docker-desktop
    pause
    exit /b 1
)

REM --- Normalise compose command (v2 preferred) ---------------------
set "COMPOSE_CMD=docker-compose"
docker compose version >nul 2>&1
if not errorlevel 1 set "COMPOSE_CMD=docker compose"

REM --- Navigate to project root -------------------------------------
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%.."

if not exist docker-compose.yml (
    echo   ERROR: docker-compose.yml not found in project root.
    pause
    exit /b 1
)

if not exist .env (
    echo   INFO: .env file not found -- creating one with defaults...
    echo # LLM Token Manager Environment Variables > .env
)

echo   Stopping any previously running containers...
%COMPOSE_CMD% --profile worker down --remove-orphans >nul 2>&1

echo.
echo   Building images and starting services...
%COMPOSE_CMD% --profile worker up -d --build

echo.
echo   Waiting for services to initialise...
timeout /t 8 /nobreak >nul

REM --- Health checks ------------------------------------------------
echo.
echo   -- Health Checks -------------------------------------------
echo.

echo   PostgreSQL ... | findstr /v "^$"
%COMPOSE_CMD% exec -T postgres pg_isready -U myuser -d mydb -q 2>nul && (
    echo     [OK] healthy
) || (
    echo     [FAIL] unavailable
)

echo   Redis      ... | findstr /v "^$"
%COMPOSE_CMD% exec -T redis redis-cli ping 2>nul | findstr "PONG" >nul && (
    echo     [OK] healthy
) || (
    echo     [FAIL] unavailable
)

echo   RabbitMQ   ... | findstr /v "^$"
%COMPOSE_CMD% exec -T rabbitmq rabbitmq-diagnostics -q ping 2>nul >nul && (
    echo     [OK] healthy
) || (
    echo     [FAIL] unavailable
)

echo   Celery     ... | findstr /v "^$"
for /f "usebackq delims=" %%i in (`docker inspect --format "{{if .State.Health}}{{.State.Health.Status}}{{else}}no healthcheck{{end}}" llm_celery_worker 2^>nul`) do set "CELERY_STATUS=%%i"
if "%CELERY_STATUS%"=="" set "CELERY_STATUS=unknown"
echo     %CELERY_STATUS%

REM --- Summary table ------------------------------------------------
call :print_infra_summary

REM --- Quick-reference footer ---------------------------------------
echo   -- Access Points ------------------------------------------
echo.
echo   PostgreSQL   :  localhost:5433   (user: myuser / db: mydb^)
echo   Redis        :  localhost:6379
echo   RabbitMQ     :  localhost:5672   (AMQP^)
echo   RabbitMQ UI  :  http://localhost:15672  (rmq_user^)
echo.
echo   -- Quick Commands ----------------------------------------
echo.
echo   %COMPOSE_CMD% --profile worker logs -f
echo   %COMPOSE_CMD% --profile worker logs -f celery_worker
echo   %COMPOSE_CMD% --profile worker ps
echo   infra_automation_scripts\stop.bat
echo   %COMPOSE_CMD% --profile infra up -d              (infra-only^)
echo.
echo   ============================================================
echo   All services are running in the background.
echo   You can safely close this window or press any key to exit.
echo   ============================================================
echo.
pause >nul
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
        REM collapse 5432/tcp->5433 into 5432->5433
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
