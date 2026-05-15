@echo off
REM ============================================================================
REM  Image AP Sport for Laser - Launcher (modo produccion v1.3+)
REM  -----------------------------------------------------------------------
REM  v1.3+ sirve el wizard como build estatico desde FastAPI en /app/.
REM  Ya NO necesitamos arrancar Vite dev (era un punto de falla con npm/PATH).
REM
REM  Flujo:
REM    1. Verifica venv + build estatico (web/build/).
REM    2. Arranca API en ventana 'Laser API' (uvicorn).
REM    3. Espera al healthcheck.
REM    4. Abre el navegador a http://127.0.0.1:18765/app/.
REM
REM  Si querés modo dev con hot-reload (vite): cd web && npm run dev
REM  (manualmente; el launcher principal va por la ruta estatica).
REM ============================================================================

setlocal EnableExtensions
chcp 65001 >nul
title Image AP Sport for Laser - Iniciando...

set "ROOT=%~dp0"
cd /d "%ROOT%"

REM Cargar .env (port custom, etc.)
if exist ".env" (
    for /f "usebackq eol=# tokens=1,* delims==" %%a in (".env") do (
        if not "%%b"=="" set "%%a=%%b"
    )
)
if not defined LASER_API_PORT set "LASER_API_PORT=18765"

echo.
echo  ============================================
echo   Image AP Sport for Laser - Wizard
echo  ============================================
echo.

REM ---------------------------------------------------------------------------
REM Verificaciones previas
REM ---------------------------------------------------------------------------
if not exist "%ROOT%.venv312\Scripts\python.exe" (
    echo [ERROR] No existe .venv312
    echo  Ejecuta Setup_LaserApp.bat ^(o Instalar_Dependencias.bat^) primero.
    pause
    exit /b 1
)

if not exist "%ROOT%web\build\index.html" (
    echo [AVISO] No existe web\build\index.html.
    echo  Intentando construir el wizard estatico ahora...
    if not exist "%ROOT%web\node_modules\" (
        echo [ERROR] Falta web\node_modules. Ejecuta Setup_LaserApp.bat.
        pause
        exit /b 1
    )
    pushd "%ROOT%web"
    call npm run build
    if errorlevel 1 (
        echo [ERROR] npm run build fallo. Revisa la consola arriba.
        popd
        pause
        exit /b 1
    )
    popd
    echo  Build OK.
)

if not exist "%ROOT%_win_start_api.bat" (
    echo [ERROR] Falta _win_start_api.bat en la raiz del repo.
    pause
    exit /b 1
)

REM ---------------------------------------------------------------------------
REM 1. API + wizard estatico en :18765
REM ---------------------------------------------------------------------------
echo [1/3] API + wizard (puerto %LASER_API_PORT%)...
powershell -NoProfile -Command "try { (Invoke-WebRequest -Uri 'http://127.0.0.1:%LASER_API_PORT%/api/health' -UseBasicParsing -TimeoutSec 2).StatusCode | Out-Null; exit 0 } catch { exit 1 }" >nul 2>&1
if not errorlevel 1 (
    echo        API ya responde - no se abre otra instancia.
    goto :open_browser
)

powershell -NoProfile -Command "if (Get-NetTCPConnection -LocalPort %LASER_API_PORT% -State Listen -ErrorAction SilentlyContinue) { exit 1 } else { exit 0 }" >nul 2>&1
if errorlevel 1 (
    echo [AVISO] Puerto %LASER_API_PORT% ocupado pero la API no responde.
    echo         Ejecuta Detener_Laser_App.bat y vuelve a intentar.
    pause
    exit /b 1
)

echo        Abriendo ventana "Laser API"...
start "Laser API" /D "%ROOT%" cmd /k call _win_start_api.bat

echo [2/3] Esperando API (puede tardar 10-20s la primera vez por LPIPS warmup)...
set /a tries=0
:wait_api
set /a tries+=1
powershell -NoProfile -Command "try { (Invoke-WebRequest -Uri 'http://127.0.0.1:%LASER_API_PORT%/api/health' -UseBasicParsing -TimeoutSec 3).StatusCode | Out-Null; exit 0 } catch { exit 1 }" >nul 2>&1
if not errorlevel 1 goto :api_ok
if %tries% geq 60 (
    echo [ERROR] API no respondio en 60 intentos. Mira la ventana "Laser API" para detalles.
    pause
    exit /b 1
)
ping 127.0.0.1 -n 3 >nul 2>&1
goto :wait_api
:api_ok
echo        API lista.

:open_browser
echo [3/3] Abriendo navegador en http://127.0.0.1:%LASER_API_PORT%/app/
start "" "http://127.0.0.1:%LASER_API_PORT%/app/"

echo.
echo  ============================================
echo   Listo. Wizard en http://127.0.0.1:%LASER_API_PORT%/app/
echo   API:               http://127.0.0.1:%LASER_API_PORT%/api/health
echo.
echo   La ventana "Laser API" debe quedar abierta.
echo   Cerrarla = detener el servidor.
echo   Alternativa: ejecutar Detener_Laser_App.bat para cerrar limpio.
echo  ============================================
echo.
pause
endlocal
exit /b 0
