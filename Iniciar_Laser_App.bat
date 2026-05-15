@echo off
setlocal EnableExtensions
chcp 65001 >nul
title Image AP Sport for Laser - Iniciando...

set "ROOT=%~dp0"
cd /d "%ROOT%"

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

if not exist "%ROOT%.venv312\Scripts\python.exe" (
    echo [ERROR] No existe .venv312
    echo  Ejecuta Instalar_Dependencias.bat
    pause
    exit /b 1
)

if not exist "%ROOT%web\node_modules\" (
    echo [ERROR] Falta web\node_modules
    echo  Ejecuta Instalar_Dependencias.bat
    pause
    exit /b 1
)

if not exist "%ROOT%_win_start_api.bat" (
    echo [ERROR] Falta _win_start_api.bat en la raiz del repo.
    pause
    exit /b 1
)

echo [1/4] API (puerto %LASER_API_PORT%)...
powershell -NoProfile -Command "try { (Invoke-WebRequest -Uri 'http://127.0.0.1:%LASER_API_PORT%/api/health' -UseBasicParsing -TimeoutSec 2).StatusCode | Out-Null; exit 0 } catch { exit 1 }" >nul 2>&1
if not errorlevel 1 (
    echo        API ya responde - no se abre otra instancia.
    goto :start_web
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

echo [2/4] Esperando API...
set /a tries=0
:wait_api
set /a tries+=1
powershell -NoProfile -Command "try { (Invoke-WebRequest -Uri 'http://127.0.0.1:%LASER_API_PORT%/api/health' -UseBasicParsing -TimeoutSec 3).StatusCode | Out-Null; exit 0 } catch { exit 1 }" >nul 2>&1
if not errorlevel 1 goto :api_ok
if %tries% geq 60 (
    echo [ERROR] API no respondio. Mira la ventana "Laser API".
    pause
    exit /b 1
)
ping 127.0.0.1 -n 3 >nul 2>&1
goto :wait_api
:api_ok
echo        API lista.

:start_web
echo [3/4] Web (puerto 5173)...
powershell -NoProfile -Command "try { (Invoke-WebRequest -Uri 'http://127.0.0.1:5173/' -UseBasicParsing -TimeoutSec 2).StatusCode | Out-Null; exit 0 } catch { exit 1 }" >nul 2>&1
if not errorlevel 1 (
    echo        Vite ya responde.
    goto :open_browser
)

powershell -NoProfile -Command "if (Get-NetTCPConnection -LocalPort 5173 -State Listen -ErrorAction SilentlyContinue) { exit 1 } else { exit 0 }" >nul 2>&1
if errorlevel 1 (
    echo [AVISO] Puerto 5173 ocupado sin respuesta HTTP. Ejecuta Detener_Laser_App.bat
    pause
    exit /b 1
)

echo        Abriendo ventana "Laser Web"...
start "Laser Web" /D "%ROOT%" cmd /k call _win_start_web.bat

echo [4/4] Esperando Vite...
set /a tries=0
:wait_web
set /a tries+=1
powershell -NoProfile -Command "try { (Invoke-WebRequest -Uri 'http://127.0.0.1:5173/' -UseBasicParsing -TimeoutSec 3).StatusCode | Out-Null; exit 0 } catch { exit 1 }" >nul 2>&1
if not errorlevel 1 goto :open_browser
if %tries% geq 45 (
    echo [ERROR] Vite no respondio. Mira la ventana "Laser Web".
    pause
    exit /b 1
)
ping 127.0.0.1 -n 3 >nul 2>&1
goto :wait_web

:open_browser
start "" "http://localhost:5173"
echo.
echo  Listo: http://localhost:5173
echo  API: http://127.0.0.1:%LASER_API_PORT%
echo  Ventanas: "Laser API" y "Laser Web" deben quedar abiertas.
echo  Cerrar todo: Detener_Laser_App.bat
echo.
pause
endlocal
exit /b 0
