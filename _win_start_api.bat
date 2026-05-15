@echo off
setlocal EnableExtensions
chcp 65001 >nul
title Laser API - Image AP

cd /d "%~dp0"
if errorlevel 1 (
    echo [ERROR] No se pudo entrar al directorio del proyecto.
    pause
    exit /b 1
)

if not exist ".venv312\Scripts\python.exe" (
    echo [ERROR] Falta .venv312 - ejecuta Instalar_Dependencias.bat
    pause
    exit /b 1
)

if exist ".env" (
    for /f "usebackq eol=# tokens=1,* delims==" %%a in (".env") do (
        if not "%%b"=="" set "%%a=%%b"
    )
)

if not defined LASER_API_PORT set "LASER_API_PORT=18765"
if not defined LASER_LPIPS_DEVICE set "LASER_LPIPS_DEVICE=auto"
if not defined LASER_CUDA_MEMORY_CAP_GIB set "LASER_CUDA_MEMORY_CAP_GIB=5.5"

echo.
echo  API FastAPI en http://127.0.0.1:%LASER_API_PORT%
echo  Health: http://127.0.0.1:%LASER_API_PORT%/api/health
echo  Cierra esta ventana para detener la API.
echo.

".venv312\Scripts\python.exe" -m uvicorn scripts.api_server:app --host 127.0.0.1 --port %LASER_API_PORT%

echo.
echo  API detenida (codigo %ERRORLEVEL%).
pause
endlocal
