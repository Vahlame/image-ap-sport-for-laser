@echo off
setlocal EnableExtensions
chcp 65001 >nul
title Laser Web - Image AP (puerto 5173)

cd /d "%~dp0web"
if errorlevel 1 (
    echo [ERROR] No existe la carpeta web\
    pause
    exit /b 1
)

if not exist "node_modules\" (
    echo [ERROR] Falta node_modules - ejecuta Instalar_Dependencias.bat
    pause
    exit /b 1
)

echo.
echo  Vite dev server en http://localhost:5173
echo  Cierra esta ventana para detener la web.
echo.

call npm run dev

echo.
echo  Web detenida (codigo %ERRORLEVEL%).
pause
endlocal
