@echo off
setlocal EnableExtensions
chcp 65001 >nul
cd /d "%~dp0"

echo.
echo  Instalacion de dependencias (primera vez o tras actualizar repo)
echo  ================================================================
echo.

where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python no esta en PATH. Instala Python 3.11 o 3.12.
    pause
    exit /b 1
)

if not exist ".venv312\Scripts\python.exe" (
    echo Creando entorno virtual .venv312 ...
    python -m venv .venv312
)

echo Instalando paquetes Python [api, perceptual, dev, sam2] ...
.venv312\Scripts\python.exe -m pip install -U pip
.venv312\Scripts\pip.exe install -e ".[api,perceptual,dev,sam2]"

echo.
echo Instalando dependencias web (npm) ...
cd web
where npm >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Node/npm no encontrado. Instala Node 18+.
    cd ..
    pause
    exit /b 1
)
call npm install
cd ..

echo.
echo  Instalacion terminada. Ahora puedes usar Iniciar_Laser_App.bat
echo.
if not exist ".env" if exist ".env.example" (
    echo  Copia .env.example a .env y pon tu HF_TOKEN si usas SAM2.
)
pause
endlocal
