@echo off
REM ============================================================================
REM  Image AP - Laser Image Prep  -  Setup completo
REM  -----------------------------------------------------------------------
REM  Este script:
REM    1. Verifica/instala Python 3.11+ via winget.
REM    2. Verifica/instala Node.js LTS via winget.
REM    3. Crea venv .venv312 + instala dependencias Python (.[api,perceptual]).
REM    4. cd web ^&^& npm install ^&^& npm run build (Wizard SvelteKit estatico).
REM    5. Crea acceso directo en el Escritorio a Iniciar_Laser_App.bat.
REM  -----------------------------------------------------------------------
REM  Doble-clic y deja que termine. Tarda ~15-30 min la primera vez (depende
REM  de Internet). Despues sera doble-clic en Iniciar_Laser_App.bat y listo.
REM ============================================================================

setlocal EnableExtensions EnableDelayedExpansion
title Image AP Laser - Setup
chcp 65001 >nul
cd /d "%~dp0"

echo.
echo  ============================================
echo   Image AP Sport for Laser - Setup
echo  ============================================
echo.
echo  Este script va a instalar todo lo necesario para correr la app.
echo  Si pide UAC (permisos administrador), aceptar.
echo.
pause

REM ---------------------------------------------------------------------------
REM 1. Python 3.11+
REM ---------------------------------------------------------------------------
echo.
echo [1/5] Verificando Python 3.11+...
where python >nul 2>nul
if errorlevel 1 (
    echo   No se encontro python en PATH. Instalando via winget...
    winget install --id Python.Python.3.11 -e --accept-source-agreements --accept-package-agreements
    if errorlevel 1 (
        echo   [ERROR] winget fallo. Instala Python 3.11+ manualmente desde:
        echo   https://www.python.org/downloads/release/python-3119/
        pause
        exit /b 1
    )
    echo   Python instalado. Reabri esta ventana para que PATH se refresque y volve a ejecutar Setup.
    pause
    exit /b 0
) else (
    python --version
    echo   OK
)

REM ---------------------------------------------------------------------------
REM 2. Node.js LTS
REM ---------------------------------------------------------------------------
echo.
echo [2/5] Verificando Node.js...
where node >nul 2>nul
if errorlevel 1 (
    echo   No se encontro node en PATH. Instalando via winget...
    winget install --id OpenJS.NodeJS.LTS -e --accept-source-agreements --accept-package-agreements
    if errorlevel 1 (
        echo   [ERROR] winget fallo. Instala Node.js LTS manualmente desde:
        echo   https://nodejs.org/en/download/prebuilt-installer
        pause
        exit /b 1
    )
    echo   Node instalado. Reabri esta ventana y volve a ejecutar Setup.
    pause
    exit /b 0
) else (
    node --version
    echo   OK
)

REM ---------------------------------------------------------------------------
REM 3. Venv Python + dependencias
REM ---------------------------------------------------------------------------
echo.
echo [3/5] Configurando .venv312 + dependencias Python...
if not exist ".venv312\Scripts\python.exe" (
    python -m venv .venv312
    if errorlevel 1 (
        echo   [ERROR] No se pudo crear venv.
        pause
        exit /b 1
    )
)
call ".venv312\Scripts\activate.bat"
echo   Actualizando pip...
python -m pip install --upgrade pip --quiet
echo   Instalando proyecto y extras [api,perceptual,dev]... (puede tardar varios minutos)
python -m pip install -e ".[api,perceptual,dev]"
if errorlevel 1 (
    echo   [ERROR] pip install fallo. Revisar conexion a Internet o pyproject.toml.
    pause
    exit /b 1
)
echo   OK

REM ---------------------------------------------------------------------------
REM 4. Wizard SvelteKit
REM ---------------------------------------------------------------------------
echo.
echo [4/5] Construyendo wizard web (SvelteKit estatico)...
pushd web

REM node_modules puede estar (a) ausente, (b) presente pero incompleto. Si falta vite,
REM reinstalamos las dependencias (npm ci). Esto cubre el caso de install previo roto.
if not exist "node_modules\.bin\vite.cmd" (
    if exist "node_modules" (
        echo   node_modules incompleto ^(falta vite^). Reinstalando dependencias npm...
    ) else (
        echo   node_modules ausente. Instalando dependencias npm...
    )
    if exist "package-lock.json" (
        call npm ci --no-audit --no-fund
    ) else (
        call npm install --no-audit --no-fund
    )
    if errorlevel 1 (
        echo   [ERROR] npm install/ci fallo. Revisa la consola arriba.
        popd
        pause
        exit /b 1
    )
)

REM Usar la ruta absoluta al shim de vite para evitar problemas de PATH dentro de
REM Program Files. Si falla, fallback a npx (que tambien resuelve binarios locales).
echo   Building estatico (vite build)...
if exist "node_modules\.bin\vite.cmd" (
    call "node_modules\.bin\vite.cmd" build
) else (
    call npx --no-install vite build
)
if errorlevel 1 (
    echo   [ERROR] vite build fallo. Revisa la consola arriba.
    popd
    pause
    exit /b 1
)
popd
echo   OK

REM ---------------------------------------------------------------------------
REM 5. Shortcut en Escritorio
REM ---------------------------------------------------------------------------
echo.
echo [5/5] Creando acceso directo en el Escritorio...
set "SHORTCUT_PS=$ws = New-Object -ComObject WScript.Shell; $sc = $ws.CreateShortcut([Environment]::GetFolderPath('Desktop') + '\Image AP Laser.lnk'); $sc.TargetPath = '%CD%\Iniciar_Laser_App.bat'; $sc.WorkingDirectory = '%CD%'; $sc.IconLocation = 'imageres.dll,108'; $sc.Save()"
powershell -NoProfile -ExecutionPolicy Bypass -Command "%SHORTCUT_PS%"
if errorlevel 1 (
    echo   [WARN] No se pudo crear shortcut. Manualmente arrastra Iniciar_Laser_App.bat al Escritorio.
) else (
    echo   OK ^(Image AP Laser en el Escritorio^)
)

echo.
echo  ============================================
echo   Setup COMPLETO.
echo  ============================================
echo.
echo  Para arrancar la app: doble-clic en "Image AP Laser" del Escritorio
echo  o en Iniciar_Laser_App.bat de este directorio.
echo.
echo  Wizard estara en http://127.0.0.1:18765/app/
echo.
pause
exit /b 0
