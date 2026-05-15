# Installer profesional (.exe) para Image AP Laser

Este directorio contiene el script Inno Setup para producir un instalador
`.exe` con UI estándar (asistente con licencia, carpeta destino, accesos
directos, desinstalador en Panel de Control).

## Compilar el .exe (Windows)

### 1. Instalar Inno Setup Compiler (una vez)

Descargar desde **https://jrsoftware.org/isdl.php** → instalar (~5 MB).
El compilador queda en `C:\Program Files (x86)\Inno Setup 6\`.

### 2. Compilar

**Opción A — UI:** Doble-clic en `LaserApp.iss` → se abre Inno Setup
Compiler → menú `Build` → `Compile` (o F9).

**Opción B — CLI:**
```powershell
& "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" .\installer\LaserApp.iss
```

El instalador resultante queda en `installer\Output\ImageAPLaser_Setup_v1.2.0.exe`.

## Qué hace el instalador para el usuario final

1. Asistente clásico Windows (licencia GPL-3.0, carpeta destino, idioma).
2. Copia toda la fuente del proyecto a `C:\Program Files\ImageAPLaser\`
   (o donde elija el usuario).
3. Crea accesos directos en Menú Inicio + Escritorio (opcional).
4. **Al terminar la instalación**, lanza `Setup_LaserApp.bat` que:
   - Verifica Python 3.11+ → si falta, instala via `winget install Python.Python.3.11`.
   - Verifica Node.js LTS → si falta, instala via `winget install OpenJS.NodeJS.LTS`.
   - Crea venv `.venv312` y pip-instala las deps Python (`[api,perceptual]`).
   - Construye el wizard SvelteKit estático.
   - Crea acceso directo en Escritorio.
5. Registra desinstalador en Panel de Control.

## Alternativa sin compilar el .exe

El usuario puede saltarse el instalador `.exe` y usar directamente:

```powershell
git clone https://github.com/Vahlame/image-ap-sport-for-laser
cd image-ap-sport-for-laser
Setup_LaserApp.bat       # configura todo
Iniciar_Laser_App.bat    # arranca la app
```

`Setup_LaserApp.bat` y `Iniciar_Laser_App.bat` están en la raíz del repo;
son la ruta "sin instalador" para devs.

## Por qué no PyInstaller `.exe` "todo en uno"

Sí intenté la ruta PyInstaller-onefile (single .exe con Python + libs bundleados):

- **PyTorch 2.6 + CUDA**: el bundle pesa ~2.5 GB y arranca en ~30 s la primera vez
  (descomprime libs a `%TEMP%`).
- **CuDNN runtime**: PyInstaller no detecta automáticamente todos los `.dll` de CUDA;
  hay que mantener un `hidden-imports` largo y frágil.
- **LPIPS pretrained weights**: hay que bundlearlas o el primer eval falla.
- **Mantenimiento alto**: cada upgrade de PyTorch rompe el spec.

El approach **Inno Setup + bootstrap** es:
- Más liviano (instalador ~10 MB sin PyTorch; las deps se bajan en setup).
- Maneja CUDA/CuDNN nativamente (pip instala wheels oficiales según el sistema).
- Compatible con futuras versiones sin re-empaquetar.

Si en el futuro hace falta `.exe` 100 % offline:
- `briefcase` (BeeWare) tiene mejor soporte para apps Python distribuidas.
- O bundlear Python embeddable + wheels pre-descargadas en un `.zip` autoextraíble.
