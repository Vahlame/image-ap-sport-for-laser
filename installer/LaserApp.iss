; Inno Setup script para Image AP Sport for Laser.
;
; Para compilar este script en un .exe instalador profesional:
;   1. Descargar Inno Setup desde https://jrsoftware.org/isdl.php (gratis).
;   2. Instalar Inno Setup Compiler.
;   3. Abrir LaserApp.iss en Inno Setup Compiler (doble-click).
;   4. Build > Compile (F9). El .exe se genera en installer/Output/.
;
; El instalador resultante:
;   - Muestra UI estandar (idioma, licencia GPL-3, carpeta destino, accesos directos).
;   - Copia toda la fuente del proyecto a Program Files (o user-selected).
;   - Crea acceso directo en Menu Inicio + Escritorio (opcional).
;   - Tras instalar, lanza Setup_LaserApp.bat para configurar venv + dependencias.
;   - Registra desinstalador en Panel de Control.
;
; Requisitos del usuario tras instalar:
;   - Conexion a Internet (para que Setup_LaserApp.bat baje Python+Node+deps).
;   - 5 GB de espacio libre (mayoria es PyTorch + cuda libs).

#define MyAppName "Image AP Sport for Laser"
#define MyAppVersion "1.2.0"
#define MyAppPublisher "Vahlame y colaboradores GPL-3.0"
#define MyAppURL "https://github.com/Vahlame/image-ap-sport-for-laser"
#define MyAppExeName "Iniciar_Laser_App.bat"
#define MyAppSetupExeName "Setup_LaserApp.bat"

[Setup]
AppId={{C8F8B8A4-1E2D-4F3B-9A1A-8F7D6C5B4E2A}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}
DefaultDirName={autopf}\ImageAPLaser
DefaultGroupName=Image AP Laser
DisableProgramGroupPage=yes
LicenseFile=..\LICENSE
OutputBaseFilename=ImageAPLaser_Setup_v{#MyAppVersion}
OutputDir=Output
SetupIconFile=
Compression=lzma2/ultra
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=admin
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
UninstallDisplayName={#MyAppName} v{#MyAppVersion}

[Languages]
Name: "spanish"; MessagesFile: "compiler:Languages\Spanish.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Crear acceso directo en el Escritorio"; GroupDescription: "Accesos directos:"; Flags: checkedonce
Name: "runsetup"; Description: "Ejecutar Setup completo al terminar (instala Python, Node y dependencias)"; GroupDescription: "Configuracion inicial:"; Flags: checkedonce

[Files]
; Toda la fuente del proyecto excepto venv, node_modules, runs, claude internals.
Source: "..\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs; \
    Excludes: ".venv312\*,.venv\*,web\node_modules\*,web\build\*,web\.svelte-kit\*,runs\_*,runs\references\foto*,runs\references\imagr_*,.claude\*,.git\*,*.pyc,__pycache__\*,.pytest_cache\*,reports\*,installer\Output\*"

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"; IconFilename: "{sys}\imageres.dll"; IconIndex: 108
Name: "{group}\Setup (re-instalar dependencias)"; Filename: "{app}\{#MyAppSetupExeName}"; WorkingDir: "{app}"
Name: "{group}\Carpeta de instalacion"; Filename: "{app}"
Name: "{group}\Desinstalar {#MyAppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"; IconFilename: "{sys}\imageres.dll"; IconIndex: 108; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppSetupExeName}"; WorkingDir: "{app}"; Description: "Configurar Python y dependencias ahora"; Flags: postinstall shellexec runascurrentuser; Tasks: runsetup

[UninstallDelete]
Type: filesandordirs; Name: "{app}\.venv312"
Type: filesandordirs; Name: "{app}\web\node_modules"
Type: filesandordirs; Name: "{app}\web\build"
Type: filesandordirs; Name: "{app}\web\.svelte-kit"

[Code]
function IsPythonInstalled: Boolean;
var
  ResultCode: Integer;
begin
  Result := Exec('cmd.exe', '/C python --version', '', SW_HIDE, ewWaitUntilTerminated, ResultCode) and (ResultCode = 0);
end;

function InitializeSetup: Boolean;
begin
  Result := True;
  if not IsPythonInstalled then begin
    if MsgBox(
      'Image AP Laser requiere Python 3.11+ y Node.js LTS.' #13#10 #13#10
      'Setup_LaserApp.bat los instalara automaticamente via winget tras la instalacion,' #13#10
      'pero necesitas conexion a Internet.' #13#10 #13#10
      'Continuar instalacion?',
      mbConfirmation, MB_YESNO) = IDNO then
      Result := False;
  end;
end;
