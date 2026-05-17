# Code Signing — Eliminar el warning de Windows SmartScreen

## El problema

Cuando un usuario descarga `ImageAPLaser_Setup_v*.exe` desde GitHub Releases, Windows
SmartScreen muestra:

> "Windows protegió su PC"
> "SmartScreen de Microsoft Defender evitó que se iniciara una aplicación desconocida"
> "Editor: Editor desconocido"

Esto pasa porque el `.exe` **no está firmado con un certificado Authenticode** emitido
por una Autoridad de Certificación (CA) confiable, y aún no acumuló "reputación"
(SmartScreen necesita ~10,000 descargas con análisis limpio).

## Por qué no es trivial firmar

Las opciones de firma para Windows en orden de costo/efectividad:

| Opción | Costo | Setup | Resultado |
|---|---|---|---|
| **Self-signed cert** | $0 | 15 min | Sigue mostrando "Editor desconocido" (cert no confiable). NO sirve. |
| **SignPath.io OSS** | $0 | 1-2 días aprobación + setup CI | Firma real con CA confiable de SignPath Foundation. **Ideal para este proyecto.** |
| **Sectigo OV** | ~$200/año | 3-5 días verificación | Firma real, requiere ~10k downloads para SmartScreen reputation |
| **Sectigo EV** | ~$400-700/año | 1-2 semanas (HSM físico) | Firma real, **SmartScreen aprueba desde la primera descarga** |

## Solución recomendada: SignPath.io (gratis para OSS)

[SignPath Foundation](https://signpath.org/) ofrece firma de código **gratis** para
proyectos open-source que cumplen estos requisitos:

- ✅ Licencia OSI-approved (tenemos **GPL-3.0** ✓)
- ✅ Sin dual-licensing comercial (✓)
- ✅ Sin componentes propietarios (✓)
- ✅ Proyecto activamente mantenido con releases (✓ — 9 releases)
- ✅ Builds reproducibles desde el source

### Pasos para aplicar

1. **Crear cuenta** en https://about.signpath.io/signup
2. **Aplicar al programa OSS** en https://signpath.io/solutions/open-source-community
3. **Completar el form** con:
   - Repo URL: `https://github.com/Vahlame/image-ap-sport-for-laser`
   - License: GPL-3.0-or-later
   - Project description: "Image processing for CO2 laser engraving (acrylic back-engrave)"
   - Maintainer: Jorge David Hidalgo Oporta (Vahlame)
4. **Esperar aprobación** (~5 días hábiles, validan que el proyecto es legítimo OSS)
5. **Setup del CI**: agregar workflow GitHub Actions que llama al API de SignPath para
   firmar el `.exe` generado por Inno Setup tras cada release.

### Workflow GitHub Actions de ejemplo

`.github/workflows/release.yml`:

```yaml
name: Release Signed Installer

on:
  push:
    tags: ['v*']

jobs:
  build-and-sign:
    runs-on: windows-latest
    steps:
      - uses: actions/checkout@v4
      - name: Compile Inno Setup
        run: |
          choco install innosetup
          ISCC.exe installer/LaserApp.iss
      - name: Submit to SignPath for signing
        uses: signpath/github-action-submit-signing-request@v1.2
        with:
          api-token: ${{ secrets.SIGNPATH_API_TOKEN }}
          organization-id: '<your-org-id>'
          project-slug: 'image-ap-sport-for-laser'
          signing-policy-slug: 'release-signing'
          artifact-configuration-slug: 'installer-exe'
          github-artifact-id: '${{ steps.upload-unsigned.outputs.artifact-id }}'
          wait-for-completion: true
          output-artifact-directory: 'installer/Output/signed'
      - name: Upload to GitHub Release
        uses: softprops/action-gh-release@v2
        with:
          files: installer/Output/signed/*.exe
```

Una vez configurado, cada `git tag vX.Y.Z && git push --tags` dispara la firma
automática y publica el `.exe` firmado en la release.

## Solución alternativa: Self-signed (NO recomendado pero útil para testing)

Si querés probar el flujo de firma sin gastar tiempo en SignPath:

```powershell
# 1. Generar certificado autofirmado (válido 5 años)
$cert = New-SelfSignedCertificate `
  -Type CodeSigningCert `
  -Subject "CN=Vahlame ImageAPLaser" `
  -KeyAlgorithm RSA -KeyLength 2048 `
  -NotAfter (Get-Date).AddYears(5) `
  -CertStoreLocation "Cert:\CurrentUser\My"

# 2. Exportar como .pfx (con password)
$pwd = ConvertTo-SecureString -String "MyPassword" -Force -AsPlainText
Export-PfxCertificate -Cert $cert -FilePath "imageaplaser.pfx" -Password $pwd

# 3. Firmar el .exe
& "C:\Program Files (x86)\Windows Kits\10\bin\10.0.22621.0\x64\signtool.exe" sign `
  /f imageaplaser.pfx /p MyPassword `
  /tr http://timestamp.digicert.com /td sha256 /fd sha256 `
  installer\Output\ImageAPLaser_Setup_v2.0.0.exe
```

**Limitación**: el usuario verá "Editor: Vahlame ImageAPLaser" en vez de "desconocido",
pero SmartScreen sigue bloqueando porque el certificado no viene de una CA confiable.

## Alternativa #2: Distribuir via Winget

Microsoft Winget tiene su propio repositorio de paquetes verificados. Una vez aceptado:

```powershell
winget install Vahlame.ImageAPLaser
```

Y no hay warning de SmartScreen (winget firma internamente sus paquetes).

Pasos para publicar en winget:

1. Crear PR a https://github.com/microsoft/winget-pkgs
2. Manifiestos requeridos: `manifests/v/Vahlame/ImageAPLaser/2.0.0/*.yaml`
3. SHA256 del `.exe` apuntando a GitHub Release URL
4. CI de Microsoft valida → mergea → disponible vía `winget install`

Esto es lo MÁS práctico a largo plazo: una vez aceptado, cada nueva versión solo
requiere un manifiesto actualizado.

## Estado actual del proyecto

- ✅ ZIP portable como alternativa **sin warning** (ver release v2.0.0+)
- ✅ Documentación clara en README sobre cómo bypassar SmartScreen
- ⏳ **PENDIENTE**: aplicar a SignPath.io OSS (acción manual del maintainer)
- ⏳ **PENDIENTE**: publicar manifiesto en winget-pkgs

Una vez que SignPath apruebe el proyecto, todos los `.exe` futuros (v2.1+) saldrán
firmados y sin warning desde el primer download.

## Referencias

- [SignPath Foundation OSS program](https://signpath.io/solutions/open-source-community)
- [SignPath terms for OSS projects](https://signpath.org/terms.html)
- [Windows Authenticode Knowledge Base](https://signpath.io/knowledge-base/windows-platform)
- [Microsoft SmartScreen behavior](https://learn.microsoft.com/en-us/windows/security/threat-protection/microsoft-defender-smartscreen/)
- [Winget package submission guide](https://learn.microsoft.com/en-us/windows/package-manager/package/)
