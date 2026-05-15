@echo off
chcp 65001 >nul
echo Deteniendo procesos en puertos 18765 (API) y 5173 (web)...

powershell -NoProfile -Command "$ports = 18765,5173; foreach ($p in $ports) { Get-NetTCPConnection -LocalPort $p -State Listen -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue } }"

echo Listo. Si quedan ventanas cmd abiertas, cierralas manualmente.
ping 127.0.0.1 -n 4 >nul 2>&1
