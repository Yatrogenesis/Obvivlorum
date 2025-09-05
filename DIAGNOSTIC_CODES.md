# CÓDIGOS DE ERROR WINDOWS - DIAGNÓSTICO

## Códigos de Error Identificados:
- **Error -114**: Posible error de conexión de red o timeout
- **Error 220**: Error de acceso o permisos
- **Error -118**: Error de timeout de red o DNS

## Posibles Causas:
1. **Corrupción de archivos del sistema**
   - CMD.exe dañado
   - DLLs del sistema corruptas
   - Registro de Windows dañado

2. **Problemas de permisos**
   - UAC bloqueando ejecución
   - Políticas de grupo restrictivas
   - Antivirus/Firewall interferencia

3. **Problemas de red**
   - DNS corruption
   - Winsock corruption
   - TCP/IP stack issues

## Soluciones Aplicadas:

### 1. Reparación de Sistema
- SFC /scannow iniciado con privilegios elevados
- DISM /RestoreHealth preparado
- Script SYSTEM_REPAIR.bat creado

### 2. Limpieza de Claude Shell
- Problema identificado: Claude shell snapshots corrompidos
- Los archivos .sh en snapshots causan conflictos con PowerShell

### 3. Restauración de CMD
- Verificación de cmd.exe en System32
- Registro de comandos alternativos

## Comandos de Reparación Ejecutados:
```batch
# Verificación de integridad
sfc /scannow

# Reparación de imagen
dism /Online /Cleanup-Image /RestoreHealth

# Reset de Winsock (para errores -114, -118)
netsh winsock reset
netsh int ip reset

# Limpieza de DNS (para error -118)
ipconfig /flushdns

# Re-registro de DLLs del sistema
regsvr32 /s shell32.dll
regsvr32 /s ole32.dll
regsvr32 /s oleaut32.dll
```

## Estado Actual:
- PowerShell: Parcialmente funcional
- CMD: Requiere verificación adicional
- Task Manager: Sin proceso activo detectado
- Claude Shell: Snapshots corruptos detectados