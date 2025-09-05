@echo off
title Detener AI Symbiote
color 0C

echo.
echo     ===================================================
echo            DETENIENDO AI SYMBIOTE
echo     ===================================================
echo.

:: Eliminar archivo de estado
del "%~dp0.symbiote_status" 2>nul

:: Cerrar procesos Python
echo     [*] Deteniendo servicios Python...
taskkill /F /IM python.exe 2>nul
taskkill /F /IM pythonw.exe 2>nul

:: Cerrar procesos Node
echo     [*] Deteniendo servicios Node.js...
taskkill /F /IM node.exe 2>nul

:: Cerrar ventanas de comando relacionadas
echo     [*] Cerrando ventanas de servicio...
taskkill /FI "WINDOWTITLE eq AI Symbiote*" 2>nul

echo.
echo     ✅ Todos los servicios detenidos
echo.
pause