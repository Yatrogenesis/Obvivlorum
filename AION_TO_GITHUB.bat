@echo off
REM =====================================================
REM Script para subir el proyecto AION a GitHub
REM =====================================================

echo ================================================================================
echo                         SUBIR AION PROTOCOL A GITHUB                          
echo ================================================================================
echo.

cd /d D:\Obvivlorum\AION

echo Inicializando repositorio Git...
git init

echo Agregando archivos al repositorio...
git add .

echo Realizando commit inicial...
git commit -m "Initial commit - AION Protocol integration with Obvivlorum"

echo Agregando repositorio remoto...
git remote add origin https://github.com/Yatrogenesis/Obvivlorum.git

echo Subiendo al repositorio remoto...
git push -u origin master

echo.
echo ================================================================================
echo                      SUBIDA COMPLETADA EXITOSAMENTE!                           
echo ================================================================================
echo.
echo Repositorio: https://github.com/Yatrogenesis/Obvivlorum
echo.

pause
