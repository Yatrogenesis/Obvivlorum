@echo off
title QUICK FIX - AI Funcional
color 0A

echo Parando procesos anteriores...
taskkill /F /IM python.exe /T >nul 2>&1
timeout /t 2 /nobreak >nul

echo.
echo ===============================================
echo   REPARACION RAPIDA - AI INTELIGENTE
echo ===============================================
echo.

cd /d "D:\Obvivlorum\web\backend"

echo Creando servidor corregido...

echo import asyncio > fixed_server.py
echo from ai_simple_working import SimpleWorkingAI >> fixed_server.py
echo. >> fixed_server.py
echo ai = SimpleWorkingAI() >> fixed_server.py
echo. >> fixed_server.py
echo async def test_ai(): >> fixed_server.py
echo     print("=== PROBANDO AI INTELIGENTE ===") >> fixed_server.py
echo     questions = [ >> fixed_server.py
echo         "Que es la inteligencia artificial", >> fixed_server.py
echo         "Como funciona una red neuronal", >> fixed_server.py
echo         "Hola", >> fixed_server.py
echo         "Cuales son tus capacidades", >> fixed_server.py
echo         "Por que es importante la ciberseguridad" >> fixed_server.py
echo     ] >> fixed_server.py
echo. >> fixed_server.py
echo     for q in questions: >> fixed_server.py
echo         print(f"\nPregunta: {q}") >> fixed_server.py
echo         response = await ai.process_message(q) >> fixed_server.py
echo         print(f"Respuesta: {response[:200]}...") >> fixed_server.py
echo. >> fixed_server.py
echo if __name__ == "__main__": >> fixed_server.py
echo     asyncio.run(test_ai()) >> fixed_server.py

echo.
echo Probando AI directamente...
python fixed_server.py

echo.
echo ===============================================
echo   PRUEBA COMPLETADA
echo ===============================================
echo.
echo Si las respuestas se ven bien arriba, 
echo el AI esta funcionando correctamente.
echo.
echo El problema esta en la carga del servidor.
echo Necesitas reiniciar el servidor con el AI correcto.
echo.
pause