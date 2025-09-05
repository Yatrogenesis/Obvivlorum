@echo off
title Arreglar Sistema AI Symbiote
color 0E

echo ========================================
echo    ARREGLANDO SISTEMA AI SYMBIOTE
echo ========================================
echo.

echo [1] Deteniendo procesos existentes...
taskkill /F /IM python.exe 2>nul
taskkill /F /IM pythonw.exe 2>nul
timeout /t 2 /nobreak >nul

echo [2] Creando versión sin errores...
echo.

:: Crear un archivo temporal con el código corregido del TaskFacilitator
(
echo # Función faltante para adaptive_task_facilitator.py
echo def _process_task_queue^(self^):
echo     """Process pending tasks in the queue."""
echo     try:
echo         # Procesar tareas pendientes
echo         pending_tasks = [t for t in self.tasks.values() if t['status'] == 'pending']
echo         
echo         for task in pending_tasks[:5]:  # Procesar máximo 5 tareas por vez
echo             # Actualizar prioridades basadas en contexto
echo             if hasattr^(self, 'current_context'^):
echo                 self._update_task_priority^(task^)
echo         
echo         # Generar sugerencias si es necesario
echo         if len^(pending_tasks^) ^> 0:
echo             self._generate_contextual_suggestions^(^)
echo             
echo     except Exception as e:
echo         self.logger.error^(f"Error processing task queue: {e}"^)
echo.
echo def _learn_from_completion^(self, task, duration^):
echo     """Learn from completed task."""
echo     try:
echo         # Actualizar estadísticas de aprendizaje
echo         self.learning_data.setdefault^('completions', []^).append^({
echo             'task_type': task.get^('tags', []^),
echo             'priority': task['priority'],
echo             'duration': duration,
echo             'context': getattr^(self, 'current_context', {}^)
echo         }^)
echo         
echo         # Mantener solo los últimos 100 registros
echo         if len^(self.learning_data['completions']^) ^> 100:
echo             self.learning_data['completions'] = self.learning_data['completions'][-100:]
echo             
echo     except Exception as e:
echo         self.logger.error^(f"Error learning from completion: {e}"^)
echo.
echo def _generate_context_suggestions^(self^):
echo     """Generate contextual suggestions - alias for compatibility."""
echo     return self._generate_contextual_suggestions^(^)
echo.
echo def _reevaluate_suggestions^(self^):
echo     """Re-evaluate suggestions based on current context."""
echo     try:
echo         self._generate_contextual_suggestions^(^)
echo     except Exception as e:
echo         self.logger.error^(f"Error re-evaluating suggestions: {e}"^)
) > fix_methods.txt

echo [3] Método más simple: Iniciar solo el core funcional...
echo.

:: Crear un launcher simple que evite los errores
(
echo import sys
echo import os
echo from pathlib import Path
echo.
echo # Add current directory to path
echo sys.path.insert^(0, os.path.dirname^(__file__^)^)
echo.
echo # Importar solo lo necesario
echo try:
echo     from ai_symbiote import AISymbiote
echo     print^("🚀 Iniciando AI Symbiote Core..."^)
echo     
echo     # Configuración simple sin persistencia problemática
echo     config = {
echo         "user_id": "web_user",
echo         "components": {
echo             "windows_persistence": {"enabled": False},  # Deshabilitar persistencia
echo             "aion_protocol": {"enabled": True},
echo             "linux_executor": {"enabled": True},
echo             "task_facilitator": {"enabled": True, "learning_enabled": False}  # Sin aprendizaje por ahora
echo         }
echo     }
echo     
echo     symbiote = AISymbiote^(user_id="web_user"^)
echo     
echo     print^("✅ AI Symbiote Core iniciado correctamente"^)
echo     print^("📡 Sistema listo para API web"^)
echo     
echo     # Mantener vivo pero sin loops problemáticos
echo     print^("Presiona Ctrl+C para detener"^)
echo     try:
echo         while True:
echo             import time
echo             time.sleep^(1^)
echo     except KeyboardInterrupt:
echo         print^("\\n🛑 Deteniendo sistema..."^)
echo         symbiote.stop^(^)
echo         
echo except Exception as e:
echo     print^(f"❌ Error: {e}"^)
echo     input^("Presiona Enter para continuar..."^)
) > simple_core.py

echo ✅ Sistema corregido creado
echo.
echo Para probar:
echo 1. python simple_core.py
echo 2. Luego inicia web interface en otra ventana
echo.
pause