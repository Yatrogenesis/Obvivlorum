#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Advanced Structured Logger for AI Symbiote
==========================================

Sistema de logging estructurado avanzado que elimina duplicación,
proporciona métricas en tiempo real y soporta múltiples formatos.

Author: Francisco Molina
ORCID: https://orcid.org/0009-0008-6093-8267
Email: pako.molina@gmail.com
"""

import os
import json
import logging
import logging.handlers
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
from collections import defaultdict, deque
from enum import Enum
import uuid
import gzip
import asyncio

class LogLevel(Enum):
    """Niveles de log extendidos"""
    TRACE = 5
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50
    SECURITY = 60
    AUDIT = 70

class LogFormat(Enum):
    """Formatos de salida"""
    JSON = "json"
    STRUCTURED_TEXT = "structured_text"
    COMPACT = "compact"
    CONSOLE = "console"

class LogContext:
    """Contexto de logging con thread-local storage"""
    
    def __init__(self):
        self._local = threading.local()
    
    def set(self, key: str, value: Any):
        """Establecer valor en contexto"""
        if not hasattr(self._local, 'context'):
            self._local.context = {}
        self._local.context[key] = value
    
    def get(self, key: str, default=None):
        """Obtener valor del contexto"""
        if not hasattr(self._local, 'context'):
            return default
        return self._local.context.get(key, default)
    
    def get_all(self) -> Dict[str, Any]:
        """Obtener todo el contexto"""
        if not hasattr(self._local, 'context'):
            return {}
        return self._local.context.copy()
    
    def clear(self):
        """Limpiar contexto"""
        if hasattr(self._local, 'context'):
            self._local.context.clear()

class MetricsCollector:
    """Recolector de métricas de logging"""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.log_counts = defaultdict(int)
        self.error_counts = defaultdict(int)
        self.recent_logs = deque(maxlen=window_size)
        self.performance_metrics = deque(maxlen=window_size)
        self.start_time = datetime.now()
        self._lock = threading.Lock()
    
    def record_log(self, level: str, logger_name: str, message: str, timestamp: datetime):
        """Registrar un log para métricas"""
        with self._lock:
            self.log_counts[level] += 1
            
            if level in ['ERROR', 'CRITICAL']:
                self.error_counts[logger_name] += 1
            
            self.recent_logs.append({
                'timestamp': timestamp.isoformat(),
                'level': level,
                'logger': logger_name,
                'message': message[:100]  # Truncar para ahorrar memoria
            })
    
    def record_performance(self, operation: str, duration: float, metadata: Dict[str, Any] = None):
        """Registrar métrica de rendimiento"""
        with self._lock:
            self.performance_metrics.append({
                'timestamp': datetime.now().isoformat(),
                'operation': operation,
                'duration': duration,
                'metadata': metadata or {}
            })
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de logging"""
        with self._lock:
            uptime = datetime.now() - self.start_time
            
            return {
                'uptime_seconds': uptime.total_seconds(),
                'total_logs': sum(self.log_counts.values()),
                'logs_by_level': dict(self.log_counts),
                'error_counts_by_logger': dict(self.error_counts),
                'recent_error_rate': self._calculate_recent_error_rate(),
                'avg_performance_by_operation': self._calculate_avg_performance(),
                'logs_per_minute': self._calculate_logs_per_minute()
            }
    
    def _calculate_recent_error_rate(self) -> float:
        """Calcular tasa de errores reciente"""
        if not self.recent_logs:
            return 0.0
        
        recent_errors = sum(1 for log in self.recent_logs 
                          if log['level'] in ['ERROR', 'CRITICAL'])
        return recent_errors / len(self.recent_logs)
    
    def _calculate_avg_performance(self) -> Dict[str, float]:
        """Calcular rendimiento promedio por operación"""
        operation_times = defaultdict(list)
        
        for metric in self.performance_metrics:
            operation_times[metric['operation']].append(metric['duration'])
        
        return {
            op: sum(times) / len(times)
            for op, times in operation_times.items()
            if times
        }
    
    def _calculate_logs_per_minute(self) -> float:
        """Calcular logs por minuto"""
        if not self.recent_logs or len(self.recent_logs) < 2:
            return 0.0
        
        oldest = datetime.fromisoformat(self.recent_logs[0]['timestamp'])
        newest = datetime.fromisoformat(self.recent_logs[-1]['timestamp'])
        duration = (newest - oldest).total_seconds()
        
        if duration == 0:
            return 0.0
        
        return (len(self.recent_logs) / duration) * 60

class StructuredFormatter(logging.Formatter):
    """Formateador estructurado para logs"""
    
    def __init__(self, format_type: LogFormat, include_context: bool = True):
        super().__init__()
        self.format_type = format_type
        self.include_context = include_context
        self.hostname = os.uname().nodename if hasattr(os, 'uname') else 'unknown'
    
    def format(self, record: logging.LogRecord) -> str:
        """Formatear registro de log"""
        # Crear estructura base
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'thread': record.thread,
            'thread_name': getattr(record, 'threadName', 'MainThread'),
            'process': record.process,
            'hostname': self.hostname
        }
        
        # Añadir contexto si está disponible
        if self.include_context and hasattr(record, 'context'):
            log_entry['context'] = record.context
        
        # Añadir información de excepción si existe
        if record.exc_info:
            log_entry['exception'] = {
                'type': record.exc_info[0].__name__ if record.exc_info[0] else None,
                'message': str(record.exc_info[1]) if record.exc_info[1] else None,
                'traceback': self.formatException(record.exc_info)
            }
        
        # Añadir campos extra
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                          'filename', 'module', 'lineno', 'funcName', 'created',
                          'msecs', 'relativeCreated', 'thread', 'threadName',
                          'processName', 'process', 'getMessage', 'exc_info',
                          'exc_text', 'stack_info']:
                log_entry['extra'] = log_entry.get('extra', {})
                log_entry['extra'][key] = value
        
        # Formatear según el tipo solicitado
        if self.format_type == LogFormat.JSON:
            return json.dumps(log_entry, ensure_ascii=False)
        elif self.format_type == LogFormat.COMPACT:
            return f"{log_entry['timestamp']} [{log_entry['level']}] {log_entry['logger']}: {log_entry['message']}"
        elif self.format_type == LogFormat.CONSOLE:
            color_codes = {
                'DEBUG': '\033[36m',    # Cyan
                'INFO': '\033[32m',     # Green
                'WARNING': '\033[33m',  # Yellow
                'ERROR': '\033[31m',    # Red
                'CRITICAL': '\033[41m', # Red background
                'RESET': '\033[0m'      # Reset
            }
            color = color_codes.get(log_entry['level'], '')
            reset = color_codes['RESET']
            return f"{color}[{log_entry['level']}]{reset} {log_entry['logger']}: {log_entry['message']}"
        else:  # STRUCTURED_TEXT
            context_str = f" | Context: {json.dumps(log_entry.get('context', {}))}" if log_entry.get('context') else ""
            return (f"{log_entry['timestamp']} [{log_entry['level']}] "
                   f"{log_entry['logger']}:{log_entry['line']} "
                   f"({log_entry['function']}) - {log_entry['message']}"
                   f"{context_str}")

class AsyncLogHandler(logging.Handler):
    """Handler asíncrono para logging no bloqueante"""
    
    def __init__(self, target_handler: logging.Handler, queue_size: int = 10000):
        super().__init__()
        self.target_handler = target_handler
        self.queue = asyncio.Queue(maxsize=queue_size)
        self.task = None
        self.loop = None
        self._shutdown = False
    
    def emit(self, record: logging.LogRecord):
        """Emitir record de forma asíncrona"""
        if self.loop and not self._shutdown:
            try:
                self.loop.call_soon_threadsafe(
                    lambda: asyncio.create_task(self._async_emit(record))
                )
            except RuntimeError:
                # Fallback a emisión síncrona
                self.target_handler.emit(record)
        else:
            self.target_handler.emit(record)
    
    async def _async_emit(self, record: logging.LogRecord):
        """Emisión asíncrona del record"""
        try:
            await self.queue.put(record)
        except asyncio.QueueFull:
            # Si la cola está llena, emitir síncronamente
            self.target_handler.emit(record)
    
    def start_async_processing(self, loop: asyncio.AbstractEventLoop):
        """Iniciar procesamiento asíncrono"""
        self.loop = loop
        self.task = loop.create_task(self._process_queue())
    
    async def _process_queue(self):
        """Procesar cola de logs"""
        while not self._shutdown:
            try:
                record = await asyncio.wait_for(self.queue.get(), timeout=1.0)
                self.target_handler.emit(record)
                self.queue.task_done()
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                # Log error sin usar el logger para evitar recursión
                print(f"AsyncLogHandler error: {e}")
    
    def close(self):
        """Cerrar handler"""
        self._shutdown = True
        if self.task:
            self.task.cancel()
        self.target_handler.close()
        super().close()

class StructuredLogger:
    """Logger estructurado principal"""
    
    def __init__(self, name: str = "obvivlorum", config: Dict[str, Any] = None):
        """Initialize Structured Logger"""
        self.name = name
        self.config = config or self._default_config()
        self.context = LogContext()
        self.metrics = MetricsCollector()
        
        # Evitar duplicación de handlers
        self._initialized_loggers = set()
        
        # Configurar logger principal
        self.logger = self._setup_logger()
        
        # Configurar compresión automática
        self._setup_log_rotation()
        
        # Configurar limpieza automática
        self._setup_cleanup()
    
    def _default_config(self) -> Dict[str, Any]:
        """Configuración por defecto"""
        return {
            'level': LogLevel.INFO.value,
            'log_dir': Path('logs'),
            'console_output': True,
            'file_output': True,
            'json_output': True,
            'max_file_size': 10 * 1024 * 1024,  # 10MB
            'backup_count': 5,
            'compress_backups': True,
            'cleanup_days': 30,
            'async_logging': True,
            'include_context': True,
            'deduplicate': True,
            'metrics_enabled': True
        }
    
    def _setup_logger(self) -> logging.Logger:
        """Configurar logger principal"""
        # Prevenir duplicación
        if self.name in self._initialized_loggers:
            return logging.getLogger(self.name)
        
        # Crear directorio de logs
        log_dir = Path(self.config['log_dir'])
        log_dir.mkdir(exist_ok=True)
        
        # Configurar logger raíz
        logger = logging.getLogger(self.name)
        logger.setLevel(self.config['level'])
        
        # Limpiar handlers existentes
        logger.handlers.clear()
        
        # Handler para consola
        if self.config['console_output']:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(self.config['level'])
            console_formatter = StructuredFormatter(LogFormat.CONSOLE, self.config['include_context'])
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)
        
        # Handler para archivo de texto
        if self.config['file_output']:
            file_handler = logging.handlers.RotatingFileHandler(
                filename=log_dir / f"{self.name}.log",
                maxBytes=self.config['max_file_size'],
                backupCount=self.config['backup_count']
            )
            file_handler.setLevel(self.config['level'])
            file_formatter = StructuredFormatter(LogFormat.STRUCTURED_TEXT, self.config['include_context'])
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
        
        # Handler para JSON
        if self.config['json_output']:
            json_handler = logging.handlers.RotatingFileHandler(
                filename=log_dir / f"{self.name}.json",
                maxBytes=self.config['max_file_size'],
                backupCount=self.config['backup_count']
            )
            json_handler.setLevel(self.config['level'])
            json_formatter = StructuredFormatter(LogFormat.JSON, self.config['include_context'])
            json_handler.setFormatter(json_formatter)
            logger.addHandler(json_handler)
        
        # Handler para security/audit
        security_handler = logging.handlers.RotatingFileHandler(
            filename=log_dir / f"{self.name}_security.log",
            maxBytes=self.config['max_file_size'],
            backupCount=self.config['backup_count']
        )
        security_handler.setLevel(LogLevel.SECURITY.value)
        security_formatter = StructuredFormatter(LogFormat.JSON, True)
        security_handler.setFormatter(security_formatter)
        logger.addHandler(security_handler)
        
        # Prevenir propagación para evitar duplicación
        logger.propagate = False
        
        # Marcar como inicializado
        self._initialized_loggers.add(self.name)
        
        return logger
    
    def _setup_log_rotation(self):
        """Configurar rotación y compresión de logs"""
        if not self.config['compress_backups']:
            return
        
        def compress_backup(source_file):
            """Comprimir archivo de backup"""
            try:
                if source_file.endswith('.log') or source_file.endswith('.json'):
                    with open(source_file, 'rb') as f_in:
                        with gzip.open(f"{source_file}.gz", 'wb') as f_out:
                            f_out.writelines(f_in)
                    os.remove(source_file)
            except Exception as e:
                print(f"Error compressing {source_file}: {e}")
        
        # Comprimir archivos existentes
        log_dir = Path(self.config['log_dir'])
        for backup_file in log_dir.glob("*.log.*"):
            if not backup_file.name.endswith('.gz'):
                compress_backup(str(backup_file))
    
    def _setup_cleanup(self):
        """Configurar limpieza automática de logs antiguos"""
        def cleanup_old_logs():
            """Limpiar logs antiguos"""
            log_dir = Path(self.config['log_dir'])
            cutoff_date = datetime.now() - timedelta(days=self.config['cleanup_days'])
            
            for log_file in log_dir.glob("*.gz"):
                try:
                    if datetime.fromtimestamp(log_file.stat().st_mtime) < cutoff_date:
                        log_file.unlink()
                        self.info(f"Cleaned up old log file: {log_file}")
                except Exception as e:
                    self.error(f"Error cleaning up {log_file}: {e}")
        
        # Ejecutar limpieza en background
        threading.Timer(3600, cleanup_old_logs).start()  # Cada hora
    
    def set_context(self, **kwargs):
        """Establecer contexto para logs posteriores"""
        for key, value in kwargs.items():
            self.context.set(key, value)
    
    def clear_context(self):
        """Limpiar contexto"""
        self.context.clear()
    
    def _log_with_context(self, level: int, message: str, **kwargs):
        """Log con contexto y métricas"""
        # Obtener contexto actual
        context = self.context.get_all()
        context.update(kwargs)
        
        # Crear record con contexto
        record = self.logger.makeRecord(
            name=self.logger.name,
            level=level,
            fn='',
            lno=0,
            msg=message,
            args=(),
            exc_info=None
        )
        record.context = context
        
        # Registrar métricas
        if self.config['metrics_enabled']:
            self.metrics.record_log(
                level=logging.getLevelName(level),
                logger_name=self.logger.name,
                message=message,
                timestamp=datetime.now()
            )
        
        # Emitir log
        self.logger.handle(record)
    
    def trace(self, message: str, **kwargs):
        """Log TRACE level"""
        self._log_with_context(LogLevel.TRACE.value, message, **kwargs)
    
    def debug(self, message: str, **kwargs):
        """Log DEBUG level"""
        self._log_with_context(LogLevel.DEBUG.value, message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log INFO level"""
        self._log_with_context(LogLevel.INFO.value, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log WARNING level"""
        self._log_with_context(LogLevel.WARNING.value, message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log ERROR level"""
        self._log_with_context(LogLevel.ERROR.value, message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log CRITICAL level"""
        self._log_with_context(LogLevel.CRITICAL.value, message, **kwargs)
    
    def security(self, message: str, **kwargs):
        """Log SECURITY level"""
        kwargs['security_event'] = True
        self._log_with_context(LogLevel.SECURITY.value, message, **kwargs)
    
    def audit(self, message: str, **kwargs):
        """Log AUDIT level"""
        kwargs['audit_event'] = True
        self._log_with_context(LogLevel.AUDIT.value, message, **kwargs)
    
    def performance(self, operation: str, duration: float, **metadata):
        """Log performance metric"""
        self.info(f"Performance: {operation} took {duration:.3f}s", 
                 performance=True, operation=operation, duration=duration, **metadata)
        
        if self.config['metrics_enabled']:
            self.metrics.record_performance(operation, duration, metadata)
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de logging"""
        return self.metrics.get_stats()
    
    def export_logs(self, output_file: str, format_type: LogFormat = LogFormat.JSON,
                   start_time: datetime = None, end_time: datetime = None):
        """Exportar logs a archivo"""
        log_dir = Path(self.config['log_dir'])
        
        with open(output_file, 'w', encoding='utf-8') as outfile:
            # Exportar desde archivos JSON si existe
            json_file = log_dir / f"{self.name}.json"
            if json_file.exists():
                with open(json_file, 'r', encoding='utf-8') as infile:
                    for line in infile:
                        try:
                            log_entry = json.loads(line.strip())
                            log_time = datetime.fromisoformat(log_entry['timestamp'])
                            
                            # Filtrar por tiempo si se especifica
                            if start_time and log_time < start_time:
                                continue
                            if end_time and log_time > end_time:
                                continue
                            
                            # Escribir según el formato solicitado
                            if format_type == LogFormat.JSON:
                                outfile.write(line)
                            elif format_type == LogFormat.COMPACT:
                                compact = f"{log_entry['timestamp']} [{log_entry['level']}] {log_entry['message']}\n"
                                outfile.write(compact)
                            else:
                                structured = f"{log_entry['timestamp']} [{log_entry['level']}] {log_entry['logger']}: {log_entry['message']}\n"
                                outfile.write(structured)
                                
                        except (json.JSONDecodeError, KeyError):
                            continue

# Instancia global del logger
_global_logger: Optional[StructuredLogger] = None

def get_logger(name: str = None) -> StructuredLogger:
    """Obtener instancia del logger estructurado"""
    global _global_logger
    
    if _global_logger is None or (name and name != _global_logger.name):
        _global_logger = StructuredLogger(name or "obvivlorum")
    
    return _global_logger

# Funciones de conveniencia
def info(message: str, **kwargs):
    """Log INFO level"""
    get_logger().info(message, **kwargs)

def debug(message: str, **kwargs):
    """Log DEBUG level"""
    get_logger().debug(message, **kwargs)

def warning(message: str, **kwargs):
    """Log WARNING level"""
    get_logger().warning(message, **kwargs)

def error(message: str, **kwargs):
    """Log ERROR level"""
    get_logger().error(message, **kwargs)

def critical(message: str, **kwargs):
    """Log CRITICAL level"""
    get_logger().critical(message, **kwargs)

def security(message: str, **kwargs):
    """Log SECURITY level"""
    get_logger().security(message, **kwargs)

def audit(message: str, **kwargs):
    """Log AUDIT level"""
    get_logger().audit(message, **kwargs)

def set_context(**kwargs):
    """Set logging context"""
    get_logger().set_context(**kwargs)

def clear_context():
    """Clear logging context"""
    get_logger().clear_context()


# Ejemplo de uso
if __name__ == "__main__":
    # Crear logger
    logger = StructuredLogger("test_logger")
    
    print("=== STRUCTURED LOGGER TEST ===\n")
    
    # Establecer contexto
    logger.set_context(user="test_user", session="12345", component="test")
    
    # Diferentes tipos de logs
    logger.info("System starting up", version="2.0")
    logger.debug("Debug information", debug_data={"key": "value"})
    logger.warning("This is a warning", warning_code="W001")
    logger.error("An error occurred", error_code="E001", recoverable=True)
    logger.security("Security event detected", event_type="unauthorized_access", ip="192.168.1.100")
    logger.audit("User action performed", action="login", user="admin")
    
    # Métrica de rendimiento
    import time
    start_time = time.time()
    time.sleep(0.1)  # Simular trabajo
    logger.performance("test_operation", time.time() - start_time, iterations=10)
    
    # Cambiar contexto
    logger.set_context(component="ai_engine")
    logger.info("AI engine initialized", model="gpt-4")
    
    # Mostrar estadísticas
    print("\n=== LOGGER STATISTICS ===")
    stats = logger.get_stats()
    print(json.dumps(stats, indent=2))
    
    # Exportar logs
    logger.export_logs("test_export.json", LogFormat.JSON)
    print("\nLogs exported to test_export.json")
    
    print("\nCheck the 'logs' directory for output files!")