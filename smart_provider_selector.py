#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Smart Provider Selector for Multi-Provider AI Engine
=====================================================

Sistema inteligente de seleccion de proveedores de IA basado en:
- Tipo de tarea detectada
- Metricas historicas de rendimiento
- Disponibilidad y costo
- Contexto de la consulta

Author: Francisco Molina
ORCID: https://orcid.org/0009-0008-6093-8267
Email: pako.molina@gmail.com
"""

import os
import json
import time
import logging
import statistics
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict, deque
import re
import tiktoken

logger = logging.getLogger("SmartProviderSelector")

class TaskType(Enum):
    """Tipos de tareas que puede realizar la IA"""
    CODING = "coding"
    CREATIVE = "creative"
    ANALYTICAL = "analytical"
    CONVERSATIONAL = "conversational"
    MATHEMATICAL = "mathematical"
    TRANSLATION = "translation"
    SUMMARIZATION = "summarization"
    SECURITY_ANALYSIS = "security_analysis"
    SYSTEM_COMMAND = "system_command"
    UNKNOWN = "unknown"

class ProviderType(Enum):
    """Tipos de proveedores disponibles"""
    LOCAL_GGUF = "local_gguf"
    OPENAI_GPT35 = "openai_gpt35"
    OPENAI_GPT4 = "openai_gpt4"
    CLAUDE_SONNET = "claude_sonnet"
    CLAUDE_HAIKU = "claude_haiku"
    FALLBACK = "fallback"

class ProviderMetrics:
    """Metricas de rendimiento de un proveedor"""
    
    def __init__(self, provider_type: ProviderType, window_size: int = 100):
        self.provider_type = provider_type
        self.response_times = deque(maxlen=window_size)
        self.success_count = 0
        self.failure_count = 0
        self.token_usage = deque(maxlen=window_size)
        self.quality_scores = deque(maxlen=window_size)
        self.task_performance = defaultdict(lambda: {"success": 0, "total": 0})
        self.last_failure_time = None
        self.consecutive_failures = 0
    
    def add_response(self, response_time: float, success: bool, 
                    tokens_used: int = 0, quality_score: float = None,
                    task_type: TaskType = None):
        """Registrar una respuesta del proveedor"""
        self.response_times.append(response_time)
        
        if success:
            self.success_count += 1
            self.consecutive_failures = 0
        else:
            self.failure_count += 1
            self.consecutive_failures += 1
            self.last_failure_time = datetime.now()
        
        if tokens_used > 0:
            self.token_usage.append(tokens_used)
        
        if quality_score is not None:
            self.quality_scores.append(quality_score)
        
        if task_type:
            self.task_performance[task_type]["total"] += 1
            if success:
                self.task_performance[task_type]["success"] += 1
    
    def get_availability_score(self) -> float:
        """Calcular score de disponibilidad (0-1)"""
        if self.success_count + self.failure_count == 0:
            return 0.5  # Sin datos, asumir disponibilidad media
        
        # Penalizar por fallos consecutivos
        base_availability = self.success_count / (self.success_count + self.failure_count)
        
        if self.consecutive_failures > 0:
            penalty = min(0.5, self.consecutive_failures * 0.1)
            base_availability *= (1 - penalty)
        
        # Penalizar si el ultimo fallo fue muy reciente
        if self.last_failure_time:
            time_since_failure = datetime.now() - self.last_failure_time
            if time_since_failure < timedelta(minutes=5):
                base_availability *= 0.7
        
        return base_availability
    
    def get_average_response_time(self) -> float:
        """Obtener tiempo de respuesta promedio"""
        if not self.response_times:
            return float('inf')
        return statistics.mean(self.response_times)
    
    def get_task_success_rate(self, task_type: TaskType) -> float:
        """Obtener tasa de exito para un tipo de tarea especifico"""
        if task_type not in self.task_performance:
            return 0.5  # Sin datos, asumir rendimiento medio
        
        perf = self.task_performance[task_type]
        if perf["total"] == 0:
            return 0.5
        
        return perf["success"] / perf["total"]
    
    def get_quality_score(self) -> float:
        """Obtener score de calidad promedio"""
        if not self.quality_scores:
            return 0.7  # Asumir calidad media por defecto
        return statistics.mean(self.quality_scores)

class SmartProviderSelector:
    """
    Selector inteligente de proveedores de IA
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize Smart Provider Selector"""
        self.config = config or self._default_config()
        self.metrics: Dict[ProviderType, ProviderMetrics] = {}
        self.task_patterns = self._init_task_patterns()
        self.provider_capabilities = self._init_provider_capabilities()
        self.cost_per_token = self._init_cost_structure()
        self.current_budget = self.config.get("monthly_budget", 100.0)
        self.budget_used = 0.0
        
        # Inicializar metricas para cada proveedor
        for provider in ProviderType:
            self.metrics[provider] = ProviderMetrics(provider)
        
        # Cargar metricas historicas si existen
        self._load_historical_metrics()
        
        logger.info("Smart Provider Selector initialized")
    
    def _default_config(self) -> Dict[str, Any]:
        """Configuracion por defecto"""
        return {
            "prefer_local": True,
            "max_response_time": 30.0,  # segundos
            "quality_threshold": 0.6,
            "availability_threshold": 0.7,
            "monthly_budget": 100.0,  # USD
            "fallback_chain": [
                ProviderType.LOCAL_GGUF,
                ProviderType.OPENAI_GPT35,
                ProviderType.CLAUDE_HAIKU
            ],
            "task_preferences": {
                TaskType.CODING: [ProviderType.OPENAI_GPT4, ProviderType.CLAUDE_SONNET],
                TaskType.CREATIVE: [ProviderType.CLAUDE_SONNET, ProviderType.OPENAI_GPT4],
                TaskType.ANALYTICAL: [ProviderType.OPENAI_GPT4, ProviderType.LOCAL_GGUF],
                TaskType.SECURITY_ANALYSIS: [ProviderType.LOCAL_GGUF, ProviderType.CLAUDE_SONNET],
                TaskType.SYSTEM_COMMAND: [ProviderType.LOCAL_GGUF]  # Preferir local para comandos
            }
        }
    
    def _init_task_patterns(self) -> Dict[TaskType, List[str]]:
        """Inicializar patrones para detectar tipos de tareas"""
        return {
            TaskType.CODING: [
                r"\b(python|java|javascript|code|function|class|debug|fix|implement|program)\b",
                r"\b(def |import |return |if |for |while |try |catch)\b",
                r"```[a-z]+\n",  # Code blocks
                r"\b(api|backend|frontend|database|sql)\b"
            ],
            TaskType.CREATIVE: [
                r"\b(story|poem|creative|imagine|design|art|music|write|compose)\b",
                r"\b(character|plot|scene|dialogue|narrative)\b",
                r"\b(beautiful|inspiring|unique|original)\b"
            ],
            TaskType.ANALYTICAL: [
                r"\b(analyze|compare|evaluate|research|study|investigate|examine)\b",
                r"\b(data|statistics|correlation|trend|pattern)\b",
                r"\b(pros?\s+and\s+cons?|advantages?|disadvantages?)\b"
            ],
            TaskType.MATHEMATICAL: [
                r"\b(calculate|compute|solve|equation|integral|derivative)\b",
                r"\b(math|algebra|geometry|calculus|statistics)\b",
                r"[0-9]+\s*[\+\-\*\/\^]\s*[0-9]+",  # Math operations
                r"\b(sum|product|factorial|logarithm|exponential)\b"
            ],
            TaskType.TRANSLATION: [
                r"\b(translate|translation|spanish|french|german|chinese|japanese)\b",
                r"\b(from\s+\w+\s+to\s+\w+)\b",
                r"\b(idioma|langue|sprache|??|??)\b"
            ],
            TaskType.SUMMARIZATION: [
                r"\b(summarize|summary|brief|concise|main\s+points|key\s+takeaways)\b",
                r"\b(tl;?dr|abstract|overview|outline)\b"
            ],
            TaskType.SECURITY_ANALYSIS: [
                r"\b(security|vulnerability|exploit|penetration|audit)\b",
                r"\b(nmap|sqlmap|metasploit|burp|wireshark)\b",
                r"\b(cve|cvss|owasp|threat|risk\s+assessment)\b",
                r"\b(firewall|ids|ips|siem|encryption)\b"
            ],
            TaskType.SYSTEM_COMMAND: [
                r"\b(bash|shell|terminal|command|execute|run)\b",
                r"\b(ls|cd|grep|awk|sed|chmod|sudo)\b",
                r"^\$\s+",  # Shell prompt
                r"\b(systemctl|service|docker|kubectl)\b"
            ],
            TaskType.CONVERSATIONAL: [
                r"\b(hello|hi|hey|how\s+are\s+you|chat|talk)\b",
                r"\b(opinion|think|feel|believe|suggest)\b",
                r"\?$",  # Questions
                r"\b(tell\s+me|explain|what\s+is|who\s+is)\b"
            ]
        }
    
    def _init_provider_capabilities(self) -> Dict[ProviderType, Dict[str, Any]]:
        """Inicializar capacidades de cada proveedor"""
        return {
            ProviderType.LOCAL_GGUF: {
                "max_tokens": 4096,
                "supports_streaming": True,
                "speed": "fast",
                "quality": "medium",
                "cost": "free",
                "best_for": [TaskType.SYSTEM_COMMAND, TaskType.SECURITY_ANALYSIS],
                "availability": 1.0  # Siempre disponible localmente
            },
            ProviderType.OPENAI_GPT35: {
                "max_tokens": 4096,
                "supports_streaming": True,
                "speed": "fast",
                "quality": "good",
                "cost": "low",
                "best_for": [TaskType.CONVERSATIONAL, TaskType.SUMMARIZATION],
                "availability": 0.95
            },
            ProviderType.OPENAI_GPT4: {
                "max_tokens": 8192,
                "supports_streaming": True,
                "speed": "medium",
                "quality": "excellent",
                "cost": "high",
                "best_for": [TaskType.CODING, TaskType.ANALYTICAL],
                "availability": 0.95
            },
            ProviderType.CLAUDE_SONNET: {
                "max_tokens": 100000,
                "supports_streaming": False,
                "speed": "medium",
                "quality": "excellent",
                "cost": "medium",
                "best_for": [TaskType.CREATIVE, TaskType.CODING],
                "availability": 0.98
            },
            ProviderType.CLAUDE_HAIKU: {
                "max_tokens": 100000,
                "supports_streaming": False,
                "speed": "fast",
                "quality": "good",
                "cost": "low",
                "best_for": [TaskType.TRANSLATION, TaskType.SUMMARIZATION],
                "availability": 0.98
            }
        }
    
    def _init_cost_structure(self) -> Dict[ProviderType, float]:
        """Inicializar estructura de costos (USD por 1K tokens)"""
        return {
            ProviderType.LOCAL_GGUF: 0.0,
            ProviderType.OPENAI_GPT35: 0.002,
            ProviderType.OPENAI_GPT4: 0.03,
            ProviderType.CLAUDE_SONNET: 0.015,
            ProviderType.CLAUDE_HAIKU: 0.001
        }
    
    def detect_task_type(self, query: str, context: Dict[str, Any] = None) -> TaskType:
        """Detectar el tipo de tarea basado en la consulta"""
        query_lower = query.lower()
        
        # Contar coincidencias para cada tipo de tarea
        task_scores = defaultdict(int)
        
        for task_type, patterns in self.task_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, query_lower, re.IGNORECASE)
                task_scores[task_type] += len(matches)
        
        # Si hay contexto, usarlo para mejorar la deteccion
        if context:
            if context.get("previous_task_type"):
                # Dar peso al tipo de tarea anterior (continuidad de conversacion)
                prev_type = context["previous_task_type"]
                if isinstance(prev_type, str):
                    try:
                        prev_type = TaskType(prev_type)
                        task_scores[prev_type] += 2
                    except ValueError:
                        pass
        
        # Determinar el tipo de tarea con mayor score
        if not task_scores:
            return TaskType.CONVERSATIONAL  # Por defecto
        
        detected_type = max(task_scores.items(), key=lambda x: x[1])[0]
        
        logger.info(f"Detected task type: {detected_type.value} (scores: {dict(task_scores)})")
        return detected_type
    
    def estimate_tokens(self, text: str) -> int:
        """Estimar numero de tokens en un texto"""
        # Usar tiktoken si esta disponible, sino estimacion simple
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
        except:
            # Estimacion simple: ~4 caracteres por token
            return len(text) // 4
    
    def calculate_cost(self, provider: ProviderType, tokens: int) -> float:
        """Calcular costo estimado para un proveedor"""
        return (tokens / 1000) * self.cost_per_token.get(provider, 0)
    
    def select_provider(self, query: str, context: Dict[str, Any] = None,
                       requirements: Dict[str, Any] = None) -> Tuple[ProviderType, Dict[str, Any]]:
        """
        Seleccionar el mejor proveedor para una consulta
        
        Returns:
            Tuple[ProviderType, metadata_dict]
        """
        # Detectar tipo de tarea
        task_type = self.detect_task_type(query, context)
        
        # Estimar tokens necesarios
        estimated_tokens = self.estimate_tokens(query)
        if context and context.get("expected_response_length"):
            estimated_tokens += context["expected_response_length"]
        
        # Obtener requirements
        requirements = requirements or {}
        max_latency = requirements.get("max_latency", self.config["max_response_time"])
        quality_required = requirements.get("quality", self.config["quality_threshold"])
        
        # Calcular scores para cada proveedor
        provider_scores = {}
        
        for provider in ProviderType:
            if provider == ProviderType.FALLBACK:
                continue
            
            score = self._calculate_provider_score(
                provider, task_type, estimated_tokens,
                max_latency, quality_required
            )
            provider_scores[provider] = score
        
        # Ordenar proveedores por score
        sorted_providers = sorted(
            provider_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Seleccionar el mejor proveedor disponible
        selected_provider = None
        selection_reason = ""
        
        for provider, score in sorted_providers:
            # Verificar disponibilidad
            availability = self.metrics[provider].get_availability_score()
            if availability < self.config["availability_threshold"]:
                logger.info(f"Skipping {provider.value} due to low availability: {availability:.2f}")
                continue
            
            # Verificar presupuesto
            estimated_cost = self.calculate_cost(provider, estimated_tokens)
            if self.budget_used + estimated_cost > self.current_budget:
                logger.info(f"Skipping {provider.value} due to budget constraints")
                continue
            
            # Proveedor seleccionado
            selected_provider = provider
            selection_reason = f"Best score: {score:.3f}"
            break
        
        # Si no hay proveedor disponible, usar fallback
        if not selected_provider:
            fallback_chain = self.config["fallback_chain"]
            for fallback in fallback_chain:
                if self.metrics[fallback].get_availability_score() > 0.3:
                    selected_provider = fallback
                    selection_reason = "Fallback due to no suitable provider"
                    break
        
        # Si aun no hay proveedor, usar LOCAL_GGUF como ultimo recurso
        if not selected_provider:
            selected_provider = ProviderType.LOCAL_GGUF
            selection_reason = "Last resort fallback"
        
        # Preparar metadata
        metadata = {
            "task_type": task_type.value,
            "estimated_tokens": estimated_tokens,
            "estimated_cost": self.calculate_cost(selected_provider, estimated_tokens),
            "selection_reason": selection_reason,
            "provider_score": provider_scores.get(selected_provider, 0),
            "alternatives": [
                {
                    "provider": p.value,
                    "score": s,
                    "availability": self.metrics[p].get_availability_score()
                }
                for p, s in sorted_providers[:3]
                if p != selected_provider
            ]
        }
        
        logger.info(f"Selected provider: {selected_provider.value} ({selection_reason})")
        
        return selected_provider, metadata
    
    def _calculate_provider_score(self, provider: ProviderType, task_type: TaskType,
                                 estimated_tokens: int, max_latency: float,
                                 quality_required: float) -> float:
        """Calcular score para un proveedor especifico"""
        score = 0.0
        weights = {
            "task_performance": 0.3,
            "quality": 0.25,
            "speed": 0.2,
            "cost": 0.15,
            "availability": 0.1
        }
        
        metrics = self.metrics[provider]
        capabilities = self.provider_capabilities[provider]
        
        # 1. Rendimiento en el tipo de tarea
        task_success_rate = metrics.get_task_success_rate(task_type)
        
        # Bonus si el proveedor es bueno para este tipo de tarea
        if task_type in capabilities.get("best_for", []):
            task_success_rate = min(1.0, task_success_rate + 0.2)
        
        score += task_success_rate * weights["task_performance"]
        
        # 2. Calidad
        quality_score = metrics.get_quality_score()
        
        # Mapear calidad de capabilities a score numerico
        quality_map = {"low": 0.4, "medium": 0.6, "good": 0.8, "excellent": 1.0}
        baseline_quality = quality_map.get(capabilities.get("quality", "medium"), 0.6)
        
        # Combinar metricas historicas con baseline
        combined_quality = (quality_score * 0.7 + baseline_quality * 0.3)
        
        if combined_quality >= quality_required:
            score += combined_quality * weights["quality"]
        else:
            # Penalizar si no cumple con la calidad requerida
            score -= 0.2
        
        # 3. Velocidad
        avg_response_time = metrics.get_average_response_time()
        
        if avg_response_time < max_latency:
            # Score inversamente proporcional al tiempo
            speed_score = max(0, 1 - (avg_response_time / max_latency))
            score += speed_score * weights["speed"]
        else:
            # Penalizar si es muy lento
            score -= 0.1
        
        # 4. Costo
        cost = self.calculate_cost(provider, estimated_tokens)
        
        # Score inversamente proporcional al costo
        if cost == 0:
            cost_score = 1.0  # Local es gratis
        else:
            # Normalizar costo (asumiendo max $1 por request)
            cost_score = max(0, 1 - min(cost, 1.0))
        
        score += cost_score * weights["cost"]
        
        # 5. Disponibilidad
        availability_score = metrics.get_availability_score()
        score += availability_score * weights["availability"]
        
        # Bonus/Penalizaciones adicionales
        
        # Preferir local si esta configurado
        if self.config.get("prefer_local") and provider == ProviderType.LOCAL_GGUF:
            score += 0.1
        
        # Penalizar si hay muchos fallos recientes
        if metrics.consecutive_failures > 3:
            score *= 0.5
        
        # Bonus por preferencia de tarea
        task_preferences = self.config.get("task_preferences", {})
        if task_type in task_preferences:
            preferred_providers = task_preferences[task_type]
            if provider in preferred_providers:
                # Bonus basado en posicion en la lista de preferencias
                position = preferred_providers.index(provider)
                score += (0.2 - position * 0.05)
        
        return min(1.0, max(0.0, score))  # Clamp entre 0 y 1
    
    def record_result(self, provider: ProviderType, success: bool,
                     response_time: float, tokens_used: int,
                     task_type: TaskType = None,
                     quality_score: float = None):
        """Registrar el resultado de usar un proveedor"""
        metrics = self.metrics[provider]
        metrics.add_response(
            response_time=response_time,
            success=success,
            tokens_used=tokens_used,
            quality_score=quality_score,
            task_type=task_type
        )
        
        # Actualizar presupuesto usado
        cost = self.calculate_cost(provider, tokens_used)
        self.budget_used += cost
        
        # Guardar metricas
        self._save_metrics()
        
        logger.info(f"Recorded result for {provider.value}: "
                   f"success={success}, time={response_time:.2f}s, "
                   f"tokens={tokens_used}, cost=${cost:.4f}")
    
    def _save_metrics(self):
        """Guardar metricas en archivo"""
        metrics_data = {}
        
        for provider, metrics in self.metrics.items():
            metrics_data[provider.value] = {
                "success_count": metrics.success_count,
                "failure_count": metrics.failure_count,
                "avg_response_time": metrics.get_average_response_time(),
                "availability": metrics.get_availability_score(),
                "quality_score": metrics.get_quality_score(),
                "task_performance": {
                    task.value: perf 
                    for task, perf in metrics.task_performance.items()
                }
            }
        
        metrics_file = Path("provider_metrics.json")
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "budget_used": self.budget_used,
                "metrics": metrics_data
            }, f, indent=2)
    
    def _load_historical_metrics(self):
        """Cargar metricas historicas si existen"""
        metrics_file = Path("provider_metrics.json")
        
        if not metrics_file.exists():
            return
        
        try:
            with open(metrics_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            self.budget_used = data.get("budget_used", 0)
            
            # Restaurar metricas basicas
            for provider_str, provider_data in data.get("metrics", {}).items():
                try:
                    provider = ProviderType(provider_str)
                    metrics = self.metrics[provider]
                    
                    metrics.success_count = provider_data.get("success_count", 0)
                    metrics.failure_count = provider_data.get("failure_count", 0)
                    
                    # Restaurar rendimiento por tarea
                    for task_str, perf in provider_data.get("task_performance", {}).items():
                        try:
                            task = TaskType(task_str)
                            metrics.task_performance[task] = perf
                        except ValueError:
                            pass
                            
                except ValueError:
                    pass
            
            logger.info("Loaded historical metrics")
            
        except Exception as e:
            logger.error(f"Error loading historical metrics: {e}")
    
    def get_provider_stats(self) -> Dict[str, Any]:
        """Obtener estadisticas de todos los proveedores"""
        stats = {
            "budget": {
                "total": self.current_budget,
                "used": self.budget_used,
                "remaining": self.current_budget - self.budget_used
            },
            "providers": {}
        }
        
        for provider, metrics in self.metrics.items():
            stats["providers"][provider.value] = {
                "availability": metrics.get_availability_score(),
                "avg_response_time": metrics.get_average_response_time(),
                "success_rate": metrics.success_count / max(1, metrics.success_count + metrics.failure_count),
                "quality_score": metrics.get_quality_score(),
                "consecutive_failures": metrics.consecutive_failures,
                "total_requests": metrics.success_count + metrics.failure_count
            }
        
        return stats


# Ejemplo de uso
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Crear selector
    selector = SmartProviderSelector()
    
    # Queries de prueba
    test_queries = [
        ("Write a Python function to calculate fibonacci", {"expected_response_length": 500}),
        ("Analyze the security vulnerabilities in this network", None),
        ("Tell me a creative story about a robot", None),
        ("Translate this to Spanish: Hello world", None),
        ("What is 2+2?", None),
        ("Execute nmap scan on localhost", None),
    ]
    
    print("=== SMART PROVIDER SELECTOR TEST ===\n")
    
    for query, context in test_queries:
        print(f"Query: {query[:50]}...")
        provider, metadata = selector.select_provider(query, context)
        
        print(f"Selected Provider: {provider.value}")
        print(f"Task Type: {metadata['task_type']}")
        print(f"Estimated Cost: ${metadata['estimated_cost']:.4f}")
        print(f"Selection Reason: {metadata['selection_reason']}")
        print("-" * 50)
        
        # Simular resultado
        import random
        success = random.random() > 0.1
        response_time = random.uniform(0.5, 5.0)
        tokens = metadata['estimated_tokens'] + random.randint(100, 500)
        
        selector.record_result(
            provider=provider,
            success=success,
            response_time=response_time,
            tokens_used=tokens,
            task_type=TaskType(metadata['task_type']),
            quality_score=random.uniform(0.6, 1.0) if success else None
        )
    
    # Mostrar estadisticas
    print("\n=== PROVIDER STATISTICS ===")
    stats = selector.get_provider_stats()
    print(json.dumps(stats, indent=2))