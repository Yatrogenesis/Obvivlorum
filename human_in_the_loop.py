#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Human-in-the-Loop Security System for AI Symbiote
==================================================

Sistema de confirmación humana para comandos de alto riesgo.
Previene ejecución no autorizada de comandos peligrosos.

Author: Francisco Molina
ORCID: https://orcid.org/0009-0008-6093-8267
Email: pako.molina@gmail.com
"""

import os
import re
import json
import hashlib
import logging
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import tkinter as tk
from tkinter import messagebox
import asyncio
from pathlib import Path

logger = logging.getLogger("HumanInTheLoop")

class RiskLevel(Enum):
    """Niveles de riesgo para comandos"""
    SAFE = "safe"           # No requiere confirmación
    LOW = "low"             # Confirmación opcional
    MEDIUM = "medium"       # Confirmación recomendada
    HIGH = "high"           # Confirmación requerida
    CRITICAL = "critical"   # Confirmación múltiple requerida
    FORBIDDEN = "forbidden" # Bloqueado completamente

class CommandCategory(Enum):
    """Categorías de comandos"""
    FILE_READ = "file_read"
    FILE_WRITE = "file_write"
    FILE_DELETE = "file_delete"
    NETWORK_SCAN = "network_scan"
    NETWORK_ATTACK = "network_attack"
    SYSTEM_MODIFY = "system_modify"
    PROCESS_CONTROL = "process_control"
    CREDENTIAL_ACCESS = "credential_access"
    DATA_EXFILTRATION = "data_exfiltration"
    EXECUTION = "execution"

class HumanInTheLoop:
    """
    Sistema de confirmación humana para comandos peligrosos
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize Human-in-the-Loop system"""
        self.config = config or self._default_config()
        self.is_active = self.config.get("enabled", True)
        self.auto_approve_safe = self.config.get("auto_approve_safe", True)
        
        # Historial de comandos
        self.command_history = []
        self.approval_history = []
        self.rejection_history = []
        
        # Cache de decisiones
        self.decision_cache = {}
        self.cache_duration = timedelta(minutes=self.config.get("cache_minutes", 5))
        
        # Patrones de riesgo
        self._init_risk_patterns()
        
        logger.info("Human-in-the-Loop system initialized")
    
    def _default_config(self) -> Dict[str, Any]:
        """Configuración por defecto"""
        return {
            "enabled": True,
            "auto_approve_safe": True,
            "require_reason_for_high_risk": True,
            "cache_minutes": 5,
            "max_auto_approvals_per_minute": 10,
            "gui_timeout_seconds": 60,
            "log_all_decisions": True,
            "blacklist_patterns": [
                r"rm\s+-rf\s+/",
                r"format\s+[cC]:",
                r"del\s+/s\s+/q\s+[cC]:\\",
                r":(){ :|:& };:",  # Fork bomb
                r"dd\s+if=/dev/zero",
                r"cryptolocker",
                r"ransomware"
            ],
            "whitelist_commands": [
                "ls", "dir", "pwd", "whoami", "date", "echo"
            ]
        }
    
    def _init_risk_patterns(self):
        """Inicializar patrones de detección de riesgo"""
        self.risk_patterns = {
            RiskLevel.FORBIDDEN: [
                # Comandos destructivos
                (r"rm\s+-rf\s+/", "System-wide deletion"),
                (r"format\s+[cC]:", "Drive formatting"),
                (r"del\s+/s\s+/q\s+[cC]:\\", "Windows system deletion"),
                (r":(){ :|:& };:", "Fork bomb"),
                (r"dd\s+if=/dev/(zero|random)\s+of=/dev/[sh]da", "Disk overwrite"),
                
                # Malware conocido
                (r"(cryptolocker|ransomware|malware)", "Known malware"),
                (r"mimikatz", "Credential theft tool"),
                
                # Exfiltración masiva
                (r"tar.*\|\s*nc\s+", "Data exfiltration via netcat"),
                (r"curl.*--data.*@/etc/passwd", "Password file exfiltration"),
            ],
            
            RiskLevel.CRITICAL: [
                # Modificación del sistema
                (r"passwd\s+root", "Root password change"),
                (r"usermod.*-aG.*sudo", "Privilege escalation"),
                (r"chmod\s+777\s+/", "Dangerous permission change"),
                (r"chown.*root", "Ownership change to root"),
                
                # Acceso a credenciales
                (r"/etc/(passwd|shadow|sudoers)", "Credential file access"),
                (r"\.ssh/.*key", "SSH key access"),
                (r"wallet\.(dat|json)", "Cryptocurrency wallet access"),
                
                # Network attacks
                (r"sqlmap.*--dump", "SQL injection attack"),
                (r"hydra.*-l.*-p", "Brute force attack"),
                (r"metasploit", "Exploitation framework"),
            ],
            
            RiskLevel.HIGH: [
                # Escaneo de red
                (r"nmap.*-sS", "Stealth port scan"),
                (r"nmap.*-O", "OS fingerprinting"),
                (r"masscan", "Mass port scanner"),
                
                # Modificación de archivos sensibles
                (r"/etc/hosts", "Hosts file modification"),
                (r"iptables.*-F", "Firewall flush"),
                (r"systemctl.*(stop|disable).*firewall", "Firewall disable"),
                
                # Ejecución remota
                (r"ssh.*@.*[0-9]{1,3}\.[0-9]{1,3}", "SSH to IP address"),
                (r"psexec", "Remote execution tool"),
            ],
            
            RiskLevel.MEDIUM: [
                # Herramientas de seguridad
                (r"nmap\s+localhost", "Local port scan"),
                (r"nikto", "Web vulnerability scanner"),
                (r"dirb", "Directory brute forcer"),
                
                # Acceso a logs
                (r"/var/log/", "Log file access"),
                (r"Event.*Log", "Windows event log access"),
                
                # Descarga de archivos
                (r"wget\s+http", "File download"),
                (r"curl.*-O", "File download with curl"),
            ],
            
            RiskLevel.LOW: [
                # Comandos de información
                (r"netstat", "Network statistics"),
                (r"ps\s+aux", "Process listing"),
                (r"df\s+-h", "Disk usage"),
                (r"top", "Process monitor"),
                
                # Lectura de archivos no sensibles
                (r"cat\s+[^/]", "Local file read"),
                (r"grep", "Text search"),
            ]
        }
    
    def analyze_command(self, command: str, context: Dict[str, Any] = None) -> Tuple[RiskLevel, str, CommandCategory]:
        """
        Analizar un comando y determinar su nivel de riesgo
        
        Returns:
            Tuple[RiskLevel, reason, CommandCategory]
        """
        command_lower = command.lower().strip()
        
        # Verificar whitelist primero
        for safe_cmd in self.config["whitelist_commands"]:
            if command_lower.startswith(safe_cmd):
                return RiskLevel.SAFE, "Whitelisted command", CommandCategory.EXECUTION
        
        # Verificar blacklist
        for pattern in self.config["blacklist_patterns"]:
            if re.search(pattern, command, re.IGNORECASE):
                return RiskLevel.FORBIDDEN, f"Blacklisted pattern: {pattern}", CommandCategory.EXECUTION
        
        # Analizar patrones de riesgo
        for risk_level, patterns in self.risk_patterns.items():
            for pattern, reason in patterns:
                if re.search(pattern, command, re.IGNORECASE):
                    category = self._determine_category(command, pattern)
                    return risk_level, reason, category
        
        # Por defecto, considerar como bajo riesgo
        return RiskLevel.LOW, "Unknown command pattern", CommandCategory.EXECUTION
    
    def _determine_category(self, command: str, pattern: str) -> CommandCategory:
        """Determinar la categoría del comando"""
        command_lower = command.lower()
        
        if any(x in command_lower for x in ["nmap", "masscan", "netstat"]):
            return CommandCategory.NETWORK_SCAN
        elif any(x in command_lower for x in ["sqlmap", "hydra", "metasploit"]):
            return CommandCategory.NETWORK_ATTACK
        elif any(x in command_lower for x in ["rm ", "del ", "format"]):
            return CommandCategory.FILE_DELETE
        elif any(x in command_lower for x in ["passwd", "shadow", "wallet", ".ssh"]):
            return CommandCategory.CREDENTIAL_ACCESS
        elif any(x in command_lower for x in ["cat", "type", "read"]):
            return CommandCategory.FILE_READ
        elif any(x in command_lower for x in ["echo", "write", ">"]):
            return CommandCategory.FILE_WRITE
        elif any(x in command_lower for x in ["systemctl", "service", "kill"]):
            return CommandCategory.PROCESS_CONTROL
        elif any(x in command_lower for x in ["tar", "zip", "curl.*--data", "nc "]):
            return CommandCategory.DATA_EXFILTRATION
        elif any(x in command_lower for x in ["chmod", "chown", "usermod"]):
            return CommandCategory.SYSTEM_MODIFY
        else:
            return CommandCategory.EXECUTION
    
    def request_approval(self, command: str, risk_level: RiskLevel, reason: str, 
                         category: CommandCategory, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Solicitar aprobación humana para un comando
        
        Returns:
            Dict con decision, reason, timestamp
        """
        # Verificar cache de decisiones
        command_hash = hashlib.sha256(command.encode()).hexdigest()
        if command_hash in self.decision_cache:
            cached = self.decision_cache[command_hash]
            if datetime.now() - cached["timestamp"] < self.cache_duration:
                logger.info(f"Using cached decision for command: {command[:50]}...")
                return cached
        
        # Auto-aprobar comandos seguros si está configurado
        if self.auto_approve_safe and risk_level == RiskLevel.SAFE:
            decision = {
                "approved": True,
                "reason": "Auto-approved (safe command)",
                "timestamp": datetime.now(),
                "risk_level": risk_level.value,
                "category": category.value
            }
            self.decision_cache[command_hash] = decision
            return decision
        
        # Bloquear comandos prohibidos
        if risk_level == RiskLevel.FORBIDDEN:
            decision = {
                "approved": False,
                "reason": f"Blocked: {reason}",
                "timestamp": datetime.now(),
                "risk_level": risk_level.value,
                "category": category.value
            }
            self.rejection_history.append({
                "command": command,
                "decision": decision,
                "context": context
            })
            return decision
        
        # Solicitar confirmación humana
        if risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            decision = self._show_approval_dialog(command, risk_level, reason, category, context)
        else:
            # Para riesgo medio y bajo, usar confirmación simple
            decision = self._show_simple_confirmation(command, risk_level, reason, category)
        
        # Cachear decisión
        self.decision_cache[command_hash] = decision
        
        # Guardar en historial
        if decision["approved"]:
            self.approval_history.append({
                "command": command,
                "decision": decision,
                "context": context
            })
        else:
            self.rejection_history.append({
                "command": command,
                "decision": decision,
                "context": context
            })
        
        # Log si está configurado
        if self.config.get("log_all_decisions", True):
            self._log_decision(command, decision, context)
        
        return decision
    
    def _show_approval_dialog(self, command: str, risk_level: RiskLevel, 
                             reason: str, category: CommandCategory, 
                             context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Mostrar diálogo de aprobación detallado para comandos de alto riesgo"""
        root = tk.Tk()
        root.withdraw()  # Ocultar ventana principal
        
        # Preparar mensaje
        message = f"""
⚠️ COMANDO DE ALTO RIESGO DETECTADO ⚠️

Nivel de Riesgo: {risk_level.value.upper()}
Categoría: {category.value.replace('_', ' ').title()}
Razón: {reason}

Comando:
{command}

Contexto:
{json.dumps(context, indent=2) if context else 'No disponible'}

¿Desea ejecutar este comando?

ADVERTENCIA: Este comando puede tener consecuencias graves e irreversibles.
"""
        
        # Mostrar diálogo
        result = messagebox.askyesno(
            "Confirmación de Seguridad Requerida",
            message,
            icon=messagebox.WARNING
        )
        
        root.destroy()
        
        # Para comandos CRÍTICOS, solicitar doble confirmación
        if risk_level == RiskLevel.CRITICAL and result:
            root2 = tk.Tk()
            root2.withdraw()
            
            second_result = messagebox.askyesno(
                "SEGUNDA CONFIRMACIÓN REQUERIDA",
                "⚠️⚠️⚠️ ADVERTENCIA CRÍTICA ⚠️⚠️⚠️\n\n"
                "Este comando es EXTREMADAMENTE peligroso.\n\n"
                "¿Está ABSOLUTAMENTE SEGURO de que desea continuar?\n\n"
                "Esta es su última oportunidad de cancelar.",
                icon=messagebox.ERROR
            )
            
            root2.destroy()
            result = second_result
        
        return {
            "approved": result,
            "reason": "Human approval" if result else "Human rejection",
            "timestamp": datetime.now(),
            "risk_level": risk_level.value,
            "category": category.value,
            "human_confirmed": True
        }
    
    def _show_simple_confirmation(self, command: str, risk_level: RiskLevel, 
                                 reason: str, category: CommandCategory) -> Dict[str, Any]:
        """Mostrar confirmación simple para comandos de riesgo medio/bajo"""
        import tkinter.simpledialog as simpledialog
        
        root = tk.Tk()
        root.withdraw()
        
        message = f"Comando: {command[:100]}{'...' if len(command) > 100 else ''}\n"
        message += f"Riesgo: {risk_level.value}\n"
        message += f"Razón: {reason}\n\n"
        message += "Escriba 'yes' para aprobar, cualquier otra cosa para rechazar:"
        
        response = simpledialog.askstring(
            "Confirmación de Comando",
            message
        )
        
        root.destroy()
        
        approved = response and response.lower() == "yes"
        
        return {
            "approved": approved,
            "reason": f"User input: {response}" if response else "User cancelled",
            "timestamp": datetime.now(),
            "risk_level": risk_level.value,
            "category": category.value,
            "human_confirmed": True
        }
    
    def _log_decision(self, command: str, decision: Dict[str, Any], context: Dict[str, Any] = None):
        """Guardar decisión en log"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "command": command,
            "decision": decision,
            "context": context
        }
        
        log_file = Path("human_decisions.jsonl")
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")
    
    def check_command(self, command: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Verificar si un comando puede ser ejecutado
        
        Returns:
            Dict con approved (bool), reason (str), y metadata
        """
        if not self.is_active:
            return {
                "approved": True,
                "reason": "Human-in-the-Loop disabled",
                "timestamp": datetime.now()
            }
        
        # Analizar comando
        risk_level, reason, category = self.analyze_command(command, context)
        
        # Solicitar aprobación según el nivel de riesgo
        decision = self.request_approval(command, risk_level, reason, category, context)
        
        # Agregar al historial
        self.command_history.append({
            "command": command,
            "risk_level": risk_level.value,
            "category": category.value,
            "decision": decision,
            "timestamp": datetime.now()
        })
        
        return decision
    
    def get_statistics(self) -> Dict[str, Any]:
        """Obtener estadísticas del sistema"""
        return {
            "total_commands": len(self.command_history),
            "approved": len(self.approval_history),
            "rejected": len(self.rejection_history),
            "cache_size": len(self.decision_cache),
            "risk_distribution": self._get_risk_distribution(),
            "category_distribution": self._get_category_distribution()
        }
    
    def _get_risk_distribution(self) -> Dict[str, int]:
        """Obtener distribución de niveles de riesgo"""
        distribution = {level.value: 0 for level in RiskLevel}
        for entry in self.command_history:
            risk = entry.get("risk_level", "unknown")
            if risk in distribution:
                distribution[risk] += 1
        return distribution
    
    def _get_category_distribution(self) -> Dict[str, int]:
        """Obtener distribución de categorías"""
        distribution = {cat.value: 0 for cat in CommandCategory}
        for entry in self.command_history:
            category = entry.get("category", "unknown")
            if category in distribution:
                distribution[category] += 1
        return distribution


# Ejemplo de uso
if __name__ == "__main__":
    # Configurar logging
    logging.basicConfig(level=logging.INFO)
    
    # Crear sistema Human-in-the-Loop
    hitl = HumanInTheLoop()
    
    # Comandos de prueba
    test_commands = [
        "ls -la",                              # Safe
        "nmap localhost",                       # Medium
        "nmap -sS 192.168.1.0/24",            # High
        "rm -rf /",                            # Forbidden
        "cat /etc/passwd",                     # Critical
        "echo 'test' > test.txt",             # Low
    ]
    
    print("=== HUMAN-IN-THE-LOOP TEST ===\n")
    
    for cmd in test_commands:
        print(f"Testing command: {cmd}")
        result = hitl.check_command(cmd, {"source": "test", "user": "developer"})
        print(f"Decision: {'✅ APPROVED' if result['approved'] else '❌ REJECTED'}")
        print(f"Reason: {result['reason']}")
        print(f"Risk Level: {result.get('risk_level', 'N/A')}")
        print("-" * 50)
    
    # Mostrar estadísticas
    print("\n=== STATISTICS ===")
    stats = hitl.get_statistics()
    print(json.dumps(stats, indent=2))