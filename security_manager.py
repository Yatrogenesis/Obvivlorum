#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Security Manager with Escalated Privileges System
=================================================

Sistema de gestión de seguridad con privilegios escalonados para AI Symbiote.
Implementa control granular de permisos basado en niveles de confianza.

Author: Francisco Molina
ORCID: https://orcid.org/0009-0008-6093-8267
Email: pako.molina@gmail.com
"""

import os
import json
import hashlib
import logging
from enum import Enum
from typing import Dict, List, Set, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path
import jwt
from cryptography.fernet import Fernet

logger = logging.getLogger("SecurityManager")

class ThreatLevel(Enum):
    """Niveles de amenaza del sistema"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4
    MAXIMUM = 5

class PrivilegeLevel(Enum):
    """Niveles de privilegios escalonados"""
    GUEST = 0        # Solo lectura, sin ejecución
    USER = 1         # Operaciones básicas
    OPERATOR = 2     # Operaciones avanzadas con restricciones
    ADMIN = 3        # Control total con auditoría
    SYSTEM = 4       # Modo kernel sin restricciones

class Permission(Enum):
    """Permisos granulares del sistema"""
    # Permisos básicos
    GUI_ACCESS = "gui_access"
    WEB_ACCESS = "web_access"
    READ_FILES = "read_files"
    READ_LOGS = "read_logs"
    
    # Permisos intermedios
    WRITE_FILES = "write_files"
    EXECUTE_SAFE_COMMANDS = "execute_safe_commands"
    NETWORK_READ = "network_read"
    API_ACCESS = "api_access"
    
    # Permisos avanzados
    EXECUTE_SYSTEM_COMMANDS = "execute_system_commands"
    NETWORK_WRITE = "network_write"
    PERSISTENCE = "persistence"
    MODIFY_CONFIG = "modify_config"
    
    # Permisos críticos
    UNRESTRICTED_COMMANDS = "unrestricted_commands"
    KERNEL_ACCESS = "kernel_access"
    SECURITY_OVERRIDE = "security_override"
    AUDIT_BYPASS = "audit_bypass"

class SecurityManager:
    """
    Gestor de seguridad con sistema de privilegios escalonados
    """
    
    def __init__(self, config_path: str = None):
        """Initialize Security Manager"""
        self.config_path = config_path or "security_config.json"
        self.config = self._load_config()
        
        # Estado actual
        self.current_privilege_level = PrivilegeLevel.GUEST
        self.current_threat_level = ThreatLevel.LOW
        self.active_permissions: Set[Permission] = set()
        self.session_token = None
        
        # Auditoría
        self.audit_log = []
        self.permission_requests = []
        self.escalation_attempts = []
        
        # Criptografía para tokens
        self.cipher_suite = Fernet(self._get_or_create_key())
        
        # Definir jerarquía de permisos
        self._init_permission_hierarchy()
        
        # Cargar estado inicial
        self._initialize_permissions()
        
        logger.info("Security Manager initialized with privilege level: %s", 
                   self.current_privilege_level.name)
    
    def _load_config(self) -> Dict[str, Any]:
        """Cargar configuración de seguridad"""
        default_config = {
            "default_privilege_level": "GUEST",
            "auto_escalation": False,
            "require_2fa_for_admin": True,
            "session_timeout_minutes": 30,
            "max_escalation_attempts": 3,
            "audit_all_actions": True,
            "threat_response": {
                "LOW": {"max_privilege": "OPERATOR"},
                "MEDIUM": {"max_privilege": "USER"},
                "HIGH": {"max_privilege": "GUEST"},
                "CRITICAL": {"max_privilege": "GUEST", "lockdown": True},
                "MAXIMUM": {"max_privilege": "NONE", "shutdown": True}
            }
        }
        
        try:
            if Path(self.config_path).exists():
                with open(self.config_path, 'r') as f:
                    loaded_config = json.load(f)
                    default_config.update(loaded_config)
        except Exception as e:
            logger.error(f"Error loading config: {e}")
        
        return default_config
    
    def _get_or_create_key(self) -> bytes:
        """Obtener o crear clave de cifrado"""
        key_file = Path(".security_key")
        
        if key_file.exists():
            with open(key_file, 'rb') as f:
                return f.read()
        else:
            key = Fernet.generate_key()
            with open(key_file, 'wb') as f:
                f.write(key)
            # Establecer permisos restrictivos
            os.chmod(key_file, 0o600)
            return key
    
    def _init_permission_hierarchy(self):
        """Inicializar jerarquía de permisos por nivel"""
        self.permission_hierarchy = {
            PrivilegeLevel.GUEST: {
                Permission.GUI_ACCESS,
                Permission.WEB_ACCESS,
                Permission.READ_FILES,
                Permission.READ_LOGS
            },
            
            PrivilegeLevel.USER: {
                # Hereda de GUEST
                Permission.GUI_ACCESS,
                Permission.WEB_ACCESS,
                Permission.READ_FILES,
                Permission.READ_LOGS,
                # Nuevos permisos
                Permission.WRITE_FILES,
                Permission.EXECUTE_SAFE_COMMANDS,
                Permission.API_ACCESS
            },
            
            PrivilegeLevel.OPERATOR: {
                # Hereda de USER
                Permission.GUI_ACCESS,
                Permission.WEB_ACCESS,
                Permission.READ_FILES,
                Permission.READ_LOGS,
                Permission.WRITE_FILES,
                Permission.EXECUTE_SAFE_COMMANDS,
                Permission.API_ACCESS,
                # Nuevos permisos
                Permission.NETWORK_READ,
                Permission.NETWORK_WRITE,
                Permission.EXECUTE_SYSTEM_COMMANDS,
                Permission.MODIFY_CONFIG
            },
            
            PrivilegeLevel.ADMIN: {
                # Hereda de OPERATOR
                Permission.GUI_ACCESS,
                Permission.WEB_ACCESS,
                Permission.READ_FILES,
                Permission.READ_LOGS,
                Permission.WRITE_FILES,
                Permission.EXECUTE_SAFE_COMMANDS,
                Permission.API_ACCESS,
                Permission.NETWORK_READ,
                Permission.NETWORK_WRITE,
                Permission.EXECUTE_SYSTEM_COMMANDS,
                Permission.MODIFY_CONFIG,
                # Nuevos permisos
                Permission.PERSISTENCE,
                Permission.UNRESTRICTED_COMMANDS,
                Permission.SECURITY_OVERRIDE
            },
            
            PrivilegeLevel.SYSTEM: {
                # Todos los permisos
                permission for permission in Permission
            }
        }
    
    def _initialize_permissions(self):
        """Inicializar permisos según el nivel actual"""
        default_level_str = self.config.get("default_privilege_level", "GUEST")
        try:
            self.current_privilege_level = PrivilegeLevel[default_level_str]
        except KeyError:
            self.current_privilege_level = PrivilegeLevel.GUEST
        
        self.active_permissions = self.permission_hierarchy.get(
            self.current_privilege_level, set()
        )
        
        logger.info(f"Initialized with {len(self.active_permissions)} permissions")
    
    def has_permission(self, permission: Permission) -> bool:
        """Verificar si se tiene un permiso específico"""
        has_perm = permission in self.active_permissions
        
        # Auditar verificación de permisos
        if self.config.get("audit_all_actions", True):
            self._audit_action("permission_check", {
                "permission": permission.value,
                "granted": has_perm,
                "privilege_level": self.current_privilege_level.name
            })
        
        return has_perm
    
    def request_privilege_escalation(self, target_level: PrivilegeLevel, 
                                    reason: str, 
                                    authentication: Dict[str, str] = None) -> bool:
        """
        Solicitar escalación de privilegios
        
        Args:
            target_level: Nivel de privilegio deseado
            reason: Razón para la escalación
            authentication: Credenciales de autenticación
        
        Returns:
            bool: True si la escalación fue exitosa
        """
        # Registrar intento
        self.escalation_attempts.append({
            "timestamp": datetime.now().isoformat(),
            "from_level": self.current_privilege_level.name,
            "to_level": target_level.name,
            "reason": reason
        })
        
        # Verificar si se excedieron los intentos
        recent_attempts = [
            a for a in self.escalation_attempts
            if datetime.fromisoformat(a["timestamp"]) > 
               datetime.now() - timedelta(minutes=5)
        ]
        
        if len(recent_attempts) > self.config.get("max_escalation_attempts", 3):
            logger.warning("Too many escalation attempts")
            self._audit_action("escalation_blocked", {
                "reason": "too_many_attempts",
                "target_level": target_level.name
            })
            return False
        
        # Verificar restricciones por nivel de amenaza
        threat_restrictions = self.config["threat_response"].get(
            self.current_threat_level.name, {}
        )
        
        max_allowed = threat_restrictions.get("max_privilege", "SYSTEM")
        if max_allowed != "SYSTEM":
            try:
                max_level = PrivilegeLevel[max_allowed]
                if target_level.value > max_level.value:
                    logger.warning(f"Escalation denied due to threat level {self.current_threat_level.name}")
                    self._audit_action("escalation_denied", {
                        "reason": "threat_level_restriction",
                        "current_threat": self.current_threat_level.name,
                        "max_allowed": max_allowed
                    })
                    return False
            except KeyError:
                pass
        
        # Verificar autenticación para niveles altos
        if target_level.value >= PrivilegeLevel.ADMIN.value:
            if not self._verify_authentication(authentication, target_level):
                logger.warning("Authentication failed for privilege escalation")
                self._audit_action("escalation_failed", {
                    "reason": "authentication_failed",
                    "target_level": target_level.name
                })
                return False
        
        # Aplicar escalación
        old_level = self.current_privilege_level
        self.current_privilege_level = target_level
        self.active_permissions = self.permission_hierarchy.get(target_level, set())
        
        # Generar nuevo token de sesión
        self.session_token = self._generate_session_token(target_level)
        
        # Auditar escalación exitosa
        self._audit_action("privilege_escalated", {
            "from_level": old_level.name,
            "to_level": target_level.name,
            "reason": reason,
            "permissions_count": len(self.active_permissions)
        })
        
        logger.info(f"Privilege escalated from {old_level.name} to {target_level.name}")
        return True
    
    def _verify_authentication(self, auth: Dict[str, str], 
                              target_level: PrivilegeLevel) -> bool:
        """Verificar autenticación para escalación"""
        if not auth:
            return False
        
        # Para ADMIN, requerir 2FA si está configurado
        if target_level == PrivilegeLevel.ADMIN and \
           self.config.get("require_2fa_for_admin", True):
            if "2fa_code" not in auth:
                return False
            # Aquí se verificaría el código 2FA real
            # Por ahora, solo verificamos que existe
        
        # Verificar credenciales (simplificado para demo)
        # En producción, esto debería verificar contra un sistema real
        if auth.get("username") == "admin" and \
           hashlib.sha256(auth.get("password", "").encode()).hexdigest() == \
           "8c6976e5b5410415bde908bd4dee15dfb167a9c873fc4bb8a81f6f2ab448a918":  # "admin"
            return True
        
        return False
    
    def _generate_session_token(self, privilege_level: PrivilegeLevel) -> str:
        """Generar token JWT de sesión"""
        payload = {
            "privilege_level": privilege_level.name,
            "permissions": [p.value for p in self.active_permissions],
            "issued_at": datetime.now().isoformat(),
            "expires_at": (datetime.now() + 
                          timedelta(minutes=self.config.get("session_timeout_minutes", 30))
                         ).isoformat()
        }
        
        # Cifrar el payload
        token_bytes = json.dumps(payload).encode()
        encrypted_token = self.cipher_suite.encrypt(token_bytes)
        
        return encrypted_token.decode()
    
    def verify_session_token(self, token: str) -> bool:
        """Verificar validez del token de sesión"""
        try:
            # Descifrar token
            decrypted = self.cipher_suite.decrypt(token.encode())
            payload = json.loads(decrypted.decode())
            
            # Verificar expiración
            expires_at = datetime.fromisoformat(payload["expires_at"])
            if datetime.now() > expires_at:
                logger.warning("Session token expired")
                return False
            
            # Verificar nivel de privilegio
            if payload["privilege_level"] != self.current_privilege_level.name:
                logger.warning("Privilege level mismatch in token")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Token verification failed: {e}")
            return False
    
    def update_threat_level(self, new_level: ThreatLevel):
        """Actualizar nivel de amenaza y ajustar permisos"""
        old_level = self.current_threat_level
        self.current_threat_level = new_level
        
        # Aplicar restricciones según el nivel de amenaza
        threat_config = self.config["threat_response"].get(new_level.name, {})
        
        # Verificar si hay lockdown
        if threat_config.get("lockdown", False):
            logger.warning(f"LOCKDOWN activated due to threat level: {new_level.name}")
            self._enter_lockdown_mode()
        
        # Verificar si hay shutdown
        if threat_config.get("shutdown", False):
            logger.critical(f"SHUTDOWN triggered due to threat level: {new_level.name}")
            self._initiate_shutdown()
        
        # Ajustar privilegios máximos
        max_privilege_str = threat_config.get("max_privilege", "SYSTEM")
        if max_privilege_str != "SYSTEM":
            try:
                max_privilege = PrivilegeLevel[max_privilege_str]
                if self.current_privilege_level.value > max_privilege.value:
                    logger.warning(f"Reducing privileges due to threat level {new_level.name}")
                    self.current_privilege_level = max_privilege
                    self.active_permissions = self.permission_hierarchy.get(max_privilege, set())
            except KeyError:
                pass
        
        # Auditar cambio
        self._audit_action("threat_level_changed", {
            "from_level": old_level.name,
            "to_level": new_level.name,
            "lockdown": threat_config.get("lockdown", False),
            "shutdown": threat_config.get("shutdown", False),
            "current_privileges": self.current_privilege_level.name
        })
    
    def _enter_lockdown_mode(self):
        """Entrar en modo lockdown - solo permisos mínimos"""
        self.current_privilege_level = PrivilegeLevel.GUEST
        self.active_permissions = {
            Permission.READ_LOGS  # Solo permitir lectura de logs
        }
        self.session_token = None  # Invalidar todas las sesiones
        
        logger.critical("System entered LOCKDOWN mode")
    
    def _initiate_shutdown(self):
        """Iniciar shutdown seguro del sistema"""
        # Guardar estado actual
        self._save_security_state()
        
        # Revocar todos los permisos
        self.active_permissions = set()
        self.session_token = None
        
        logger.critical("System SHUTDOWN initiated")
        
        # En un sistema real, aquí se iniciaría el proceso de shutdown
        # Por seguridad, no implementamos el shutdown real en este demo
    
    def _audit_action(self, action: str, details: Dict[str, Any]):
        """Registrar acción en log de auditoría"""
        audit_entry = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "privilege_level": self.current_privilege_level.name,
            "threat_level": self.current_threat_level.name,
            "details": details
        }
        
        self.audit_log.append(audit_entry)
        
        # Guardar en archivo si está configurado
        if self.config.get("audit_all_actions", True):
            audit_file = Path("security_audit.jsonl")
            with open(audit_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(audit_entry) + "\n")
    
    def _save_security_state(self):
        """Guardar estado de seguridad actual"""
        state = {
            "timestamp": datetime.now().isoformat(),
            "privilege_level": self.current_privilege_level.name,
            "threat_level": self.current_threat_level.name,
            "active_permissions": [p.value for p in self.active_permissions],
            "audit_summary": {
                "total_actions": len(self.audit_log),
                "escalation_attempts": len(self.escalation_attempts),
                "permission_requests": len(self.permission_requests)
            }
        }
        
        state_file = Path("security_state.json")
        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
    
    def get_security_status(self) -> Dict[str, Any]:
        """Obtener estado actual de seguridad"""
        return {
            "privilege_level": self.current_privilege_level.name,
            "threat_level": self.current_threat_level.name,
            "active_permissions": [p.value for p in self.active_permissions],
            "session_valid": self.verify_session_token(self.session_token) if self.session_token else False,
            "audit_entries": len(self.audit_log),
            "recent_escalations": len([
                a for a in self.escalation_attempts
                if datetime.fromisoformat(a["timestamp"]) > 
                   datetime.now() - timedelta(hours=1)
            ])
        }


# Ejemplo de uso
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Crear Security Manager
    security = SecurityManager()
    
    print("=== SECURITY MANAGER TEST ===\n")
    
    # Mostrar estado inicial
    print("Initial Status:")
    print(json.dumps(security.get_security_status(), indent=2))
    print()
    
    # Intentar operación sin permisos
    print("Checking UNRESTRICTED_COMMANDS permission:")
    print(f"Has permission: {security.has_permission(Permission.UNRESTRICTED_COMMANDS)}")
    print()
    
    # Solicitar escalación a OPERATOR
    print("Requesting escalation to OPERATOR...")
    success = security.request_privilege_escalation(
        PrivilegeLevel.OPERATOR,
        "Testing escalation",
        {"username": "admin", "password": "admin"}
    )
    print(f"Escalation success: {success}")
    print()
    
    # Verificar nuevos permisos
    print("New permissions:")
    for perm in security.active_permissions:
        print(f"  - {perm.value}")
    print()
    
    # Simular amenaza alta
    print("Simulating HIGH threat level...")
    security.update_threat_level(ThreatLevel.HIGH)
    print(f"Current privilege level: {security.current_privilege_level.name}")
    print()
    
    # Mostrar estado final
    print("Final Status:")
    print(json.dumps(security.get_security_status(), indent=2))