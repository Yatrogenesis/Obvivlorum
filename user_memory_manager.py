#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
User Memory Manager - Multi-User Vector Storage
===============================================

Manages user-specific encrypted memory storage for multi-user OAuth system:
- Individual vector memory per user
- Encrypted storage per user session
- Conversation isolation
- Secure memory management
- User-specific embeddings and context
"""

import os
import json
import logging
import hashlib
import base64
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict
import threading
import time

# Encryption
try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    logging.warning("cryptography not available - using basic encoding")

# Vector storage
try:
    import numpy as np
    import pickle
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    logging.warning("numpy not available - using basic storage")

logger = logging.getLogger("UserMemoryManager")

@dataclass
class UserSession:
    """User session information."""
    user_id: str
    oauth_provider: Optional[str] = None
    user_email: Optional[str] = None
    user_name: Optional[str] = None
    session_start: datetime = None
    last_activity: datetime = None
    session_token: Optional[str] = None
    
    def __post_init__(self):
        if self.session_start is None:
            self.session_start = datetime.now()
        if self.last_activity is None:
            self.last_activity = datetime.now()

@dataclass 
class ConversationEntry:
    """Single conversation entry."""
    timestamp: datetime
    user_message: str
    ai_response: str
    user_id: str
    session_id: str
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class VectorMemory:
    """Vector memory storage for a user."""
    user_id: str
    embeddings: Optional[Any] = None  # numpy array if available
    conversation_vectors: List[Any] = None
    semantic_index: Dict[str, Any] = None
    last_updated: datetime = None
    
    def __post_init__(self):
        if self.conversation_vectors is None:
            self.conversation_vectors = []
        if self.semantic_index is None:
            self.semantic_index = {}
        if self.last_updated is None:
            self.last_updated = datetime.now()

class UserMemoryManager:
    """Manages user-specific encrypted memory storage."""
    
    def __init__(self, storage_dir: str = ".user_memory"):
        """Initialize user memory manager."""
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        
        # User sessions
        self.active_sessions: Dict[str, UserSession] = {}
        self.user_memories: Dict[str, VectorMemory] = {}
        self.conversation_histories: Dict[str, List[ConversationEntry]] = {}
        
        # Encryption keys per user
        self.user_keys: Dict[str, bytes] = {}
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Session timeout (24 hours)
        self.session_timeout = timedelta(hours=24)
        
        # Load existing data
        self._load_user_data()
        
        # Start cleanup thread
        self._start_cleanup_thread()
    
    def _generate_user_key(self, user_id: str, password: str = None) -> bytes:
        """Generate encryption key for user."""
        if not CRYPTO_AVAILABLE:
            return base64.b64encode(user_id.encode()).ljust(32)[:32]
        
        # Use user_id as base, add optional password
        salt = hashlib.sha256(user_id.encode()).digest()[:16]
        
        if password:
            key_material = password.encode()
        else:
            # Use user_id as default key material
            key_material = user_id.encode()
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        
        return base64.urlsafe_b64encode(kdf.derive(key_material))
    
    def _get_encryption_key(self, user_id: str) -> bytes:
        """Get or generate encryption key for user."""
        if user_id not in self.user_keys:
            self.user_keys[user_id] = self._generate_user_key(user_id)
        return self.user_keys[user_id]
    
    def _encrypt_data(self, data: str, user_id: str) -> str:
        """Encrypt data for specific user."""
        if not CRYPTO_AVAILABLE:
            # Basic base64 encoding as fallback
            return base64.b64encode(data.encode()).decode()
        
        try:
            key = self._get_encryption_key(user_id)
            fernet = Fernet(key)
            encrypted = fernet.encrypt(data.encode())
            return base64.b64encode(encrypted).decode()
        except Exception as e:
            logger.error(f"Encryption failed for user {user_id}: {e}")
            # Fallback to base64
            return base64.b64encode(data.encode()).decode()
    
    def _decrypt_data(self, encrypted_data: str, user_id: str) -> str:
        """Decrypt data for specific user."""
        if not CRYPTO_AVAILABLE:
            # Basic base64 decoding as fallback
            try:
                return base64.b64decode(encrypted_data.encode()).decode()
            except:
                return encrypted_data  # Return as-is if decode fails
        
        try:
            key = self._get_encryption_key(user_id)
            fernet = Fernet(key)
            encrypted_bytes = base64.b64decode(encrypted_data.encode())
            decrypted = fernet.decrypt(encrypted_bytes)
            return decrypted.decode()
        except Exception as e:
            logger.error(f"Decryption failed for user {user_id}: {e}")
            # Fallback to base64
            try:
                return base64.b64decode(encrypted_data.encode()).decode()
            except:
                return encrypted_data
    
    def create_user_session(self, user_id: str, oauth_provider: str = None, 
                          user_email: str = None, user_name: str = None) -> str:
        """Create new user session."""
        with self.lock:
            # Generate session token
            session_token = hashlib.sha256(
                f"{user_id}_{datetime.now().isoformat()}_{os.urandom(16).hex()}".encode()
            ).hexdigest()
            
            session = UserSession(
                user_id=user_id,
                oauth_provider=oauth_provider,
                user_email=user_email,
                user_name=user_name,
                session_token=session_token
            )
            
            self.active_sessions[session_token] = session
            
            # Initialize user memory if not exists
            if user_id not in self.user_memories:
                self.user_memories[user_id] = VectorMemory(user_id=user_id)
            
            if user_id not in self.conversation_histories:
                self.conversation_histories[user_id] = []
            
            logger.info(f"Created session for user {user_id} ({oauth_provider})")
            return session_token
    
    def get_user_session(self, session_token: str) -> Optional[UserSession]:
        """Get user session by token."""
        with self.lock:
            session = self.active_sessions.get(session_token)
            if session:
                # Check if session expired
                if datetime.now() - session.last_activity > self.session_timeout:
                    self._expire_session(session_token)
                    return None
                
                # Update activity
                session.last_activity = datetime.now()
                return session
            return None
    
    def _expire_session(self, session_token: str):
        """Expire a session."""
        if session_token in self.active_sessions:
            session = self.active_sessions[session_token]
            logger.info(f"Session expired for user {session.user_id}")
            del self.active_sessions[session_token]
    
    def add_conversation_entry(self, session_token: str, user_message: str, 
                             ai_response: str, metadata: Dict = None) -> bool:
        """Add conversation entry for user."""
        session = self.get_user_session(session_token)
        if not session:
            return False
        
        with self.lock:
            entry = ConversationEntry(
                timestamp=datetime.now(),
                user_message=user_message,
                ai_response=ai_response,
                user_id=session.user_id,
                session_id=session_token,
                metadata=metadata or {}
            )
            
            if session.user_id not in self.conversation_histories:
                self.conversation_histories[session.user_id] = []
            
            self.conversation_histories[session.user_id].append(entry)
            
            # Keep last 100 conversations per user
            if len(self.conversation_histories[session.user_id]) > 100:
                self.conversation_histories[session.user_id] = \
                    self.conversation_histories[session.user_id][-100:]
            
            # Update vector memory if possible
            self._update_vector_memory(session.user_id, user_message, ai_response)
            
            # Save to disk periodically
            if len(self.conversation_histories[session.user_id]) % 10 == 0:
                self._save_user_data(session.user_id)
            
            return True
    
    def get_user_conversation_history(self, session_token: str, limit: int = 50) -> List[ConversationEntry]:
        """Get conversation history for user."""
        session = self.get_user_session(session_token)
        if not session:
            return []
        
        with self.lock:
            history = self.conversation_histories.get(session.user_id, [])
            return history[-limit:] if limit else history
    
    def _update_vector_memory(self, user_id: str, user_message: str, ai_response: str):
        """Update vector memory for user (placeholder for now)."""
        # This would integrate with actual vector embedding models
        # For now, just track message count and keywords
        
        if user_id not in self.user_memories:
            self.user_memories[user_id] = VectorMemory(user_id=user_id)
        
        memory = self.user_memories[user_id]
        
        # Simple keyword extraction (could be replaced with proper embeddings)
        keywords = set()
        for text in [user_message, ai_response]:
            words = text.lower().split()
            keywords.update(word for word in words if len(word) > 4)
        
        # Update semantic index
        for keyword in keywords:
            if keyword not in memory.semantic_index:
                memory.semantic_index[keyword] = 0
            memory.semantic_index[keyword] += 1
        
        memory.last_updated = datetime.now()
    
    def search_user_memory(self, session_token: str, query: str, limit: int = 10) -> List[ConversationEntry]:
        """Search user's conversation memory."""
        session = self.get_user_session(session_token)
        if not session:
            return []
        
        with self.lock:
            history = self.conversation_histories.get(session.user_id, [])
            query_lower = query.lower()
            
            # Simple text search (could be enhanced with vector similarity)
            matching = []
            for entry in history:
                if (query_lower in entry.user_message.lower() or 
                    query_lower in entry.ai_response.lower()):
                    matching.append(entry)
            
            # Return most recent matches
            return matching[-limit:] if matching else []
    
    def get_user_memory_stats(self, session_token: str) -> Dict[str, Any]:
        """Get memory statistics for user."""
        session = self.get_user_session(session_token)
        if not session:
            return {}
        
        with self.lock:
            history = self.conversation_histories.get(session.user_id, [])
            memory = self.user_memories.get(session.user_id)
            
            stats = {
                "user_id": session.user_id,
                "conversation_count": len(history),
                "session_duration": str(datetime.now() - session.session_start),
                "last_activity": session.last_activity.isoformat(),
                "memory_keywords": len(memory.semantic_index) if memory else 0
            }
            
            if memory and memory.semantic_index:
                # Top keywords
                sorted_keywords = sorted(
                    memory.semantic_index.items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                stats["top_keywords"] = sorted_keywords[:10]
            
            return stats
    
    def _load_user_data(self):
        """Load user data from disk."""
        try:
            # Load user memories
            for user_file in self.storage_dir.glob("memory_*.json"):
                user_id = user_file.stem.replace("memory_", "")
                try:
                    with open(user_file, 'r', encoding='utf-8') as f:
                        encrypted_data = f.read()
                    
                    decrypted_data = self._decrypt_data(encrypted_data, user_id)
                    data = json.loads(decrypted_data)
                    
                    # Reconstruct memory object
                    memory = VectorMemory(
                        user_id=user_id,
                        semantic_index=data.get('semantic_index', {}),
                        last_updated=datetime.fromisoformat(data.get('last_updated', datetime.now().isoformat()))
                    )
                    self.user_memories[user_id] = memory
                    
                except Exception as e:
                    logger.error(f"Failed to load memory for user {user_id}: {e}")
            
            # Load conversation histories  
            for conv_file in self.storage_dir.glob("conversations_*.json"):
                user_id = conv_file.stem.replace("conversations_", "")
                try:
                    with open(conv_file, 'r', encoding='utf-8') as f:
                        encrypted_data = f.read()
                    
                    decrypted_data = self._decrypt_data(encrypted_data, user_id)
                    data = json.loads(decrypted_data)
                    
                    # Reconstruct conversation entries
                    conversations = []
                    for entry_data in data.get('conversations', []):
                        entry = ConversationEntry(
                            timestamp=datetime.fromisoformat(entry_data['timestamp']),
                            user_message=entry_data['user_message'],
                            ai_response=entry_data['ai_response'],
                            user_id=entry_data['user_id'],
                            session_id=entry_data['session_id'],
                            metadata=entry_data.get('metadata', {})
                        )
                        conversations.append(entry)
                    
                    self.conversation_histories[user_id] = conversations
                    
                except Exception as e:
                    logger.error(f"Failed to load conversations for user {user_id}: {e}")
                    
            logger.info(f"Loaded data for {len(self.user_memories)} users")
            
        except Exception as e:
            logger.error(f"Failed to load user data: {e}")
    
    def _save_user_data(self, user_id: str):
        """Save user data to disk."""
        try:
            # Save memory
            if user_id in self.user_memories:
                memory = self.user_memories[user_id]
                data = {
                    'semantic_index': memory.semantic_index,
                    'last_updated': memory.last_updated.isoformat()
                }
                
                encrypted_data = self._encrypt_data(json.dumps(data), user_id)
                
                memory_file = self.storage_dir / f"memory_{user_id}.json"
                with open(memory_file, 'w', encoding='utf-8') as f:
                    f.write(encrypted_data)
            
            # Save conversations
            if user_id in self.conversation_histories:
                conversations = self.conversation_histories[user_id]
                data = {
                    'conversations': [
                        {
                            'timestamp': entry.timestamp.isoformat(),
                            'user_message': entry.user_message,
                            'ai_response': entry.ai_response,
                            'user_id': entry.user_id,
                            'session_id': entry.session_id,
                            'metadata': entry.metadata
                        }
                        for entry in conversations
                    ]
                }
                
                encrypted_data = self._encrypt_data(json.dumps(data), user_id)
                
                conv_file = self.storage_dir / f"conversations_{user_id}.json"
                with open(conv_file, 'w', encoding='utf-8') as f:
                    f.write(encrypted_data)
                    
        except Exception as e:
            logger.error(f"Failed to save data for user {user_id}: {e}")
    
    def save_all_user_data(self):
        """Save all user data to disk."""
        with self.lock:
            for user_id in set(list(self.user_memories.keys()) + list(self.conversation_histories.keys())):
                self._save_user_data(user_id)
            logger.info("Saved all user data")
    
    def _start_cleanup_thread(self):
        """Start background thread for session cleanup."""
        def cleanup():
            while True:
                try:
                    time.sleep(3600)  # Check every hour
                    with self.lock:
                        expired_sessions = []
                        for token, session in self.active_sessions.items():
                            if datetime.now() - session.last_activity > self.session_timeout:
                                expired_sessions.append(token)
                        
                        for token in expired_sessions:
                            self._expire_session(token)
                        
                        if expired_sessions:
                            logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
                            
                except Exception as e:
                    logger.error(f"Session cleanup error: {e}")
        
        cleanup_thread = threading.Thread(target=cleanup, daemon=True)
        cleanup_thread.start()
    
    def get_active_users(self) -> List[Dict[str, Any]]:
        """Get list of active users."""
        with self.lock:
            users = []
            for session in self.active_sessions.values():
                users.append({
                    'user_id': session.user_id,
                    'user_name': session.user_name,
                    'user_email': session.user_email,
                    'oauth_provider': session.oauth_provider,
                    'session_start': session.session_start.isoformat(),
                    'last_activity': session.last_activity.isoformat()
                })
            return users
    
    def delete_user_data(self, user_id: str) -> bool:
        """Delete all data for a user (GDPR compliance)."""
        try:
            with self.lock:
                # Remove from memory
                if user_id in self.user_memories:
                    del self.user_memories[user_id]
                
                if user_id in self.conversation_histories:
                    del self.conversation_histories[user_id]
                
                if user_id in self.user_keys:
                    del self.user_keys[user_id]
                
                # Remove active sessions for this user
                sessions_to_remove = [
                    token for token, session in self.active_sessions.items()
                    if session.user_id == user_id
                ]
                for token in sessions_to_remove:
                    del self.active_sessions[token]
                
                # Remove files
                memory_file = self.storage_dir / f"memory_{user_id}.json"
                conv_file = self.storage_dir / f"conversations_{user_id}.json"
                
                if memory_file.exists():
                    memory_file.unlink()
                
                if conv_file.exists():
                    conv_file.unlink()
                
                logger.info(f"Deleted all data for user {user_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to delete data for user {user_id}: {e}")
            return False

# Global instance
_user_memory_manager = None

def get_user_memory_manager(storage_dir: str = ".user_memory") -> UserMemoryManager:
    """Get global user memory manager instance."""
    global _user_memory_manager
    if _user_memory_manager is None:
        _user_memory_manager = UserMemoryManager(storage_dir)
    return _user_memory_manager

if __name__ == "__main__":
    # Test the user memory manager
    manager = UserMemoryManager()
    
    # Create test session
    session_token = manager.create_user_session(
        user_id="test_user",
        oauth_provider="google",
        user_email="test@example.com",
        user_name="Test User"
    )
    
    # Add test conversation
    manager.add_conversation_entry(
        session_token,
        "Hello, how are you?",
        "I'm doing well, thank you for asking!"
    )
    
    # Get stats
    stats = manager.get_user_memory_stats(session_token)
    print("User stats:", json.dumps(stats, indent=2))
    
    # Save data
    manager.save_all_user_data()
    
    print("User memory manager test completed!")