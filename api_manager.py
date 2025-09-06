#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API Manager - Multi-Provider API Key Management
===============================================

Manages API keys for multiple AI providers:
- OpenAI (GPT models)
- Anthropic (Claude models)  
- Google (Gemini models)
- Others (extensible)

Supports multiple loading methods:
- Direct input by user
- Environment variables (.env)
- API container files (AutoGPT style)
- Secure key storage
"""

import os
import json
import logging
import asyncio
import aiohttp
from typing import Dict, List, Optional, Any
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger("APIManager")

class AIProvider(Enum):
    """Supported AI providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    AZURE = "azure"
    COHERE = "cohere"
    HUGGINGFACE = "huggingface"
    LOCAL = "local"

@dataclass
class APIConfiguration:
    """API configuration for a provider."""
    provider: AIProvider
    api_key: Optional[str] = None
    endpoint: Optional[str] = None
    model: Optional[str] = None
    max_tokens: int = 2000
    temperature: float = 0.7
    enabled: bool = False

class APIManager:
    """Manages API keys and configurations for multiple providers."""
    
    def __init__(self, config_dir: str = "."):
        """Initialize API Manager."""
        self.config_dir = Path(config_dir)
        self.configs: Dict[AIProvider, APIConfiguration] = {}
        self.current_provider: Optional[AIProvider] = None
        
        # Default API endpoints
        self.default_endpoints = {
            AIProvider.OPENAI: "https://api.openai.com/v1",
            AIProvider.ANTHROPIC: "https://api.anthropic.com/v1",
            AIProvider.GOOGLE: "https://generativelanguage.googleapis.com/v1",
            AIProvider.AZURE: None,  # Configured per deployment
            AIProvider.COHERE: "https://api.cohere.ai/v1",
            AIProvider.HUGGINGFACE: "https://api-inference.huggingface.co/models",
            AIProvider.LOCAL: "http://localhost:8000"
        }
        
        # Default models
        self.default_models = {
            AIProvider.OPENAI: "gpt-3.5-turbo",
            AIProvider.ANTHROPIC: "claude-3-sonnet-20240229",
            AIProvider.GOOGLE: "gemini-pro",
            AIProvider.AZURE: "gpt-35-turbo",
            AIProvider.COHERE: "command",
            AIProvider.HUGGINGFACE: "microsoft/DialoGPT-medium",
            AIProvider.LOCAL: "llama-2-7b-chat"
        }
        
        self._load_configurations()
    
    def _load_configurations(self):
        """Load API configurations from multiple sources."""
        logger.info("Loading API configurations...")
        
        # 1. Try .env file
        self._load_from_env()
        
        # 2. Try api_keys.json (AutoGPT style)
        self._load_from_json()
        
        # 3. Try environment variables
        self._load_from_environment()
        
        # 4. Try config.json
        self._load_from_config()
        
        logger.info(f"Loaded {len([c for c in self.configs.values() if c.enabled])} active API configurations")
    
    def _load_from_env(self):
        """Load from .env file."""
        env_file = self.config_dir / ".env"
        if env_file.exists():
            logger.info("Loading from .env file...")
            try:
                with open(env_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if '=' in line and not line.startswith('#'):
                            key, value = line.split('=', 1)
                            key, value = key.strip(), value.strip().strip('"\'')
                            self._process_env_var(key, value)
            except Exception as e:
                logger.warning(f"Failed to load .env file: {e}")
    
    def _load_from_json(self):
        """Load from api_keys.json (AutoGPT style)."""
        json_file = self.config_dir / "api_keys.json"
        if json_file.exists():
            logger.info("Loading from api_keys.json...")
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                for provider_name, config in data.items():
                    try:
                        provider = AIProvider(provider_name.lower())
                        api_config = APIConfiguration(
                            provider=provider,
                            api_key=config.get('api_key'),
                            endpoint=config.get('endpoint', self.default_endpoints[provider]),
                            model=config.get('model', self.default_models[provider]),
                            max_tokens=config.get('max_tokens', 2000),
                            temperature=config.get('temperature', 0.7),
                            enabled=config.get('enabled', bool(config.get('api_key')))
                        )
                        self.configs[provider] = api_config
                    except ValueError:
                        logger.warning(f"Unknown provider: {provider_name}")
                        
            except Exception as e:
                logger.warning(f"Failed to load api_keys.json: {e}")
    
    def _load_from_environment(self):
        """Load from environment variables."""
        logger.info("Loading from environment variables...")
        
        env_mappings = {
            'OPENAI_API_KEY': AIProvider.OPENAI,
            'ANTHROPIC_API_KEY': AIProvider.ANTHROPIC,
            'GOOGLE_API_KEY': AIProvider.GOOGLE,
            'AZURE_OPENAI_KEY': AIProvider.AZURE,
            'COHERE_API_KEY': AIProvider.COHERE,
            'HUGGINGFACE_API_TOKEN': AIProvider.HUGGINGFACE
        }
        
        for env_var, provider in env_mappings.items():
            api_key = os.getenv(env_var)
            if api_key:
                self._process_provider_config(provider, api_key)
    
    def _load_from_config(self):
        """Load from config.json."""
        config_file = self.config_dir / "config.json"
        if config_file.exists():
            logger.info("Loading from config.json...")
            try:
                with open(config_file, 'r') as f:
                    data = json.load(f)
                
                api_configs = data.get('api_providers', {})
                for provider_name, config in api_configs.items():
                    try:
                        provider = AIProvider(provider_name.lower())
                        if provider not in self.configs or not self.configs[provider].enabled:
                            api_config = APIConfiguration(
                                provider=provider,
                                api_key=config.get('api_key'),
                                endpoint=config.get('endpoint', self.default_endpoints[provider]),
                                model=config.get('model', self.default_models[provider]),
                                max_tokens=config.get('max_tokens', 2000),
                                temperature=config.get('temperature', 0.7),
                                enabled=config.get('enabled', bool(config.get('api_key')))
                            )
                            self.configs[provider] = api_config
                    except ValueError:
                        logger.warning(f"Unknown provider: {provider_name}")
                        
            except Exception as e:
                logger.warning(f"Failed to load config.json: {e}")
    
    def _process_env_var(self, key: str, value: str):
        """Process environment variable."""
        key_upper = key.upper()
        
        if key_upper == 'OPENAI_API_KEY':
            self._process_provider_config(AIProvider.OPENAI, value)
        elif key_upper == 'ANTHROPIC_API_KEY':
            self._process_provider_config(AIProvider.ANTHROPIC, value)
        elif key_upper == 'GOOGLE_API_KEY':
            self._process_provider_config(AIProvider.GOOGLE, value)
        elif key_upper == 'AZURE_OPENAI_KEY':
            self._process_provider_config(AIProvider.AZURE, value)
        elif key_upper == 'COHERE_API_KEY':
            self._process_provider_config(AIProvider.COHERE, value)
        elif key_upper == 'HUGGINGFACE_API_TOKEN':
            self._process_provider_config(AIProvider.HUGGINGFACE, value)
    
    def _process_provider_config(self, provider: AIProvider, api_key: str):
        """Process provider configuration."""
        if provider not in self.configs or not self.configs[provider].enabled:
            config = APIConfiguration(
                provider=provider,
                api_key=api_key,
                endpoint=self.default_endpoints[provider],
                model=self.default_models[provider],
                enabled=True
            )
            self.configs[provider] = config
            logger.info(f"Configured {provider.value} with API key")
    
    def add_api_key(self, provider: AIProvider, api_key: str, **kwargs) -> bool:
        """Add API key for a provider."""
        try:
            config = APIConfiguration(
                provider=provider,
                api_key=api_key,
                endpoint=kwargs.get('endpoint', self.default_endpoints[provider]),
                model=kwargs.get('model', self.default_models[provider]),
                max_tokens=kwargs.get('max_tokens', 2000),
                temperature=kwargs.get('temperature', 0.7),
                enabled=True
            )
            self.configs[provider] = config
            logger.info(f"Added API key for {provider.value}")
            return True
        except Exception as e:
            logger.error(f"Failed to add API key for {provider.value}: {e}")
            return False
    
    def get_available_providers(self) -> List[AIProvider]:
        """Get list of available providers."""
        return [provider for provider, config in self.configs.items() if config.enabled]
    
    def select_provider(self, provider: AIProvider) -> bool:
        """Select active provider."""
        if provider in self.configs and self.configs[provider].enabled:
            self.current_provider = provider
            logger.info(f"Selected provider: {provider.value}")
            return True
        else:
            logger.warning(f"Provider {provider.value} not available")
            return False
    
    def get_current_config(self) -> Optional[APIConfiguration]:
        """Get current provider configuration."""
        if self.current_provider:
            return self.configs.get(self.current_provider)
        return None
    
    def create_sample_files(self):
        """Create sample configuration files."""
        # Create sample .env
        env_sample = self.config_dir / ".env.sample"
        with open(env_sample, 'w') as f:
            f.write("""# API Keys for AI Providers
# Copy this file to .env and add your actual API keys

# OpenAI
OPENAI_API_KEY=your_openai_key_here

# Anthropic (Claude)
ANTHROPIC_API_KEY=your_anthropic_key_here

# Google (Gemini)
GOOGLE_API_KEY=your_google_key_here

# Azure OpenAI
AZURE_OPENAI_KEY=your_azure_key_here

# Cohere
COHERE_API_KEY=your_cohere_key_here

# HuggingFace
HUGGINGFACE_API_TOKEN=your_huggingface_token_here
""")
        
        # Create sample api_keys.json
        json_sample = self.config_dir / "api_keys.json.sample"
        sample_config = {
            "openai": {
                "api_key": "your_openai_key_here",
                "model": "gpt-3.5-turbo",
                "max_tokens": 2000,
                "temperature": 0.7,
                "enabled": False
            },
            "anthropic": {
                "api_key": "your_anthropic_key_here", 
                "model": "claude-3-sonnet-20240229",
                "max_tokens": 2000,
                "temperature": 0.7,
                "enabled": False
            },
            "google": {
                "api_key": "your_google_key_here",
                "model": "gemini-pro",
                "max_tokens": 2000,
                "temperature": 0.7,
                "enabled": False
            }
        }
        
        with open(json_sample, 'w') as f:
            json.dump(sample_config, f, indent=2)
        
        logger.info("Created sample configuration files")
    
    def interactive_setup(self):
        """Interactive API key setup."""
        print("=== API Manager - Interactive Setup ===")
        print("\nAvailable AI Providers:")
        
        providers = list(AIProvider)
        for i, provider in enumerate(providers, 1):
            status = "CONFIGURED" if provider in self.configs and self.configs[provider].enabled else "NOT CONFIGURED"
            print(f"{i}. {provider.value.title()}: {status}")
        
        print("\nOptions:")
        print("1. Add API key manually")
        print("2. Create sample configuration files")
        print("3. List current configuration")
        print("4. Select active provider")
        print("0. Exit")
        
        choice = input("\nChoice: ").strip()
        
        if choice == "1":
            self._manual_api_setup()
        elif choice == "2":
            self.create_sample_files()
            print("Sample files created: .env.sample and api_keys.json.sample")
        elif choice == "3":
            self._list_configuration()
        elif choice == "4":
            self._select_provider_interactive()
    
    def _manual_api_setup(self):
        """Manual API key setup."""
        print("\nSelect provider to configure:")
        providers = list(AIProvider)
        for i, provider in enumerate(providers, 1):
            print(f"{i}. {provider.value.title()}")
        
        try:
            choice = int(input("Provider number: "))
            if 1 <= choice <= len(providers):
                provider = providers[choice - 1]
                api_key = input(f"Enter API key for {provider.value.title()}: ").strip()
                
                if api_key:
                    if self.add_api_key(provider, api_key):
                        print(f"Successfully configured {provider.value.title()}")
                    else:
                        print(f"Failed to configure {provider.value.title()}")
                else:
                    print("No API key provided")
        except (ValueError, IndexError):
            print("Invalid selection")
    
    def _list_configuration(self):
        """List current configuration."""
        print("\n=== Current Configuration ===")
        
        if not self.configs:
            print("No API configurations found")
            return
        
        for provider, config in self.configs.items():
            status = "ENABLED" if config.enabled else "DISABLED"
            masked_key = f"{config.api_key[:8]}..." if config.api_key else "NOT SET"
            current = " (CURRENT)" if provider == self.current_provider else ""
            
            print(f"{provider.value.title()}: {status}{current}")
            print(f"  API Key: {masked_key}")
            print(f"  Model: {config.model}")
            print(f"  Endpoint: {config.endpoint}")
            print()
    
    def _select_provider_interactive(self):
        """Interactive provider selection."""
        available = self.get_available_providers()
        
        if not available:
            print("No providers configured")
            return
        
        print("\nAvailable providers:")
        for i, provider in enumerate(available, 1):
            current = " (CURRENT)" if provider == self.current_provider else ""
            print(f"{i}. {provider.value.title()}{current}")
        
        try:
            choice = int(input("Select provider: "))
            if 1 <= choice <= len(available):
                provider = available[choice - 1]
                if self.select_provider(provider):
                    print(f"Selected {provider.value.title()}")
                else:
                    print("Failed to select provider")
        except (ValueError, IndexError):
            print("Invalid selection")
    
    def has_provider(self, provider_name: str) -> bool:
        """Check if a provider is configured and enabled."""
        try:
            provider = AIProvider(provider_name.lower())
            return provider in self.configs and self.configs[provider].enabled
        except ValueError:
            return False
    
    def add_provider(self, provider_name: str, api_key: str, **kwargs) -> bool:
        """Add a provider configuration."""
        try:
            provider = AIProvider(provider_name.lower())
            return self.add_api_key(provider, api_key, **kwargs)
        except ValueError:
            logger.error(f"Unknown provider: {provider_name}")
            return False
    
    async def send_message(self, message: str, provider: str = None, model: str = None, **kwargs) -> str:
        """Send message to AI provider."""
        try:
            # Determine provider
            if provider:
                target_provider = AIProvider(provider.lower())
            elif self.current_provider:
                target_provider = self.current_provider
            else:
                # Use first available provider
                available = self.get_available_providers()
                if not available:
                    raise Exception("No AI providers configured")
                target_provider = available[0]
            
            # Get configuration
            if target_provider not in self.configs or not self.configs[target_provider].enabled:
                raise Exception(f"Provider {target_provider.value} not configured or enabled")
            
            config = self.configs[target_provider]
            
            # Use provided model or default
            selected_model = model or config.model
            
            # Call appropriate API
            if target_provider == AIProvider.ANTHROPIC:
                return await self._call_anthropic_api(message, config, selected_model, **kwargs)
            elif target_provider == AIProvider.OPENAI:
                return await self._call_openai_api(message, config, selected_model, **kwargs)
            elif target_provider == AIProvider.GOOGLE:
                return await self._call_google_api(message, config, selected_model, **kwargs)
            else:
                raise Exception(f"Provider {target_provider.value} not implemented yet")
                
        except Exception as e:
            logger.error(f"API call failed: {e}")
            raise
    
    async def _call_anthropic_api(self, message: str, config: APIConfiguration, model: str, **kwargs) -> str:
        """Call Anthropic Claude API."""
        if not config.api_key:
            raise Exception("Anthropic API key not configured")
        
        headers = {
            'Content-Type': 'application/json',
            'x-api-key': config.api_key,
            'anthropic-version': '2023-06-01'
        }
        
        data = {
            'model': model,
            'max_tokens': kwargs.get('max_tokens', config.max_tokens),
            'messages': [
                {
                    'role': 'user',
                    'content': message
                }
            ],
            'temperature': kwargs.get('temperature', config.temperature)
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{config.endpoint}/messages",
                    headers=headers,
                    json=data,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        if result.get('content') and len(result['content']) > 0:
                            return result['content'][0].get('text', 'No response text')
                        else:
                            return "Empty response from Claude"
                    else:
                        error_text = await response.text()
                        logger.error(f"Anthropic API error {response.status}: {error_text}")
                        raise Exception(f"Anthropic API error: {response.status} - {error_text}")
        except asyncio.TimeoutError:
            raise Exception("Anthropic API request timed out")
        except aiohttp.ClientError as e:
            raise Exception(f"Network error calling Anthropic API: {e}")
    
    async def _call_openai_api(self, message: str, config: APIConfiguration, model: str, **kwargs) -> str:
        """Call OpenAI API."""
        if not config.api_key:
            raise Exception("OpenAI API key not configured")
        
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {config.api_key}'
        }
        
        data = {
            'model': model,
            'messages': [
                {
                    'role': 'user',
                    'content': message
                }
            ],
            'max_tokens': kwargs.get('max_tokens', config.max_tokens),
            'temperature': kwargs.get('temperature', config.temperature)
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{config.endpoint}/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        if result.get('choices') and len(result['choices']) > 0:
                            return result['choices'][0]['message']['content']
                        else:
                            return "Empty response from OpenAI"
                    else:
                        error_text = await response.text()
                        logger.error(f"OpenAI API error {response.status}: {error_text}")
                        raise Exception(f"OpenAI API error: {response.status} - {error_text}")
        except asyncio.TimeoutError:
            raise Exception("OpenAI API request timed out")
        except aiohttp.ClientError as e:
            raise Exception(f"Network error calling OpenAI API: {e}")
    
    async def _call_google_api(self, message: str, config: APIConfiguration, model: str, **kwargs) -> str:
        """Call Google Gemini API."""
        if not config.api_key:
            raise Exception("Google API key not configured")
        
        data = {
            'contents': [{
                'parts': [{
                    'text': message
                }]
            }],
            'generationConfig': {
                'maxOutputTokens': kwargs.get('max_tokens', config.max_tokens),
                'temperature': kwargs.get('temperature', config.temperature)
            }
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{config.endpoint}/models/{model}:generateContent?key={config.api_key}",
                    json=data,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        if result.get('candidates') and len(result['candidates']) > 0:
                            candidate = result['candidates'][0]
                            if candidate.get('content') and candidate['content'].get('parts'):
                                return candidate['content']['parts'][0]['text']
                        return "Empty response from Google"
                    else:
                        error_text = await response.text()
                        logger.error(f"Google API error {response.status}: {error_text}")
                        raise Exception(f"Google API error: {response.status} - {error_text}")
        except asyncio.TimeoutError:
            raise Exception("Google API request timed out")
        except aiohttp.ClientError as e:
            raise Exception(f"Network error calling Google API: {e}")

# Global instance
_api_manager = None

def get_api_manager(config_dir: str = ".") -> APIManager:
    """Get global API manager instance."""
    global _api_manager
    if _api_manager is None:
        _api_manager = APIManager(config_dir)
    return _api_manager

if __name__ == "__main__":
    # Interactive setup
    manager = APIManager()
    manager.interactive_setup()