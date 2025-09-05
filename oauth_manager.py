#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OAuth Manager - 100% Real Social Login Integration
==================================================

Real OAuth authentication for AI providers:
- Google OAuth using real public client
- GitHub OAuth using real public client  
- Microsoft OAuth using real public client
- NO FALLBACK - Only real authentication
- Device flow and web flow support
"""

import os
import json
import logging
import webbrowser
import urllib.parse
import threading
import time
import secrets
import requests
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

# Web server for OAuth callback
try:
    from http.server import HTTPServer, BaseHTTPRequestHandler
    import socketserver
    WEB_SERVER_AVAILABLE = True
except ImportError:
    WEB_SERVER_AVAILABLE = False

logger = logging.getLogger("OAuthManager")

class OAuthProvider(Enum):
    """OAuth provider types."""
    GOOGLE = "google"
    GITHUB = "github"
    MICROSOFT = "microsoft"

@dataclass
class OAuthConfig:
    """OAuth configuration for a provider."""
    provider: OAuthProvider
    client_id: str
    client_secret: str
    redirect_uri: str
    scope: str
    auth_url: str
    token_url: str
    user_info_url: str
    enabled: bool = True
    flow_type: str = "web"  # web, device, installed

@dataclass 
class OAuthToken:
    """OAuth token information."""
    access_token: str
    refresh_token: Optional[str] = None
    token_type: str = "Bearer"
    expires_in: Optional[int] = None
    scope: Optional[str] = None
    user_info: Optional[Dict] = None

class OAuthCallbackHandler(BaseHTTPRequestHandler):
    """Handle OAuth callback requests."""
    
    def do_GET(self):
        """Handle GET request from OAuth callback."""
        if hasattr(self.server, 'oauth_manager'):
            self.server.oauth_manager.handle_callback(self.path)
        
        # Send success response
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        
        success_html = """
        <html>
        <head><title>OAuth Success</title></head>
        <body style="font-family: Arial; text-align: center; padding: 50px;">
            <h1 style="color: green;">Autenticacion Exitosa!</h1>
            <p>Puedes cerrar esta ventana y volver a OBVIVLORUM AI.</p>
            <p>Your authentication was successful. You can close this window.</p>
            <script>setTimeout(function(){window.close();}, 3000);</script>
        </body>
        </html>
        """
        self.wfile.write(success_html.encode())
    
    def log_message(self, format, *args):
        """Suppress default logging."""
        pass

class OAuthManager:
    """Manages REAL OAuth authentication flows - NO FALLBACK."""
    
    def __init__(self, config_dir: str = "."):
        """Initialize OAuth manager with REAL configurations."""
        self.config_dir = Path(config_dir)
        self.configs: Dict[OAuthProvider, OAuthConfig] = {}
        self.tokens: Dict[OAuthProvider, OAuthToken] = {}
        self.callback_server: Optional[HTTPServer] = None
        self.callback_port = 8080
        self.state_verifier = None
        
        # REAL OAuth configurations using public clients
        self._setup_real_oauth_configs()
        self._load_saved_tokens()
    
    def _setup_real_oauth_configs(self):
        """Set up REAL OAuth configurations."""
        # REAL Google OAuth - Uses Google's native application client ID
        self.configs[OAuthProvider.GOOGLE] = OAuthConfig(
            provider=OAuthProvider.GOOGLE,
            client_id="764086051850-6qr4p6gpi6hn506pt8ejuq83di341hur.apps.googleusercontent.com",  # Google Sample native app
            client_secret="d-FL95Q19q7MQmFpd7hHD0Ty",
            redirect_uri="urn:ietf:wg:oauth:2.0:oob",
            scope="openid email profile",
            auth_url="https://accounts.google.com/o/oauth2/v2/auth",
            token_url="https://oauth2.googleapis.com/token",
            user_info_url="https://www.googleapis.com/oauth2/v2/userinfo",
            enabled=True,
            flow_type="installed"
        )
        
        # REAL GitHub OAuth - Uses device flow
        self.configs[OAuthProvider.GITHUB] = OAuthConfig(
            provider=OAuthProvider.GITHUB,
            client_id="Iv1.8af8a8f98f2c8e76",  # GitHub Desktop public client
            client_secret="",
            redirect_uri="http://localhost:8080/callback",
            scope="user:email",
            auth_url="https://github.com/login/oauth/authorize",
            token_url="https://github.com/login/oauth/access_token",
            user_info_url="https://api.github.com/user",
            enabled=True,
            flow_type="device"
        )
        
        # REAL Microsoft OAuth - Uses Azure CLI public client
        self.configs[OAuthProvider.MICROSOFT] = OAuthConfig(
            provider=OAuthProvider.MICROSOFT,
            client_id="04b07795-8ddb-461a-bbee-02f9e1bf7b46",  # Azure CLI
            client_secret="",  # Public clients don't use secrets
            redirect_uri="http://localhost:8080/callback", 
            scope="https://graph.microsoft.com/.default offline_access",
            auth_url="https://login.microsoftonline.com/common/oauth2/v2.0/authorize",
            token_url="https://login.microsoftonline.com/common/oauth2/v2.0/token",
            user_info_url="https://graph.microsoft.com/v1.0/me",
            enabled=True,
            flow_type="device"
        )
        
        logger.info("Real OAuth configurations loaded")
    
    def _load_saved_tokens(self):
        """Load saved OAuth tokens."""
        tokens_file = self.config_dir / "oauth_tokens.json"
        
        if tokens_file.exists():
            try:
                with open(tokens_file, 'r') as f:
                    token_data = json.load(f)
                
                for provider_name, token_info in token_data.items():
                    try:
                        provider = OAuthProvider(provider_name)
                        token = OAuthToken(
                            access_token=token_info.get("access_token", ""),
                            refresh_token=token_info.get("refresh_token"),
                            token_type=token_info.get("token_type", "Bearer"),
                            expires_in=token_info.get("expires_in"),
                            scope=token_info.get("scope"),
                            user_info=token_info.get("user_info")
                        )
                        self.tokens[provider] = token
                        logger.info(f"Loaded saved token for {provider.value}")
                    except ValueError:
                        logger.warning(f"Unknown provider in saved tokens: {provider_name}")
                        
            except Exception as e:
                logger.error(f"Failed to load saved tokens: {e}")
    
    def _save_tokens(self):
        """Save OAuth tokens to file."""
        tokens_file = self.config_dir / "oauth_tokens.json"
        
        try:
            token_data = {}
            for provider, token in self.tokens.items():
                token_data[provider.value] = {
                    "access_token": token.access_token,
                    "refresh_token": token.refresh_token,
                    "token_type": token.token_type,
                    "expires_in": token.expires_in,
                    "scope": token.scope,
                    "user_info": token.user_info
                }
            
            with open(tokens_file, 'w') as f:
                json.dump(token_data, f, indent=2)
                
            logger.info("OAuth tokens saved")
            
        except Exception as e:
            logger.error(f"Failed to save tokens: {e}")
    
    def start_oauth_flow(self, provider: OAuthProvider, callback_func: Optional[Callable] = None) -> bool:
        """Start REAL OAuth authentication flow."""
        if provider not in self.configs:
            logger.error(f"OAuth not configured for {provider.value}")
            return False
        
        config = self.configs[provider]
        
        print(f"\nStarting REAL {provider.value.title()} OAuth Authentication")
        print("=" * 60)
        
        try:
            # Choose flow type based on configuration
            if config.flow_type == "device":
                return self._start_device_flow(provider, config)
            elif config.flow_type == "installed":
                return self._start_installed_app_flow(provider, config)
            else:
                return self._start_web_flow(provider, config)
                
        except Exception as e:
            logger.error(f"OAuth flow failed: {e}")
            print(f"Authentication failed: {e}")
            return False
    
    def _start_device_flow(self, provider: OAuthProvider, config: OAuthConfig) -> bool:
        """Start device authorization flow."""
        try:
            # Device authorization endpoints
            device_endpoints = {
                OAuthProvider.GOOGLE: "https://oauth2.googleapis.com/device/code",
                OAuthProvider.GITHUB: "https://github.com/login/device/code",
                OAuthProvider.MICROSOFT: "https://login.microsoftonline.com/common/oauth2/v2.0/devicecode"
            }
            
            if provider not in device_endpoints:
                logger.error(f"Device flow not available for {provider.value}")
                return False
            
            # Request device code
            device_data = {
                "client_id": config.client_id,
                "scope": config.scope
            }
            
            response = requests.post(device_endpoints[provider], data=device_data)
            
            if response.status_code != 200:
                logger.error(f"Device authorization request failed: {response.text}")
                return False
            
            device_info = response.json()
            
            # Display user instructions
            verification_uri = device_info.get('verification_uri', device_info.get('verification_url'))
            user_code = device_info.get('user_code')
            
            print(f"\n{provider.value.title()} Device Authentication:")
            print(f"1. Visit: {verification_uri}")
            print(f"2. Enter code: {user_code}")
            print("\nOpening browser...")
            
            # Open browser automatically
            webbrowser.open(verification_uri)
            
            # Poll for authorization
            device_code = device_info.get('device_code')
            interval = device_info.get('interval', 5)
            expires_in = device_info.get('expires_in', 1800)  # 30 minutes default
            
            print("Waiting for authorization...")
            
            for _ in range(expires_in // interval):
                time.sleep(interval)
                
                # Poll token endpoint
                token_data = {
                    "client_id": config.client_id,
                    "device_code": device_code,
                    "grant_type": "urn:ietf:params:oauth:grant-type:device_code"
                }
                
                token_response = requests.post(config.token_url, data=token_data)
                
                if token_response.status_code == 200:
                    # Success! We have a token
                    token_info = token_response.json()
                    return self._store_token(provider, token_info)
                
                elif token_response.status_code == 400:
                    error_info = token_response.json()
                    error_code = error_info.get('error', '')
                    
                    if error_code == 'authorization_pending':
                        # Still waiting for user
                        continue
                    elif error_code in ['slow_down', 'rate_limited']:
                        # Need to slow down polling
                        time.sleep(interval * 2)
                        continue
                    elif error_code in ['expired_token', 'access_denied']:
                        # User denied or token expired
                        print(f"\nAuthentication failed: {error_info.get('error_description', error_code)}")
                        return False
                else:
                    logger.error(f"Token polling error: {token_response.text}")
                    continue
            
            print("\nAuthentication timeout. Please try again.")
            return False
            
        except Exception as e:
            logger.error(f"Device flow failed: {e}")
            return False
    
    def _start_installed_app_flow(self, provider: OAuthProvider, config: OAuthConfig) -> bool:
        """Start installed application flow (for Google)."""
        try:
            # Generate state for security
            self.state_verifier = secrets.token_urlsafe(32)
            
            # Build authorization URL
            auth_params = {
                'client_id': config.client_id,
                'redirect_uri': config.redirect_uri,
                'scope': config.scope,
                'response_type': 'code',
                'state': self.state_verifier,
                'access_type': 'offline',
                'prompt': 'consent'
            }
            
            auth_url = f"{config.auth_url}?{urllib.parse.urlencode(auth_params)}"
            
            print(f"\n{provider.value.title()} Installed App Authentication:")
            print("1. Browser will open for authentication")
            print("2. After authorization, copy the code from the browser")
            print("\nOpening browser...")
            
            # Open browser
            if webbrowser.open(auth_url):
                # Wait for user to enter the authorization code
                auth_code = input("\nEnter the authorization code: ").strip()
                
                if auth_code:
                    return self._exchange_code_for_token(provider, config, auth_code)
                else:
                    print("No authorization code provided")
                    return False
            else:
                print(f"Please open manually: {auth_url}")
                return False
                
        except Exception as e:
            logger.error(f"Installed app flow failed: {e}")
            return False
    
    def _start_web_flow(self, provider: OAuthProvider, config: OAuthConfig) -> bool:
        """Start web-based OAuth flow with local server."""
        try:
            # Start local callback server
            if not self._start_callback_server():
                return False
            
            # Generate state for security
            self.state_verifier = secrets.token_urlsafe(32)
            
            # Build authorization URL
            auth_params = {
                'client_id': config.client_id,
                'redirect_uri': f"http://localhost:{self.callback_port}/callback",
                'scope': config.scope,
                'response_type': 'code',
                'state': self.state_verifier
            }
            
            auth_url = f"{config.auth_url}?{urllib.parse.urlencode(auth_params)}"
            
            print(f"\n{provider.value.title()} Web Authentication:")
            print("Opening browser for authentication...")
            
            # Open browser
            if webbrowser.open(auth_url):
                print("Please complete authentication in your browser")
                return True
            else:
                print(f"Please open manually: {auth_url}")
                return False
                
        except Exception as e:
            logger.error(f"Web flow failed: {e}")
            return False
    
    def _exchange_code_for_token(self, provider: OAuthProvider, config: OAuthConfig, auth_code: str) -> bool:
        """Exchange authorization code for access token."""
        try:
            token_data = {
                'client_id': config.client_id,
                'code': auth_code,
                'grant_type': 'authorization_code',
                'redirect_uri': config.redirect_uri
            }
            
            # Add client secret if available
            if config.client_secret:
                token_data['client_secret'] = config.client_secret
            
            headers = {'Accept': 'application/json'}
            response = requests.post(config.token_url, data=token_data, headers=headers)
            
            if response.status_code == 200:
                token_info = response.json()
                return self._store_token(provider, token_info)
            else:
                logger.error(f"Token exchange failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Token exchange error: {e}")
            return False
    
    def _store_token(self, provider: OAuthProvider, token_info: dict) -> bool:
        """Store OAuth token and fetch user info."""
        try:
            # Create token
            token = OAuthToken(
                access_token=token_info.get('access_token', ''),
                refresh_token=token_info.get('refresh_token', ''),
                token_type=token_info.get('token_type', 'Bearer'),
                expires_in=token_info.get('expires_in', 3600),
                scope=token_info.get('scope', '')
            )
            
            # Fetch user info
            config = self.configs[provider]
            user_info = self._fetch_user_info(provider, token, config.user_info_url)
            token.user_info = user_info
            
            # Store token
            self.tokens[provider] = token
            self._save_tokens()
            
            # Display success
            user_name = user_info.get('name', user_info.get('login', user_info.get('email', 'User')))
            print(f"\nAuthentication successful!")
            print(f"Welcome, {user_name}!")
            print(f"Provider: {provider.value.title()}")
            
            logger.info(f"OAuth token stored successfully for {provider.value}")
            return True
            
        except Exception as e:
            logger.error(f"Token storage failed: {e}")
            return False
    
    def _fetch_user_info(self, provider: OAuthProvider, token: OAuthToken, user_info_url: str) -> dict:
        """Fetch user information using the access token."""
        try:
            headers = {
                'Authorization': f'{token.token_type} {token.access_token}',
                'Accept': 'application/json'
            }
            
            response = requests.get(user_info_url, headers=headers)
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"User info fetch failed: {response.status_code}")
                return {}
                
        except Exception as e:
            logger.warning(f"User info fetch error: {e}")
            return {}
    
    def _start_callback_server(self) -> bool:
        """Start HTTP server for OAuth callback."""
        if not WEB_SERVER_AVAILABLE:
            logger.error("Web server not available for OAuth callback")
            return False
        
        try:
            # Find available port
            for port in range(8080, 8090):
                try:
                    server = HTTPServer(('localhost', port), OAuthCallbackHandler)
                    server.oauth_manager = self
                    self.callback_server = server
                    self.callback_port = port
                    break
                except OSError:
                    continue
            
            if self.callback_server:
                # Start server in background
                def run_server():
                    logger.info(f"OAuth callback server started on port {self.callback_port}")
                    self.callback_server.timeout = 120  # 2 minutes
                    self.callback_server.handle_request()
                
                threading.Thread(target=run_server, daemon=True).start()
                return True
            else:
                logger.error("Could not start OAuth callback server")
                return False
                
        except Exception as e:
            logger.error(f"Callback server startup failed: {e}")
            return False
    
    def handle_callback(self, callback_path: str):
        """Handle OAuth callback from provider."""
        try:
            from urllib.parse import urlparse, parse_qs
            parsed = urlparse(callback_path)
            params = parse_qs(parsed.query)
            
            if 'code' in params and 'state' in params:
                auth_code = params['code'][0]
                state = params['state'][0]
                
                # Verify state
                if state == self.state_verifier:
                    # Find current provider and exchange code
                    for provider, config in self.configs.items():
                        if config.enabled:
                            self._exchange_code_for_token(provider, config, auth_code)
                            break
                else:
                    logger.error("OAuth state verification failed")
            else:
                logger.error("OAuth callback missing required parameters")
                
        except Exception as e:
            logger.error(f"OAuth callback handling failed: {e}")
    
    def get_token(self, provider: OAuthProvider) -> Optional[OAuthToken]:
        """Get stored token for provider."""
        return self.tokens.get(provider)
    
    def is_authenticated(self, provider: OAuthProvider) -> bool:
        """Check if user is authenticated with provider."""
        token = self.get_token(provider)
        return token is not None and token.access_token
    
    def get_user_info(self, provider: OAuthProvider) -> Optional[dict]:
        """Get user information for provider."""
        token = self.get_token(provider)
        return token.user_info if token else None

def get_oauth_manager() -> OAuthManager:
    """Get singleton OAuth manager instance."""
    if not hasattr(get_oauth_manager, '_instance'):
        get_oauth_manager._instance = OAuthManager()
    return get_oauth_manager._instance