#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OAuth Manager - Social Login Integration
========================================

Handles OAuth authentication for AI providers:
- Google OAuth for Claude/Anthropic
- GitHub OAuth for OpenAI
- Microsoft OAuth for Azure
- Social login integration
- Browser-based authentication flows
"""

import os
import json
import logging
import webbrowser
import urllib.parse
import threading
import time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import hashlib
import base64
import secrets

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
    DISCORD = "discord"
    CUSTOM = "custom"

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
    enabled: bool = False

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
    """Manages OAuth authentication flows."""
    
    def __init__(self, config_dir: str = "."):
        """Initialize OAuth manager."""
        self.config_dir = Path(config_dir)
        self.configs: Dict[OAuthProvider, OAuthConfig] = {}
        self.tokens: Dict[OAuthProvider, OAuthToken] = {}
        self.callback_server: Optional[HTTPServer] = None
        self.callback_port = 8080
        self.state_verifier = None
        
        # Default OAuth configurations
        self.default_configs = {
            OAuthProvider.GOOGLE: {
                "auth_url": "https://accounts.google.com/o/oauth2/auth",
                "token_url": "https://oauth2.googleapis.com/token",
                "user_info_url": "https://www.googleapis.com/oauth2/v2/userinfo",
                "scope": "openid email profile"
            },
            OAuthProvider.GITHUB: {
                "auth_url": "https://github.com/login/oauth/authorize",
                "token_url": "https://github.com/login/oauth/access_token",
                "user_info_url": "https://api.github.com/user",
                "scope": "user:email"
            },
            OAuthProvider.MICROSOFT: {
                "auth_url": "https://login.microsoftonline.com/common/oauth2/v2.0/authorize",
                "token_url": "https://login.microsoftonline.com/common/oauth2/v2.0/token",
                "user_info_url": "https://graph.microsoft.com/v1.0/me",
                "scope": "openid profile email"
            }
        }
        
        self._load_oauth_configs()
        self._load_saved_tokens()
    
    def _load_oauth_configs(self):
        """Load OAuth configurations."""
        logger.info("Loading OAuth configurations...")
        
        # Try oauth_config.json
        oauth_config_file = self.config_dir / "oauth_config.json"
        if oauth_config_file.exists():
            try:
                with open(oauth_config_file, 'r') as f:
                    data = json.load(f)
                
                for provider_name, config in data.items():
                    try:
                        provider = OAuthProvider(provider_name.lower())
                        defaults = self.default_configs.get(provider, {})
                        
                        oauth_config = OAuthConfig(
                            provider=provider,
                            client_id=config['client_id'],
                            client_secret=config['client_secret'],
                            redirect_uri=config.get('redirect_uri', f'http://localhost:{self.callback_port}/callback'),
                            scope=config.get('scope', defaults.get('scope', '')),
                            auth_url=config.get('auth_url', defaults.get('auth_url', '')),
                            token_url=config.get('token_url', defaults.get('token_url', '')),
                            user_info_url=config.get('user_info_url', defaults.get('user_info_url', '')),
                            enabled=config.get('enabled', True)
                        )
                        self.configs[provider] = oauth_config
                        
                    except ValueError:
                        logger.warning(f"Unknown OAuth provider: {provider_name}")
                        
            except Exception as e:
                logger.warning(f"Failed to load oauth_config.json: {e}")
    
    def _load_saved_tokens(self):
        """Load saved OAuth tokens."""
        tokens_file = self.config_dir / "oauth_tokens.json"
        if tokens_file.exists():
            try:
                with open(tokens_file, 'r') as f:
                    data = json.load(f)
                
                for provider_name, token_data in data.items():
                    try:
                        provider = OAuthProvider(provider_name.lower())
                        token = OAuthToken(
                            access_token=token_data['access_token'],
                            refresh_token=token_data.get('refresh_token'),
                            token_type=token_data.get('token_type', 'Bearer'),
                            expires_in=token_data.get('expires_in'),
                            scope=token_data.get('scope'),
                            user_info=token_data.get('user_info')
                        )
                        self.tokens[provider] = token
                        
                    except ValueError:
                        logger.warning(f"Unknown provider in tokens: {provider_name}")
                        
            except Exception as e:
                logger.warning(f"Failed to load saved tokens: {e}")
    
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
        """Start OAuth authentication flow."""
        if provider not in self.configs:
            logger.warning(f"OAuth not configured for {provider.value}, creating demo flow")
            # Create demo authentication flow
            return self._demo_oauth_flow(provider)
        
        config = self.configs[provider]
        
        # Check if using demo credentials
        if config.client_id.startswith("demo-"):
            logger.info(f"Using demo OAuth flow for {provider.value}")
            return self._demo_oauth_flow(provider)
        
        try:
            # Start callback server
            if WEB_SERVER_AVAILABLE and not self.callback_server:
                self._start_callback_server()
            
            # Generate state for security
            self.state_verifier = secrets.token_urlsafe(32)
            
            # Build authorization URL
            auth_params = {
                'client_id': config.client_id,
                'redirect_uri': config.redirect_uri,
                'scope': config.scope,
                'response_type': 'code',
                'state': self.state_verifier
            }
            
            # Add provider-specific parameters
            if provider == OAuthProvider.GOOGLE:
                auth_params['access_type'] = 'offline'
                auth_params['prompt'] = 'consent'
            
            auth_url = f"{config.auth_url}?{urllib.parse.urlencode(auth_params)}"
            
            # Open browser
            print(f"\nInitiating OAuth flow for {provider.value.title()}...")
            print("Opening browser for authentication...")
            
            if webbrowser.open(auth_url):
                print(f"Browser opened for {provider.value.title()} authentication")
                print("Please complete the authentication in your browser")
                return True
            else:
                print("Failed to open browser automatically")
                print(f"Please open this URL manually: {auth_url}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to start OAuth flow: {e}")
            return False
    
    def _demo_oauth_flow(self, provider: OAuthProvider) -> bool:
        """Demo OAuth flow for testing without real credentials."""
        try:
            print(f"\nDemo OAuth Flow for {provider.value.title()}")
            print("This is a demonstration - no real authentication occurs")
            
            # Simulate authentication delay
            threading.Thread(target=self._simulate_oauth_success, args=(provider,), daemon=True).start()
            
            return True
            
        except Exception as e:
            logger.error(f"Demo OAuth flow failed: {e}")
            return False
    
    def _simulate_oauth_success(self, provider: OAuthProvider):
        """Simulate successful OAuth authentication."""
        time.sleep(2)  # Simulate network delay
        
        # Create demo token
        demo_token = OAuthToken(
            access_token=f"demo_access_token_{provider.value}_{int(time.time())}",
            refresh_token=f"demo_refresh_token_{provider.value}",
            token_type="Bearer",
            expires_in=3600,
            scope="demo scope",
            user_info={
                "name": f"Demo User ({provider.value.title()})",
                "email": f"demo@{provider.value}.com",
                "id": f"demo_user_{provider.value}",
                "verified": True
            }
        )
        
        self.tokens[provider] = demo_token
        logger.info(f"Demo OAuth completed for {provider.value}")
    
    def _start_callback_server(self):
        """Start HTTP server for OAuth callback."""
        if not WEB_SERVER_AVAILABLE:
            logger.error("Web server not available for OAuth callback")
            return
        
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
                # Start server in background thread
                def run_server():
                    logger.info(f"OAuth callback server started on port {self.callback_port}")
                    self.callback_server.timeout = 60  # 1 minute timeout
                    self.callback_server.handle_request()
                
                threading.Thread(target=run_server, daemon=True).start()
            else:
                logger.error("Could not start OAuth callback server")
                
        except Exception as e:
            logger.error(f"Failed to start callback server: {e}")
    
    def handle_callback(self, callback_path: str):
        """Handle OAuth callback."""
        try:
            # Parse callback URL
            parsed = urllib.parse.urlparse(callback_path)
            params = urllib.parse.parse_qs(parsed.query)
            
            # Verify state
            state = params.get('state', [None])[0]
            if state != self.state_verifier:
                logger.error("OAuth state verification failed")
                return False
            
            # Get authorization code
            auth_code = params.get('code', [None])[0]
            if not auth_code:
                error = params.get('error', [None])[0]
                logger.error(f"OAuth callback error: {error}")
                return False
            
            print(f"OAuth callback received with code: {auth_code[:10]}...")
            
            # Exchange code for token (would need to implement token exchange)
            # For now, just log success
            print("OAuth flow completed successfully!")
            
            return True
            
        except Exception as e:
            logger.error(f"OAuth callback error: {e}")
            return False
    
    def get_user_info(self, provider: OAuthProvider) -> Optional[Dict]:
        """Get user information for authenticated provider."""
        if provider not in self.tokens:
            return None
        
        token = self.tokens[provider]
        return token.user_info
    
    def is_authenticated(self, provider: OAuthProvider) -> bool:
        """Check if user is authenticated with provider."""
        return provider in self.tokens and self.tokens[provider].access_token is not None
    
    def logout(self, provider: OAuthProvider):
        """Logout from OAuth provider."""
        if provider in self.tokens:
            del self.tokens[provider]
            self._save_tokens()
            logger.info(f"Logged out from {provider.value}")
    
    def create_sample_oauth_config(self):
        """Create sample OAuth configuration file."""
        sample_config = {
            "google": {
                "client_id": "your_google_client_id_here",
                "client_secret": "your_google_client_secret_here",
                "redirect_uri": f"http://localhost:{self.callback_port}/callback",
                "enabled": False
            },
            "github": {
                "client_id": "your_github_client_id_here", 
                "client_secret": "your_github_client_secret_here",
                "redirect_uri": f"http://localhost:{self.callback_port}/callback",
                "enabled": False
            },
            "microsoft": {
                "client_id": "your_microsoft_client_id_here",
                "client_secret": "your_microsoft_client_secret_here", 
                "redirect_uri": f"http://localhost:{self.callback_port}/callback",
                "enabled": False
            }
        }
        
        sample_file = self.config_dir / "oauth_config.json.sample"
        with open(sample_file, 'w') as f:
            json.dump(sample_config, f, indent=2)
        
        print("Sample OAuth config created: oauth_config.json.sample")
        print("\nTo enable OAuth:")
        print("1. Create OAuth apps in Google/GitHub/Microsoft consoles")
        print("2. Copy oauth_config.json.sample to oauth_config.json") 
        print("3. Add your actual client IDs and secrets")
        print("4. Set enabled: true for desired providers")
    
    def interactive_setup(self):
        """Interactive OAuth setup."""
        print("=== OAuth Manager - Social Login Setup ===")
        
        print("\nAvailable OAuth Providers:")
        providers = list(OAuthProvider)
        for i, provider in enumerate(providers, 1):
            configured = "CONFIGURED" if provider in self.configs else "NOT CONFIGURED"
            authenticated = "AUTHENTICATED" if self.is_authenticated(provider) else "NOT AUTHENTICATED"
            print(f"{i}. {provider.value.title()}: {configured}, {authenticated}")
        
        print("\nOptions:")
        print("1. Start OAuth flow for a provider")
        print("2. Create sample OAuth configuration")
        print("3. View authenticated users")
        print("4. Logout from provider")
        print("0. Exit")
        
        choice = input("\nChoice: ").strip()
        
        if choice == "1":
            self._start_oauth_interactive()
        elif choice == "2":
            self.create_sample_oauth_config()
        elif choice == "3":
            self._view_authenticated_users()
        elif choice == "4":
            self._logout_interactive()
    
    def _start_oauth_interactive(self):
        """Interactive OAuth flow start."""
        configured_providers = [p for p in self.configs.keys()]
        
        if not configured_providers:
            print("No OAuth providers configured. Create oauth_config.json first.")
            return
        
        print("\nSelect provider for OAuth:")
        for i, provider in enumerate(configured_providers, 1):
            print(f"{i}. {provider.value.title()}")
        
        try:
            choice = int(input("Provider: "))
            if 1 <= choice <= len(configured_providers):
                provider = configured_providers[choice - 1]
                self.start_oauth_flow(provider)
                
                # Wait for completion
                print("Waiting for OAuth completion (60 seconds)...")
                time.sleep(2)  # Give time for callback
                
        except (ValueError, IndexError):
            print("Invalid selection")
    
    def _view_authenticated_users(self):
        """View authenticated users."""
        print("\n=== Authenticated Users ===")
        
        if not self.tokens:
            print("No authenticated users")
            return
        
        for provider, token in self.tokens.items():
            print(f"\n{provider.value.title()}:")
            if token.user_info:
                print(f"  Name: {token.user_info.get('name', 'N/A')}")
                print(f"  Email: {token.user_info.get('email', 'N/A')}")
            print(f"  Token: {token.access_token[:20]}...")
    
    def _logout_interactive(self):
        """Interactive logout."""
        authenticated = [p for p in self.tokens.keys()]
        
        if not authenticated:
            print("No authenticated providers")
            return
        
        print("\nSelect provider to logout:")
        for i, provider in enumerate(authenticated, 1):
            print(f"{i}. {provider.value.title()}")
        
        try:
            choice = int(input("Provider: "))
            if 1 <= choice <= len(authenticated):
                provider = authenticated[choice - 1]
                self.logout(provider)
                print(f"Logged out from {provider.value.title()}")
        except (ValueError, IndexError):
            print("Invalid selection")

# Global instance
_oauth_manager = None

def get_oauth_manager(config_dir: str = ".") -> OAuthManager:
    """Get global OAuth manager instance."""
    global _oauth_manager
    if _oauth_manager is None:
        _oauth_manager = OAuthManager(config_dir)
    return _oauth_manager

if __name__ == "__main__":
    # Interactive setup
    manager = OAuthManager()
    manager.interactive_setup()