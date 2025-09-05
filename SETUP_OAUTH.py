#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OAuth Setup Assistant
====================

Helps users set up OAuth authentication for OBVIVLORUM AI system.
Provides step-by-step instructions for registering OAuth applications
with Google, GitHub, and Microsoft.
"""

import json
import os
import webbrowser
from pathlib import Path

def print_banner():
    """Print setup banner."""
    print("\n" + "=" * 60)
    print("OBVIVLORUM AI - OAuth Setup Assistant")
    print("=" * 60)
    print("Configure OAuth authentication for AI providers")
    print()

def setup_google_oauth():
    """Guide for Google OAuth setup."""
    print("GOOGLE OAUTH SETUP")
    print("-" * 30)
    print("1. Go to Google Cloud Console: https://console.cloud.google.com/")
    print("2. Create a new project or select existing project")
    print("3. Enable the Google+ API and Google OAuth2 API")
    print("4. Go to 'Credentials' -> 'Create Credentials' -> 'OAuth client ID'")
    print("5. Select 'Desktop application' as application type")
    print("6. Add authorized redirect URI: http://localhost:8080/callback")
    print("7. Copy the Client ID and Client Secret")
    print()
    
    open_browser = input("Open Google Cloud Console now? (y/N): ").lower().strip()
    if open_browser == 'y':
        webbrowser.open("https://console.cloud.google.com/apis/credentials")
    
    client_id = input("Enter your Google Client ID: ").strip()
    client_secret = input("Enter your Google Client Secret: ").strip()
    
    return client_id, client_secret

def setup_github_oauth():
    """Guide for GitHub OAuth setup."""
    print("GITHUB OAUTH SETUP")
    print("-" * 30)
    print("1. Go to GitHub Settings: https://github.com/settings/developers")
    print("2. Click 'New OAuth App'")
    print("3. Fill in application details:")
    print("   - Application name: OBVIVLORUM AI")
    print("   - Homepage URL: https://github.com/Yatrogenesis/Obvivlorum")
    print("   - Authorization callback URL: http://localhost:8080/callback")
    print("4. Copy the Client ID and generate/copy Client Secret")
    print()
    
    open_browser = input("Open GitHub Developer Settings now? (y/N): ").lower().strip()
    if open_browser == 'y':
        webbrowser.open("https://github.com/settings/developers")
    
    client_id = input("Enter your GitHub Client ID: ").strip()
    client_secret = input("Enter your GitHub Client Secret: ").strip()
    
    return client_id, client_secret

def setup_microsoft_oauth():
    """Guide for Microsoft OAuth setup."""
    print("MICROSOFT OAUTH SETUP")
    print("-" * 30)
    print("1. Go to Azure App Registrations: https://portal.azure.com/#blade/Microsoft_AAD_RegisteredApps")
    print("2. Click 'New registration'")
    print("3. Fill in application details:")
    print("   - Name: OBVIVLORUM AI")
    print("   - Supported account types: Accounts in any organizational directory and personal Microsoft accounts")
    print("   - Redirect URI: Web -> http://localhost:8080/callback")
    print("4. Go to 'Certificates & secrets' and create new client secret")
    print("5. Copy Application (client) ID and the client secret value")
    print()
    
    open_browser = input("Open Azure Portal now? (y/N): ").lower().strip()
    if open_browser == 'y':
        webbrowser.open("https://portal.azure.com/#blade/Microsoft_AAD_RegisteredApps/ApplicationsListBlade")
    
    client_id = input("Enter your Microsoft Application (client) ID: ").strip()
    client_secret = input("Enter your Microsoft Client Secret: ").strip()
    
    return client_id, client_secret

def update_oauth_config(google_creds=None, github_creds=None, microsoft_creds=None):
    """Update the OAuth configuration file."""
    config_file = Path("oauth_config.json")
    
    # Load existing config
    if config_file.exists():
        with open(config_file, 'r') as f:
            config = json.load(f)
    else:
        config = {}
    
    # Update configurations
    if google_creds and google_creds[0] and google_creds[1]:
        config["google"] = {
            "client_id": google_creds[0],
            "client_secret": google_creds[1],
            "redirect_uri": "http://localhost:8080/callback",
            "scope": "openid email profile",
            "enabled": True
        }
    
    if github_creds and github_creds[0] and github_creds[1]:
        config["github"] = {
            "client_id": github_creds[0],
            "client_secret": github_creds[1],
            "redirect_uri": "http://localhost:8080/callback",
            "scope": "user:email",
            "enabled": True
        }
    
    if microsoft_creds and microsoft_creds[0] and microsoft_creds[1]:
        config["microsoft"] = {
            "client_id": microsoft_creds[0],
            "client_secret": microsoft_creds[1],
            "redirect_uri": "http://localhost:8080/callback",
            "scope": "openid profile email",
            "enabled": True
        }
    
    # Save updated config
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"\nOAuth configuration updated: {config_file}")
    return config

def main():
    """Main setup function."""
    print_banner()
    
    google_creds = None
    github_creds = None
    microsoft_creds = None
    
    print("Which OAuth providers would you like to set up?")
    print("(You can set up multiple providers)")
    print()
    
    # Google setup
    setup_google = input("Set up Google OAuth? (Y/n): ").lower().strip()
    if setup_google != 'n':
        try:
            google_creds = setup_google_oauth()
        except KeyboardInterrupt:
            print("\nSkipped Google OAuth setup")
    
    print()
    
    # GitHub setup
    setup_github = input("Set up GitHub OAuth? (Y/n): ").lower().strip()
    if setup_github != 'n':
        try:
            github_creds = setup_github_oauth()
        except KeyboardInterrupt:
            print("\nSkipped GitHub OAuth setup")
    
    print()
    
    # Microsoft setup
    setup_microsoft = input("Set up Microsoft OAuth? (Y/n): ").lower().strip()
    if setup_microsoft != 'n':
        try:
            microsoft_creds = setup_microsoft_oauth()
        except KeyboardInterrupt:
            print("\nSkipped Microsoft OAuth setup")
    
    # Update configuration
    if any([google_creds, github_creds, microsoft_creds]):
        print("\nUpdating OAuth configuration...")
        config = update_oauth_config(google_creds, github_creds, microsoft_creds)
        
        # Show summary
        print("\n" + "=" * 60)
        print("OAUTH SETUP COMPLETE")
        print("=" * 60)
        
        enabled_providers = []
        for provider, settings in config.items():
            if isinstance(settings, dict) and settings.get('enabled', False):
                enabled_providers.append(provider.title())
        
        if enabled_providers:
            print(f"Enabled providers: {', '.join(enabled_providers)}")
            print("\nYou can now use OAuth authentication in OBVIVLORUM AI!")
            print("The system will automatically open your browser for authentication.")
        else:
            print("No providers were fully configured.")
            print("Run this script again to complete the setup.")
    else:
        print("\nNo OAuth providers were configured.")
        print("Run this script again when you're ready to set up OAuth authentication.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nOAuth setup cancelled by user.")