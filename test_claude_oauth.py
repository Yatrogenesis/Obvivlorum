#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Claude Direct OAuth Authentication
======================================

Test script for direct OAuth authentication with Anthropic/Claude
without using localhost redirect.
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, '.')

def test_claude_oauth():
    """Test Claude OAuth direct authentication."""
    try:
        from oauth_manager import OAuthManager, OAuthProvider
        
        print("OBVIVLORUM AI - Claude Direct OAuth Test")
        print("=" * 50)
        print()
        
        oauth = OAuthManager()
        
        print("Testing Claude/Anthropic direct authentication...")
        print("This will connect directly to Claude.ai without localhost redirect")
        print()
        
        # Test Claude OAuth
        result = oauth.start_oauth_flow(OAuthProvider.CLAUDE)
        
        if result:
            print("\n" + "=" * 50)
            print("AUTHENTICATION SUCCESSFUL!")
            print("=" * 50)
            
            # Get token and user info
            token = oauth.get_token(OAuthProvider.CLAUDE)
            if token and token.user_info:
                user_info = token.user_info
                print(f"Welcome to OBVIVLORUM AI!")
                print(f"User: {user_info.get('name', user_info.get('email', 'Claude User'))}")
                print(f"Token Type: {token.token_type}")
                print(f"Scope: {token.scope}")
                print()
                print("You can now use OBVIVLORUM AI with your Anthropic account!")
            else:
                print("Authentication successful but no user info available")
        else:
            print("\n" + "=" * 50)
            print("AUTHENTICATION FAILED")
            print("=" * 50)
            print("Please try again or check your Anthropic account")
            
    except KeyboardInterrupt:
        print("\nAuthentication cancelled by user")
    except Exception as e:
        print(f"\nError during Claude OAuth test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_claude_oauth()