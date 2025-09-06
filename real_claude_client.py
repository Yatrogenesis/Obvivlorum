#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real Claude Client - Browser Automation Based Claude Integration
===============================================================

This module implements real Claude.ai integration using browser automation
with Playwright, similar to how Claude Code actually works.

Features:
- Real browser session with claude.ai
- OAuth authentication handling
- Session persistence
- Real Claude responses (not simulated)
- Cloudflare bypass through real browser
- Message context management
"""

import asyncio
import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Dict, Optional, Any, List
from dataclasses import dataclass, asdict

# Playwright imports
try:
    from playwright.async_api import async_playwright, Browser, Page, BrowserContext
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False

logger = logging.getLogger("RealClaudeClient")

@dataclass
class ConversationContext:
    """Conversation context for Claude chat."""
    uuid: str
    organization_uuid: Optional[str] = None
    name: str = "OBVIVLORUM Chat"
    created_at: float = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()

@dataclass
class ClaudeSession:
    """Claude session information."""
    authenticated: bool = False
    user_email: Optional[str] = None
    organization_uuid: Optional[str] = None
    session_token: Optional[str] = None
    cookies: Dict[str, str] = None
    current_conversation: Optional[ConversationContext] = None
    
    def __post_init__(self):
        if self.cookies is None:
            self.cookies = {}

class RealClaudeClient:
    """Real Claude client using browser automation."""
    
    def __init__(self, session_file: str = "claude_session.json", headless: bool = True):
        """Initialize the real Claude client."""
        if not PLAYWRIGHT_AVAILABLE:
            raise ImportError("Playwright is required. Install with: pip install playwright")
            
        self.session_file = Path(session_file)
        self.headless = headless
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None
        self.session = ClaudeSession()
        
        # Load existing session if available
        self.load_session()
        
    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    def load_session(self):
        """Load session from file."""
        try:
            if self.session_file.exists():
                with open(self.session_file, 'r') as f:
                    data = json.load(f)
                    self.session = ClaudeSession(**data)
                    if self.session.current_conversation:
                        self.session.current_conversation = ConversationContext(
                            **self.session.current_conversation
                        )
                    logger.info("Loaded existing Claude session")
        except Exception as e:
            logger.warning(f"Could not load session: {e}")
            self.session = ClaudeSession()
    
    def save_session(self):
        """Save session to file."""
        try:
            # Convert to dict for JSON serialization
            data = asdict(self.session)
            with open(self.session_file, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info("Claude session saved")
        except Exception as e:
            logger.error(f"Could not save session: {e}")
    
    async def initialize(self):
        """Initialize browser and page."""
        try:
            playwright = await async_playwright().start()
            
            # Launch browser with proper arguments
            self.browser = await playwright.chromium.launch(
                headless=self.headless,
                args=[
                    '--disable-blink-features=AutomationControlled',
                    '--disable-dev-shm-usage',
                    '--disable-extensions',
                    '--no-sandbox',
                    '--disable-setuid-sandbox',
                    '--disable-web-security',
                    '--disable-features=VizDisplayCompositor'
                ]
            )
            
            # Create context with realistic user agent
            self.context = await self.browser.new_context(
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                viewport={'width': 1280, 'height': 720},
                locale='en-US',
                timezone_id='America/New_York'
            )
            
            # Add cookies if available
            if self.session.cookies:
                await self.context.add_cookies([
                    {
                        'name': name,
                        'value': value,
                        'domain': 'claude.ai',
                        'path': '/',
                        'httpOnly': True,
                        'secure': True
                    }
                    for name, value in self.session.cookies.items()
                ])
            
            # Create page
            self.page = await self.context.new_page()
            
            # Set up page event handlers
            await self.setup_page_handlers()
            
            logger.info("Real Claude client initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize browser: {e}")
            raise
    
    async def setup_page_handlers(self):
        """Set up page event handlers."""
        # Handle console messages
        self.page.on("console", lambda msg: logger.debug(f"Browser console: {msg.text}"))
        
        # Handle page errors
        self.page.on("pageerror", lambda error: logger.error(f"Page error: {error}"))
    
    async def authenticate(self) -> bool:
        """Authenticate with Claude.ai using browser automation."""
        try:
            logger.info("Starting Claude.ai authentication...")
            
            # Navigate to Claude.ai
            await self.page.goto("https://claude.ai/login", wait_until="networkidle")
            
            # Wait for page to load
            await self.page.wait_for_timeout(2000)
            
            # Check if already authenticated
            if await self.check_authentication_status():
                logger.info("Already authenticated with Claude.ai")
                return True
            
            logger.info("Please complete authentication in the browser...")
            
            # Wait for user to complete authentication
            # We'll wait for the chat page to load as indication of successful auth
            try:
                await self.page.wait_for_url("**/chats**", timeout=300000)  # 5 minutes
                logger.info("Authentication successful!")
                
                # Extract session information
                await self.extract_session_info()
                
                self.session.authenticated = True
                self.save_session()
                
                return True
                
            except Exception as e:
                logger.error(f"Authentication timeout or failed: {e}")
                return False
                
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return False
    
    async def check_authentication_status(self) -> bool:
        """Check if user is already authenticated."""
        try:
            # Try to navigate to chats
            await self.page.goto("https://claude.ai/chats", wait_until="networkidle")
            
            # Wait a bit
            await self.page.wait_for_timeout(3000)
            
            # If we're still on the chats page, we're authenticated
            current_url = self.page.url
            if "/chats" in current_url:
                await self.extract_session_info()
                self.session.authenticated = True
                return True
                
            return False
            
        except Exception:
            return False
    
    async def extract_session_info(self):
        """Extract session information from the authenticated page."""
        try:
            # Get cookies
            cookies = await self.context.cookies()
            self.session.cookies = {
                cookie['name']: cookie['value'] 
                for cookie in cookies 
                if cookie['domain'] == 'claude.ai' or cookie['domain'] == '.claude.ai'
            }
            
            # Try to get user information from the page
            try:
                # This might need adjustment based on Claude.ai's actual structure
                user_info = await self.page.evaluate("""
                    () => {
                        // Try to extract user info from window object or DOM
                        if (window.__CLAUDE_USER__) {
                            return window.__CLAUDE_USER__;
                        }
                        // Fallback: try to get from DOM elements
                        const userEmail = document.querySelector('[data-testid="user-email"]');
                        return {
                            email: userEmail ? userEmail.textContent : null
                        };
                    }
                """)
                
                if user_info and user_info.get('email'):
                    self.session.user_email = user_info['email']
                    
            except Exception as e:
                logger.debug(f"Could not extract detailed user info: {e}")
            
            logger.info("Session information extracted")
            
        except Exception as e:
            logger.error(f"Failed to extract session info: {e}")
    
    async def create_conversation(self, name: str = None) -> Optional[ConversationContext]:
        """Create a new conversation."""
        try:
            if not self.session.authenticated:
                logger.error("Not authenticated")
                return None
            
            # Navigate to chats if not already there
            if "/chats" not in self.page.url:
                await self.page.goto("https://claude.ai/chats", wait_until="networkidle")
            
            # Look for "Start new chat" button or similar
            try:
                # Try multiple selectors for new chat button
                selectors = [
                    'button[data-testid="new-chat"]',
                    'button[aria-label="Start new chat"]',
                    'button:has-text("New chat")',
                    'a[href="/chats/new"]',
                    '[data-testid="new-conversation"]'
                ]
                
                clicked = False
                for selector in selectors:
                    try:
                        element = await self.page.wait_for_selector(selector, timeout=2000)
                        if element:
                            await element.click()
                            clicked = True
                            break
                    except:
                        continue
                
                if not clicked:
                    # Fallback: navigate directly to new chat
                    await self.page.goto("https://claude.ai/chats/new", wait_until="networkidle")
                
                # Wait for new conversation to load
                await self.page.wait_for_timeout(2000)
                
                # Extract conversation UUID from URL
                current_url = self.page.url
                if "/chats/" in current_url:
                    conv_id = current_url.split("/chats/")[-1].split("?")[0]
                    if conv_id and conv_id != "new":
                        conversation = ConversationContext(
                            uuid=conv_id,
                            name=name or f"OBVIVLORUM Chat {int(time.time())}",
                            organization_uuid=self.session.organization_uuid
                        )
                        self.session.current_conversation = conversation
                        self.save_session()
                        logger.info(f"Created conversation: {conv_id}")
                        return conversation
                
                # Fallback: generate UUID
                conversation = ConversationContext(
                    uuid=str(uuid.uuid4()),
                    name=name or f"OBVIVLORUM Chat {int(time.time())}",
                    organization_uuid=self.session.organization_uuid
                )
                self.session.current_conversation = conversation
                self.save_session()
                logger.info("Created fallback conversation")
                return conversation
                
            except Exception as e:
                logger.error(f"Failed to create conversation: {e}")
                return None
                
        except Exception as e:
            logger.error(f"Conversation creation error: {e}")
            return None
    
    async def send_message(self, message: str, model: str = "claude-3-sonnet") -> str:
        """Send message to Claude and get real response."""
        try:
            if not self.session.authenticated:
                return "Error: Not authenticated with Claude.ai"
            
            # Ensure we have a conversation
            if not self.session.current_conversation:
                conv = await self.create_conversation()
                if not conv:
                    return "Error: Could not create conversation"
            
            # Navigate to the conversation if not already there
            conv_url = f"https://claude.ai/chats/{self.session.current_conversation.uuid}"
            if self.page.url != conv_url:
                await self.page.goto(conv_url, wait_until="networkidle")
                await self.page.wait_for_timeout(2000)
            
            # Find the message input field
            input_selectors = [
                'textarea[placeholder*="Talk to Claude"]',
                'textarea[data-testid="chat-input"]',
                'div[contenteditable="true"][role="textbox"]',
                'textarea[aria-label*="message"]',
                'input[type="text"][placeholder*="message"]'
            ]
            
            message_input = None
            for selector in input_selectors:
                try:
                    message_input = await self.page.wait_for_selector(selector, timeout=3000)
                    if message_input:
                        break
                except:
                    continue
            
            if not message_input:
                return "Error: Could not find message input field"
            
            # Clear and type the message
            await message_input.click()
            await message_input.fill("")  # Clear
            await message_input.fill(message)
            
            # Find and click send button
            send_selectors = [
                'button[data-testid="send-button"]',
                'button[aria-label="Send message"]',
                'button:has-text("Send")',
                'button[type="submit"]',
                'svg[data-testid="send-icon"]'
            ]
            
            send_button = None
            for selector in send_selectors:
                try:
                    send_button = await self.page.wait_for_selector(selector, timeout=2000)
                    if send_button:
                        # Check if button is enabled
                        is_enabled = await send_button.is_enabled()
                        if is_enabled:
                            break
                except:
                    continue
            
            if not send_button:
                # Fallback: try pressing Enter
                await message_input.press("Enter")
            else:
                await send_button.click()
            
            # Wait for Claude's response
            logger.info("Message sent, waiting for Claude's response...")
            
            # Wait for response to appear (look for streaming or complete response)
            response_selectors = [
                '[data-testid="claude-message"]',
                '.message-content',
                '[role="assistant"]',
                '.assistant-message',
                '.response-message'
            ]
            
            # Wait for response to start appearing
            await self.page.wait_for_timeout(3000)
            
            # Try to wait for response completion (look for indicators)
            try:
                # Wait for typing indicator to disappear or response to complete
                await self.page.wait_for_function(
                    """
                    () => {
                        // Look for completion indicators
                        const stopButton = document.querySelector('button[aria-label*="Stop"]');
                        const typingIndicator = document.querySelector('[data-testid="typing-indicator"]');
                        
                        // If no stop button and no typing indicator, response is likely complete
                        return !stopButton && !typingIndicator;
                    }
                    """,
                    timeout=60000  # Wait up to 1 minute
                )
            except:
                # If we can't detect completion, wait a reasonable time
                await self.page.wait_for_timeout(5000)
            
            # Extract the response text
            response_text = await self.extract_latest_response()
            
            if response_text:
                logger.info("Received Claude response")
                return response_text
            else:
                return "Received response from Claude (could not extract text)"
                
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            return f"Error: {e}"
    
    async def extract_latest_response(self) -> str:
        """Extract the latest response from Claude."""
        try:
            # Try multiple strategies to extract the response
            strategies = [
                # Strategy 1: Look for specific data attributes
                """
                () => {
                    const messages = document.querySelectorAll('[data-testid="claude-message"], [role="assistant"]');
                    const latest = messages[messages.length - 1];
                    return latest ? latest.innerText || latest.textContent : null;
                }
                """,
                # Strategy 2: Look for message containers
                """
                () => {
                    const messages = document.querySelectorAll('.message-content, .assistant-message');
                    const latest = messages[messages.length - 1];
                    return latest ? latest.innerText || latest.textContent : null;
                }
                """,
                # Strategy 3: Look for any recent text that might be the response
                """
                () => {
                    const allElements = document.querySelectorAll('div, p, span');
                    let latestResponse = '';
                    let maxLength = 0;
                    
                    for (let elem of allElements) {
                        const text = elem.innerText || elem.textContent;
                        if (text && text.length > maxLength && text.length > 10) {
                            // Basic heuristic: longer text is more likely to be the response
                            maxLength = text.length;
                            latestResponse = text;
                        }
                    }
                    
                    return latestResponse || null;
                }
                """
            ]
            
            for strategy in strategies:
                try:
                    result = await self.page.evaluate(strategy)
                    if result and len(result.strip()) > 0:
                        return result.strip()
                except:
                    continue
            
            # Fallback: get all text content and try to find the response
            page_text = await self.page.text_content('body')
            if page_text:
                # This is a very basic fallback - in real implementation
                # you'd want more sophisticated parsing
                lines = page_text.split('\n')
                for line in reversed(lines):
                    if line.strip() and len(line.strip()) > 20:
                        return line.strip()
            
            return None
            
        except Exception as e:
            logger.error(f"Could not extract response: {e}")
            return None
    
    async def close(self):
        """Close browser and cleanup."""
        try:
            if self.context:
                await self.context.close()
            if self.browser:
                await self.browser.close()
            logger.info("Real Claude client closed")
        except Exception as e:
            logger.error(f"Error closing client: {e}")

# Convenience functions
async def create_claude_client(headless: bool = True) -> RealClaudeClient:
    """Create and initialize a Claude client."""
    client = RealClaudeClient(headless=headless)
    await client.initialize()
    return client

async def test_claude_client():
    """Test the Claude client."""
    async with RealClaudeClient(headless=False) as client:
        # Authenticate
        if await client.authenticate():
            print("✅ Authentication successful")
            
            # Send test message
            response = await client.send_message("Hello! Please respond with 'Real Claude AI is working!' to confirm this is a real connection.")
            print(f"Claude response: {response}")
        else:
            print("❌ Authentication failed")

if __name__ == "__main__":
    # Test the client
    asyncio.run(test_claude_client())