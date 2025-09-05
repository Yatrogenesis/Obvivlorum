# OBVIVLORUM AI - API Setup Guide

## 🚀 Quick Start

### Option 1: Social Login (OAuth) - RECOMMENDED
**Easy, secure, no API keys to manage!**

1. **Run the OAuth-enabled GUI:**
   ```bash
   python ai_gui_with_oauth.py
   ```

2. **Click "Login with Google/GitHub/Microsoft"**
   - Browser opens automatically
   - Login with your existing account
   - Uses your existing paid plans
   - No API keys exposed in project

3. **Enjoy real AI chat!**

### Option 2: Manual API Keys
**For users who want direct API control**

1. **Interactive setup:**
   ```bash
   python api_manager.py
   ```

2. **Or create .env file:**
   ```
   OPENAI_API_KEY=your_openai_key_here
   ANTHROPIC_API_KEY=your_anthropic_key_here
   GOOGLE_API_KEY=your_google_key_here
   ```

3. **Or create api_keys.json:**
   ```json
   {
     "openai": {
       "api_key": "your_key_here",
       "enabled": true
     },
     "anthropic": {
       "api_key": "your_key_here", 
       "enabled": true
     }
   }
   ```

## 🔐 Security Benefits of OAuth

### Why OAuth is Better:
- ✅ **No API keys in code** - safer for version control
- ✅ **Uses existing paid accounts** - no duplicate billing
- ✅ **Secure token management** - handled by providers
- ✅ **Easy account linking** - familiar login flow
- ✅ **Infrastructure security** - leverages Google/Microsoft security

### How It Works:
1. **You click "Login with Google"**
2. **Browser opens** → Google login page
3. **You authenticate** with your Google account
4. **OAuth token obtained** - no API key needed
5. **AI access granted** through your existing account

## 📁 Available Files

### Main Applications:
- **`ai_gui_with_oauth.py`** - Complete GUI with social login
- **`ai_simple_gui_working.py`** - Simple GUI (rule-based only)
- **`ai_engine_with_apis.py`** - Core AI engine with API support

### Configuration Tools:
- **`api_manager.py`** - Interactive API key management
- **`oauth_manager.py`** - Interactive OAuth setup
- **`UNICODE_CLEANUP_SCRIPT.py`** - Unicode cleaning tool

### Setup Scripts:
- **`CREATE_DESCRIPTIVE_SHORTCUTS.bat`** - Create desktop shortcuts
- **`SINGLE_CLICK_LAUNCHER.py`** - Mode selector launcher

## 🖥️ Available Interfaces

### GUI Applications:
1. **OAuth GUI** - `ai_gui_with_oauth.py`
   - Social login integration
   - Real AI connectivity
   - Provider selection
   - Secure token management

2. **Simple GUI** - `ai_simple_gui_working.py`
   - Lightweight interface
   - Rule-based responses only
   - No API keys required
   - Always works

3. **Traditional GUI** - `ai_symbiote_gui.py`
   - Original interface
   - Multiple AI engines
   - Advanced configuration

### Command Line:
- **AI Engine Test** - `python ai_engine_with_apis.py`
- **API Setup** - `python api_manager.py`
- **OAuth Setup** - `python oauth_manager.py`

## 🔧 Setup for OAuth (One-time)

If you want to enable OAuth social login:

### 1. Create OAuth Applications:

**Google OAuth:**
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create new project or use existing
3. Enable Google+ API
4. Create OAuth 2.0 credentials
5. Add `http://localhost:8080/callback` as redirect URI

**GitHub OAuth:**
1. Go to GitHub Settings → Developer settings → OAuth Apps
2. Create new OAuth app
3. Set callback URL: `http://localhost:8080/callback`

**Microsoft OAuth:**
1. Go to [Azure App Registrations](https://portal.azure.com/)
2. Create new app registration
3. Add `http://localhost:8080/callback` as redirect URI

### 2. Configure OAuth:
```bash
python oauth_manager.py
# Choose option 2: "Create sample OAuth configuration"
# Edit oauth_config.json with your client IDs and secrets
```

### 3. Test OAuth:
```bash
python oauth_manager.py
# Choose option 1: "Start OAuth flow for a provider"
```

## ❓ Troubleshooting

### "No API providers configured"
- Run `python api_manager.py` to set up API keys
- Or use OAuth social login for automatic setup

### "OAuth not configured for provider"
- Run `python oauth_manager.py` to set up OAuth
- Create oauth_config.json with your app credentials

### GUI "blinking" (starts and closes)
- Unicode issues were fixed in latest version
- Use `ai_gui_with_oauth.py` or `ai_simple_gui_working.py`

### API calls failing
- Check your API key validity
- Verify provider endpoints
- Check network connection

## 📞 Support

For help:
- Check GitHub issues: https://github.com/Yatrogenesis/Obvivlorum/issues
- Run diagnostic: `python test_system_final.py`
- Check logs for error details

---

**Created by:** Francisco Molina  
**ORCID:** https://orcid.org/0009-0008-6093-8267  
**Email:** pako.molina@gmail.com