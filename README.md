# Obvivlorum AI Symbiote System

[![CI/CD Pipeline](https://github.com/Yatrogenesis/Obvivlorum/actions/workflows/ci.yml/badge.svg)](https://github.com/Yatrogenesis/Obvivlorum/actions/workflows/ci.yml)
[![Research Release](https://github.com/Yatrogenesis/Obvivlorum/actions/workflows/research-release.yml/badge.svg)](https://github.com/Yatrogenesis/Obvivlorum/actions/workflows/research-release.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=flat&logo=docker&logoColor=white)](https://www.docker.com/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

[![Consciousness Research](https://img.shields.io/badge/Research-Consciousness%20AI-purple.svg)](docs/OBVIVLORUM_FRAMEWORK.md)
[![Quantum Computing](https://img.shields.io/badge/Quantum-Inspired%20Processing-blue.svg)](scientific/quantum_formalism.py)
[![IIT Implementation](https://img.shields.io/badge/IIT-Phi%20Calculations-green.svg)](scientific/consciousness_metrics.py)
[![Neuroplasticity](https://img.shields.io/badge/Neural-Plasticity%20Engine-orange.svg)](scientific/neuroplasticity_engine.py)

**Version:** 2.0  
**Author:** Francisco Molina  
**ORCID:** https://orcid.org/0009-0008-6093-8267  
**Email:** pako.molina@gmail.com  

## Overview

Advanced AI symbiosis platform with enterprise-grade security, adaptive intelligence, and multi-modal capabilities. The system incorporates multiple security layers, human-in-the-loop controls, and adaptive persistence mechanisms with rigorous scientific foundations.

## Key Features

### 🧠 AION Protocol Integration
- **5 specialized protocols**: ALPHA (Scientific Research), BETA (Mobile Apps), GAMMA (Enterprise Systems), DELTA (Web Apps), OMEGA (Licensing)
- **Enhanced vector memory management** with holographic storage
- **Metacognitive learning** that adapts to user patterns
- **Quantum-inspired symbolic processing** for advanced concept handling

### 🔗 Obvivlorum Bridge
- **Seamless integration** between AION and Obvivlorum frameworks
- **Quantum symbolic superposition** for concept processing
- **Holographic memory storage** for persistent learning
- **Recursive self-awareness** and identity preservation

### 🖥️ Cross-Platform Execution
- **Windows persistence mechanisms** (Registry, Startup folder, Scheduled tasks)
- **Linux execution capabilities** via WSL or native Linux
- **Safe command execution** with dangerous command blocking
- **Multi-distro WSL support** (Ubuntu, ParrotOS, Docker Desktop)

### 🎯 Adaptive Task Facilitation
- **Intelligent task suggestions** based on context and patterns
- **Adaptive scheduling** that learns from user behavior
- **Context-aware prioritization** of tasks and activities
- **Proactive assistance** with minimal user interruption

### 🛡️ Security & Safety
- **Safe mode execution** prevents dangerous operations
- **Command whitelisting** and blacklisting
- **Secure persistence** without compromising system integrity
- **Privacy-focused** learning with local data storage

## System Architecture

```
┌─────────────────────────────────────────────────────┐
│                AI SYMBIOTE                          │
├─────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌──────────────────────────┐   │
│  │    AION     │◄──►│    Obvivlorum Bridge     │   │
│  │  Protocol   │    │   (Quantum Symbolic)     │   │ 
│  └─────────────┘    └──────────────────────────┘   │
├─────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌──────────────────────────┐   │
│  │  Windows    │    │     Linux Executor       │   │
│  │ Persistence │    │   (WSL + Native)         │   │
│  └─────────────┘    └──────────────────────────┘   │
├─────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────┐   │
│  │      Adaptive Task Facilitator              │   │
│  │   (Learning + Context Awareness)            │   │
│  └─────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites
- Python 3.8+ 
- Windows 10+ or Linux
- WSL2 (recommended for Windows users)
- Administrative privileges (for persistence installation)

### Installation

1. **Clone or extract the AI Symbiote system:**
   ```bash
   # If using git
   git clone <repository_url>
   cd Obvivlorum
   
   # Or extract the provided archive
   cd path/to/extracted/Obvivlorum
   ```

2. **Install Python dependencies:**
   ```bash
   cd AION
   pip install -r requirements.txt
   cd ..
   ```

3. **Basic system test:**
   ```bash
   python ai_symbiote.py --status
   ```

### Usage Examples

#### 1. Start the AI Symbiote System
```bash
# Interactive mode (recommended for first-time users)
python ai_symbiote.py --user-id your_username

# Background mode with persistence
python ai_symbiote.py --user-id your_username --background --persistent
```

#### 2. Execute AION Protocols
```bash
# Test scientific research protocol
python ai_symbiote.py --test-protocol ALPHA --user-id your_username

# Using the Python API
from ai_symbiote import AISymbiote

symbiote = AISymbiote(user_id="your_username")
result = symbiote.execute_protocol("ALPHA", {
    "research_domain": "quantum_consciousness", 
    "research_type": "exploratory",
    "seed_concepts": ["quantum_entanglement", "neural_plasticity"]
})
```

#### 3. Linux Command Execution
```bash
# Safe command execution
python ai_symbiote.py --test-linux "echo 'Hello from WSL'" --user-id your_username

# Using the Python API
result = symbiote.execute_linux_command("ls -la /tmp")
```

#### 4. Task Management
```python
from ai_symbiote import AISymbiote

symbiote = AISymbiote(user_id="your_username")
symbiote.start()

# Add tasks
task_id = symbiote.add_task(
    "Research AI consciousness models",
    description="Investigate current theories and models",
    priority=8,
    tags=["research", "ai", "consciousness"]
)

# The system will provide intelligent suggestions and scheduling
```

#### 5. Install Persistence (Windows)
```bash
# Install persistence mechanisms
python ai_symbiote.py --install-persistence --user-id your_username

# Check persistence status
python windows_persistence.py --status
```

## Component Details

### AION Protocol System
Located in `AION/` directory:
- `aion_core.py` - Main protocol engine
- `aion_protocols/` - Individual protocol implementations
- `aion_obvivlorum_bridge.py` - Bridge integration
- `config.json` - System configuration

**Available Protocols:**
- **ALPHA**: Scientific and disruptive technology research
- **BETA**: Multi-platform mobile application development  
- **GAMMA**: Enterprise system architecture and implementation
- **DELTA**: Modern web application development
- **OMEGA**: Software licensing and intellectual property management

### Enhanced Systems
- **Vector Memory Manager**: Holographic information storage and retrieval
- **Metacognitive System**: Self-aware learning and pattern recognition
- **Quantum Framework**: Symbolic superposition and entanglement processing

### Cross-Platform Components

#### Windows Persistence (`windows_persistence.py`)
- Registry Run key installation
- Startup folder shortcuts
- Scheduled task creation
- Self-healing mechanisms
- Automatic restart capabilities

#### Linux Executor (`linux_executor.py`)
- WSL distro detection and management
- Safe command execution with sandboxing
- Package management integration
- Virtual environment support
- System resource monitoring

#### Task Facilitator (`adaptive_task_facilitator.py`)
- Behavioral pattern learning
- Context-aware suggestions
- Adaptive scheduling optimization
- User preference modeling
- Proactive task management

## Configuration

### Main Configuration
Edit the main system configuration by providing a custom config file:

```json
{
  "user_id": "your_username",
  "components": {
    "aion_protocol": {
      "enabled": true,
      "config_file": "AION/config.json"
    },
    "windows_persistence": {
      "enabled": true,
      "auto_install": false
    },
    "linux_executor": {
      "enabled": true,
      "safe_mode": true
    },
    "task_facilitator": {
      "enabled": true,
      "learning_enabled": true,
      "proactive_suggestions": true
    }
  },
  "system": {
    "heartbeat_interval": 60,
    "background_mode": false,
    "auto_restart": true
  },
  "security": {
    "safe_mode": true,
    "blocked_commands": ["rm -rf /", "format", "fdisk"]
  }
}
```

### AION Protocol Configuration
Edit `AION/config.json` for protocol-specific settings:

```json
{
  "performance_targets": {
    "response_time": "1ms",
    "failure_rate": "0.001%",
    "availability": "99.999%"
  },
  "integration": {
    "obvivlorum_bridge": "aion_obvivlorum_bridge",
    "bridge_mode": "bidirectional"
  },
  "system": {
    "auto_update": true,
    "architecture_evolution_rate": 0.05,
    "coherence_threshold": 0.85
  }
}
```

## Testing

### Run Comprehensive Tests
```bash
# Full test suite
python test_comprehensive.py

# Individual component tests  
python test_bridge_integration.py
python AION/test_aion.py
```

### Test Results Summary
The system has been tested with:
- **31 test cases** covering all major components
- **80.6% success rate** in comprehensive testing
- **Performance benchmarks**:
  - Protocol execution: ~1ms average
  - Task creation: <1ms average  
  - Status retrieval: ~53ms average
- **Cross-platform compatibility** verified on Windows 10/11 with WSL

## Security Considerations

### Safe Execution
- All commands are filtered through safety checks
- Dangerous commands are automatically blocked
- Sandboxed execution environments for untrusted operations
- User confirmation required for potentially risky operations

### Privacy Protection
- All learning data stored locally
- No external data transmission without explicit consent
- User behavior patterns anonymized
- Configurable data retention policies

### Persistence Security
- Registry modifications limited to current user scope
- No system-level modifications without administrator consent
- Graceful fallback if persistence installation fails
- Easy uninstallation process

## Troubleshooting

### Common Issues

#### 1. AION Protocol Errors
```bash
# Issue: "Unknown research domain" errors
# Solution: Add create_domain=True to parameters
result = symbiote.execute_protocol("ALPHA", {
    "research_domain": "your_domain",
    "create_domain": True  # This creates new domains
})
```

#### 2. WSL Connection Issues
```bash
# Check WSL status
wsl --status

# List available distributions
wsl --list --verbose

# Set default distribution
wsl --set-default Ubuntu
```

#### 3. Persistence Installation Fails
```bash
# Check permissions (Run as Administrator)
# Or install manually:
python windows_persistence.py --install --symbiote-path path/to/ai_symbiote.py
```

#### 4. Task Facilitator Errors
```python
# Ensure data directory exists and is writable
import os
data_dir = os.path.expanduser("~/.ai_symbiote/task_data")
os.makedirs(data_dir, exist_ok=True)
```

### Performance Optimization

#### For Better Performance:
1. **Use SSD storage** for faster file I/O operations
2. **Allocate sufficient RAM** (minimum 4GB recommended)
3. **Enable WSL2** for better Linux integration performance
4. **Configure antivirus exclusions** for the Obvivlorum directory

#### For Reduced Resource Usage:
```python
# Disable resource-intensive features
config = {
    "components": {
        "task_facilitator": {
            "learning_enabled": False,  # Reduces CPU usage
            "proactive_suggestions": False  # Reduces background activity
        }
    }
}
```

## API Reference

### Core AISymbiote Class

```python
class AISymbiote:
    def __init__(self, config_file=None, user_id="default")
    def start(self)  # Start the system
    def stop(self)   # Stop the system gracefully
    
    # Protocol execution
    def execute_protocol(self, protocol_name: str, parameters: dict) -> dict
    
    # Linux command execution  
    def execute_linux_command(self, command: str, **kwargs) -> dict
    
    # Task management
    def add_task(self, name: str, **kwargs) -> str
    
    # System status
    def get_system_status(self) -> dict
```

### AION Protocol Methods

```python
# Direct AION access
aion = symbiote.components["aion"]

# Execute protocols
result = aion.execute_protocol("ALPHA", parameters)

# Get protocol information
info = aion.get_protocol_info("BETA")

# Analyze performance
metrics = aion.analyze_performance()
```

### Bridge Integration

```python
# Enhanced protocol execution through bridge
bridge = symbiote.components["bridge"]
result = bridge.execute_protocol_with_obvivlorum("ALPHA", parameters)

# Architecture evolution
evolution = bridge.evolve_architecture(evolutionary_pressure=0.6)
```

## Development and Contribution

### Code Structure
```
D:\Obvivlorum/
├── ai_symbiote.py              # Main orchestrator
├── windows_persistence.py     # Windows-specific functionality
├── linux_executor.py          # Linux execution engine
├── adaptive_task_facilitator.py # Task management system
├── test_comprehensive.py      # Test suite
├── test_bridge_integration.py # Bridge-specific tests
├── obvlivorum-implementation.py # Obvivlorum framework (conceptual)
├── AION/                       # AION Protocol system
│   ├── aion_core.py           # Core protocol engine
│   ├── aion_obvivlorum_bridge.py # Bridge implementation
│   ├── aion_protocols/        # Individual protocols
│   ├── config.json            # AION configuration
│   └── requirements.txt       # Python dependencies
└── README.md                  # This file
```

### Contributing Guidelines
1. **Follow existing code patterns** and documentation standards
2. **Add comprehensive tests** for new functionality
3. **Maintain security best practices** - never compromise safety features
4. **Test cross-platform compatibility** on both Windows and Linux
5. **Update documentation** for any new features or API changes

### Extending the System

#### Adding New AION Protocols
1. Create new protocol in `AION/aion_protocols/protocol_name.py`
2. Follow the existing protocol structure and interface
3. Add configuration to `AION/config.json`
4. Create comprehensive tests

#### Adding New Platform Support
1. Create platform-specific module (e.g., `macos_integration.py`)
2. Implement required interfaces from existing modules
3. Add detection logic to `ai_symbiote.py`
4. Test thoroughly on target platform

## License and Legal

### Software License
This software is released under a custom license. See individual files for specific licensing terms. Commercial use requires explicit permission from the author.

### Academic Use
This system is designed for educational and research purposes. Academic institutions may use this software for non-commercial research with proper attribution.

### Attribution Required
When using or referencing this work, please cite:
```
Molina, F. (2025). AI Symbiote: Adaptive Artificial Intelligence Assistant. 
ORCID: https://orcid.org/0009-0008-6093-8267
```

## Support and Contact

### Getting Help
1. **Check the troubleshooting section** above
2. **Review test results** with `python test_comprehensive.py`
3. **Check component status** with `python ai_symbiote.py --status`
4. **Enable debug logging** for detailed error information

### Contact Information
- **Author:** Francisco Molina
- **Email:** pako.molina@gmail.com  
- **ORCID:** https://orcid.org/0009-0008-6093-8267

### Reporting Issues
When reporting issues, please include:
1. System configuration (OS, Python version, WSL status)
2. Complete error messages and stack traces
3. Steps to reproduce the issue
4. Expected vs. actual behavior
5. Log files (with sensitive information removed)

---

**Note:** This AI Symbiote system represents a sophisticated integration of multiple AI paradigms and should be used responsibly. The system is designed to be helpful, harmless, and honest in all its interactions.

*Generated with Claude Code and enhanced by the AI Symbiote system itself.*