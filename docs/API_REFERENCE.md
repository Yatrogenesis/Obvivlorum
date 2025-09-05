# API Reference - Obvivlorum AI Symbiote System v2.0

## Core API Overview

This document provides comprehensive API documentation for all major components of the Obvivlorum AI Symbiote System.

## Core Orchestrator API

### CoreOrchestrator Class

The central system management component that coordinates all other services.

```python
from core_orchestrator import CoreOrchestrator, SystemStatus

# Initialize orchestrator
orchestrator = CoreOrchestrator(config_path="config_optimized.json")

# Start the system
await orchestrator.start_system()

# Get system status
status = orchestrator.get_status()
print(f"System running: {status.running}")
print(f"Active components: {status.active_components}")
```

#### Methods

##### `__init__(config_path: str = "config_optimized.json")`
Initialize the core orchestrator with configuration.

**Parameters:**
- `config_path`: Path to configuration file

##### `async start_system() -> None`
Start the complete system with all components.

**Raises:**
- `Exception`: If system initialization fails

##### `async shutdown() -> None`
Gracefully shutdown the system and all components.

##### `get_status() -> SystemStatus`
Get current system status.

**Returns:**
- `SystemStatus`: Object containing system state information

##### `async queue_command(command: Dict[str, Any]) -> None`
Queue a command for asynchronous processing.

**Parameters:**
- `command`: Dictionary containing command type and data

##### `async set_security_level(level: PrivilegeLevel) -> bool`
Set system security/privilege level.

**Parameters:**
- `level`: Target privilege level

**Returns:**
- `bool`: True if successful

## Security Manager API

### SecurityManager Class

Handles privilege management, authentication, and threat assessment.

```python
from security_manager import SecurityManager, PrivilegeLevel, ThreatLevel

# Initialize security manager
security = SecurityManager()

# Set privilege level
await security.set_privilege_level(PrivilegeLevel.OPERATOR)

# Assess threat level
threat = await security.assess_threat_level("rm -rf /", {})
print(f"Threat level: {threat}")
```

#### Enumerations

##### PrivilegeLevel
- `GUEST`: Read-only access
- `USER`: Standard user operations  
- `OPERATOR`: Advanced operations with monitoring
- `ADMIN`: Administrative functions
- `SYSTEM`: Full system access

##### ThreatLevel
- `MINIMAL`: No significant risk
- `LOW`: Minor risk
- `MEDIUM`: Moderate risk requiring attention
- `HIGH`: High risk requiring approval
- `CRITICAL`: Critical risk requiring verification
- `EXTREME`: Maximum risk requiring multi-factor approval

#### Methods

##### `__init__(config_path: str = None)`
Initialize security manager.

**Parameters:**
- `config_path`: Optional path to security configuration

##### `async set_privilege_level(level: PrivilegeLevel) -> bool`
Set current privilege level.

**Parameters:**
- `level`: Target privilege level

**Returns:**
- `bool`: True if successful

##### `async assess_threat_level(operation: str, context: Dict[str, Any]) -> ThreatLevel`
Assess threat level of an operation.

**Parameters:**
- `operation`: Operation string to assess
- `context`: Additional context information

**Returns:**
- `ThreatLevel`: Assessed threat level

##### `async generate_jwt_token(user_id: str, privilege_level: PrivilegeLevel) -> str`
Generate JWT authentication token.

**Parameters:**
- `user_id`: User identifier
- `privilege_level`: Privilege level for token

**Returns:**
- `str`: JWT token

##### `async execute_privileged_operation(operation: str, required_level: PrivilegeLevel, context: Dict[str, Any]) -> Any`
Execute operation with privilege verification.

**Parameters:**
- `operation`: Operation to execute
- `required_level`: Required privilege level
- `context`: Operation context

**Returns:**
- `Any`: Operation result

## Human-in-the-Loop API

### HumanInTheLoopManager Class

Manages human approval workflows for risky operations.

```python
from human_in_the_loop import HumanInTheLoopManager, RiskLevel, ApprovalRequest

# Initialize HITL manager
hitl = HumanInTheLoopManager(config, security_manager, logger)

# Request approval for operation
result = await hitl.request_approval(
    command="sudo apt-get install package",
    context={"user": "admin", "environment": "production"},
    requester="system"
)

print(f"Approved: {result.approved}")
```

#### Enumerations

##### RiskLevel
- `MINIMAL`: Auto-approved operations
- `LOW`: Low risk operations
- `MEDIUM`: Moderate risk requiring basic approval
- `HIGH`: High risk requiring detailed approval
- `CRITICAL`: Critical risk requiring verification
- `EXTREME`: Maximum risk requiring multi-factor approval

#### Classes

##### ApprovalRequest
```python
@dataclass
class ApprovalRequest:
    command: str
    risk_level: RiskLevel
    context: Dict[str, Any]
    requester: str
    timestamp: datetime
    reasoning: str
```

##### ApprovalResult
```python
@dataclass
class ApprovalResult:
    approved: bool
    risk_level: RiskLevel
    required_privilege_level: PrivilegeLevel
    reasoning: str
    approval_time: datetime
    expires_at: datetime
```

#### Methods

##### `__init__(config: Dict[str, Any], security_manager: SecurityManager, logger: StructuredLogger)`
Initialize HITL manager.

**Parameters:**
- `config`: Configuration dictionary
- `security_manager`: Security manager instance
- `logger`: Logger instance

##### `async request_approval(command: str, context: Dict[str, Any], requester: str) -> ApprovalResult`
Request approval for a command.

**Parameters:**
- `command`: Command requiring approval
- `context`: Command context
- `requester`: User/system requesting approval

**Returns:**
- `ApprovalResult`: Approval decision and metadata

##### `async assess_risk_level(command: str, context: Dict[str, Any]) -> RiskLevel`
Assess risk level of a command.

**Parameters:**
- `command`: Command to assess
- `context`: Command context

**Returns:**
- `RiskLevel`: Assessed risk level

## Smart Provider Selector API

### SmartProviderSelector Class

Intelligently selects AI providers based on task requirements and performance.

```python
from smart_provider_selector import SmartProviderSelector, ProviderType, TaskType

# Initialize provider selector
selector = SmartProviderSelector(config, logger)

# Get completion from optimal provider
response = await selector.get_completion(
    prompt="Explain quantum computing",
    task_type=TaskType.EXPLANATION,
    context={"complexity": "advanced"}
)

print(f"Provider used: {response['provider']}")
print(f"Response: {response['content']}")
```

#### Enumerations

##### ProviderType
- `OPENAI`: OpenAI API (GPT models)
- `CLAUDE`: Anthropic Claude API  
- `LOCAL_GGUF`: Local GGUF models
- `OLLAMA`: Ollama local models
- `HUGGINGFACE`: HuggingFace models

##### TaskType
- `GENERAL`: General purpose tasks
- `CODING`: Code generation and analysis
- `ANALYSIS`: Data analysis and reasoning
- `CREATIVE`: Creative writing and ideation
- `TECHNICAL`: Technical documentation and explanations
- `CONVERSATION`: Conversational interactions

#### Methods

##### `__init__(config: Dict[str, Any], logger: StructuredLogger = None)`
Initialize provider selector.

**Parameters:**
- `config`: Configuration dictionary
- `logger`: Optional logger instance

##### `async get_completion(prompt: str, task_type: TaskType = TaskType.GENERAL, context: Dict[str, Any] = None) -> Dict[str, Any]`
Get completion from optimal provider.

**Parameters:**
- `prompt`: Input prompt
- `task_type`: Type of task for optimization
- `context`: Additional context

**Returns:**
- `Dict[str, Any]`: Response with content and metadata

##### `async select_optimal_provider(task_type: TaskType, context: Dict[str, Any]) -> ProviderType`
Select optimal provider for task.

**Parameters:**
- `task_type`: Type of task
- `context`: Task context

**Returns:**
- `ProviderType`: Selected provider

##### `get_provider_metrics() -> Dict[str, Any]`
Get performance metrics for all providers.

**Returns:**
- `Dict[str, Any]`: Provider performance data

## Structured Logger API

### StructuredLogger Class

Advanced logging system with metrics and structured output.

```python
from structured_logger import StructuredLogger, LogLevel

# Initialize logger
logger = StructuredLogger("my_component", config)

# Log messages with different levels
logger.info("System started")
logger.warning("High memory usage detected", extra={
    "memory_usage": 85.5,
    "threshold": 80.0
})
logger.error("Database connection failed", extra={
    "error_code": "DB_CONN_TIMEOUT",
    "retry_count": 3
})

# Log with structured data
logger.log_structured({
    "event": "user_action",
    "action": "login",
    "user_id": "user123",
    "timestamp": datetime.now(),
    "success": True
})
```

#### Enumerations

##### LogLevel
- `DEBUG`: Detailed debugging information
- `INFO`: General informational messages
- `WARNING`: Warning messages
- `ERROR`: Error messages
- `CRITICAL`: Critical system errors

#### Methods

##### `__init__(name: str = "obvivlorum", config: Dict[str, Any] = None)`
Initialize structured logger.

**Parameters:**
- `name`: Logger name
- `config`: Logger configuration

##### `info(message: str, extra: Dict[str, Any] = None) -> None`
Log info message.

**Parameters:**
- `message`: Log message
- `extra`: Additional structured data

##### `warning(message: str, extra: Dict[str, Any] = None) -> None`
Log warning message.

##### `error(message: str, extra: Dict[str, Any] = None) -> None`
Log error message.

##### `critical(message: str, extra: Dict[str, Any] = None) -> None`
Log critical message.

##### `debug(message: str, extra: Dict[str, Any] = None) -> None`
Log debug message.

##### `log_structured(data: Dict[str, Any], level: LogLevel = LogLevel.INFO) -> None`
Log structured data.

**Parameters:**
- `data`: Structured data to log
- `level`: Log level

##### `get_metrics() -> Dict[str, Any]`
Get logging metrics.

**Returns:**
- `Dict[str, Any]`: Metrics data

##### `is_healthy() -> bool`
Check logger health status.

**Returns:**
- `bool`: True if healthy

##### `async shutdown() -> None`
Gracefully shutdown logger.

## Unified Launcher API

### UnifiedLauncher Class

Consolidated system launcher for all startup modes.

```python
from unified_launcher import UnifiedLauncher

# Initialize launcher
launcher = UnifiedLauncher()

# List available modes
launcher.list_modes()

# Launch specific mode
success = launcher.launch_mode("core", ["--verbose"])

# Check running processes
launcher.status()

# Stop all processes
launcher.stop_all()
```

#### Classes

##### LaunchMode
```python
@dataclass
class LaunchMode:
    name: str
    description: str
    command: List[str]
    environment: Dict[str, str]
    working_dir: Optional[str] = None
    requires_admin: bool = False
    background: bool = False
```

#### Methods

##### `__init__()`
Initialize unified launcher.

##### `list_modes() -> None`
Display all available launch modes.

##### `launch_mode(mode_name: str, args: List[str] = None) -> bool`
Launch a specific mode.

**Parameters:**
- `mode_name`: Name of mode to launch
- `args`: Additional arguments

**Returns:**
- `bool`: True if successful

##### `status() -> None`
Show status of running processes.

##### `stop_all() -> None`
Stop all running processes.

##### `interactive_menu() -> None`
Show interactive mode selection menu.

## Configuration API

### Configuration Structure

```json
{
  "logging": {
    "level": "INFO",
    "format": "structured",
    "enable_metrics": true,
    "max_file_size": "10MB",
    "backup_count": 3
  },
  "security": {
    "default_privilege_level": "user",
    "enable_threat_detection": true,
    "jwt_secret": "your-secret-key",
    "session_timeout": 3600
  },
  "ai_providers": {
    "default_provider": "smart_selection",
    "enable_performance_tracking": true,
    "providers": {
      "openai": {
        "enabled": true,
        "api_key": "your-openai-key",
        "model": "gpt-4"
      },
      "claude": {
        "enabled": true,
        "api_key": "your-claude-key",
        "model": "claude-3-sonnet"
      },
      "local_gguf": {
        "enabled": true,
        "model_path": "models/llama-7b.gguf"
      }
    }
  },
  "human_in_the_loop": {
    "enabled": true,
    "risk_threshold": 0.7,
    "timeout": 30,
    "require_justification": true
  },
  "persistence": {
    "enabled": false,
    "adaptive_mode": true,
    "check_interval": 60,
    "threat_threshold": 0.5
  }
}
```

## Error Handling

### Common Exceptions

#### `SystemInitializationError`
Raised when system initialization fails.

#### `SecurityViolationError`
Raised when security constraints are violated.

#### `ProviderNotAvailableError`
Raised when requested AI provider is not available.

#### `ApprovalTimeoutError`
Raised when human approval times out.

#### `ConfigurationError`
Raised when configuration is invalid.

### Example Error Handling

```python
from core_orchestrator import CoreOrchestrator
from security_manager import SecurityViolationError

try:
    orchestrator = CoreOrchestrator()
    await orchestrator.start_system()
except SystemInitializationError as e:
    logger.error(f"Failed to initialize system: {e}")
    # Handle initialization failure
except SecurityViolationError as e:
    logger.error(f"Security violation: {e}")
    # Handle security violation
except Exception as e:
    logger.error(f"Unexpected error: {e}")
    # Handle unexpected errors
```

## Integration Examples

### Complete System Integration

```python
import asyncio
from core_orchestrator import CoreOrchestrator
from security_manager import PrivilegeLevel

async def main():
    # Initialize system
    orchestrator = CoreOrchestrator("config_optimized.json")
    
    try:
        # Start system
        await orchestrator.start_system()
        
        # Set security level
        await orchestrator.set_security_level(PrivilegeLevel.OPERATOR)
        
        # Queue commands for processing
        await orchestrator.queue_command({
            "type": "query_ai",
            "data": {
                "prompt": "Analyze system performance",
                "task_type": "analysis"
            }
        })
        
        # Monitor system status
        while True:
            status = orchestrator.get_status()
            if not status.running:
                break
            await asyncio.sleep(1)
            
    finally:
        # Graceful shutdown
        await orchestrator.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
```

### Custom Component Integration

```python
from structured_logger import StructuredLogger
from security_manager import SecurityManager
from smart_provider_selector import SmartProviderSelector, TaskType

class MyCustomComponent:
    def __init__(self):
        self.logger = StructuredLogger("custom_component")
        self.security = SecurityManager()
        self.ai_selector = SmartProviderSelector({
            "default_provider": "openai",
            "providers": {"openai": {"enabled": True}}
        }, self.logger)
    
    async def process_request(self, request: str):
        # Log request
        self.logger.info("Processing request", extra={
            "request_id": "req_123",
            "request": request
        })
        
        # Security check
        threat_level = await self.security.assess_threat_level(request, {})
        if threat_level.value in ["critical", "extreme"]:
            self.logger.warning("High threat request blocked", extra={
                "threat_level": threat_level.value,
                "request": request
            })
            return {"error": "Request blocked due to security concerns"}
        
        # Process with AI
        response = await self.ai_selector.get_completion(
            request, TaskType.GENERAL
        )
        
        # Log success
        self.logger.info("Request processed successfully", extra={
            "request_id": "req_123",
            "provider": response["provider"]
        })
        
        return response
```

## Performance Considerations

### Optimization Guidelines

1. **Async Operations**: Use async/await for all I/O operations
2. **Resource Pooling**: Reuse connections and resources where possible
3. **Caching**: Implement caching for frequently accessed data
4. **Logging Levels**: Use appropriate logging levels in production
5. **Memory Management**: Properly clean up resources and connections

### Performance Monitoring

```python
# Monitor system performance
status = orchestrator.get_status()
metrics = logger.get_metrics()
provider_metrics = ai_selector.get_provider_metrics()

print(f"Active components: {len(status.active_components)}")
print(f"Log entries: {metrics['total_entries']}")
print(f"AI requests: {provider_metrics['total_requests']}")
```

---

*This API reference covers the core functionality of the Obvivlorum AI Symbiote System v2.0. For additional examples and advanced usage patterns, refer to the individual component documentation and source code.*