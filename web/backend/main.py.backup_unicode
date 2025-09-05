#!/usr/bin/env python3
"""
AI Symbiote Web API - Production Version
=======================================

Modern FastAPI backend with lifespan events and full functionality.
"""

import os
import sys
from typing import Dict, List, Any, Optional
from datetime import datetime
import asyncio
import json
import random
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Mock data for demonstration
MOCK_PROTOCOLS = {
    "ALPHA": {
        "name": "ALPHA",
        "version": "1.0",
        "description": "Scientific & Disruptive Research Protocol",
        "capabilities": ["research", "analysis", "innovation", "quantum_computing"]
    },
    "BETA": {
        "name": "BETA", 
        "version": "1.0",
        "description": "Mobile Application Development Protocol",
        "capabilities": ["mobile", "flutter", "react_native", "cross_platform"]
    },
    "GAMMA": {
        "name": "GAMMA",
        "version": "1.0", 
        "description": "Enterprise System Architecture Protocol",
        "capabilities": ["microservices", "scalability", "enterprise", "kubernetes"]
    },
    "DELTA": {
        "name": "DELTA",
        "version": "1.0",
        "description": "Web Application Development Protocol", 
        "capabilities": ["web", "frontend", "backend", "full_stack"]
    },
    "OMEGA": {
        "name": "OMEGA",
        "version": "1.0",
        "description": "Software Licensing & IP Management Protocol",
        "capabilities": ["licensing", "legal", "compliance", "patents"]
    }
}

MOCK_TASKS = [
    {
        "task_id": "task_001",
        "name": "Analyze quantum computing trends",
        "description": "Research current quantum computing developments and applications",
        "status": "in_progress",
        "priority": 8,
        "progress": 0.65,
        "created_at": "2025-08-04T10:00:00",
        "tags": ["research", "quantum", "analysis"]
    },
    {
        "task_id": "task_002", 
        "name": "Develop mobile app prototype",
        "description": "Create MVP for AI-powered mobile application",
        "status": "pending",
        "priority": 6,
        "progress": 0.0,
        "created_at": "2025-08-04T09:30:00",
        "tags": ["mobile", "prototype", "mvp"]
    },
    {
        "task_id": "task_003",
        "name": "Design microservices architecture",
        "description": "Plan scalable enterprise system architecture",
        "status": "completed",
        "priority": 9,
        "progress": 1.0,
        "created_at": "2025-08-03T14:00:00",
        "tags": ["architecture", "microservices", "enterprise"]
    }
]

MOCK_SYSTEM_STATUS = {
    "is_running": True,
    "user_id": "web_user",
    "components": {
        "aion_protocol": {"is_active": True, "status": "running", "version": "2.0"},
        "bridge": {"is_active": True, "status": "connected", "connection_id": "bridge_001"}, 
        "linux_executor": {"is_active": True, "wsl_available": True, "distros": ["Ubuntu", "ParrotOS"]},
        "task_facilitator": {"is_active": True, "tasks_count": len(MOCK_TASKS), "learning_enabled": True},
        "web_interface": {"is_active": True, "status": "online", "version": "1.0"}
    },
    "uptime": 3600,  # 1 hora
    "memory_usage": {
        "total": 8192,
        "used": 2048,
        "free": 6144
    }
}

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"WebSocket connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            print(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")

    async def broadcast(self, message: dict):
        if not self.active_connections:
            return
            
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                print(f"Failed to send message to WebSocket: {e}")
                disconnected.append(connection)
        
        # Remove disconnected clients
        for conn in disconnected:
            self.disconnect(conn)

manager = ConnectionManager()

# Background task to simulate real-time updates
async def send_periodic_updates():
    """Send periodic updates to connected clients."""
    while True:
        await asyncio.sleep(30)  # Every 30 seconds
        
        # Update mock metrics
        MOCK_SYSTEM_STATUS["uptime"] += 30
        
        await manager.broadcast({
            "type": "system_heartbeat",
            "data": {
                "timestamp": datetime.now().isoformat(),
                "status": "running",
                "uptime": MOCK_SYSTEM_STATUS["uptime"],
                "active_connections": len(manager.active_connections)
            }
        })

# Lifespan event handler (replaces deprecated on_event)
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application lifespan events."""
    # Startup
    print("🚀 AI Symbiote API starting up...")
    
    # Start background task
    task = asyncio.create_task(send_periodic_updates())
    
    print("✅ AI Symbiote API started successfully")
    print("🌐 Frontend should connect to: http://localhost:8000")
    print("📚 API Documentation: http://localhost:8000/api/docs")
    
    yield
    
    # Shutdown
    print("🛑 AI Symbiote API shutting down...")
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    print("✅ AI Symbiote API shutdown complete")

# Create FastAPI app with lifespan
app = FastAPI(
    title="AI Symbiote API",
    description="Modern REST API and WebSocket interface for AI Symbiote system",
    version="1.1.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class SystemStatus(BaseModel):
    is_running: bool
    user_id: str
    components: Dict[str, Any]
    uptime: Optional[float] = None
    memory_usage: Optional[Dict[str, float]] = None

class ProtocolExecutionRequest(BaseModel):
    protocol: str = Field(..., pattern="^(ALPHA|BETA|GAMMA|DELTA|OMEGA)$")
    parameters: Dict[str, Any]

class ProtocolExecutionResponse(BaseModel):
    status: str
    execution_id: Optional[str] = None
    result: Dict[str, Any]
    execution_time: Optional[float] = None
    timestamp: str

class TaskCreateRequest(BaseModel):
    name: str
    description: Optional[str] = None
    priority: int = Field(5, ge=1, le=10)
    tags: List[str] = []
    due_date: Optional[str] = None

class TaskResponse(BaseModel):
    task_id: str
    name: str
    status: str
    created_at: str
    priority: int

class LinuxCommandRequest(BaseModel):
    command: str
    distro: Optional[str] = None
    timeout: int = Field(30, ge=1, le=300)

class LinuxCommandResponse(BaseModel):
    status: str
    stdout: Optional[str] = None
    stderr: Optional[str] = None
    return_code: Optional[int] = None
    execution_time: float

# API Endpoints
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "AI Symbiote API",
        "version": "1.1.0",
        "status": "online",
        "endpoints": {
            "docs": "/api/docs",
            "health": "/api/health",
            "system": "/api/system/status",
            "protocols": "/api/protocols",
            "tasks": "/api/tasks",
            "websocket": "/ws"
        },
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/system/status", response_model=SystemStatus)
async def get_system_status():
    """Get current system status."""
    return SystemStatus(**MOCK_SYSTEM_STATUS)

@app.post("/api/protocols/{protocol_name}/execute", response_model=ProtocolExecutionResponse)
async def execute_protocol(protocol_name: str, request: ProtocolExecutionRequest):
    """Execute an AION protocol."""
    protocol_upper = protocol_name.upper()
    
    if protocol_upper not in MOCK_PROTOCOLS:
        raise HTTPException(status_code=400, detail=f"Unknown protocol: {protocol_name}")
    
    if protocol_upper != request.protocol.upper():
        raise HTTPException(status_code=400, detail="Protocol name in path and body must match")
    
    # Simulate execution time
    execution_time = round(random.uniform(0.1, 1.0), 3)
    await asyncio.sleep(execution_time)
    
    execution_id = f"{protocol_upper}-{random.randint(1000000, 9999999)}"
    timestamp = datetime.now().isoformat()
    
    # Create realistic result based on protocol
    results = {
        "ALPHA": {
            "research_domain": request.parameters.get("research_domain", "general"),
            "concepts_discovered": random.randint(5, 20),
            "innovation_score": round(random.uniform(0.7, 0.95), 2),
            "recommendations": ["Explore quantum applications", "Consider AI integration"]
        },
        "BETA": {
            "app_name": request.parameters.get("app_name", "Mobile App"),
            "platforms": request.parameters.get("platforms", ["iOS", "Android"]),
            "features_implemented": random.randint(8, 15),
            "performance_score": round(random.uniform(0.8, 0.98), 2)
        },
        "GAMMA": {
            "project_name": request.parameters.get("project_name", "Enterprise System"),
            "services_created": random.randint(3, 12),
            "scalability_rating": round(random.uniform(0.85, 0.99), 2),
            "architecture_pattern": "microservices"
        },
        "DELTA": {
            "app_name": request.parameters.get("app_name", "Web Application"),
            "tech_stack": request.parameters.get("tech_stack", "React"),
            "components_created": random.randint(10, 25),
            "lighthouse_score": random.randint(85, 100)
        },
        "OMEGA": {
            "project_name": request.parameters.get("project_name", "Software Project"),
            "license_type": request.parameters.get("license_type", "MIT"),
            "compliance_score": round(random.uniform(0.9, 1.0), 2),
            "legal_review": "approved"
        }
    }
    
    result = {
        "status": "success",
        "execution_id": execution_id,
        "result": {
            "protocol": protocol_upper,
            "parameters": request.parameters,
            **results[protocol_upper]
        },
        "execution_time": execution_time,
        "timestamp": timestamp
    }
    
    # Broadcast execution event
    await manager.broadcast({
        "type": "protocol_executed",
        "data": {
            "protocol": protocol_upper,
            "status": "success",
            "execution_id": execution_id,
            "timestamp": timestamp
        }
    })
    
    return ProtocolExecutionResponse(**result)

@app.get("/api/protocols")
async def list_protocols():
    """List available AION protocols."""
    return {"protocols": MOCK_PROTOCOLS}

@app.get("/api/protocols/{protocol_name}")
async def get_protocol_info(protocol_name: str):
    """Get detailed information about a specific protocol."""
    protocol_upper = protocol_name.upper()
    
    if protocol_upper not in MOCK_PROTOCOLS:
        raise HTTPException(status_code=404, detail=f"Protocol {protocol_name} not found")
    
    return {
        "status": "success",
        **MOCK_PROTOCOLS[protocol_upper],
        "examples": {
            "ALPHA": {
                "research_domain": "quantum_computing",
                "research_type": "exploratory",
                "seed_concepts": ["quantum_entanglement", "neural_networks"]
            },
            "BETA": {
                "app_name": "MyApp",
                "architecture": "flutter",
                "platforms": ["Android", "iOS"]
            }
        }.get(protocol_upper, {})
    }

@app.post("/api/tasks", response_model=TaskResponse)
async def create_task(request: TaskCreateRequest):
    """Create a new task."""
    task_id = f"task_{random.randint(100, 999)}"
    timestamp = datetime.now().isoformat()
    
    new_task = {
        "task_id": task_id,
        "name": request.name,
        "description": request.description,
        "status": "pending",
        "priority": request.priority,
        "tags": request.tags,
        "progress": 0.0,
        "created_at": timestamp,
        "due_date": request.due_date
    }
    
    MOCK_TASKS.append(new_task)
    
    # Broadcast task creation
    await manager.broadcast({
        "type": "task_created",
        "data": {
            "task_id": task_id,
            "name": request.name,
            "priority": request.priority,
            "timestamp": timestamp
        }
    })
    
    return TaskResponse(
        task_id=task_id,
        name=request.name,
        status="pending",
        created_at=timestamp,
        priority=request.priority
    )

@app.get("/api/tasks")
async def list_tasks():
    """List all tasks."""
    return {
        "tasks": MOCK_TASKS, 
        "total": len(MOCK_TASKS),
        "active": len([t for t in MOCK_TASKS if t["status"] != "completed"]),
        "completed": len([t for t in MOCK_TASKS if t["status"] == "completed"])
    }

@app.get("/api/tasks/{task_id}")
async def get_task(task_id: str):
    """Get a specific task."""
    task = next((t for t in MOCK_TASKS if t["task_id"] == task_id), None)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task

@app.put("/api/tasks/{task_id}")
async def update_task(task_id: str, updates: Dict[str, Any]):
    """Update a task."""
    task = next((t for t in MOCK_TASKS if t["task_id"] == task_id), None)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    # Update allowed fields
    allowed_fields = ["name", "description", "priority", "status", "progress", "tags"]
    for field, value in updates.items():
        if field in allowed_fields:
            task[field] = value
    
    task["updated_at"] = datetime.now().isoformat()
    
    # Broadcast task update
    await manager.broadcast({
        "type": "task_updated",
        "data": {
            "task_id": task_id,
            "updates": updates,
            "timestamp": task["updated_at"]
        }
    })
    
    return task

@app.delete("/api/tasks/{task_id}")
async def delete_task(task_id: str):
    """Delete a task."""
    global MOCK_TASKS
    task = next((t for t in MOCK_TASKS if t["task_id"] == task_id), None)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    MOCK_TASKS = [t for t in MOCK_TASKS if t["task_id"] != task_id]
    
    # Broadcast task deletion
    await manager.broadcast({
        "type": "task_deleted",
        "data": {
            "task_id": task_id,
            "timestamp": datetime.now().isoformat()
        }
    })
    
    return {"message": "Task deleted successfully"}

@app.post("/api/linux/execute", response_model=LinuxCommandResponse)
async def execute_linux_command(request: LinuxCommandRequest):
    """Execute a Linux command via WSL."""
    start_time = asyncio.get_event_loop().time()
    
    # Simulate command execution with realistic delay
    await asyncio.sleep(min(request.timeout / 10, 2.0))
    
    # Mock different outputs based on command
    command = request.command.lower()
    if "ls" in command:
        output = "total 42\ndrwxr-xr-x 1 user user 4096 Aug  4 10:00 .\ndrwxr-xr-x 1 user user 4096 Aug  4 09:00 ..\n-rw-r--r-- 1 user user  123 Aug  4 10:00 file.txt\n-rw-r--r-- 1 user user  456 Aug  4 10:00 script.py"
    elif "whoami" in command:
        output = "web_user"
    elif "pwd" in command:
        output = "/home/web_user"
    elif "ps" in command:
        output = "  PID TTY          TIME CMD\n 1234 pts/0    00:00:01 bash\n 5678 pts/0    00:00:00 python"
    elif "uname" in command:
        output = f"Linux {request.distro or 'Ubuntu'} 5.15.0 #1 SMP x86_64 GNU/Linux"
    elif "echo" in command:
        # Extract text after echo
        import re
        match = re.search(r'echo\s+(.+)', request.command)
        output = match.group(1).strip("'\"") if match else "Hello World"
    elif "cat" in command and "/etc/os-release" in command:
        distro_name = request.distro or "Ubuntu"
        output = f'PRETTY_NAME="{distro_name} 22.04 LTS"\nNAME="{distro_name}"\nVERSION_ID="22.04"'
    elif any(danger in command for danger in ["rm -rf", "format", "fdisk", "del", "rm /*"]):
        # Block dangerous commands
        return LinuxCommandResponse(
            status="blocked",
            stdout=None,
            stderr="Command blocked for security reasons",
            return_code=1,
            execution_time=0.0
        )
    else:
        output = f"Command '{request.command}' executed successfully on {request.distro or 'default distro'}"
    
    execution_time = asyncio.get_event_loop().time() - start_time
    
    return LinuxCommandResponse(
        status="success",
        stdout=output,
        stderr=None,
        return_code=0,
        execution_time=round(execution_time, 3)
    )

@app.get("/api/metrics")
async def get_metrics():
    """Get system metrics and performance data."""
    return {
        "timestamp": datetime.now().isoformat(),
        "components": {
            "aion": {
                "total_executions": random.randint(50, 200),
                "average_execution_time": round(random.uniform(0.1, 2.0), 3),
                "success_rate": round(random.uniform(0.85, 0.99), 3),
                "protocols": {
                    "ALPHA": {"total_executions": random.randint(10, 50), "success_rate": 0.95},
                    "BETA": {"total_executions": random.randint(8, 30), "success_rate": 0.92},
                    "GAMMA": {"total_executions": random.randint(5, 25), "success_rate": 0.88},
                    "DELTA": {"total_executions": random.randint(12, 40), "success_rate": 0.93},
                    "OMEGA": {"total_executions": random.randint(3, 15), "success_rate": 0.97}
                }
            },
            "tasks": {
                "active_tasks": len([t for t in MOCK_TASKS if t["status"] != "completed"]),
                "completed_tasks": len([t for t in MOCK_TASKS if t["status"] == "completed"]),
                "suggestions_generated": random.randint(15, 50),
                "average_completion_time": round(random.uniform(2.5, 8.0), 1)
            },
            "system": {
                "uptime": MOCK_SYSTEM_STATUS["uptime"],
                "memory_usage": MOCK_SYSTEM_STATUS["memory_usage"],
                "active_connections": len(manager.active_connections)
            }
        }
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    await manager.connect(websocket)
    try:
        # Send initial status
        await websocket.send_json({
            "type": "connection_established",
            "data": MOCK_SYSTEM_STATUS
        })
        
        # Keep connection alive and handle messages
        while True:
            try:
                data = await websocket.receive_json()
                
                if data.get("type") == "ping":
                    await websocket.send_json({
                        "type": "pong",
                        "timestamp": datetime.now().isoformat()
                    })
                elif data.get("type") == "get_status":
                    await websocket.send_json({
                        "type": "system_status",
                        "data": MOCK_SYSTEM_STATUS
                    })
                    
            except WebSocketDisconnect:
                break
            except Exception as e:
                print(f"WebSocket error: {e}")
                break
                
    finally:
        manager.disconnect(websocket)

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.1.0",
        "uptime": MOCK_SYSTEM_STATUS["uptime"],
        "active_connections": len(manager.active_connections),
        "components": {
            "api": "online",
            "websocket": "online",
            "database": "mock",
            "protocols": "available"
        }
    }

if __name__ == "__main__":
    print("🚀 Starting AI Symbiote API Server...")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )