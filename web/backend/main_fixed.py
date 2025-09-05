#!/usr/bin/env python3
"""
AI Symbiote Web API - Version Corregida
=======================================

Version sin dependencias del core AI Symbiote para evitar errores.
Proporciona una API funcional con datos mock para la interfaz web.
"""

import os
import sys
from typing import Dict, List, Any, Optional
from datetime import datetime
import asyncio
import json
import random

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Create FastAPI app
app = FastAPI(
    title="AI Symbiote API - Fixed Version",
    description="REST API for AI Symbiote system with mock data",
    version="1.0.1",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mock data
MOCK_PROTOCOLS = {
    "ALPHA": {
        "name": "ALPHA",
        "version": "1.0",
        "description": "Scientific & Disruptive Research Protocol",
        "capabilities": ["research", "analysis", "innovation"]
    },
    "BETA": {
        "name": "BETA", 
        "version": "1.0",
        "description": "Mobile Application Development Protocol",
        "capabilities": ["mobile", "flutter", "react-native"]
    },
    "GAMMA": {
        "name": "GAMMA",
        "version": "1.0", 
        "description": "Enterprise System Architecture Protocol",
        "capabilities": ["microservices", "scalability", "enterprise"]
    },
    "DELTA": {
        "name": "DELTA",
        "version": "1.0",
        "description": "Web Application Development Protocol", 
        "capabilities": ["web", "frontend", "backend"]
    },
    "OMEGA": {
        "name": "OMEGA",
        "version": "1.0",
        "description": "Software Licensing & IP Management Protocol",
        "capabilities": ["licensing", "legal", "compliance"]
    }
}

MOCK_TASKS = [
    {
        "task_id": "task_001",
        "name": "Analyze quantum computing trends",
        "status": "in_progress",
        "priority": 8,
        "progress": 0.65,
        "created_at": "2025-08-04T10:00:00",
        "tags": ["research", "quantum"]
    },
    {
        "task_id": "task_002", 
        "name": "Develop mobile app prototype",
        "status": "pending",
        "priority": 6,
        "progress": 0.0,
        "created_at": "2025-08-04T09:30:00",
        "tags": ["mobile", "prototype"]
    }
]

MOCK_SYSTEM_STATUS = {
    "is_running": True,
    "user_id": "web_user",
    "components": {
        "aion_protocol": {"is_active": True, "status": "running"},
        "bridge": {"is_active": True, "status": "connected"}, 
        "linux_executor": {"is_active": True, "wsl_available": True},
        "task_facilitator": {"is_active": True, "tasks_count": len(MOCK_TASKS)},
        "web_interface": {"is_active": True, "status": "online"}
    },
    "uptime": 3600  # 1 hora
}

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections[:]:
            try:
                await connection.send_json(message)
            except:
                self.active_connections.remove(connection)

manager = ConnectionManager()

# Pydantic models
class SystemStatus(BaseModel):
    is_running: bool
    user_id: str
    components: Dict[str, Any]
    uptime: Optional[float] = None

class ProtocolExecutionRequest(BaseModel):
    protocol: str = Field(..., pattern="^(ALPHA|BETA|GAMMA|DELTA|OMEGA)$")
    parameters: Dict[str, Any]

class TaskCreateRequest(BaseModel):
    name: str
    description: Optional[str] = None
    priority: int = Field(5, ge=1, le=10)
    tags: List[str] = []

class LinuxCommandRequest(BaseModel):
    command: str
    distro: Optional[str] = None
    timeout: int = Field(30, ge=1, le=300)

# API Endpoints
@app.get("/")
async def root():
    return {
        "name": "AI Symbiote API - Fixed Version",
        "version": "1.0.1",
        "status": "online",
        "docs": "/api/docs"
    }

@app.get("/api/system/status", response_model=SystemStatus)
async def get_system_status():
    return SystemStatus(**MOCK_SYSTEM_STATUS)

@app.post("/api/protocols/{protocol_name}/execute")
async def execute_protocol(protocol_name: str, request: ProtocolExecutionRequest):
    if protocol_name.upper() not in MOCK_PROTOCOLS:
        raise HTTPException(status_code=400, detail="Unknown protocol")
    
    # Simulate execution
    await asyncio.sleep(0.1)  # Simulate processing time
    
    execution_id = f"{protocol_name.upper()}-{random.randint(1000000, 9999999)}"
    
    result = {
        "status": "success",
        "execution_id": execution_id,
        "protocol": protocol_name.upper(),
        "parameters": request.parameters,
        "execution_time": 0.1,
        "timestamp": datetime.now().isoformat()
    }
    
    # Broadcast execution event
    await manager.broadcast({
        "type": "protocol_executed",
        "data": {
            "protocol": protocol_name.upper(),
            "status": "success",
            "timestamp": datetime.now().isoformat()
        }
    })
    
    return result

@app.get("/api/protocols")
async def list_protocols():
    return {"protocols": MOCK_PROTOCOLS}

@app.post("/api/tasks")
async def create_task(request: TaskCreateRequest):
    task_id = f"task_{random.randint(100, 999)}"
    
    new_task = {
        "task_id": task_id,
        "name": request.name,
        "description": request.description,
        "status": "pending",
        "priority": request.priority,
        "tags": request.tags,
        "progress": 0.0,
        "created_at": datetime.now().isoformat()
    }
    
    MOCK_TASKS.append(new_task)
    
    await manager.broadcast({
        "type": "task_created",
        "data": {
            "task_id": task_id,
            "name": request.name,
            "timestamp": datetime.now().isoformat()
        }
    })
    
    return {
        "task_id": task_id,
        "name": request.name,
        "status": "pending",
        "created_at": datetime.now(),
        "priority": request.priority
    }

@app.get("/api/tasks")
async def list_tasks():
    return {"tasks": MOCK_TASKS, "total": len(MOCK_TASKS)}

@app.post("/api/linux/execute")
async def execute_linux_command(request: LinuxCommandRequest):
    # Simulate command execution
    await asyncio.sleep(0.2)
    
    # Mock different outputs based on command
    if "ls" in request.command:
        output = "total 42\ndrwxr-xr-x 1 user user 4096 Aug  4 10:00 .\ndrwxr-xr-x 1 user user 4096 Aug  4 09:00 ..\n-rw-r--r-- 1 user user  123 Aug  4 10:00 file.txt"
    elif "whoami" in request.command:
        output = "web_user"
    elif "pwd" in request.command:
        output = "/home/web_user"
    elif "echo" in request.command:
        output = request.command.replace("echo ", "").strip("'\"")
    else:
        output = f"Command '{request.command}' executed successfully"
    
    return {
        "status": "success",
        "stdout": output,
        "stderr": None,
        "return_code": 0,
        "execution_time": 0.2
    }

@app.get("/api/metrics")
async def get_metrics():
    return {
        "timestamp": datetime.now().isoformat(),
        "components": {
            "aion": {
                "total_executions": random.randint(10, 100),
                "average_execution_time": round(random.uniform(0.1, 2.0), 3),
                "success_rate": round(random.uniform(0.85, 0.99), 2),
                "protocols": {
                    "ALPHA": {"total_executions": random.randint(5, 25), "success_rate": 0.95},
                    "BETA": {"total_executions": random.randint(3, 15), "success_rate": 0.92},
                    "GAMMA": {"total_executions": random.randint(2, 10), "success_rate": 0.88},
                    "DELTA": {"total_executions": random.randint(4, 20), "success_rate": 0.93},
                    "OMEGA": {"total_executions": random.randint(1, 8), "success_rate": 0.97}
                }
            },
            "tasks": {
                "active_tasks": len([t for t in MOCK_TASKS if t["status"] != "completed"]),
                "completed_tasks": len([t for t in MOCK_TASKS if t["status"] == "completed"]),
                "suggestions_generated": random.randint(5, 20)
            }
        }
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        # Send initial status
        await websocket.send_json({
            "type": "connection_established",
            "data": MOCK_SYSTEM_STATUS
        })
        
        # Keep connection alive
        while True:
            try:
                data = await websocket.receive_json()
                
                if data.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
                elif data.get("type") == "get_status":
                    await websocket.send_json({
                        "type": "system_status",
                        "data": MOCK_SYSTEM_STATUS
                    })
                    
            except WebSocketDisconnect:
                break
            except Exception as e:
                print(f"WebSocket error: {e}")
                
    finally:
        manager.disconnect(websocket)

@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.1-fixed"
    }

# Background task to simulate real-time updates
async def send_periodic_updates():
    while True:
        await asyncio.sleep(30)  # Every 30 seconds
        await manager.broadcast({
            "type": "system_heartbeat",
            "data": {
                "timestamp": datetime.now().isoformat(),
                "status": "running"
            }
        })

@app.on_event("startup")
async def startup_event():
    # Start background task
    asyncio.create_task(send_periodic_updates())
    print(" AI Symbiote API (Fixed Version) started successfully")
    print(" Frontend should connect to: http://localhost:8000")
    print(" API Documentation: http://localhost:8000/api/docs")

if __name__ == "__main__":
    print(" Starting AI Symbiote API Server (Fixed Version)...")
    uvicorn.run(
        "main_fixed:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )