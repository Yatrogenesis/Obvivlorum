# AI Symbiote Deployment Guide

**Complete setup and deployment instructions for the AI Symbiote system**

## Pre-Deployment Checklist

### System Requirements
- [x] Windows 10/11 or Linux (Ubuntu 18.04+)
- [x] Python 3.8 or higher
- [x] At least 4GB RAM available
- [x] 2GB free disk space
- [x] Administrative privileges (for persistence features)
- [x] WSL2 installed and configured (Windows users)

### Network Requirements
- [x] Internet connection for initial setup
- [x] Firewall exceptions if needed for local services
- [x] Proxy configuration if in corporate environment

## Installation Process

### Step 1: Environment Preparation

#### Windows Users
```powershell
# Enable WSL2 (Run as Administrator)
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart

# Restart system, then set WSL2 as default
wsl --set-default-version 2

# Install Ubuntu distribution
wsl --install -d Ubuntu

# Verify installation
wsl --list --verbose
```

#### Linux Users
```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install required system packages
sudo apt install -y python3 python3-pip python3-venv git curl wget

# Verify Python installation
python3 --version
```

### Step 2: AI Symbiote Installation

```bash
# Navigate to deployment directory
cd /path/to/Obvivlorum

# Set up Python virtual environment (recommended)
python3 -m venv ai_symbiote_env
source ai_symbiote_env/bin/activate  # Linux/WSL
# or
ai_symbiote_env\Scripts\activate     # Windows CMD

# Install AION dependencies
cd AION
pip install -r requirements.txt
cd ..

# Install additional system packages if needed
pip install psutil requests
```

### Step 3: Initial Configuration

```bash
# Create user-specific configuration
cp AION/config.json AION/config_production.json

# Edit configuration for production
nano AION/config_production.json
```

**Production Configuration Example:**
```json
{
  "performance_targets": {
    "response_time": "500ms",
    "failure_rate": "0.01%",
    "availability": "99.9%",
    "memory_usage": "<100MB baseline",
    "throughput": "1000+ requests/second"
  },
  "integration": {
    "obvivlorum_bridge": "aion_obvivlorum_bridge",
    "bridge_mode": "bidirectional",
    "shared_memory_size": "512MB"
  },
  "security": {
    "encryption_level": "AES-256",
    "certificate_pinning": true,
    "runtime_protection": true,
    "quantum_resistant": true
  },
  "logging": {
    "level": "INFO",
    "file_path": "/var/log/ai_symbiote/aion_protocol.log",
    "rotation": "daily",
    "retention": "7 days"
  },
  "system": {
    "auto_update": false,
    "architecture_evolution_rate": 0.02,
    "coherence_threshold": 0.90,
    "emergency_rollback": true
  }
}
```

### Step 4: System Validation

```bash
# Run comprehensive tests
python test_comprehensive.py

# Verify all components
python ai_symbiote.py --status --user-id production_user

# Test core functionality
python ai_symbiote.py --test-protocol ALPHA --user-id production_user
```

## Production Deployment

### Deployment Option 1: Standalone Service

#### Create System Service (Linux)
```bash
# Create service user
sudo useradd -r -s /bin/false ai_symbiote

# Create service directories
sudo mkdir -p /opt/ai_symbiote
sudo mkdir -p /var/log/ai_symbiote
sudo mkdir -p /var/lib/ai_symbiote

# Copy application files
sudo cp -r /path/to/Obvivlorum/* /opt/ai_symbiote/
sudo chown -R ai_symbiote:ai_symbiote /opt/ai_symbiote
sudo chown -R ai_symbiote:ai_symbiote /var/log/ai_symbiote
sudo chown -R ai_symbiote:ai_symbiote /var/lib/ai_symbiote

# Create systemd service file
sudo tee /etc/systemd/system/ai_symbiote.service > /dev/null <<EOF
[Unit]
Description=AI Symbiote - Adaptive AI Assistant
After=network.target
Wants=network.target

[Service]
Type=simple
User=ai_symbiote
Group=ai_symbiote
WorkingDirectory=/opt/ai_symbiote
Environment=PYTHONPATH=/opt/ai_symbiote
ExecStart=/usr/bin/python3 /opt/ai_symbiote/ai_symbiote.py --user-id production --background --persistent
ExecReload=/bin/kill -HUP \$MAINPID
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable ai_symbiote
sudo systemctl start ai_symbiote

# Check service status
sudo systemctl status ai_symbiote
```

#### Create Windows Service
```powershell
# Using NSSM (Non-Sucking Service Manager)
# Download NSSM from https://nssm.cc/

# Install service
nssm install "AI Symbiote" "C:\Python39\python.exe"
nssm set "AI Symbiote" AppParameters "D:\Obvivlorum\ai_symbiote.py --user-id production --background --persistent"
nssm set "AI Symbiote" AppDirectory "D:\Obvivlorum"
nssm set "AI Symbiote" DisplayName "AI Symbiote - Adaptive AI Assistant"
nssm set "AI Symbiote" Description "Advanced AI system with cross-platform capabilities"
nssm set "AI Symbiote" Start SERVICE_AUTO_START

# Start service
net start "AI Symbiote"
```

### Deployment Option 2: Docker Container

#### Create Dockerfile
```dockerfile
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create application user
RUN useradd -r -s /bin/false ai_symbiote

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY AION/requirements.txt /app/AION/
RUN pip install -r AION/requirements.txt

# Copy application code
COPY . /app/
RUN chown -R ai_symbiote:ai_symbiote /app

# Switch to application user
USER ai_symbiote

# Expose any required ports (if needed)
# EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python ai_symbiote.py --status --user-id docker || exit 1

# Start the application
CMD ["python", "ai_symbiote.py", "--user-id", "docker", "--background"]
```

#### Docker Compose Configuration
```yaml
version: '3.8'

services:
  ai_symbiote:
    build: .
    container_name: ai_symbiote
    restart: unless-stopped
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    environment:
      - PYTHONUNBUFFERED=1
      - AI_SYMBIOTE_USER_ID=docker
      - AI_SYMBIOTE_CONFIG=/app/config/production.json
    healthcheck:
      test: ["CMD", "python", "ai_symbiote.py", "--status", "--user-id", "docker"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    networks:
      - ai_symbiote_network

networks:
  ai_symbiote_network:
    driver: bridge
```

## Monitoring and Management

### Logging Configuration

#### Centralized Logging (Linux)
```bash
# Configure rsyslog
sudo tee /etc/rsyslog.d/ai_symbiote.conf > /dev/null <<EOF
# AI Symbiote logging
if \$programname == 'ai_symbiote' then /var/log/ai_symbiote/application.log
& stop
EOF

# Restart rsyslog
sudo systemctl restart rsyslog

# Configure log rotation
sudo tee /etc/logrotate.d/ai_symbiote > /dev/null <<EOF
/var/log/ai_symbiote/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    create 0644 ai_symbiote ai_symbiote
    postrotate
        systemctl reload ai_symbiote
    endscript
}
EOF
```

#### Log Monitoring Script
```bash
#!/bin/bash
# save as monitor_ai_symbiote.sh

LOG_FILE="/var/log/ai_symbiote/application.log"
ERROR_THRESHOLD=5
WARNING_THRESHOLD=10

# Count recent errors and warnings
ERRORS=$(tail -n 1000 "$LOG_FILE" | grep -c "ERROR")
WARNINGS=$(tail -n 1000 "$LOG_FILE" | grep -c "WARNING")

echo "AI Symbiote Health Check - $(date)"
echo "Recent errors: $ERRORS (threshold: $ERROR_THRESHOLD)"
echo "Recent warnings: $WARNINGS (threshold: $WARNING_THRESHOLD)"

if [ $ERRORS -gt $ERROR_THRESHOLD ]; then
    echo "ALERT: Error threshold exceeded!"
    # Send notification (email, Slack, etc.)
fi

if [ $WARNINGS -gt $WARNING_THRESHOLD ]; then
    echo "WARNING: Warning threshold exceeded!"
    # Send notification
fi

# Check if service is running
if systemctl is-active --quiet ai_symbiote; then
    echo "Service status: RUNNING"
else
    echo "Service status: STOPPED"
    echo "CRITICAL: AI Symbiote service is not running!"
    # Restart service and send alert
    sudo systemctl start ai_symbiote
fi
```

### Performance Monitoring

#### System Resource Monitor
```python
#!/usr/bin/env python3
# save as monitor_resources.py

import psutil
import json
import time
from datetime import datetime

def collect_metrics():
    """Collect system performance metrics."""
    return {
        "timestamp": datetime.now().isoformat(),
        "cpu_percent": psutil.cpu_percent(interval=1),
        "memory_percent": psutil.virtual_memory().percent,
        "disk_usage": psutil.disk_usage('/').percent,
        "network_io": dict(psutil.net_io_counters()._asdict()),
        "process_count": len(psutil.pids()),
        "boot_time": datetime.fromtimestamp(psutil.boot_time()).isoformat()
    }

def main():
    """Main monitoring loop."""
    while True:
        try:
            metrics = collect_metrics()
            
            # Save to file
            with open('/var/log/ai_symbiote/metrics.json', 'a') as f:
                f.write(json.dumps(metrics) + '\n')
            
            # Check thresholds
            if metrics["cpu_percent"] > 80:
                print(f"HIGH CPU: {metrics['cpu_percent']:.1f}%")
            
            if metrics["memory_percent"] > 85:
                print(f"HIGH MEMORY: {metrics['memory_percent']:.1f}%")
            
            if metrics["disk_usage"] > 90:
                print(f"HIGH DISK: {metrics['disk_usage']:.1f}%")
            
            time.sleep(60)  # Collect metrics every minute
            
        except Exception as e:
            print(f"Monitoring error: {e}")
            time.sleep(60)

if __name__ == "__main__":
    main()
```

## Security Hardening

### File Permissions
```bash
# Set appropriate file permissions
sudo chmod 750 /opt/ai_symbiote
sudo chmod 640 /opt/ai_symbiote/*.py
sudo chmod 600 /opt/ai_symbiote/AION/config*.json
sudo chmod 700 /var/log/ai_symbiote
sudo chmod 700 /var/lib/ai_symbiote
```

### Firewall Configuration
```bash
# Configure UFW (Uncomplicated Firewall)
sudo ufw enable
sudo ufw default deny incoming
sudo ufw default allow outgoing

# Allow SSH (if needed)
sudo ufw allow ssh

# Allow specific ports if AI Symbiote needs them
# sudo ufw allow 8080/tcp
```

### AppArmor Profile (Advanced)
```bash
# Create AppArmor profile
sudo tee /etc/apparmor.d/ai_symbiote > /dev/null <<EOF
#include <tunables/global>

/usr/bin/python3 {
  #include <abstractions/base>
  #include <abstractions/python>
  
  /opt/ai_symbiote/** r,
  /var/log/ai_symbiote/** rw,
  /var/lib/ai_symbiote/** rw,
  
  # Network access
  network inet stream,
  network inet dgram,
  
  # System access (limited)
  /proc/sys/kernel/random/uuid r,
  /sys/class/net/ r,
  
  # Deny dangerous paths
  deny /etc/passwd r,
  deny /etc/shadow r,
  deny /** w,
}
EOF

# Load profile
sudo apparmor_parser -r /etc/apparmor.d/ai_symbiote
```

## Backup and Recovery

### Automated Backup Script
```bash
#!/bin/bash
# save as backup_ai_symbiote.sh

BACKUP_DIR="/backup/ai_symbiote"
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_NAME="ai_symbiote_backup_$DATE"

# Create backup directory
mkdir -p "$BACKUP_DIR"

# Stop service
sudo systemctl stop ai_symbiote

# Create backup
tar -czf "$BACKUP_DIR/$BACKUP_NAME.tar.gz" \
    /opt/ai_symbiote \
    /var/lib/ai_symbiote \
    /etc/systemd/system/ai_symbiote.service

# Restart service
sudo systemctl start ai_symbiote

# Clean old backups (keep last 30 days)
find "$BACKUP_DIR" -name "ai_symbiote_backup_*.tar.gz" -mtime +30 -delete

echo "Backup completed: $BACKUP_DIR/$BACKUP_NAME.tar.gz"
```

### Recovery Procedure
```bash
#!/bin/bash
# save as restore_ai_symbiote.sh

BACKUP_FILE="$1"

if [ -z "$BACKUP_FILE" ]; then
    echo "Usage: $0 <backup_file.tar.gz>"
    exit 1
fi

# Stop service
sudo systemctl stop ai_symbiote

# Backup current installation
tar -czf "/tmp/ai_symbiote_current_$(date +%Y%m%d_%H%M%S).tar.gz" \
    /opt/ai_symbiote \
    /var/lib/ai_symbiote

# Restore from backup
tar -xzf "$BACKUP_FILE" -C /

# Fix permissions
sudo chown -R ai_symbiote:ai_symbiote /opt/ai_symbiote
sudo chown -R ai_symbiote:ai_symbiote /var/lib/ai_symbiote

# Reload systemd and start service
sudo systemctl daemon-reload
sudo systemctl start ai_symbiote

echo "Recovery completed from: $BACKUP_FILE"
```

## Maintenance Procedures

### Regular Maintenance Tasks

#### Weekly Maintenance Script
```bash
#!/bin/bash
# save as weekly_maintenance.sh

echo "AI Symbiote Weekly Maintenance - $(date)"

# Update system packages (optional)
# sudo apt update && sudo apt upgrade -y

# Clean old log files
find /var/log/ai_symbiote -name "*.log.*" -mtime +7 -delete

# Clean temporary files
sudo -u ai_symbiote find /var/lib/ai_symbiote/temp -type f -mtime +3 -delete

# Restart service to clear memory
sudo systemctl restart ai_symbiote

# Wait for service to start
sleep 10

# Run health check
python3 /opt/ai_symbiote/ai_symbiote.py --status --user-id production

echo "Weekly maintenance completed"
```

#### Database Maintenance (if applicable)
```bash
#!/bin/bash
# save as db_maintenance.sh

# Compact task facilitator databases
sudo -u ai_symbiote python3 -c "
import sqlite3
import os
from pathlib import Path

data_dirs = [
    '/var/lib/ai_symbiote/task_data',
    '/home/ai_symbiote/.ai_symbiote/task_data'
]

for data_dir in data_dirs:
    if os.path.exists(data_dir):
        for db_file in Path(data_dir).glob('*.db'):
            print(f'Compacting {db_file}')
            conn = sqlite3.connect(str(db_file))
            conn.execute('VACUUM')
            conn.close()
"

echo "Database maintenance completed"
```

## Troubleshooting Production Issues

### Common Production Issues

#### 1. Service Won't Start
```bash
# Check service status
sudo systemctl status ai_symbiote

# Check logs
sudo journalctl -u ai_symbiote -f

# Check configuration
python3 /opt/ai_symbiote/ai_symbiote.py --status --user-id production

# Check file permissions
ls -la /opt/ai_symbiote/
ls -la /var/lib/ai_symbiote/
```

#### 2. High Memory Usage
```bash
# Check memory usage
free -h
ps aux | grep ai_symbiote

# Check for memory leaks in logs
grep -i "memory" /var/log/ai_symbiote/application.log

# Restart service to clear memory
sudo systemctl restart ai_symbiote
```

#### 3. Network Connectivity Issues
```bash
# Check network configuration
ip addr show
netstat -tlnp | grep python

# Test external connectivity
curl -I https://www.google.com

# Check firewall rules
sudo ufw status verbose
```

### Emergency Procedures

#### Safe Mode Boot
```bash
# Start in safe mode (minimal functionality)
sudo systemctl stop ai_symbiote
sudo -u ai_symbiote python3 /opt/ai_symbiote/ai_symbiote.py \
    --user-id emergency \
    --config /opt/ai_symbiote/config/safe_mode.json
```

#### Complete System Reset
```bash
#!/bin/bash
# save as emergency_reset.sh

echo "EMERGENCY RESET - This will reset all AI Symbiote data!"
read -p "Are you sure? (type 'RESET' to confirm): " confirm

if [ "$confirm" != "RESET" ]; then
    echo "Reset cancelled"
    exit 1
fi

# Stop service
sudo systemctl stop ai_symbiote

# Backup current data
tar -czf "/tmp/ai_symbiote_emergency_backup_$(date +%Y%m%d_%H%M%S).tar.gz" \
    /var/lib/ai_symbiote

# Clear data directories
sudo rm -rf /var/lib/ai_symbiote/*
sudo rm -rf /var/log/ai_symbiote/*

# Recreate directories
sudo mkdir -p /var/lib/ai_symbiote/task_data
sudo mkdir -p /var/log/ai_symbiote

# Fix permissions
sudo chown -R ai_symbiote:ai_symbiote /var/lib/ai_symbiote
sudo chown -R ai_symbiote:ai_symbiote /var/log/ai_symbiote

# Start service
sudo systemctl start ai_symbiote

echo "Emergency reset completed"
```

## Performance Optimization

### Production Tuning
```bash
# Increase file descriptor limits
echo "ai_symbiote soft nofile 65536" | sudo tee -a /etc/security/limits.conf
echo "ai_symbiote hard nofile 65536" | sudo tee -a /etc/security/limits.conf

# Optimize Python for production
export PYTHONOPTIMIZE=1
export PYTHONDONTWRITEBYTECODE=1

# Configure swap usage
echo "vm.swappiness=10" | sudo tee -a /etc/sysctl.conf
```

### Resource Limits
```bash
# Create systemd override directory
sudo mkdir -p /etc/systemd/system/ai_symbiote.service.d

# Create resource limits override
sudo tee /etc/systemd/system/ai_symbiote.service.d/resources.conf > /dev/null <<EOF
[Service]
MemoryMax=2G
CPUQuota=200%
TasksMax=1000
LimitNOFILE=65536
EOF

# Reload systemd
sudo systemctl daemon-reload
sudo systemctl restart ai_symbiote
```

## Conclusion

This deployment guide provides comprehensive instructions for deploying the AI Symbiote system in production environments. Follow the security hardening steps and monitoring procedures to ensure reliable operation.

For additional support or advanced deployment scenarios, refer to the main README.md or contact the development team.

**Remember:** Always test deployments in a staging environment before applying to production systems.