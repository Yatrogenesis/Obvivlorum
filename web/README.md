# AI Symbiote Web Interface

Modern web interface for the AI Symbiote system with real-time monitoring, protocol execution, and task management.

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Node.js 16+
- npm or yarn

### Backend Setup

1. Navigate to backend directory:
```bash
cd web/backend
```

2. Create virtual environment:
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the backend server:
```bash
python main.py
```

The API will be available at `http://localhost:8000`
- API docs: `http://localhost:8000/api/docs`
- ReDoc: `http://localhost:8000/api/redoc`

### Frontend Setup

1. Navigate to frontend directory:
```bash
cd web/frontend
```

2. Install dependencies:
```bash
npm install
# or
yarn install
```

3. Start development server:
```bash
npm run dev
# or
yarn dev
```

The UI will be available at `http://localhost:3000`

## 📚 API Documentation

### REST Endpoints

#### System Status
```http
GET /api/system/status
```
Returns current system status including running state and component information.

#### Execute Protocol
```http
POST /api/protocols/{protocol_name}/execute
Content-Type: application/json

{
  "protocol": "ALPHA",
  "parameters": {
    "research_domain": "test",
    "research_type": "exploratory"
  }
}
```

#### List Protocols
```http
GET /api/protocols
```

#### Create Task
```http
POST /api/tasks
Content-Type: application/json

{
  "name": "Task name",
  "description": "Task description",
  "priority": 5,
  "tags": ["tag1", "tag2"]
}
```

#### Execute Linux Command
```http
POST /api/linux/execute
Content-Type: application/json

{
  "command": "ls -la",
  "distro": "Ubuntu",
  "timeout": 30
}
```

### WebSocket Connection

Connect to `ws://localhost:8000/ws` for real-time updates.

Message types:
- `system_status` - System status updates
- `protocol_executed` - Protocol execution notifications
- `task_created` - Task creation notifications

## 🎨 UI Features

### Dashboard
- Real-time system status monitoring
- Component health visualization
- Quick action buttons
- Recent activity feed
- Protocol status cards

### Protocols Page
- Interactive protocol execution
- JSON parameter editor
- Real-time execution results
- Protocol documentation

### Tasks Page
- Task creation and management
- Priority and tag management
- Progress tracking
- Due date scheduling

### Terminal Page
- Web-based terminal interface
- Linux command execution
- WSL distro selection
- Command history

### Metrics Page
- Performance visualizations
- Protocol execution statistics
- Task completion rates
- System resource usage

## 🛠️ Development

### Backend Development

The backend uses FastAPI with async support. Key files:
- `main.py` - Main application and endpoints
- `models.py` - Pydantic models
- `services/` - Business logic
- `utils/` - Utility functions

### Frontend Development

The frontend uses React with TypeScript. Key directories:
- `components/` - Reusable UI components
- `pages/` - Page components
- `hooks/` - Custom React hooks
- `services/` - API services
- `store/` - Zustand state management

### Adding New Features

1. Backend: Add endpoint in `main.py`
2. Frontend: Add API call in `services/api.ts`
3. Create/update components as needed
4. Update store if state management needed

## 🐛 Troubleshooting

### Backend Issues

1. **Port already in use**:
   ```bash
   # Find process using port 8000
   netstat -ano | findstr :8000
   # Kill the process
   taskkill /PID <process_id> /F
   ```

2. **Module not found**:
   Ensure you're in the virtual environment and all dependencies are installed.

### Frontend Issues

1. **Dependencies not installing**:
   ```bash
   # Clear cache and reinstall
   rm -rf node_modules package-lock.json
   npm install
   ```

2. **WebSocket connection failed**:
   Ensure backend is running and CORS is properly configured.

## 📦 Production Deployment

### Backend
```bash
# Use production server
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Frontend
```bash
# Build for production
npm run build
# Serve with static server
npx serve -s dist -p 3000
```

## 🔒 Security Considerations

- Enable HTTPS in production
- Use environment variables for sensitive data
- Implement proper authentication
- Configure CORS appropriately
- Use rate limiting for API endpoints