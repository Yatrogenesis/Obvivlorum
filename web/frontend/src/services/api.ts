import axios from 'axios'
import toast from 'react-hot-toast'

const API_BASE_URL = 'http://localhost:8000/api'

const axiosInstance = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Request interceptor
axiosInstance.interceptors.request.use(
  (config) => {
    // Add auth token if available
    const token = localStorage.getItem('auth_token')
    if (token) {
      config.headers.Authorization = `Bearer ${token}`
    }
    return config
  },
  (error) => {
    return Promise.reject(error)
  }
)

// Response interceptor
axiosInstance.interceptors.response.use(
  (response) => response.data,
  (error) => {
    const message = error.response?.data?.detail || error.message || 'An error occurred'
    
    if (error.response?.status === 503) {
      toast.error('AI Symbiote service unavailable')
    } else if (error.response?.status === 500) {
      toast.error('Server error: ' + message)
    } else if (error.response?.status === 400) {
      toast.error('Invalid request: ' + message)
    } else {
      toast.error(message)
    }
    
    return Promise.reject(error)
  }
)

export const api = {
  // System
  getSystemStatus: () => axiosInstance.get('/system/status'),
  getHealth: () => axiosInstance.get('/health'),
  getMetrics: () => axiosInstance.get('/metrics'),

  // Protocols
  getProtocols: () => axiosInstance.get('/protocols'),
  executeProtocol: (protocol: string, parameters: any) =>
    axiosInstance.post(`/protocols/${protocol}/execute`, {
      protocol,
      parameters,
    }),

  // Tasks
  getTasks: () => axiosInstance.get('/tasks'),
  createTask: (task: {
    name: string
    description?: string
    priority?: number
    tags?: string[]
    due_date?: string
  }) => axiosInstance.post('/tasks', task),
  updateTask: (taskId: string, updates: any) =>
    axiosInstance.put(`/tasks/${taskId}`, updates),
  deleteTask: (taskId: string) => axiosInstance.delete(`/tasks/${taskId}`),

  // Linux commands
  executeLinuxCommand: (command: string, distro?: string, timeout?: number) =>
    axiosInstance.post('/linux/execute', {
      command,
      distro,
      timeout: timeout || 30,
    }),
}

export default api