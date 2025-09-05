import { useEffect, useRef, useCallback } from 'react'
import { useSystemStore } from '../store/systemStore'
import toast from 'react-hot-toast'

const WS_URL = 'ws://localhost:8000/ws'

export function useWebSocket() {
  const ws = useRef<WebSocket | null>(null)
  const reconnectTimeout = useRef<NodeJS.Timeout>()
  const { setConnectionStatus, setSystemStatus, addActivity } = useSystemStore()

  const connect = useCallback(() => {
    try {
      ws.current = new WebSocket(WS_URL)

      ws.current.onopen = () => {
        setConnectionStatus('connected')
        toast.success('Connected to AI Symbiote')
        console.log('WebSocket connected')
        
        // Clear any reconnect timeout
        if (reconnectTimeout.current) {
          clearTimeout(reconnectTimeout.current)
        }
      }

      ws.current.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data)
          
          switch (data.type) {
            case 'system_status':
              setSystemStatus(data.data)
              break
              
            case 'connection_established':
              setSystemStatus(data.data)
              break
              
            case 'protocol_executed':
              addActivity({
                type: 'protocol',
                message: `Protocol ${data.data.protocol} executed: ${data.data.status}`,
                timestamp: data.data.timestamp,
              })
              break
              
            case 'task_created':
              addActivity({
                type: 'task',
                message: `Task created: ${data.data.name}`,
                timestamp: data.data.timestamp,
              })
              break
              
            default:
              console.log('Unknown WebSocket message type:', data.type)
          }
        } catch (error) {
          console.error('Error parsing WebSocket message:', error)
        }
      }

      ws.current.onerror = (error) => {
        console.error('WebSocket error:', error)
        toast.error('Connection error')
      }

      ws.current.onclose = () => {
        setConnectionStatus('disconnected')
        console.log('WebSocket disconnected')
        
        // Attempt to reconnect after 5 seconds
        reconnectTimeout.current = setTimeout(() => {
          console.log('Attempting to reconnect...')
          setConnectionStatus('connecting')
          connect()
        }, 5000)
      }
    } catch (error) {
      console.error('Error creating WebSocket:', error)
      setConnectionStatus('disconnected')
    }
  }, [setConnectionStatus, setSystemStatus, addActivity])

  const disconnect = useCallback(() => {
    if (ws.current) {
      ws.current.close()
      ws.current = null
    }
    
    if (reconnectTimeout.current) {
      clearTimeout(reconnectTimeout.current)
    }
  }, [])

  const sendMessage = useCallback((message: any) => {
    if (ws.current && ws.current.readyState === WebSocket.OPEN) {
      ws.current.send(JSON.stringify(message))
    } else {
      console.error('WebSocket is not connected')
    }
  }, [])

  // Heartbeat to keep connection alive
  useEffect(() => {
    const interval = setInterval(() => {
      sendMessage({ type: 'ping' })
    }, 30000) // Every 30 seconds

    return () => clearInterval(interval)
  }, [sendMessage])

  return {
    connect,
    disconnect,
    sendMessage,
    isConnected: ws.current?.readyState === WebSocket.OPEN,
  }
}