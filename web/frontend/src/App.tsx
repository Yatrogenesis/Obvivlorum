import React, { useEffect } from 'react'
import { Routes, Route } from 'react-router-dom'
import { useWebSocket } from './hooks/useWebSocket'
import { useSystemStore } from './store/systemStore'
import Layout from './components/Layout'
import Dashboard from './pages/Dashboard'
import Protocols from './pages/Protocols'
import Tasks from './pages/Tasks'
import Terminal from './pages/Terminal'
import Metrics from './pages/Metrics'
import Settings from './pages/Settings'

function App() {
  const { connect, disconnect } = useWebSocket()
  const { setConnectionStatus } = useSystemStore()

  useEffect(() => {
    // Connect to WebSocket on app mount
    connect()
    setConnectionStatus('connecting')

    return () => {
      disconnect()
      setConnectionStatus('disconnected')
    }
  }, [])

  return (
    <Layout>
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/protocols" element={<Protocols />} />
        <Route path="/tasks" element={<Tasks />} />
        <Route path="/terminal" element={<Terminal />} />
        <Route path="/metrics" element={<Metrics />} />
        <Route path="/settings" element={<Settings />} />
      </Routes>
    </Layout>
  )
}

export default App