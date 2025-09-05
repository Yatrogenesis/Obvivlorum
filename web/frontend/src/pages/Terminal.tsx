import React, { useState, useRef, useEffect } from 'react'
import { motion } from 'framer-motion'
import { CommandLineIcon, ArrowUpIcon } from '@heroicons/react/24/outline'
import { useMutation } from '@tanstack/react-query'
import toast from 'react-hot-toast'
import { api } from '../services/api'

interface CommandHistory {
  id: string
  command: string
  output: string
  error?: string
  timestamp: Date
  distro?: string
}

export default function Terminal() {
  const [command, setCommand] = useState('')
  const [distro, setDistro] = useState<string>('Ubuntu')
  const [history, setHistory] = useState<CommandHistory[]>([])
  const [historyIndex, setHistoryIndex] = useState(-1)
  const terminalRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  const executeMutation = useMutation({
    mutationFn: (data: { command: string; distro?: string }) =>
      api.executeLinuxCommand(data.command, data.distro),
    onSuccess: (data, variables) => {
      const newEntry: CommandHistory = {
        id: Date.now().toString(),
        command: variables.command,
        output: data.stdout || '',
        error: data.stderr,
        timestamp: new Date(),
        distro: variables.distro,
      }
      setHistory((prev) => [...prev, newEntry])
      setCommand('')
      setHistoryIndex(-1)
    },
    onError: (error: any) => {
      toast.error('Command execution failed')
      const newEntry: CommandHistory = {
        id: Date.now().toString(),
        command,
        output: '',
        error: error.message || 'Command execution failed',
        timestamp: new Date(),
        distro,
      }
      setHistory((prev) => [...prev, newEntry])
    },
  })

  useEffect(() => {
    // Scroll to bottom when new output is added
    if (terminalRef.current) {
      terminalRef.current.scrollTop = terminalRef.current.scrollHeight
    }
  }, [history])

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (!command.trim()) return

    executeMutation.mutate({ command, distro })
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'ArrowUp') {
      e.preventDefault()
      if (historyIndex < history.length - 1) {
        const newIndex = historyIndex + 1
        setHistoryIndex(newIndex)
        setCommand(history[history.length - 1 - newIndex].command)
      }
    } else if (e.key === 'ArrowDown') {
      e.preventDefault()
      if (historyIndex > 0) {
        const newIndex = historyIndex - 1
        setHistoryIndex(newIndex)
        setCommand(history[history.length - 1 - newIndex].command)
      } else if (historyIndex === 0) {
        setHistoryIndex(-1)
        setCommand('')
      }
    }
  }

  const clearTerminal = () => {
    setHistory([])
    setCommand('')
    setHistoryIndex(-1)
  }

  return (
    <div className="space-y-6 h-full">
      {/* Header */}
      <div className="bg-gradient-to-r from-gray-800 to-gray-900 rounded-lg shadow-lg p-6 text-white">
        <h1 className="text-3xl font-bold flex items-center">
          <CommandLineIcon className="h-8 w-8 mr-2" />
          Terminal
        </h1>
        <p className="mt-2 text-gray-300">
          Execute Linux commands via WSL integration
        </p>
      </div>

      {/* Terminal Container */}
      <div className="bg-gray-900 rounded-lg shadow-lg overflow-hidden flex flex-col" style={{ height: 'calc(100vh - 300px)' }}>
        {/* Terminal Header */}
        <div className="bg-gray-800 px-4 py-2 flex items-center justify-between border-b border-gray-700">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-red-500"></div>
            <div className="w-3 h-3 rounded-full bg-yellow-500"></div>
            <div className="w-3 h-3 rounded-full bg-green-500"></div>
            <span className="ml-4 text-sm text-gray-400">AI Symbiote Terminal - {distro}</span>
          </div>
          <div className="flex items-center gap-4">
            <select
              value={distro}
              onChange={(e) => setDistro(e.target.value)}
              className="bg-gray-700 text-gray-300 px-3 py-1 rounded text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="Ubuntu">Ubuntu</option>
              <option value="ParrotOS">ParrotOS</option>
              <option value="docker-desktop">Docker Desktop</option>
            </select>
            <button
              onClick={clearTerminal}
              className="text-gray-400 hover:text-white text-sm"
            >
              Clear
            </button>
          </div>
        </div>

        {/* Terminal Output */}
        <div
          ref={terminalRef}
          className="flex-1 overflow-y-auto p-4 font-mono text-sm"
          onClick={() => inputRef.current?.focus()}
        >
          {history.length === 0 && (
            <div className="text-gray-500">
              <p>Welcome to AI Symbiote Terminal</p>
              <p>Type 'help' for available commands</p>
              <p className="mt-2">Connected to: {distro}</p>
            </div>
          )}

          {history.map((entry) => (
            <motion.div
              key={entry.id}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="mb-4"
            >
              <div className="flex items-start">
                <span className="text-green-400 mr-2">➜</span>
                <span className="text-blue-400 mr-2">~</span>
                <span className="text-white flex-1">{entry.command}</span>
                {entry.distro && (
                  <span className="text-gray-500 text-xs ml-2">[{entry.distro}]</span>
                )}
              </div>
              {entry.output && (
                <pre className="text-gray-300 mt-1 whitespace-pre-wrap pl-6">
                  {entry.output}
                </pre>
              )}
              {entry.error && (
                <pre className="text-red-400 mt-1 whitespace-pre-wrap pl-6">
                  {entry.error}
                </pre>
              )}
            </motion.div>
          ))}

          {executeMutation.isPending && (
            <div className="flex items-center text-gray-400">
              <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-gray-400 mr-2"></div>
              Executing command...
            </div>
          )}
        </div>

        {/* Terminal Input */}
        <form onSubmit={handleSubmit} className="border-t border-gray-700 p-4">
          <div className="flex items-center">
            <span className="text-green-400 mr-2">➜</span>
            <span className="text-blue-400 mr-2">~</span>
            <input
              ref={inputRef}
              type="text"
              value={command}
              onChange={(e) => setCommand(e.target.value)}
              onKeyDown={handleKeyDown}
              disabled={executeMutation.isPending}
              className="flex-1 bg-transparent text-white outline-none font-mono"
              placeholder="Enter command..."
              autoFocus
            />
            <button
              type="submit"
              disabled={executeMutation.isPending || !command.trim()}
              className="ml-2 p-2 text-gray-400 hover:text-white disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <ArrowUpIcon className="h-4 w-4 rotate-90" />
            </button>
          </div>
        </form>
      </div>

      {/* Help Text */}
      <div className="bg-gray-800 rounded-lg p-4 text-sm text-gray-400">
        <p className="font-semibold mb-2">Terminal Tips:</p>
        <ul className="space-y-1">
          <li>• Use ↑/↓ arrow keys to navigate command history</li>
          <li>• Select different WSL distributions from the dropdown</li>
          <li>• Commands are executed in a safe environment with timeout protection</li>
          <li>• Dangerous commands are automatically blocked for safety</li>
        </ul>
      </div>
    </div>
  )
}