import React from 'react'
import { motion } from 'framer-motion'
import { useQuery } from '@tanstack/react-query'
import { CpuChipIcon } from '@heroicons/react/24/outline'
import { api } from '../services/api'

const protocolColors = {
  ALPHA: 'bg-purple-500',
  BETA: 'bg-blue-500',
  GAMMA: 'bg-green-500',
  DELTA: 'bg-orange-500',
  OMEGA: 'bg-red-500',
}

const protocolDescriptions = {
  ALPHA: 'Scientific & Disruptive Research',
  BETA: 'Mobile App Development',
  GAMMA: 'Enterprise Architecture',
  DELTA: 'Web Application Development',
  OMEGA: 'Licensing & IP Management',
}

export default function ProtocolStatus() {
  const { data: protocols, isLoading } = useQuery({
    queryKey: ['protocols'],
    queryFn: api.getProtocols,
  })

  const { data: metrics } = useQuery({
    queryKey: ['metrics'],
    queryFn: api.getMetrics,
  })

  if (isLoading) {
    return (
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
        <h2 className="text-lg font-semibold mb-4">AION Protocols</h2>
        <div className="animate-pulse space-y-3">
          {[1, 2, 3, 4, 5].map((i) => (
            <div key={i} className="h-20 bg-gray-200 dark:bg-gray-700 rounded-lg"></div>
          ))}
        </div>
      </div>
    )
  }

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6"
    >
      <h2 className="text-lg font-semibold mb-4 flex items-center">
        <CpuChipIcon className="h-5 w-5 mr-2 text-blue-600" />
        AION Protocols
      </h2>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {protocols?.protocols && Object.entries(protocols.protocols).map(([name, protocol]: [string, any], index) => {
          const stats = metrics?.components?.aion?.protocols?.[name]
          const executions = stats?.total_executions || 0
          const successRate = stats?.success_rate || 0
          
          return (
            <motion.div
              key={name}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1 }}
              className="bg-gray-50 dark:bg-gray-700/50 rounded-lg p-4 border border-gray-200 dark:border-gray-600 hover:shadow-md transition-shadow"
            >
              <div className="flex items-start justify-between mb-3">
                <div className="flex items-center">
                  <div className={`w-3 h-3 rounded-full ${protocolColors[name as keyof typeof protocolColors]} mr-2`}></div>
                  <h3 className="font-semibold text-gray-900 dark:text-gray-100">
                    {name}
                  </h3>
                </div>
                <span className="text-xs text-gray-500 dark:text-gray-400">
                  v{protocol.version}
                </span>
              </div>
              
              <p className="text-sm text-gray-600 dark:text-gray-300 mb-3">
                {protocolDescriptions[name as keyof typeof protocolDescriptions]}
              </p>
              
              <div className="space-y-2">
                <div className="flex justify-between text-xs">
                  <span className="text-gray-500 dark:text-gray-400">Executions</span>
                  <span className="font-medium">{executions}</span>
                </div>
                <div className="flex justify-between text-xs">
                  <span className="text-gray-500 dark:text-gray-400">Success Rate</span>
                  <span className={`font-medium ${
                    successRate > 0.9 ? 'text-green-600' : 
                    successRate > 0.7 ? 'text-orange-600' : 'text-red-600'
                  }`}>
                    {(successRate * 100).toFixed(1)}%
                  </span>
                </div>
              </div>

              {protocol.capabilities && protocol.capabilities.length > 0 && (
                <div className="mt-3 pt-3 border-t border-gray-200 dark:border-gray-600">
                  <div className="flex flex-wrap gap-1">
                    {protocol.capabilities.slice(0, 3).map((cap: string, idx: number) => (
                      <span
                        key={idx}
                        className="text-xs px-2 py-1 bg-gray-200 dark:bg-gray-600 rounded-full"
                      >
                        {cap}
                      </span>
                    ))}
                    {protocol.capabilities.length > 3 && (
                      <span className="text-xs text-gray-500 dark:text-gray-400">
                        +{protocol.capabilities.length - 3}
                      </span>
                    )}
                  </div>
                </div>
              )}
            </motion.div>
          )
        })}
      </div>
    </motion.div>
  )
}