import React from 'react'
import { motion } from 'framer-motion'
import { CheckCircleIcon, XCircleIcon, CpuChipIcon } from '@heroicons/react/24/outline'
import { useSystemStore } from '../store/systemStore'

export default function SystemStatusCard() {
  const { systemStatus } = useSystemStore()

  if (!systemStatus) {
    return (
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
        <h2 className="text-lg font-semibold mb-4">System Status</h2>
        <div className="flex items-center justify-center h-32">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-gray-900 dark:border-gray-100"></div>
        </div>
      </div>
    )
  }

  const components = Object.entries(systemStatus.components || {})

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6"
    >
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-semibold flex items-center">
          <CpuChipIcon className="h-5 w-5 mr-2 text-blue-600" />
          System Components
        </h2>
        <span className={`px-2 py-1 text-xs rounded-full ${
          systemStatus.is_running 
            ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200'
            : 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200'
        }`}>
          {systemStatus.is_running ? 'Running' : 'Stopped'}
        </span>
      </div>

      <div className="space-y-3">
        {components.map(([name, status], index) => {
          const isActive = typeof status === 'object' 
            ? status.is_active || status.status === 'active' || status.enabled
            : status === true

          return (
            <motion.div
              key={name}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: index * 0.05 }}
              className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-700/50 rounded-lg"
            >
              <div className="flex items-center">
                {isActive ? (
                  <CheckCircleIcon className="h-5 w-5 text-green-500 mr-3" />
                ) : (
                  <XCircleIcon className="h-5 w-5 text-gray-400 mr-3" />
                )}
                <span className="font-medium capitalize">
                  {name.replace(/_/g, ' ')}
                </span>
              </div>
              <span className={`text-sm ${
                isActive ? 'text-green-600 dark:text-green-400' : 'text-gray-500'
              }`}>
                {isActive ? 'Active' : 'Inactive'}
              </span>
            </motion.div>
          )
        })}
      </div>

      {systemStatus.uptime && (
        <div className="mt-4 pt-4 border-t border-gray-200 dark:border-gray-700">
          <div className="flex justify-between text-sm">
            <span className="text-gray-600 dark:text-gray-400">Uptime</span>
            <span className="font-medium">
              {Math.floor(systemStatus.uptime / 3600)}h {Math.floor((systemStatus.uptime % 3600) / 60)}m
            </span>
          </div>
        </div>
      )}
    </motion.div>
  )
}