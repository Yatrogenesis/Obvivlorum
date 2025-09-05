import React, { useState } from 'react'
import { motion } from 'framer-motion'
import { 
  PlayIcon, 
  StopIcon, 
  ArrowPathIcon,
  CommandLineIcon,
  PlusIcon,
  BoltIcon
} from '@heroicons/react/24/outline'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import toast from 'react-hot-toast'
import { api } from '../services/api'

export default function QuickActions() {
  const queryClient = useQueryClient()
  const [isExecuting, setIsExecuting] = useState(false)

  const executeMutation = useMutation({
    mutationFn: (data: { protocol: string; parameters: any }) => 
      api.executeProtocol(data.protocol, data.parameters),
    onSuccess: () => {
      toast.success('Protocol executed successfully')
      queryClient.invalidateQueries({ queryKey: ['metrics'] })
    },
    onError: () => {
      toast.error('Failed to execute protocol')
    },
  })

  const quickActions = [
    {
      name: 'Run ALPHA Protocol',
      icon: PlayIcon,
      color: 'text-green-600',
      bgColor: 'bg-green-100',
      action: () => {
        executeMutation.mutate({
          protocol: 'ALPHA',
          parameters: {
            research_domain: 'quick_test',
            research_type: 'exploratory',
            create_domain: true,
          },
        })
      },
    },
    {
      name: 'Create Task',
      icon: PlusIcon,
      color: 'text-blue-600',
      bgColor: 'bg-blue-100',
      action: () => {
        // This would open a modal in a real app
        toast.info('Task creation dialog would open here')
      },
    },
    {
      name: 'Open Terminal',
      icon: CommandLineIcon,
      color: 'text-purple-600',
      bgColor: 'bg-purple-100',
      action: () => {
        window.location.href = '/terminal'
      },
    },
    {
      name: 'Refresh Status',
      icon: ArrowPathIcon,
      color: 'text-orange-600',
      bgColor: 'bg-orange-100',
      action: () => {
        queryClient.invalidateQueries()
        toast.success('Status refreshed')
      },
    },
  ]

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6"
    >
      <h2 className="text-lg font-semibold mb-4 flex items-center">
        <BoltIcon className="h-5 w-5 mr-2 text-blue-600" />
        Quick Actions
      </h2>

      <div className="grid grid-cols-2 gap-3">
        {quickActions.map((action) => (
          <motion.button
            key={action.name}
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            onClick={action.action}
            disabled={executeMutation.isPending}
            className={`p-4 rounded-lg border border-gray-200 dark:border-gray-700 hover:shadow-md transition-all ${
              executeMutation.isPending ? 'opacity-50 cursor-not-allowed' : ''
            }`}
          >
            <div className={`inline-flex p-2 rounded-lg ${action.bgColor} dark:bg-opacity-20 mb-2`}>
              <action.icon className={`h-5 w-5 ${action.color}`} />
            </div>
            <p className="text-sm font-medium text-gray-900 dark:text-gray-100">
              {action.name}
            </p>
          </motion.button>
        ))}
      </div>

      {executeMutation.isPending && (
        <div className="mt-4 flex items-center justify-center">
          <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-600"></div>
          <span className="ml-2 text-sm text-gray-600 dark:text-gray-400">
            Executing...
          </span>
        </div>
      )}
    </motion.div>
  )
}