import React from 'react'
import { motion } from 'framer-motion'
import { useQuery } from '@tanstack/react-query'
import {
  CpuChipIcon,
  ClipboardDocumentListIcon,
  ServerStackIcon,
  ChartBarIcon,
  BoltIcon,
  ArrowTrendingUpIcon,
} from '@heroicons/react/24/outline'
import { api } from '../services/api'
import { useSystemStore } from '../store/systemStore'
import SystemStatusCard from '../components/SystemStatusCard'
import RecentActivity from '../components/RecentActivity'
import QuickActions from '../components/QuickActions'
import ProtocolStatus from '../components/ProtocolStatus'

export default function Dashboard() {
  const { systemStatus } = useSystemStore()

  const { data: metrics } = useQuery({
    queryKey: ['metrics'],
    queryFn: api.getMetrics,
    refetchInterval: 5000, // Refresh every 5 seconds
  })

  const { data: tasks } = useQuery({
    queryKey: ['tasks'],
    queryFn: api.getTasks,
    refetchInterval: 10000, // Refresh every 10 seconds
  })

  const stats = [
    {
      name: 'System Status',
      value: systemStatus?.is_running ? 'Online' : 'Offline',
      icon: ServerStackIcon,
      color: systemStatus?.is_running ? 'text-green-600' : 'text-red-600',
      bgColor: systemStatus?.is_running ? 'bg-green-100' : 'bg-red-100',
    },
    {
      name: 'Active Protocols',
      value: Object.keys(systemStatus?.components || {}).filter(c => c.includes('protocol')).length || 5,
      icon: CpuChipIcon,
      color: 'text-blue-600',
      bgColor: 'bg-blue-100',
    },
    {
      name: 'Active Tasks',
      value: tasks?.tasks?.length || 0,
      icon: ClipboardDocumentListIcon,
      color: 'text-purple-600',
      bgColor: 'bg-purple-100',
    },
    {
      name: 'Success Rate',
      value: `${Math.round((metrics?.components?.aion?.success_rate || 0) * 100)}%`,
      icon: ChartBarIcon,
      color: 'text-orange-600',
      bgColor: 'bg-orange-100',
    },
  ]

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-gradient-to-r from-blue-600 to-purple-600 rounded-lg shadow-lg p-6 text-white">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold flex items-center">
              <BoltIcon className="h-8 w-8 mr-2" />
              AI Symbiote Dashboard
            </h1>
            <p className="mt-2 text-blue-100">
              Advanced AI System with AION Protocol Integration
            </p>
          </div>
          <ArrowTrendingUpIcon className="h-12 w-12 text-blue-200" />
        </div>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-4">
        {stats.map((stat, index) => (
          <motion.div
            key={stat.name}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3, delay: index * 0.1 }}
          >
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6 card-hover">
              <div className="flex items-center">
                <div className={`p-3 rounded-lg ${stat.bgColor} dark:bg-opacity-20`}>
                  <stat.icon className={`h-6 w-6 ${stat.color}`} />
                </div>
                <div className="ml-4">
                  <p className="text-sm font-medium text-gray-600 dark:text-gray-400">
                    {stat.name}
                  </p>
                  <p className={`text-2xl font-bold ${stat.color}`}>
                    {stat.value}
                  </p>
                </div>
              </div>
            </div>
          </motion.div>
        ))}
      </div>

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        {/* System Status */}
        <div className="lg:col-span-2">
          <SystemStatusCard />
        </div>

        {/* Quick Actions */}
        <div>
          <QuickActions />
        </div>

        {/* Protocol Status */}
        <div className="lg:col-span-2">
          <ProtocolStatus />
        </div>

        {/* Recent Activity */}
        <div>
          <RecentActivity />
        </div>
      </div>
    </div>
  )
}