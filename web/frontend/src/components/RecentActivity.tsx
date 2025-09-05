import React from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  ClockIcon, 
  CheckCircleIcon, 
  XCircleIcon,
  InformationCircleIcon 
} from '@heroicons/react/24/outline'
import { useSystemStore } from '../store/systemStore'
import { format } from 'date-fns'

const activityIcons = {
  protocol: CheckCircleIcon,
  task: ClockIcon,
  error: XCircleIcon,
  info: InformationCircleIcon,
}

const activityColors = {
  protocol: 'text-green-600 bg-green-100',
  task: 'text-blue-600 bg-blue-100',
  error: 'text-red-600 bg-red-100',
  info: 'text-gray-600 bg-gray-100',
}

export default function RecentActivity() {
  const { activities } = useSystemStore()
  const recentActivities = activities.slice(0, 10)

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6"
    >
      <h2 className="text-lg font-semibold mb-4 flex items-center">
        <ClockIcon className="h-5 w-5 mr-2 text-blue-600" />
        Recent Activity
      </h2>

      {recentActivities.length === 0 ? (
        <p className="text-center text-gray-500 dark:text-gray-400 py-8">
          No recent activity
        </p>
      ) : (
        <div className="space-y-3 max-h-96 overflow-y-auto">
          <AnimatePresence>
            {recentActivities.map((activity, index) => {
              const Icon = activityIcons[activity.type as keyof typeof activityIcons] || InformationCircleIcon
              const colorClass = activityColors[activity.type as keyof typeof activityColors] || activityColors.info
              
              return (
                <motion.div
                  key={activity.id}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: 20 }}
                  transition={{ delay: index * 0.05 }}
                  className="flex items-start space-x-3"
                >
                  <div className={`p-1 rounded-full ${colorClass} dark:bg-opacity-20`}>
                    <Icon className="h-4 w-4" />
                  </div>
                  <div className="flex-1">
                    <p className="text-sm text-gray-900 dark:text-gray-100">
                      {activity.message}
                    </p>
                    <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                      {activity.timestamp 
                        ? format(new Date(activity.timestamp), 'HH:mm:ss')
                        : 'Just now'
                      }
                    </p>
                  </div>
                </motion.div>
              )
            })}
          </AnimatePresence>
        </div>
      )}

      {activities.length > 10 && (
        <div className="mt-4 pt-4 border-t border-gray-200 dark:border-gray-700">
          <p className="text-sm text-gray-500 dark:text-gray-400 text-center">
            Showing 10 of {activities.length} activities
          </p>
        </div>
      )}
    </motion.div>
  )
}