import { create } from 'zustand'
import { devtools } from 'zustand/middleware'

interface SystemStatus {
  is_running: boolean
  user_id: string
  components: Record<string, any>
  uptime?: number
  memory_usage?: Record<string, number>
}

interface SystemState {
  connectionStatus: 'connected' | 'connecting' | 'disconnected'
  systemStatus: SystemStatus | null
  activities: Array<{
    id: string
    type: string
    message: string
    timestamp: string
  }>
  setConnectionStatus: (status: 'connected' | 'connecting' | 'disconnected') => void
  setSystemStatus: (status: SystemStatus) => void
  addActivity: (activity: Omit<SystemState['activities'][0], 'id'>) => void
  clearActivities: () => void
}

export const useSystemStore = create<SystemState>()(
  devtools(
    (set) => ({
      connectionStatus: 'disconnected',
      systemStatus: null,
      activities: [],
      
      setConnectionStatus: (status) => set({ connectionStatus: status }),
      
      setSystemStatus: (status) => set({ systemStatus: status }),
      
      addActivity: (activity) =>
        set((state) => ({
          activities: [
            {
              id: Math.random().toString(36).substr(2, 9),
              ...activity,
            },
            ...state.activities.slice(0, 99), // Keep last 100 activities
          ],
        })),
      
      clearActivities: () => set({ activities: [] }),
    }),
    {
      name: 'system-store',
    }
  )
)