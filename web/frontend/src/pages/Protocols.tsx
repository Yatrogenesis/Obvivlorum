import React, { useState } from 'react'
import { motion } from 'framer-motion'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { CpuChipIcon, PlayIcon, CodeBracketIcon } from '@heroicons/react/24/outline'
import toast from 'react-hot-toast'
import { api } from '../services/api'

const protocolExamples = {
  ALPHA: {
    research_domain: "quantum_computing",
    research_type: "exploratory",
    seed_concepts: ["quantum_entanglement", "neural_networks"],
    create_domain: true
  },
  BETA: {
    app_name: "MyMobileApp",
    architecture: "flutter",
    platforms: ["Android", "iOS"],
    features: ["authentication", "real-time_sync"]
  },
  GAMMA: {
    project_name: "EnterpriseSystem",
    architecture: "microservices",
    components: ["api_gateway", "auth_service", "data_service"]
  },
  DELTA: {
    app_name: "WebDashboard",
    tech_stack: "next_js",
    features: ["ssr", "api_routes", "authentication"]
  },
  OMEGA: {
    project_name: "MyProject",
    action: "generate_license",
    license_type: "COMMERCIAL_BASIC",
    software_name: "MyApplication"
  }
}

export default function Protocols() {
  const queryClient = useQueryClient()
  const [selectedProtocol, setSelectedProtocol] = useState<string>('ALPHA')
  const [parameters, setParameters] = useState<string>(
    JSON.stringify(protocolExamples.ALPHA, null, 2)
  )
  const [executionResults, setExecutionResults] = useState<any>(null)

  const { data: protocols, isLoading } = useQuery({
    queryKey: ['protocols'],
    queryFn: api.getProtocols,
  })

  const executeMutation = useMutation({
    mutationFn: (data: { protocol: string; parameters: any }) => 
      api.executeProtocol(data.protocol, data.parameters),
    onSuccess: (data) => {
      toast.success('Protocol executed successfully')
      setExecutionResults(data)
      queryClient.invalidateQueries({ queryKey: ['metrics'] })
    },
    onError: (error: any) => {
      toast.error('Failed to execute protocol')
      console.error(error)
    },
  })

  const handleProtocolSelect = (protocol: string) => {
    setSelectedProtocol(protocol)
    setParameters(JSON.stringify(
      protocolExamples[protocol as keyof typeof protocolExamples], 
      null, 
      2
    ))
    setExecutionResults(null)
  }

  const handleExecute = () => {
    try {
      const parsedParams = JSON.parse(parameters)
      executeMutation.mutate({
        protocol: selectedProtocol,
        parameters: parsedParams,
      })
    } catch (error) {
      toast.error('Invalid JSON parameters')
    }
  }

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-gray-900"></div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-gradient-to-r from-purple-600 to-blue-600 rounded-lg shadow-lg p-6 text-white">
        <h1 className="text-3xl font-bold flex items-center">
          <CpuChipIcon className="h-8 w-8 mr-2" />
          AION Protocols
        </h1>
        <p className="mt-2 text-purple-100">
          Execute and manage AI protocols for various development scenarios
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Protocol List */}
        <div className="lg:col-span-1">
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
            <h2 className="text-lg font-semibold mb-4">Available Protocols</h2>
            <div className="space-y-2">
              {protocols?.protocols && Object.entries(protocols.protocols).map(([name, protocol]: [string, any]) => (
                <motion.button
                  key={name}
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  onClick={() => handleProtocolSelect(name)}
                  className={`w-full text-left p-4 rounded-lg border transition-all ${
                    selectedProtocol === name
                      ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
                      : 'border-gray-200 dark:border-gray-700 hover:border-gray-300'
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <div>
                      <h3 className="font-semibold">{name}</h3>
                      <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                        {protocol.description}
                      </p>
                    </div>
                    <span className="text-xs text-gray-500">v{protocol.version}</span>
                  </div>
                </motion.button>
              ))}
            </div>
          </div>
        </div>

        {/* Protocol Executor */}
        <div className="lg:col-span-2 space-y-6">
          {/* Parameters Editor */}
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-lg font-semibold flex items-center">
                <CodeBracketIcon className="h-5 w-5 mr-2" />
                Protocol Parameters
              </h2>
              <button
                onClick={handleExecute}
                disabled={executeMutation.isPending}
                className="flex items-center px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                <PlayIcon className="h-4 w-4 mr-2" />
                {executeMutation.isPending ? 'Executing...' : 'Execute'}
              </button>
            </div>
            
            <div className="relative">
              <textarea
                value={parameters}
                onChange={(e) => setParameters(e.target.value)}
                className="w-full h-64 p-4 font-mono text-sm bg-gray-50 dark:bg-gray-900 border border-gray-200 dark:border-gray-700 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                placeholder="Enter protocol parameters as JSON..."
              />
            </div>

            <div className="mt-4 flex items-center justify-between">
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Protocol: <span className="font-semibold">{selectedProtocol}</span>
              </p>
              <button
                onClick={() => setParameters(JSON.stringify(
                  protocolExamples[selectedProtocol as keyof typeof protocolExamples], 
                  null, 
                  2
                ))}
                className="text-sm text-blue-600 hover:text-blue-700"
              >
                Reset to example
              </button>
            </div>
          </div>

          {/* Execution Results */}
          {executionResults && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6"
            >
              <h2 className="text-lg font-semibold mb-4">Execution Results</h2>
              <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4 overflow-x-auto">
                <pre className="text-sm font-mono">
                  {JSON.stringify(executionResults, null, 2)}
                </pre>
              </div>
            </motion.div>
          )}
        </div>
      </div>
    </div>
  )
}