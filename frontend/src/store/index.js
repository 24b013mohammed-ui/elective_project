import { create } from 'zustand'

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:5066'

const useStore = create((set, get) => ({
  // Training data
  trainingHistory: {
    epochs: [],
    train_loss: [],
    val_loss: [],
    best_epoch: null,
    best_val_loss: null,
    total_epochs_trained: 0
  },

  // Evaluation data
  predictions: [],
  actualValues: [],
  errors: [],
  metrics: {
    normalized_scale: {},
    original_scale_inr: {}
  },
  predictionDates: [],
  predictionTickers: [],
  evaluationResults: null,

  // UI state
  currentPage: 'dashboard',
  isLoading: false,
  isTraining: false,
  trainingProgress: 0,
  error: null,
  success: null,
  selectedModel: 'latest',
  selectedTicker: null,
  selectedComparisonTickers: [],

  // Config for training
  trainConfig: {
    tickers: ['RELIANCE.NS', 'TCS.NS', 'INFY.NS'],
    start_date: '2020-01-01',
    end_date: '2024-12-31',
    batch_size: 16,
    num_epochs: 50,
    learning_rate: 0.001
  },

  // Multiple training runs for comparison
  trainingRuns: [],

  // ===== MUTATIONS =====

  setCurrentPage: (page) => set({ currentPage: page }),

  setTrainConfig: (config) => set((state) => ({
    trainConfig: { ...state.trainConfig, ...config }
  })),

  setSelectedTicker: (ticker) => set({ selectedTicker: ticker }),

  setError: (error) => {
    set({ error })
    setTimeout(() => set({ error: null }), 5000)
  },

  setSuccess: (message) => {
    set({ success: message })
    setTimeout(() => set({ success: null }), 4000)
  },

  clearError: () => set({ error: null }),

  // ===== API CALLS =====

  fetchTrainingHistory: async () => {
    try {
      set({ isLoading: true })
      const response = await fetch(`${API_BASE_URL}/api/training-history`)
      if (!response.ok) throw new Error('Failed to fetch training history')
      const data = await response.json()
      set({ trainingHistory: data, isLoading: false })
    } catch (error) {
      set({ error: error.message, isLoading: false })
    }
  },

  fetchMetrics: async () => {
    try {
      set({ isLoading: true })
      const response = await fetch(`${API_BASE_URL}/api/metrics`)
      if (!response.ok) throw new Error('Failed to fetch metrics')
      const data = await response.json()
      set({ metrics: data, isLoading: false })
    } catch (error) {
      set({ error: error.message, isLoading: false })
    }
  },

  fetchEvaluation: async () => {
    try {
      set({ isLoading: true })
      const response = await fetch(`${API_BASE_URL}/api/evaluate`)
      if (!response.ok) throw new Error('Failed to fetch evaluation results')
      
      const data = await response.json()
      
      // Debug logging
      console.log('[Store] Fetched evaluation data:', {
        predictionsCount: data.predictions_original?.length || 0,
        actualsCount: data.targets_original?.length || 0,
        errorsCount: data.errors_original?.length || 0,
        datesCount: data.dates?.length || 0,
        tickersCount: data.tickers?.length || 0,
        errorsExample: data.errors_original?.slice(0, 5) || []
      })
      
      // Filter errors more relaxed - accept any number whether finite or not
      const rawErrors = data.errors_original || []
      const filteredErrors = rawErrors.filter(e => typeof e === 'number')
      console.log('[Store] Error filtering - Raw count:', rawErrors.length, ', Filtered count:', filteredErrors.length)
      
      set({
        evaluationResults: data,
        predictions: data.predictions_original || [],
        actualValues: data.targets_original || [],
        errors: filteredErrors,
        predictionDates: data.dates || [],
        predictionTickers: data.tickers || [],
        metrics: {
          normalized_scale: data.metrics_normalized || {},
          original_scale_inr: data.metrics_original || {}
        },
        isLoading: false
      })
    } catch (error) {
      console.error('[Store] Error fetching evaluation:', error)
      set({ error: error.message, isLoading: false })
    }
  },

  fetchResultsWithRetry: async (maxRetries = 3, delayMs = 1000) => {
    for (let attempt = 0; attempt < maxRetries; attempt++) {
      try {
        // Wait before attempting (exponential backoff)
        await new Promise((resolve) => setTimeout(resolve, delayMs * (attempt + 1)))
        
        // Fetch all results sequentially
        console.log(`[Attempt ${attempt + 1}/${maxRetries}] Fetching training results...`)
        await get().fetchTrainingHistory()
        await get().fetchMetrics()
        await get().fetchEvaluation()
        
        console.log('✓ Training results fetched successfully!')
        set({ success: 'Training results loaded!' })
        return true
      } catch (error) {
        console.warn(`[Attempt ${attempt + 1}/${maxRetries}] Failed to fetch:`, error.message)
        if (attempt === maxRetries - 1) {
          set({ error: `Failed to fetch results after ${maxRetries} attempts. Try manual refresh.` })
          return false
        }
      }
    }
  },

  manuallyFetchResults: async () => {
    console.log('🔄 Manually fetching training results...')
    set({ isLoading: true })
    const success = await get().fetchResultsWithRetry(5, 500)
    if (!success) {
      set({ isLoading: false })
    }
  },

  checkPipelineStatus: async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/pipeline-status`)
      if (!response.ok) throw new Error('Failed to check pipeline status')
      
      const data = await response.json()
      
      set({
        isTraining: data.status === 'training',
        trainingProgress: data.progress || 0
      })

      // If training completed, fetch results WITH RETRY LOGIC
      if (data.status === 'completed' && get().evaluationResults === null) {
        console.log('✓ Training completed! Fetching results with retries (1-3s delays)...')
        // Add initial delay to ensure backend has finished writing results
        setTimeout(() => {
          get().fetchResultsWithRetry(3, 1000)
        }, 1500)
      }

      return data.status
    } catch (error) {
      console.error('Status check error:', error)
      return null
    }
  },

  triggerTraining: async (config = null) => {
    try {
      set({ isTraining: true, trainingProgress: 0, error: null })
      
      const trainConfig = config || get().trainConfig
      
      const response = await fetch(`${API_BASE_URL}/api/train`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(trainConfig)
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.error || 'Failed to start training')
      }

      const data = await response.json()
      set({ success: 'Training pipeline initiated!' })

      // Poll for training status every 2 seconds
      const statusCheckInterval = setInterval(async () => {
        const status = await get().checkPipelineStatus()
        
        if (status === 'error') {
          clearInterval(statusCheckInterval)
          set({ isTraining: false, error: 'Training failed' })
        } else if (status === 'completed') {
          clearInterval(statusCheckInterval)
          set({ 
            isTraining: false, 
            trainingProgress: 100,
            success: 'Training completed successfully!'
          })
        }
      }, 2000)

      return data
    } catch (error) {
      set({ error: error.message, isTraining: false })
      throw error
    }
  },

  resetPipeline: async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/reset`, {
        method: 'POST'
      })
      
      if (!response.ok) throw new Error('Failed to reset pipeline')
      
      set({
        trainingHistory: { epochs: [], train_loss: [], val_loss: [] },
        predictions: [],
        metrics: { normalized_scale: {}, original_scale_inr: {} },
        predictionDates: [],
        predictionTickers: [],
        evaluationResults: null,
        trainingProgress: 0,
        isTraining: false,
        success: 'Pipeline reset successfully'
      })
    } catch (error) {
      set({ error: error.message })
    }
  },

  // Save current training run for comparison
  addTrainingRun: () => {
    const state = get()
    const newRun = {
      id: Date.now(),
      timestamp: new Date().toLocaleString(),
      config: { ...state.trainConfig },
      history: { ...state.trainingHistory },
      metrics: {
        normalized_scale: { ...state.metrics.normalized_scale },
        original_scale_inr: { ...state.metrics.original_scale_inr }
      },
      predictions: [...state.predictions],
      actualValues: [...state.actualValues],
      errors: [...state.errors],
      predictionCount: state.predictions.length
    }
    
    set((state) => ({
      trainingRuns: [...state.trainingRuns, newRun]
    }))
    
    console.log(`✓ Training run saved! Total runs: ${get().trainingRuns.length}`)
  },

  clearTrainingRuns: () => set({ trainingRuns: [] }),

  deleteTrainingRun: (id) => set((state) => ({
    trainingRuns: state.trainingRuns.filter(run => run.id !== id)
  })),

  // Smoothly animate number changes
  animateNumber: (from, to, duration = 1000) => {
    const steps = Math.ceil(duration / 16)
    const increment = (to - from) / steps
    let current = from
    let step = 0

    return new Promise((resolve) => {
      const timer = setInterval(() => {
        current += increment
        step++
        
        if (step >= steps) {
          clearInterval(timer)
          resolve(to)
        }
      }, 16)
    })
  }
}))

export default useStore
