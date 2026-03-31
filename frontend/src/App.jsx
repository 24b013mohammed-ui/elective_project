import React, { useEffect } from 'react'
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import useStore from './store'
import Sidebar from './components/Sidebar'
import Footer from './components/Footer'
import Dashboard from './pages/Dashboard'
import TrainingVisualization from './pages/TrainingVisualization'
import PredictionAnalysis from './pages/PredictionAnalysis'
import SpectrogramViewer from './pages/SpectrogramViewer'
import MetricsDashboard from './pages/MetricsDashboard'
import ErrorAnalysis from './pages/ErrorAnalysis'
import ConfigPanel from './pages/ConfigPanel'

function App() {
  const { isTraining, checkPipelineStatus } = useStore()

  useEffect(() => {
    if (isTraining) {
      const interval = setInterval(() => checkPipelineStatus(), 2000)
      return () => clearInterval(interval)
    }
  }, [isTraining, checkPipelineStatus])

  return (
    <Router>
      <div className="min-h-screen bg-slate-50 text-slate-900 flex flex-row">
        {/* Left Navigation */}
        <Sidebar  />
        
        {/* Main Content Area */}
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', height: '100vh', overflowY: 'auto' }}>
          <main style={{ padding: '32px 40px', maxWidth: '1440px', width: '100%', margin: '0 auto', flexGrow: 1 }}>
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/training" element={<TrainingVisualization />} />
              <Route path="/predictions" element={<PredictionAnalysis />} />
              <Route path="/spectrograms" element={<SpectrogramViewer />} />
              <Route path="/metrics" element={<MetricsDashboard />} />
              <Route path="/errors" element={<ErrorAnalysis />} />
              <Route path="/config" element={<ConfigPanel />} />
            </Routes>
          </main>
          
          <Footer />
        </div>
      </div>
    </Router>
  )
}

export default App
