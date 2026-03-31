import React, { useEffect } from 'react'
import { motion } from 'framer-motion'
import { Link } from 'react-router-dom'
import useStore from '../store'

const Dashboard = () => {
  const { metrics, isTraining, trainingProgress, fetchMetrics, setCurrentPage, addTrainingRun } = useStore()
  const [displayVals, setDisplayVals] = React.useState({ r2: '0.00', mae: '0.00', mape: '0.00', loss: '0.0000' })
  const [prevTraining, setPrevTraining] = React.useState(false)

  useEffect(() => { setCurrentPage('dashboard'); fetchMetrics() }, [setCurrentPage, fetchMetrics])

  useEffect(() => {
    const hasNorm = metrics?.normalized_scale && Object.keys(metrics.normalized_scale).length > 0
    const hasOrig = metrics?.original_scale_inr && Object.keys(metrics.original_scale_inr).length > 0
    if (prevTraining && !isTraining && (hasNorm || hasOrig)) { addTrainingRun() }
    setPrevTraining(isTraining)
  }, [isTraining, metrics, addTrainingRun, prevTraining])

  useEffect(() => {
    const d = metrics.original_scale_inr || {}
    setDisplayVals({
      r2:   (d.r2 * 100 || 0).toFixed(2),
      mae:  (d.mae || 0).toFixed(2),
      mape: (d.mape || 0).toFixed(2),
      loss: (metrics.normalized_scale?.mse || 0).toFixed(4),
    })
  }, [metrics])

  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="animate-pageEnter flex flex-col gap-8">
      
      {/* Top Banner Area */}
      <div className="flex justify-between items-start border-b border-slate-200 pb-6">
        <div>
          <h2 style={{ fontSize: '2.5rem', fontWeight: 800, color: '#0f172a', letterSpacing: '-0.02em', lineHeight: 1.2 }}>
            Prediction Command Center
          </h2>
          <p style={{ fontSize: '1.125rem', color: '#64748b', fontWeight: 500, marginTop: 4 }}>
            System Engineered <span style={{ color: '#10b981', fontWeight: 700 }}>-By Shifaz</span>
          </p>
        </div>
        <button className="btn-secondary" onClick={() => useStore.getState().manuallyFetchResults()}>
          Refresh Data View
        </button>
      </div>

      {isTraining && (
        <div style={{ background: '#f0fdf4', border: '1px solid #10b981', borderRadius: 8, padding: 16 }}>
          <div className="flex justify-between mb-2">
            <span style={{ color: '#047857', fontWeight: 700 }}>Model compiling...</span>
            <span style={{ color: '#047857', fontWeight: 700 }}>{trainingProgress}%</span>
          </div>
          <div className="progress-track" style={{ height: 8, background: '#a7f3d0' }}>
            <motion.div className="progress-fill" style={{ background: '#10b981' }} animate={{ width: `${trainingProgress}%` }} />
          </div>
        </div>
      )}

      {/* Main Split Layout */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 320px', gap: '32px' }}>
        
        {/* Left Col - Context & Actions */}
        <div className="flex flex-col gap-6">
          <div className="surface" style={{ padding: 32, flex: 1, display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
            <h3 style={{ fontSize: 24, fontWeight: 700, marginBottom: 16 }}>Welcome to the Predictive Engine</h3>
            <p style={{ color: '#475569', fontSize: 16, lineHeight: 1.7, marginBottom: 24 }}>
              This platform provides a robust analytical interface for evaluating deep convolutional networks over sequential data. 
              Built strictly for high-fidelity quantitative analysis, empowering rapid model iteration and frequency transforms evaluation.
            </p>
            <div style={{ display: 'flex', gap: 12 }}>
              <Link to="/config" className="btn-primary">Configure Model</Link>
              <Link to="/training" className="btn-secondary">View History</Link>
            </div>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
            {[
              { title: 'Predictions Map', path: '/predictions', desc: 'Forecast horizons vs actuals' },
              { title: 'Frequency Vis', path: '/spectrograms', desc: 'Examine spectral data mappings' },
            ].map(l => (
              <Link key={l.title} to={l.path} className="surface hover-lift" style={{ padding: 24, textDecoration: 'none' }}>
                <p style={{ fontSize: 18, fontWeight: 700, color: '#0f172a', marginBottom: 4 }}>{l.title}</p>
                <p style={{ fontSize: 14, color: '#64748b' }}>{l.desc}</p>
              </Link>
            ))}
          </div>
        </div>

        {/* Right Col - Stacked Metrics */}
        <div className="flex flex-col gap-4">
          <h4 style={{ fontSize: 14, fontWeight: 700, textTransform: 'uppercase', color: '#94a3b8', letterSpacing: 1 }}>
            Engine Analytics
          </h4>
          
          {[
            { label: 'R² Accuracy Factor', value: displayVals.r2 + '%' },
            { label: 'Absolute Deviation', value: '₹' + displayVals.mae },
            { label: 'Percentage Errors', value: displayVals.mape + '%' },
            { label: 'Val Loss Floor', value: displayVals.loss },
          ].map((m, i) => (
            <div key={i} className="surface" style={{ padding: '20px 24px', borderLeft: '4px solid #10b981' }}>
              <p style={{ fontSize: 13, color: '#64748b', fontWeight: 600, marginBottom: 8 }}>{m.label}</p>
              <p style={{ fontSize: 32, fontWeight: 800, color: '#0f172a', fontFamily: 'JetBrains Mono', lineHeight: 1 }}>
                {m.value}
              </p>
            </div>
          ))}

          <div className="surface" style={{ padding: '20px', marginTop: 'auto', background: '#f8fafc' }}>
            <p style={{ fontSize: 12, color: '#64748b', fontWeight: 600 }}>SYSTEM IDENT</p>
            <p style={{ fontSize: 14, fontWeight: 700, color: '#0f172a' }}>SHIFAZ-ENGINE-V2</p>
          </div>
        </div>

      </div>
    </motion.div>
  )
}

export default Dashboard
