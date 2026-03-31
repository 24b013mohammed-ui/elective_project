import React, { useEffect } from 'react'
import { motion } from 'framer-motion'
import useStore from '../store'

const FeatureBlock = ({ label, description }) => (
  <div style={{ padding: '24px', border: '1px solid #e2e8f0', borderRadius: '8px', background: '#ffffff', display: 'flex', flexDirection: 'column', gap: '8px' }}>
    <p style={{ fontSize: '14px', fontWeight: 800, color: '#0f172a', textTransform: 'uppercase', letterSpacing: '0.05em' }}>{label}</p>
    <p style={{ fontSize: '13px', color: '#64748b', lineHeight: 1.6 }}>{description}</p>
  </div>
)

const SpectrogramViewer = () => {
  const { setCurrentPage, fetchEvaluation, isLoading, error } = useStore()

  useEffect(() => {
    setCurrentPage('spectrograms')
    fetchEvaluation().catch(() => {})
  }, [setCurrentPage, fetchEvaluation])

  if (isLoading) return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flex flex-col gap-6">
      <h1 style={{ fontSize: '28px', fontWeight: 800, color: '#0f172a' }}>Frequency Field Visualizer</h1>
      <div className="surface" style={{ padding: '40px', textAlign: 'center' }}>
        <span className="spinner" style={{ margin: '0 auto' }} />
      </div>
    </motion.div>
  )

  if (error) return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flex flex-col gap-6">
      <h1 style={{ fontSize: '28px', fontWeight: 800, color: '#0f172a' }}>Frequency Field Visualizer</h1>
      <div className="surface" style={{ padding: '40px', textAlign: 'center' }}>
        <p style={{ color: '#ef4444' }}>{error}</p>
      </div>
    </motion.div>
  )

  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flex flex-col gap-8">
      
      <div className="border-b border-slate-200 pb-4">
        <h1 style={{ fontSize: '28px', fontWeight: 800, color: '#0f172a' }}>Frequency Field Visualizer</h1>
        <p style={{ color: '#64748b' }}>Exploration of raw signal transformations mapping time-domain inputs into spatial grids.</p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) 320px', gap: '32px' }}>
        
        {/* Main Content */}
        <div className="flex flex-col gap-6">
          <div className="surface" style={{ padding: 0 }}>
            <div style={{ padding: '24px', borderBottom: '1px solid #e2e8f0', background: '#f8fafc' }}>
              <h3 style={{ fontSize: '15px', fontWeight: 700, color: '#0f172a' }}>Feature Extraction Methodology</h3>
            </div>
            
            <div style={{ padding: '24px', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '24px' }}>
              <FeatureBlock 
                label="STFT Operations" 
                description="Short-Time Fourier Transforms are utilized to convert 1D time-series data sequences into 2D structural representations, enhancing feature separability." 
              />
              <FeatureBlock 
                label="Multi-Band Parsing" 
                description="Inputs are split across 5 independent dimension channels before rendering, feeding parallel vectors into the initial convolutional layers." 
              />
              <FeatureBlock 
                label="Spectral Density" 
                description="Power density fluctuations define visual gradients in the transformed input maps, revealing micro-periodicities invisible to linear modeling." 
              />
              <FeatureBlock 
                label="Temporal Shift" 
                description="Rolling 20-step input windows overlap significantly, ensuring smooth structural tracking across epochs." 
              />
            </div>
          </div>

          <div className="surface-2" style={{ borderLeft: '4px solid #f59e0b', padding: '20px', background: '#fffbeb' }}>
            <p style={{ fontSize: '14px', fontWeight: 800, color: '#b45309', marginBottom: '8px' }}>Processing Pending</p>
            <p style={{ fontSize: '13px', color: '#78350f', lineHeight: 1.6 }}>
              Visual grids are calculated during backend forward passes and cached to the host node. In-browser 3D projection renders will be appended in a later subsystem patch.
            </p>
          </div>
        </div>

        {/* Sidebar Info */}
        <div className="flex flex-col gap-6">
          <div className="surface" style={{ padding: '24px' }}>
            <h4 style={{ fontSize: '12px', fontWeight: 700, color: '#64748b', textTransform: 'uppercase', marginBottom: '20px' }}>
              Tensor Architecture
            </h4>
            <div className="flex flex-col gap-4">
              {[
                { label: 'Calculated Shape', value: '5 × 128 × 18' },
                { label: 'Spectral Bins', value: '128' },
                { label: 'Sample Frames', value: '18' },
                { label: 'Lookback Window', value: '20 intervals' }
              ].map((item, idx) => (
                <div key={idx} style={{ paddingBottom: '12px', borderBottom: idx !== 3 ? '1px solid #f1f5f9' : 'none' }}>
                  <p style={{ fontSize: '11px', color: '#94a3b8', marginBottom: '4px', textTransform: 'uppercase' }}>{item.label}</p>
                  <p style={{ fontSize: '15px', fontWeight: 700, color: '#0f172a', fontFamily: 'JetBrains Mono' }}>{item.value}</p>
                </div>
              ))}
            </div>
          </div>
        </div>

      </div>
    </motion.div>
  )
}

export default SpectrogramViewer
