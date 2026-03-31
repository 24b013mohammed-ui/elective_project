import React, { useEffect, useState } from 'react'
import { motion } from 'framer-motion'
import useStore from '../store'

const MetricsDashboard = () => {
  const { metrics, setCurrentPage, fetchMetrics, fetchEvaluation, isLoading, error } = useStore()
  const [activeTab, setActiveTab] = useState('normalized')

  useEffect(() => {
    setCurrentPage('metrics')
    fetchMetrics()
    fetchEvaluation()
  }, [setCurrentPage, fetchMetrics, fetchEvaluation])

  const displayMetrics = activeTab === 'normalized' ? metrics.normalized_scale : metrics.original_scale_inr
  const hasMetrics = displayMetrics && Object.keys(displayMetrics).length > 0

  if (isLoading || error) return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flex flex-col gap-6">
      <h1 style={{ fontSize: '28px', fontWeight: 800, color: '#0f172a' }}>Evaluation Protocol</h1>
      <div className="surface" style={{ padding: '40px', textAlign: 'center' }}>
        {isLoading ? <span className="spinner" style={{ margin: '0 auto' }} /> : <p style={{ color: '#ef4444' }}>{error}</p>}
      </div>
    </motion.div>
  )

  if (!hasMetrics) return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flex flex-col gap-6">
      <h1 style={{ fontSize: '28px', fontWeight: 800, color: '#0f172a' }}>Evaluation Protocol</h1>
      <div className="surface" style={{ padding: '80px 40px', textAlign: 'center', background: '#f8fafc', border: '1px dashed #cbd5e1' }}>
        <p style={{ fontSize: '16px', fontWeight: 600, color: '#475569', marginBottom: '8px' }}>Metrics Unavailable</p>
        <p style={{ fontSize: '13px', color: '#94a3b8' }}>Awaiting engine initialization.</p>
      </div>
    </motion.div>
  )

  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flex flex-col gap-8">
      <div className="border-b border-slate-200 pb-4">
        <h1 style={{ fontSize: '28px', fontWeight: 800, color: '#0f172a' }}>Evaluation Protocol</h1>
        <p style={{ color: '#64748b' }}>Absolute quantitative metrics assessing model deviation constraints.</p>
      </div>

      <div style={{ display: 'flex', gap: '8px', marginBottom: '16px' }}>
        {[{ id: 'normalized', label: 'Internal (0-1 Scale)' }, { id: 'original', label: 'Market Scale' }].map(t => (
          <button
            key={t.id}
            style={{
              padding: '10px 16px', borderRadius: '6px', fontSize: '13px', fontWeight: 600, cursor: 'pointer',
              background: activeTab === t.id ? '#0f172a' : '#f1f5f9',
              color: activeTab === t.id ? '#ffffff' : '#64748b',
              border: 'none', transition: 'all 0.2s'
            }}
            onClick={() => setActiveTab(t.id)}
          >
            {t.label}
          </button>
        ))}
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '24px' }}>
        {Object.entries(displayMetrics).map(([key, val]) => (
          <div key={key} className="surface transform transition-transform hover:-translate-y-1" style={{ padding: '24px', position: 'relative', overflow: 'hidden' }}>
            <div style={{ width: '4px', background: '#10b981', position: 'absolute', left: 0, top: 0, bottom: 0 }} />
            <p style={{ fontSize: '12px', fontWeight: 800, color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
              {key === 'r2' ? 'Coefficient of Determination' : key.toUpperCase() + ' Metric'}
            </p>
            <p style={{ fontSize: '36px', fontWeight: 900, color: '#0f172a', fontFamily: 'JetBrains Mono', marginTop: '12px' }}>
              {typeof val === 'number' ? val.toFixed(5) : val}
            </p>
          </div>
        ))}
      </div>

      <div className="surface" style={{ marginTop: '24px', padding: '24px', background: '#f8fafc' }}>
        <h3 style={{ fontSize: '14px', fontWeight: 800, color: '#0f172a', marginBottom: '12px' }}>Protocol Context</h3>
        <p style={{ fontSize: '14px', color: '#475569', lineHeight: 1.7 }}>
          The metrics displayed assess spatial correlation and tensor derivation accuracy. 
          Use Internal constraints to assess pure gradient descent convergence, and Market Scale constraints for forward-pass deviation sizing.
        </p>
      </div>
    </motion.div>
  )
}

export default MetricsDashboard
