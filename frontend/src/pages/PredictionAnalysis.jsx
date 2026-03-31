import React, { useEffect } from 'react'
import { motion } from 'framer-motion'
import useStore from '../store'

const EmptyState = () => (
  <div style={{ padding: '80px 40px', textAlign: 'center', background: '#f8fafc', borderRadius: '8px', border: '1px dashed #cbd5e1' }}>
    <div style={{ fontSize: '32px', marginBottom: '16px' }}>📊</div>
    <p style={{ fontSize: '16px', fontWeight: 600, color: '#475569', marginBottom: '8px' }}>Execution Data Unavailable</p>
    <p style={{ fontSize: '13px', color: '#94a3b8' }}>Run the Engine sequence in Configuration to generate horizons.</p>
  </div>
)

const PredictionAnalysis = () => {
  const { setCurrentPage, fetchEvaluation, predictions, actualValues, errors, predictionDates, predictionTickers, isLoading, error } = useStore()

  useEffect(() => { setCurrentPage('predictions'); fetchEvaluation() }, [setCurrentPage, fetchEvaluation])

  if (isLoading || error) {
    return (
      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flex flex-col gap-6">
        <h1 style={{ fontSize: '28px', fontWeight: 800, color: '#0f172a' }}>Forecast Horizons</h1>
        <div className="surface" style={{ padding: '40px', textAlign: 'center' }}>
          {isLoading ? <span className="spinner" style={{ margin: '0 auto' }} /> : <p style={{ color: '#ef4444' }}>{error}</p>}
        </div>
      </motion.div>
    )
  }

  if (!predictions || predictions.length === 0) {
    return (
      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flex flex-col gap-6">
        <h1 style={{ fontSize: '28px', fontWeight: 800, color: '#0f172a' }}>Forecast Horizons</h1>
        <EmptyState />
      </motion.div>
    )
  }

  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flex flex-col gap-8">
      
      <div className="border-b border-slate-200 pb-4">
        <h1 style={{ fontSize: '28px', fontWeight: 800, color: '#0f172a' }}>Forecast Horizons</h1>
        <p style={{ color: '#64748b' }}>Post-execution analysis of predicted tensor values vs actual market truths.</p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) 300px', gap: '32px' }}>
        
        {/* Left Col: Table */}
        <div className="surface" style={{ padding: 0, overflow: 'hidden' }}>
          <div style={{ padding: '20px 24px', borderBottom: '1px solid #e2e8f0', background: '#f8fafc' }}>
            <h3 style={{ fontSize: '15px', fontWeight: 700, color: '#0f172a' }}>Tensor Output Matrix</h3>
          </div>
          <div style={{ maxHeight: '600px', overflowY: 'auto' }}>
            <table className="data-table">
              <thead>
                <tr>
                  <th>Timestamp</th>
                  <th>Identifier</th>
                  <th style={{ textAlign: 'right' }}>Calculated (T+5)</th>
                  <th style={{ textAlign: 'right' }}>True Output</th>
                  <th style={{ textAlign: 'right' }}>Delta</th>
                </tr>
              </thead>
              <tbody>
                {predictions.map((pred, idx) => {
                  const e = errors?.[idx]
                  const validErr = typeof e === 'number' && isFinite(e)
                  return (
                    <tr key={idx}>
                      <td style={{ color: '#64748b', fontSize: '13px' }}>{predictionDates[idx] || '—'}</td>
                      <td>
                        <span style={{ fontSize: '11px', fontWeight: 700, background: '#e2e8f0', color: '#475569', padding: '4px 8px', borderRadius: '4px' }}>
                          {predictionTickers[idx] || '—'}
                        </span>
                      </td>
                      <td style={{ textAlign: 'right', fontWeight: 700, color: '#10b981' }}>{typeof pred === 'number' ? pred.toFixed(2) : '—'}</td>
                      <td style={{ textAlign: 'right', fontWeight: 600, color: '#0f172a' }}>{actualValues[idx] !== undefined ? actualValues[idx].toFixed(2) : '—'}</td>
                      <td style={{ textAlign: 'right', color: validErr ? (Math.abs(e) < 50 ? '#10b981' : '#ef4444') : '#94a3b8' }}>
                        {validErr ? (e > 0 ? '+' : '') + e.toFixed(2) : '—'}
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </div>

        {/* Right Col: Stats */}
        <div className="flex flex-col gap-6">
          <div className="surface" style={{ padding: '24px' }}>
            <h4 style={{ fontSize: '12px', fontWeight: 700, color: '#64748b', textTransform: 'uppercase', marginBottom: '16px' }}>
              Execution Summary
            </h4>
            <div className="flex flex-col gap-4">
              <div>
                <p style={{ fontSize: '11px', color: '#94a3b8', textTransform: 'uppercase' }}>Tensor Count</p>
                <p style={{ fontSize: '24px', fontWeight: 800, color: '#0f172a' }}>{predictions.length}</p>
              </div>
              <div>
                <p style={{ fontSize: '11px', color: '#94a3b8', textTransform: 'uppercase' }}>Time Steps</p>
                <p style={{ fontSize: '24px', fontWeight: 800, color: '#0f172a' }}>{new Set(predictionDates).size}</p>
              </div>
              <div>
                <p style={{ fontSize: '11px', color: '#94a3b8', textTransform: 'uppercase' }}>Unique Assets</p>
                <p style={{ fontSize: '24px', fontWeight: 800, color: '#0f172a' }}>{new Set(predictionTickers).size}</p>
              </div>
            </div>
          </div>

          <div className="surface-2" style={{ padding: '24px', background: '#f1f5f9' }}>
            <p style={{ fontSize: '13px', fontWeight: 700, color: '#0f172a', marginBottom: '8px' }}>Validation Protocol</p>
            <p style={{ fontSize: '13px', color: '#475569', lineHeight: 1.6 }}>
              The matrix displays the denormalized numerical output of the network compared to the holdout sequence. Delta represents the raw deviation constraint per step.
            </p>
          </div>
        </div>

      </div>
    </motion.div>
  )
}

export default PredictionAnalysis
