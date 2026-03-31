import React, { useEffect, useMemo } from 'react'
import { motion } from 'framer-motion'
import { ResponsiveContainer, ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ReferenceLine, BarChart, Bar } from 'recharts'
import useStore from '../store'

const ErrorAnalysis = () => {
  const { setCurrentPage, fetchEvaluation, errors, predictions, isLoading, error } = useStore()

  useEffect(() => { setCurrentPage('errors'); fetchEvaluation() }, [setCurrentPage, fetchEvaluation])

  const errorStats = useMemo(() => {
    if (!errors || errors.length === 0) return null
    const valid = errors.filter(e => typeof e === 'number' && isFinite(e))
    if (!valid.length) return null
    const mean = valid.reduce((a, b) => a + b, 0) / valid.length
    const mae = valid.reduce((a, b) => a + Math.abs(b), 0) / valid.length
    return { mean, mae, min: Math.min(...valid), max: Math.max(...valid), count: valid.length }
  }, [errors])

  const histogramData = useMemo(() => {
    if (!errors || errors.length < 10) return []
    const valid = errors.filter(e => typeof e === 'number' && isFinite(e))
    if (!valid.length) return []
    const mn = Math.min(...valid), mx = Math.max(...valid)
    const bucketSize = (mx - mn || 1) / 10
    const hist = Array.from({ length: 10 }, (_, i) => ({
      range: `${(mn + i * bucketSize).toFixed(0)}`, count: 0,
    }))
    valid.forEach(e => { const idx = Math.min(Math.floor((e - mn) / bucketSize), 9); hist[idx].count++ })
    return hist.filter(h => h.count > 0)
  }, [errors])

  const residualsData = useMemo(() => {
    if (!predictions || !errors || !predictions.length) return []
    return predictions.map((pred, idx) => ({ pred, err: errors[idx] }))
      .filter(d => typeof d.err === 'number' && isFinite(d.err))
      .slice(0, 100)
  }, [predictions, errors])

  if (isLoading || !errorStats) return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flex flex-col gap-6">
      <h1 style={{ fontSize: '28px', fontWeight: 800, color: '#0f172a' }}>Log & Deviation Diagnostics</h1>
      <div className="surface" style={{ padding: '40px', textAlign: 'center' }}>
        {isLoading ? <span className="spinner" style={{ margin: '0 auto' }} /> : 'Engine awaiting output signals.'}
      </div>
    </motion.div>
  )

  const tTipStyle = {
    contentStyle: { background: '#ffffff', border: '1px solid #cbd5e1', borderRadius: '4px', fontSize: '12px' },
    labelStyle: { color: '#64748b' }
  }

  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flex flex-col gap-8">
      <div className="border-b border-slate-200 pb-4">
        <h1 style={{ fontSize: '28px', fontWeight: 800, color: '#0f172a' }}>Log & Deviation Diagnostics</h1>
        <p style={{ color: '#64748b' }}>Detailed variance and residual analysis for identifying systematic output failures.</p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '16px' }}>
        {[
          { l: 'Base Variance', v: errorStats.mean }, { l: 'Absolute Flux', v: errorStats.mae },
          { l: 'Max Positive', v: errorStats.max }, { l: 'Max Negative', v: errorStats.min },
        ].map(s => (
          <div key={s.l} className="surface" style={{ padding: '20px', borderLeft: '4px solid #3b82f6' }}>
            <p style={{ fontSize: '11px', color: '#64748b', fontWeight: 700, textTransform: 'uppercase' }}>{s.l}</p>
            <p style={{ fontSize: '24px', fontWeight: 800, color: '#0f172a', marginTop: '8px' }}>{s.v.toFixed(3)}</p>
          </div>
        ))}
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '32px' }}>
        
        {/* Histogram */}
        <div className="surface" style={{ padding: '24px' }}>
          <h3 style={{ fontSize: '15px', fontWeight: 800, color: '#0f172a', marginBottom: '20px' }}>Variance Density Map</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={histogramData} margin={{ left: -20, right: 10 }}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e2e8f0" />
              <XAxis dataKey="range" stroke="#94a3b8" tick={{ fontSize: 11 }} tickLine={false} axisLine={false} />
              <YAxis stroke="#94a3b8" tick={{ fontSize: 11 }} tickLine={false} axisLine={false} />
              <Tooltip {...tTipStyle} cursor={{ fill: '#f1f5f9' }} />
              <Bar dataKey="count" fill="#3b82f6" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Scatter */}
        <div className="surface" style={{ padding: '24px' }}>
          <h3 style={{ fontSize: '15px', fontWeight: 800, color: '#0f172a', marginBottom: '20px' }}>Residual Drift</h3>
          <ResponsiveContainer width="100%" height={300}>
            <ScatterChart margin={{ left: -20, right: 10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis dataKey="pred" type="number" name="Output" stroke="#94a3b8" tick={{ fontSize: 11 }} tickLine={false} />
              <YAxis dataKey="err" type="number" name="Drift" stroke="#94a3b8" tick={{ fontSize: 11 }} tickLine={false} axisLine={false} />
              <Tooltip {...tTipStyle} cursor={{ strokeDasharray: '3 3' }} />
              <ReferenceLine y={0} stroke="#ef4444" strokeDasharray="3 3" />
              <Scatter data={residualsData} fill="#10b981" fillOpacity={0.6} />
            </ScatterChart>
          </ResponsiveContainer>
        </div>

      </div>

    </motion.div>
  )
}

export default ErrorAnalysis
