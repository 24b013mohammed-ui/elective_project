import React, { useEffect, useState } from 'react'
import { motion } from 'framer-motion'
import useStore from '../store'

const InputRow = ({ label, name, type, value, onChange, disabled, hint }) => (
  <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
    <label style={{ fontSize: '12px', fontWeight: 600, color: '#475569', textTransform: 'uppercase' }}>
      {label}
    </label>
    <input
      type={type}
      name={name}
      value={value}
      onChange={onChange}
      disabled={disabled}
      className="input-glass"
      style={{ background: disabled ? '#f8fafc' : '#ffffff', border: '1px solid #cbd5e1' }}
    />
    {hint && <span style={{ fontSize: '11px', color: '#94a3b8' }}>{hint}</span>}
  </div>
)

const ConfigPanel = () => {
  const { trainConfig, setTrainConfig, triggerTraining, isTraining, trainingProgress, setCurrentPage } = useStore()
  const [formData, setFormData] = useState(trainConfig)

  useEffect(() => setCurrentPage('config'), [setCurrentPage])

  const handleChange = (e) => {
    const { name, value } = e.target
    setFormData({
      ...formData,
      [name]: isNaN(value) ? value : (name === 'tickers' ? value.split(',').map(t => t.trim()) : Number(value)),
    })
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    setTrainConfig(formData)
    await triggerTraining(formData)
  }

  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flex flex-col gap-6">
      <div className="border-b border-slate-200 pb-4">
        <h1 style={{ fontSize: '28px', fontWeight: 800, color: '#0f172a' }}>Engine Parameters</h1>
        <p style={{ color: '#64748b' }}>Define the structural bounds of the Shifaz Prediction Engine before initiating the run.</p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 350px', gap: '32px' }}>
        
        {/* Left Column: Form */}
        <form onSubmit={handleSubmit} className="surface" style={{ padding: '32px', display: 'flex', flexDirection: 'column', gap: '24px' }}>
          
          <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: '24px' }}>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
              <label style={{ fontSize: '12px', fontWeight: 600, color: '#475569', textTransform: 'uppercase' }}>
                Asset Identifiers (Tickers)
              </label>
              <textarea
                name="tickers"
                value={formData.tickers.join(', ')}
                onChange={handleChange}
                disabled={isTraining}
                className="input-glass"
                style={{ minHeight: '80px', background: isTraining ? '#f8fafc' : '#ffffff', border: '1px solid #cbd5e1' }}
              />
              <span style={{ fontSize: '11px', color: '#94a3b8' }}>Requires valid market identifiers separated by commas.</span>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
              <InputRow label="History Start" name="start_date" type="date" value={formData.start_date} onChange={handleChange} disabled={isTraining} />
              <InputRow label="History End" name="end_date" type="date" value={formData.end_date} onChange={handleChange} disabled={isTraining} />
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '16px' }}>
              <InputRow label="Batch Count" name="batch_size" type="number" value={formData.batch_size} onChange={handleChange} disabled={isTraining} />
              <InputRow label="Max Iterations" name="num_epochs" type="number" value={formData.num_epochs} onChange={handleChange} disabled={isTraining} />
              <InputRow label="Pace (LR)" name="learning_rate" type="number" value={formData.learning_rate} onChange={handleChange} disabled={isTraining} />
            </div>
          </div>

          <hr style={{ borderTop: '1px solid #e2e8f0', margin: '8px 0' }} />

          <button 
            type="submit" 
            disabled={isTraining} 
            className="btn-primary" 
            style={{ width: '100%', justifyContent: 'center', padding: '14px', fontSize: '16px' }}
          >
            {isTraining ? `Engine Running: ${trainingProgress}%` : 'Initiate Engine Sequence'}
          </button>

          {isTraining && (
            <div className="progress-track" style={{ height: '6px', marginTop: '8px' }}>
              <motion.div className="progress-fill" animate={{ width: `${trainingProgress}%` }} style={{ background: '#10b981' }} />
            </div>
          )}

        </form>

        {/* Right Column: Execution Information */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
          
          <div className="surface" style={{ padding: '24px', background: '#f8fafc' }}>
            <h3 style={{ fontSize: '14px', fontWeight: 800, color: '#0f172a', marginBottom: '16px' }}>Execution Rules</h3>
            <ul style={{ paddingLeft: '20px', fontSize: '13px', color: '#475569', display: 'flex', flexDirection: 'column', gap: '12px' }}>
              <li>The engine enforces strict normalization across all specified assets.</li>
              <li>A 5-day horizon boundary is applied explicitly to all target tensors.</li>
              <li>Wait for complete convergence; Early Stop algorithms are active.</li>
            </ul>
          </div>

          <div className="surface-2" style={{ padding: '24px', borderLeft: '4px solid #6366f1' }}>
            <p style={{ fontSize: '12px', fontWeight: 700, color: '#6366f1', textTransform: 'uppercase' }}>Notice</p>
            <p style={{ fontSize: '13px', color: '#475569', marginTop: '8px' }}>
              If you modify the identifiers, ensure they resolve cleanly. The Engine will automatically download required context upon initiation.
            </p>
          </div>

        </div>

      </div>
    </motion.div>
  )
}

export default ConfigPanel
