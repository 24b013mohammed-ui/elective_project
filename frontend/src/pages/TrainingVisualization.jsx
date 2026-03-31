import React, { useEffect } from 'react'
import { motion } from 'framer-motion'
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'
import useStore from '../store'

const EpochStat = ({ label, value }) => (
  <div style={{ padding: '16px', borderLeft: '3px solid #6366f1', background: '#f8fafc', borderRadius: '0 8px 8px 0' }}>
    <p style={{ fontSize: '11px', color: '#64748b', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.05em' }}>{label}</p>
    <p style={{ fontSize: '24px', fontWeight: 800, color: '#0f172a', fontFamily: 'JetBrains Mono', marginTop: '4px' }}>{value}</p>
  </div>
)

const TrainingVisualization = () => {
  const { trainingHistory, fetchTrainingHistory, setCurrentPage } = useStore()

  useEffect(() => {
    setCurrentPage('training')
    fetchTrainingHistory()
  }, [setCurrentPage, fetchTrainingHistory])

  const chartData = (trainingHistory.epochs || []).map((epoch, idx) => ({
    iteration: epoch,
    train_err: trainingHistory.train_loss?.[idx] || 0,
    val_err:   trainingHistory.val_loss?.[idx] || 0,
  }))

  const tooltipStyle = {
    contentStyle: { background: '#ffffff', border: '1px solid #e2e8f0', borderRadius: '8px', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)', fontSize: '12px' },
    labelStyle: { color: '#475569', fontWeight: 600, marginBottom: '4px' }
  }

  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flex flex-col gap-8">
      
      <div className="border-b border-slate-200 pb-4">
        <h1 style={{ fontSize: '28px', fontWeight: 800, color: '#0f172a' }}>Convergence Control</h1>
        <p style={{ color: '#64748b' }}>Iterative loss tracking and early-stopping diagnostics.</p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) 280px', gap: '32px' }}>
        
        {/* Left Col: Chart */}
        <div className="surface" style={{ padding: '24px', minHeight: '400px', display: 'flex', flexDirection: 'column' }}>
          <div style={{ marginBottom: '24px', display: 'flex', justifyContent: 'space-between' }}>
            <div>
              <h3 style={{ fontSize: '16px', fontWeight: 700, color: '#0f172a' }}>Loss Trajectory Map</h3>
              <p style={{ fontSize: '13px', color: '#64748b' }}>Gradient descent progress across iterations</p>
            </div>
          </div>

          <div style={{ flex: 1 }}>
            {chartData.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%" minHeight={350}>
                <AreaChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                  <defs>
                    <linearGradient id="trainFill" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#94a3b8" stopOpacity={0.3}/>
                      <stop offset="95%" stopColor="#94a3b8" stopOpacity={0}/>
                    </linearGradient>
                    <linearGradient id="valFill" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#10b981" stopOpacity={0.4}/>
                      <stop offset="95%" stopColor="#10b981" stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e2e8f0" />
                  <XAxis dataKey="iteration" stroke="#94a3b8" tick={{ fontSize: 11 }} tickLine={false} axisLine={false} />
                  <YAxis stroke="#94a3b8" tick={{ fontSize: 11 }} tickLine={false} axisLine={false} />
                  <Tooltip {...tooltipStyle} />
                  <Legend iconType="circle" wrapperStyle={{ fontSize: '12px', paddingTop: '20px' }} />
                  <Area type="monotone" dataKey="train_err" name="Internal Error" stroke="#64748b" strokeWidth={2} fillOpacity={1} fill="url(#trainFill)" />
                  <Area type="monotone" dataKey="val_err" name="Validation Error" stroke="#10b981" strokeWidth={3} fillOpacity={1} fill="url(#valFill)" />
                </AreaChart>
              </ResponsiveContainer>
            ) : (
              <div style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#94a3b8', fontSize: '14px' }}>
                Engine idle. Awaiting configuration input.
              </div>
            )}
          </div>
        </div>

        {/* Right Col: Stats */}
        <div className="flex flex-col gap-4">
          <div className="surface" style={{ padding: '24px' }}>
            <h4 style={{ fontSize: '13px', fontWeight: 800, textTransform: 'uppercase', color: '#0f172a', marginBottom: '20px', letterSpacing: '0.05em' }}>
              Execution Data
            </h4>
            <div className="flex flex-col gap-4">
              <EpochStat label="Optimal Threshold" value={trainingHistory.best_epoch || 'N/A'} />
              <EpochStat label="Min Val Error" value={(trainingHistory.best_val_loss || 0).toFixed(6)} />
              <EpochStat label="Iterations Ran" value={trainingHistory.total_epochs_trained || 0} />
            </div>
          </div>

          <div className="surface-2" style={{ padding: '24px', border: '1px solid #cbd5e1', background: '#ffffff' }}>
            <h4 style={{ fontSize: '13px', fontWeight: 800, textTransform: 'uppercase', color: '#0f172a', marginBottom: '12px' }}>
              Diagnostic Context
            </h4>
            <p style={{ fontSize: '13px', color: '#475569', lineHeight: 1.6 }}>
              The Engine leverages an early-stopping callback set to 5 generations of patience. Overfitting is prevented by rolling back weights to the optimal threshold automatically once criteria are met.
            </p>
          </div>
        </div>

      </div>
    </motion.div>
  )
}

export default TrainingVisualization
