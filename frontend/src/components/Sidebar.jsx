import React from 'react'
import { Link, useLocation } from 'react-router-dom'
import { motion } from 'framer-motion'

const navLinks = [
  { name: 'Dashboard',    path: '/'},
  { name: 'Model Control',path: '/training'},
  { name: 'Predictions',  path: '/predictions'},
  { name: 'Visualizer',   path: '/spectrograms'},
  { name: 'Evaluation',   path: '/metrics'},
  { name: 'Log & Errors', path: '/errors'},
  { name: 'Parameters',   path: '/config'}
]

const Sidebar = () => {
  const location = useLocation()
  
  return (
    <aside style={{
      width: '260px',
      height: '100vh',
      position: 'sticky',
      top: 0,
      background: '#ffffff',
      borderRight: '1px solid #e2e8f0',
      display: 'flex',
      flexDirection: 'column',
      padding: '24px 16px',
      flexShrink: 0
    }}>
      {/* Brand */}
      <div style={{ padding: '0 8px 32px 8px' }}>
        <h1 style={{ fontSize: 22, fontWeight: 800, color: '#0f172a', letterSpacing: '-0.02em', lineHeight: 1.1 }}>
          Data Forecaster
        </h1>
        <p style={{ fontSize: 12, color: '#10b981', fontWeight: 700, marginTop: 4 }}>
          —By Shifaz
        </p>
      </div>

      {/* Nav List */}
      <nav style={{ display: 'flex', flexDirection: 'column', gap: 6, flex: 1 }}>
        {navLinks.map((link) => {
          const active = location.pathname === link.path
          return (
            <Link
              key={link.path}
              to={link.path}
              style={{
                position: 'relative',
                padding: '12px 16px',
                borderRadius: 8,
                fontSize: 14,
                fontWeight: active ? 700 : 500,
                color: active ? '#ffffff' : '#475569',
                background: active ? '#10b981' : 'transparent',
                textDecoration: 'none',
                transition: 'all 0.2s ease',
                display: 'flex',
                alignItems: 'center'
              }}
              onMouseEnter={e => { if(!active) { e.currentTarget.style.background = '#f1f5f9'; e.currentTarget.style.color = '#0f172a' } }}
              onMouseLeave={e => { if(!active) { e.currentTarget.style.background = 'transparent'; e.currentTarget.style.color = '#475569' } }}
            >
              <div style={{ zIndex: 1 }}>{link.name}</div>
            </Link>
          )
        })}
      </nav>

      {/* Footer info */}
      <div style={{ marginTop: 'auto', padding: '16px 8px', borderTop: '1px solid #e2e8f0' }}>
        <p style={{ fontSize: 11, color: '#94a3b8', fontWeight: 600 }}>v2.0.0 Built for Forecasting</p>
      </div>
    </aside>
  )
}

export default Sidebar
