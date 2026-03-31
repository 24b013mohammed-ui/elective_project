import React from 'react'

const Footer = () => {
  return (
    <footer style={{
      background: '#f8fafc',
      borderTop: '1px solid #e2e8f0',
      padding: '24px 32px',
      marginTop: 'auto',
      display: 'flex',
      flexDirection: 'column',
      gap: '16px'
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '16px' }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
            <div style={{ width: 24, height: 24, background: '#10b981', borderRadius: '4px', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#fff', fontSize: '12px', fontWeight: 800 }}>
              SF
            </div>
            <span style={{ fontSize: '14px', fontWeight: 700, color: '#0f172a' }}>Shifaz Forecaster Engine</span>
          </div>
          <p style={{ fontSize: '13px', color: '#64748b' }}>Advanced structural evaluation mapping system.</p>
        </div>

        <div style={{ display: 'flex', gap: '24px' }}>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
            <span style={{ fontSize: '11px', fontWeight: 700, color: '#94a3b8', textTransform: 'uppercase' }}>Status</span>
            <span style={{ fontSize: '13px', fontWeight: 600, color: '#10b981' }}>System Integrity Normal</span>
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
            <span style={{ fontSize: '11px', fontWeight: 700, color: '#94a3b8', textTransform: 'uppercase' }}>Build Version</span>
            <span style={{ fontSize: '13px', fontWeight: 600, color: '#475569' }}>v2.4.0-SFZ</span>
          </div>
        </div>
      </div>

      <div style={{ borderTop: '1px solid #e2e8f0', paddingTop: '16px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <p style={{ fontSize: '12px', color: '#94a3b8' }}>
          © {new Date().getFullYear()} Shifaz Diagnostics — For Internal Use Only.
        </p>
        <div style={{ display: 'flex', gap: '8px' }}>
          {['React', 'Vite', 'Recharts', 'Zustand'].map(t => (
            <span key={t} style={{ fontSize: '11px', padding: '2px 8px', background: '#f1f5f9', borderRadius: '4px', color: '#64748b', fontWeight: 600 }}>
              {t}
            </span>
          ))}
        </div>
      </div>
    </footer>
  )
}

export default Footer
