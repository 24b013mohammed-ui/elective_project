import React, { useState } from 'react'
import { Link, useLocation } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'

/* ── SVG Icon set ──────────────────────────────────────────────── */
const Icons = {
  Dashboard: () => (
    <svg viewBox="0 0 20 20" fill="currentColor" className="w-4 h-4">
      <path d="M3 4a1 1 0 011-1h4a1 1 0 011 1v4a1 1 0 01-1 1H4a1 1 0 01-1-1V4zm0 8a1 1 0 011-1h4a1 1 0 011 1v4a1 1 0 01-1 1H4a1 1 0 01-1-1v-4zm8-8a1 1 0 011-1h4a1 1 0 011 1v4a1 1 0 01-1 1h-4a1 1 0 01-1-1V4zm0 8a1 1 0 011-1h4a1 1 0 011 1v4a1 1 0 01-1 1h-4a1 1 0 01-1-1v-4z"/>
    </svg>
  ),
  Training: () => (
    <svg viewBox="0 0 20 20" fill="currentColor" className="w-4 h-4">
      <path fillRule="evenodd" d="M3 3a1 1 0 000 2v8a2 2 0 002 2h2.586l-1.293 1.293a1 1 0 101.414 1.414L10 15.414l2.293 2.293a1 1 0 001.414-1.414L12.414 15H15a2 2 0 002-2V5a1 1 0 100-2H3zm11 4a1 1 0 10-2 0v4a1 1 0 102 0V7zm-3 1a1 1 0 10-2 0v3a1 1 0 102 0V8zM8 9a1 1 0 00-2 0v2a1 1 0 102 0V9z" clipRule="evenodd"/>
    </svg>
  ),
  Predictions: () => (
    <svg viewBox="0 0 20 20" fill="currentColor" className="w-4 h-4">
      <path d="M2 11a1 1 0 011-1h2a1 1 0 011 1v5a1 1 0 01-1 1H3a1 1 0 01-1-1v-5zM8 7a1 1 0 011-1h2a1 1 0 011 1v9a1 1 0 01-1 1H9a1 1 0 01-1-1V7zM14 4a1 1 0 011-1h2a1 1 0 011 1v12a1 1 0 01-1 1h-2a1 1 0 01-1-1V4z"/>
    </svg>
  ),
  Spectrograms: () => (
    <svg viewBox="0 0 20 20" fill="currentColor" className="w-4 h-4">
      <path fillRule="evenodd" d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z" clipRule="evenodd"/>
    </svg>
  ),
  Metrics: () => (
    <svg viewBox="0 0 20 20" fill="currentColor" className="w-4 h-4">
      <path d="M9 2a1 1 0 000 2h2a1 1 0 100-2H9z"/>
      <path fillRule="evenodd" d="M4 5a2 2 0 012-2 3 3 0 003 3h2a3 3 0 003-3 2 2 0 012 2v11a2 2 0 01-2 2H6a2 2 0 01-2-2V5zm3 4a1 1 0 000 2h.01a1 1 0 100-2H7zm3 0a1 1 0 000 2h3a1 1 0 100-2h-3zm-3 4a1 1 0 100 2h.01a1 1 0 100-2H7zm3 0a1 1 0 100 2h3a1 1 0 100-2h-3z" clipRule="evenodd"/>
    </svg>
  ),
  Errors: () => (
    <svg viewBox="0 0 20 20" fill="currentColor" className="w-4 h-4">
      <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clipRule="evenodd"/>
    </svg>
  ),
  Config: () => (
    <svg viewBox="0 0 20 20" fill="currentColor" className="w-4 h-4">
      <path fillRule="evenodd" d="M11.49 3.17c-.38-1.56-2.6-1.56-2.98 0a1.532 1.532 0 01-2.286.948c-1.372-.836-2.942.734-2.106 2.106.54.886.061 2.042-.947 2.287-1.561.379-1.561 2.6 0 2.978a1.532 1.532 0 01.947 2.287c-.836 1.372.734 2.942 2.106 2.106a1.532 1.532 0 012.287.947c.379 1.561 2.6 1.561 2.978 0a1.533 1.533 0 012.287-.947c1.372.836 2.942-.734 2.106-2.106a1.533 1.533 0 01.947-2.287c1.561-.379 1.561-2.6 0-2.978a1.532 1.532 0 01-.947-2.287c.836-1.372-.734-2.942-2.106-2.106a1.532 1.532 0 01-2.287-.947zM10 13a3 3 0 100-6 3 3 0 000 6z" clipRule="evenodd"/>
    </svg>
  ),
  Menu: ({ open }) => open ? (
    <svg viewBox="0 0 20 20" fill="currentColor" className="w-5 h-5">
      <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd"/>
    </svg>
  ) : (
    <svg viewBox="0 0 20 20" fill="currentColor" className="w-5 h-5">
      <path fillRule="evenodd" d="M3 5a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zM3 10a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zM3 15a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1z" clipRule="evenodd"/>
    </svg>
  ),
}

const navItems = [
  { name: 'Overview',     path: '/',             Icon: Icons.Dashboard    },
  { name: 'Model Builder',path: '/training',     Icon: Icons.Training     },
  { name: 'Live Market',  path: '/predictions',  Icon: Icons.Predictions  },
  { name: 'Data Visuals', path: '/spectrograms', Icon: Icons.Spectrograms },
  { name: 'Analytics',    path: '/metrics',      Icon: Icons.Metrics      },
  { name: 'Diagnostics',  path: '/errors',       Icon: Icons.Errors       },
  { name: 'Settings',     path: '/config',       Icon: Icons.Config       },
]

const Navigation = () => {
  const location = useLocation()
  const [isOpen, setIsOpen] = useState(false)
  const isActive = (path) => location.pathname === path

  return (
    <nav
      style={{
        position: 'sticky', top: 0, zIndex: 50,
        background: 'rgba(255, 255, 255, 0.9)',
        backdropFilter: 'blur(20px)',
        WebkitBackdropFilter: 'blur(20px)',
        borderBottom: '1px solid #e2e8f0',
      }}
    >
      {/* Top line removed */}
      <div style={{
        position: 'absolute', top: 0, left: 0, right: 0, height: '2px',
        background: '#10b981',
      }} />

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">

          {/* ── Logo ── */}
          <Link to="/" className="flex items-center gap-3 group" style={{ textDecoration: 'none' }}>
            <div style={{
              width: 40, height: 40,
              background: '#0f172a',
              borderRadius: 8,
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              fontWeight: 900, fontSize: 14, color: '#ffffff',
              boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
              letterSpacing: '0.02em',
              flexShrink: 0,
              transition: 'box-shadow 0.25s ease, transform 0.25s ease',
            }}>
              QA
            </div>
            <div className="hidden sm:block">
              <p style={{ fontSize: 16, fontWeight: 700, lineHeight: 1.1, letterSpacing: '-0.02em', color: '#0f172a' }}>
                QuantAnalytics
              </p>
              <p style={{ fontSize: 11, color: '#64748b', fontWeight: 600, letterSpacing: '0.04em', marginTop: 1, textTransform: 'uppercase' }}>
                Core Engine
              </p>
            </div>
          </Link>

          {/* ── Desktop Nav ── */}
          <div className="hidden md:flex items-center gap-0.5">
            {navItems.map(({ name, path, Icon }) => {
              const active = isActive(path)
              return (
                <Link
                  key={path}
                  to={path}
                  style={{
                    position: 'relative',
                    display: 'flex', alignItems: 'center', gap: 6,
                    padding: '6px 12px',
                    borderRadius: 8,
                    fontSize: 13,
                    fontWeight: active ? 600 : 500,
                    color: active ? '#10b981' : '#64748b',
                    textDecoration: 'none',
                    transition: 'color 0.2s, background 0.2s',
                    background: active ? '#f0fdf4' : 'transparent',
                    boxShadow: active ? '0 0 0 1px #bbf7d0' : 'none',
                  }}
                  onMouseEnter={e => { if (!active) { e.currentTarget.style.color = '#0f172a'; e.currentTarget.style.background = '#f1f5f9'; } }}
                  onMouseLeave={e => { if (!active) { e.currentTarget.style.color = '#64748b'; e.currentTarget.style.background = 'transparent'; } }}
                >
                  <Icon />
                  <span>{name}</span>
                  {active && (
                    <motion.div
                      layoutId="activeNav"
                      style={{
                        position: 'absolute', bottom: -1, left: '15%', right: '15%',
                        height: 2, borderRadius: 1,
                        background: '#10b981',
                      }}
                      initial={{ scaleX: 0 }}
                      animate={{ scaleX: 1 }}
                    />
                  )}
                </Link>
              )
            })}
          </div>

          {/* ── Mobile Toggle ── */}
          <button
            onClick={() => setIsOpen(!isOpen)}
            style={{
              display: 'none', padding: 8, borderRadius: 8, border: '1px solid #e2e8f0', cursor: 'pointer',
              background: '#f8fafc', color: '#64748b', transition: 'background 0.2s',
            }}
            className="md:hidden"
          >
            <Icons.Menu open={isOpen} />
          </button>
        </div>

        {/* ── Mobile Menu ── */}
        <AnimatePresence>
          {isOpen && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              transition={{ duration: 0.2 }}
              style={{ overflow: 'hidden', borderTop: '1px solid rgba(99,179,237,0.1)' }}
            >
              <div style={{ padding: '8px 0 12px' }}>
                {navItems.map(({ name, path, Icon }) => {
                  const active = isActive(path)
                  return (
                    <Link
                      key={path}
                      to={path}
                      onClick={() => setIsOpen(false)}
                      style={{
                        display: 'flex', alignItems: 'center', gap: 10,
                        padding: '10px 12px',
                        borderRadius: 8,
                        margin: '2px 0',
                        fontSize: 14,
                        fontWeight: active ? 600 : 500,
                        color: active ? '#10b981' : '#64748b',
                        background: active ? '#f0fdf4' : 'transparent',
                        textDecoration: 'none',
                        transition: 'all 0.15s ease',
                      }}
                    >
                      <Icon />
                      {name}
                    </Link>
                  )
                })}
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </nav>
  )
}

export default Navigation
