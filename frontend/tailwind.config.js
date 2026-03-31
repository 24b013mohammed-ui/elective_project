/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,jsx}",
  ],
  theme: {
    extend: {
      colors: {
        'primary': '#10b981', // Emerald 500
        'secondary': '#0f172a', // Slate 900
        'accent': '#6366f1', // Indigo 500
        'accent-dark': '#4f46e5', // Indigo 600
        'purple-accent': '#8b5cf6', // Violet 500
        'pink-accent': '#ec4899', // Pink 500
      },
      backgroundImage: {
        'gradient-premium': 'linear-gradient(135deg, #f8fafc 0%, #f1f5f9 50%, #e2e8f0 100%)',
        'gradient-accent': 'linear-gradient(135deg, #10b981 0%, #6366f1 100%)',
      },
      animation: {
        'pageEnter': 'pageEnter 0.4s ease-out',
        'fadeIn': 'fadeIn 0.3s ease-out',
        'slideInLeft': 'slideInLeft 0.5s ease-out',
        'slideInRight': 'slideInRight 0.5s ease-out',
        'slideInTop': 'slideInTop 0.4s ease-out',
        'glow': 'glow 2s ease-in-out infinite',
        'subtlePulse': 'subtlePulse 2.5s ease-in-out infinite',
        'shimmer': 'shimmer 2s infinite',
        'float': 'float 3s ease-in-out infinite',
        'scaleIn': 'scaleIn 0.4s ease-out',
      },
      keyframes: {
        pageEnter: {
          'from': { opacity: '0', transform: 'translateY(20px)' },
          'to': { opacity: '1', transform: 'translateY(0)' },
        },
        fadeIn: {
          'from': { opacity: '0' },
          'to': { opacity: '1' },
        },
        slideInLeft: {
          'from': { opacity: '0', transform: 'translateX(-30px)' },
          'to': { opacity: '1', transform: 'translateX(0)' },
        },
        slideInRight: {
          'from': { opacity: '0', transform: 'translateX(30px)' },
          'to': { opacity: '1', transform: 'translateX(0)' },
        },
        slideInTop: {
          'from': { opacity: '0', transform: 'translateY(-20px)' },
          'to': { opacity: '1', transform: 'translateY(0)' },
        },
        glow: {
          '0%, 100%': {
            textShadow: '0 0 5px rgba(18, 130, 162, 0.6)',
            boxShadow: '0 0 10px rgba(18, 130, 162, 0.4)',
          },
          '50%': {
            textShadow: '0 0 20px rgba(18, 130, 162, 0.9)',
            boxShadow: '0 0 20px rgba(18, 130, 162, 0.7)',
          },
        },
        subtlePulse: {
          '0%, 100%': { opacity: '1' },
          '50%': { opacity: '0.85' },
        },
        shimmer: {
          '0%': { backgroundPosition: '-1000px 0' },
          '100%': { backgroundPosition: '1000px 0' },
        },
        float: {
          '0%, 100%': { transform: 'translateY(0px)' },
          '50%': { transform: 'translateY(-8px)' },
        },
        scaleIn: {
          'from': { opacity: '0', transform: 'scale(0.95)' },
          'to': { opacity: '1', transform: 'scale(1)' },
        },
      },
    },
  },
  plugins: [],
  safelist: [
    'animate-pageEnter',
    'animate-fadeIn',
    'animate-slideInLeft',
    'animate-slideInRight',
    'animate-slideInTop',
    'animate-glow',
    'animate-subtlePulse',
    'animate-shimmer',
    'animate-float',
    'animate-scaleIn',
  ],
}
