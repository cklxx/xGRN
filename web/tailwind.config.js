/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        bg: '#F4F2EE',
        card: '#FFFFFF',
        sub: '#FAF8F4',
        line: '#E5E1D8',
        line2: '#DBD7CE',
        ink: '#1F1F1F',
        muted: '#6B6660',
        soft: '#A39E96',
        terra: {
          DEFAULT: '#C96442',
          hover: '#B5573A',
          50: '#FDF5F1',
          100: '#FBE6DD',
          200: '#F5CAB6',
          300: '#EAA386',
          400: '#DD8260',
          500: '#C96442',
          600: '#B5573A',
          700: '#94462F',
          800: '#73362A',
          900: '#5B2D24',
        },
        teal: {
          DEFAULT: '#5E8B7E',
          50: '#F3F7F5',
          100: '#E1EAE6',
          200: '#C4D4CD',
          300: '#A3BBB1',
          400: '#82A193',
          500: '#5E8B7E',
          600: '#4D7367',
          700: '#3F5D54',
        },
        pre: '#2A2520',
      },
      fontFamily: {
        sans: ['Inter var', 'Inter', 'ui-sans-serif', 'system-ui', 'sans-serif'],
        mono: ['"IBM Plex Mono"', 'ui-monospace', 'SFMono-Regular', 'monospace'],
      },
      fontSize: {
        '2xs': ['0.7rem', { lineHeight: '1.05rem' }],
      },
      borderRadius: {
        sm: '6px',
        DEFAULT: '8px',
        md: '10px',
        lg: '12px',
        xl: '16px',
      },
      boxShadow: {
        soft: '0 1px 2px rgba(28,25,23,0.04), 0 8px 24px -16px rgba(28,25,23,0.08)',
        terra: '0 2px 6px rgba(201,100,66,0.18)',
        terraHover: '0 4px 14px rgba(201,100,66,0.28)',
        ring: '0 0 0 3px rgba(201,100,66,0.14)',
      },
      letterSpacing: {
        cap: '0.08em',
      },
      maxWidth: {
        page: '1500px',
      },
      animation: {
        'pulse-slow': 'pulse 2.4s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'fade-in': 'fadeIn 0.25s ease-out',
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0', transform: 'translateY(4px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
      },
    },
  },
  plugins: [],
}
