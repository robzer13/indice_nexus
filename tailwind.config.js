/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './app/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        cockpit: {
          bg: '#07111f',
          sidebar: '#0a1424',
          panel: '#0d1a2b',
          elevated: '#122238',
          hover: '#172b43',
          active: '#1b344c',
        },
        ink: {
          primary: '#e6eef8',
          secondary: '#a6b6c9',
          muted: '#71839a',
          disabled: '#4e6076',
        },
        state: {
          accent: '#39d4df',
          success: '#43d49a',
          warning: '#f2b84b',
          danger: '#f2778f',
          info: '#75a9ff',
          neutral: '#8291a5',
        },
      },
      borderRadius: {
        panel: '14px',
        control: '9px',
      },
      spacing: {
        '18': '4.5rem',
        '22': '5.5rem',
      },
      fontFamily: {
        sans: ['var(--font-geist-sans)', 'ui-sans-serif', 'system-ui', 'sans-serif'],
        mono: ['var(--font-geist-mono)', 'ui-monospace', 'SFMono-Regular', 'monospace'],
      },
      boxShadow: {
        panel: '0 12px 40px rgba(0,0,0,0.22)',
        'panel-soft': '0 8px 28px rgba(2, 10, 24, 0.24)',
      },
    },
  },
  plugins: [],
};
