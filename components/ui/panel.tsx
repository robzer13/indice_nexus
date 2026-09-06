import type { HTMLAttributes, ReactNode } from 'react';

type PanelProps = HTMLAttributes<HTMLElement> & {
  children: ReactNode;
  as?: 'div' | 'section' | 'article';
  elevated?: boolean;
};

export function Panel({ children, as = 'div', elevated = false, className = '', ...props }: PanelProps) {
  const Component = as;
  return (
    <Component
      className={`rounded-panel border border-slate-700/60 ${elevated ? 'bg-cockpit-elevated shadow-panel-soft' : 'bg-cockpit-panel/80'} transition-colors duration-150 hover:border-slate-600/80 ${className}`}
      {...props}
    >
      {children}
    </Component>
  );
}
