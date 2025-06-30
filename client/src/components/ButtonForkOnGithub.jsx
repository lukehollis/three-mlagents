import React from 'react';
import { Github } from '@geist-ui/icons';

/**
 * A reusable "Fork on GitHub" button rendered as a fixed-position anchor.
 *
 * Props:
 *   position: object — CSS position properties (e.g., { top: '20px', right: '20px' }).
 *   style:    object — Additional style overrides.
 */
export default function ButtonForkOnGithub({ position = { bottom: '20px', right: '20px' }, style = {} }) {
  const baseStyle = {
    position: 'fixed',
    backgroundColor: 'rgba(17, 17, 17, 0.8)',
    color: '#fff',
    padding: '10px 16px',
    borderRadius: '6px',
    fontWeight: 500,
    textDecoration: 'none',
    backdropFilter: 'blur(8px)',
    zIndex: 1000,
    display: 'flex',
    alignItems: 'center',
    gap: '6px',
    border: '1px solid #fff',
    ...position,
    ...style,
  };

  return (
    <a
      href="https://github.com/lukehollis/three-mlagents"
      target="_blank"
      rel="noopener noreferrer"
      style={baseStyle}
    >
      <Github size={12} style={{ border: '1px solid #fff' }} />
      Fork on GitHub
    </a>
  );
} 