import React from 'react';
import { Github } from '@geist-ui/icons';
import { useMediaQuery } from '@geist-ui/core';

/**
 * A reusable "Fork on GitHub" button rendered as a fixed-position anchor.
 *
 * Props:
 *   position: object — CSS position properties (e.g., { top: '20px', right: '20px' }).
 *   style:    object — Additional style overrides.
 */
export default function ButtonForkOnGithub({ position = { bottom: '10px', right: '10px' }, style = {} }) {
  const isMobile = useMediaQuery('xs');

  const baseStyle = {
    position: 'fixed',
    backgroundColor: '#000',
    color: '#fff',
    padding: isMobile ? '4px 6px' : '10px 16px',
    borderRadius: 0,
    fontWeight: 600,
    textDecoration: 'none',
    zIndex: 1000,
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    border: '1px solid #fff',
    fontSize: isMobile ? '10px' : '0.8rem',
    fontFamily: '"Roboto Mono", monospace',
    textTransform: 'uppercase',
    letterSpacing: '0.1em',
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
      <Github size={isMobile ? 10 : 12} />

      Fork on GitHub
    </a>
  );
} 