import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { useResponsive } from '../hooks/useResponsive';

export default function HomeButton() {
  const [hover, setHover] = useState(false);
  const { isMobile } = useResponsive();

  return (
    <Link
      to="/"
      style={{
        fontFamily: '"Roboto Mono", monospace',
        color: '#fff',
        textDecoration: hover ? 'none' : 'underline',
        display: 'inline-block',
        fontSize: isMobile ? '12px' : '14px',
        textTransform: 'uppercase',
        letterSpacing: '0.1em',
        marginBottom: '4px',
        cursor: 'pointer'
      }}
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
    >
      {"<< "} HOME
    </Link>
  );
}
