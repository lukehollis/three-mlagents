import React, { useEffect, useRef } from 'react';
import { useMediaQuery } from '@geist-ui/core';

export default function DebugConsole({ logs }) {
  const containerRef = useRef(null);
  const isMobile = useMediaQuery('sm');

  useEffect(() => {
    const el = containerRef.current;
    if (el) {
      el.scrollTop = el.scrollHeight;
    }
  }, [logs]);

  return (
    <div
      ref={containerRef}
      style={{
        width: '100%',
        height: '140px',
        overflowY: 'scroll',
        background: 'rgba(0,0,0,0.95)',
        color: '#0f0',
        fontFamily: 'monospace',
        fontSize: isMobile ? 8 : 10,
        padding: '8px',
        borderRadius: '8px',
        border: '1px solid rgba(255,255,255,0.2)',
      }}
    >
      {logs.map((ln, i) => {
        // Attempt to pretty-print JSON objects so they read like
        // "key: value, key2: value2". If ln is already an object we use it
        // directly, otherwise we try to parse if it looks like JSON.
        let obj = ln;
        if (typeof ln === 'string') {
          const trimmed = ln.trim();
          if ((trimmed.startsWith('{') && trimmed.endsWith('}')) ||
              (trimmed.startsWith('[') && trimmed.endsWith(']'))) {
            obj = JSON.parse(trimmed);
          }
        }

        const formatted = typeof obj === 'object' && obj !== null && !Array.isArray(obj)
          ? Object.entries(obj)
              .map(([key, value]) => `${key}: ${typeof value === 'object' ? JSON.stringify(value) : value}`)
              .join(', ')
          : (typeof obj === 'string' ? obj : JSON.stringify(obj));

        return <div key={i}>{formatted}</div>;
      })}
    </div>
  );
} 