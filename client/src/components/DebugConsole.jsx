import React, { useEffect, useRef } from 'react';

export default function DebugConsole({ logs }) {
  const containerRef = useRef(null);

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
        position: 'absolute',
        bottom: 10,
        right: 10,
        width: '30%',
        height: '140px',
        overflowY: 'scroll',
        background: 'rgba(0,0,0,0.95)',
        color: '#0f0',
        fontFamily: 'monospace',
        fontSize: 10,
        padding: 4,
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