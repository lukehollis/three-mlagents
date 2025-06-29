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
      {logs.map((ln, i) => (
        <div key={i}>{ln}</div>
      ))}
    </div>
  );
} 