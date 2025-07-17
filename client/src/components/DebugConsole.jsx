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
          let trimmed = ln.trim();
          
          // Handle "Received data: {...}..." format
          if (trimmed.startsWith('Received data: ')) {
            const jsonStart = trimmed.indexOf('{');
            if (jsonStart !== -1) {
              // Extract JSON part, remove trailing "..."
              let jsonPart = trimmed.substring(jsonStart);
              if (jsonPart.endsWith('...')) {
                jsonPart = jsonPart.slice(0, -3);
              }
              try {
                obj = JSON.parse(jsonPart);
              } catch (e) {
                // If parsing fails, keep original string
                obj = ln;
              }
            }
          } else if ((trimmed.startsWith('{') && trimmed.endsWith('}')) ||
                     (trimmed.startsWith('[') && trimmed.endsWith(']'))) {
            try {
              obj = JSON.parse(trimmed);
            } catch (e) {
              obj = ln;
            }
          }
        }

        // Special formatting for logs with "message" key: [type]: message
        let formatted;
        if (typeof obj === 'object' && obj !== null && !Array.isArray(obj) && obj.message) {
          formatted = `[${obj.type || 'log'}]: ${obj.message}`;
        } else if (typeof obj === 'object' && obj !== null && !Array.isArray(obj)) {
          formatted = Object.entries(obj)
            .map(([key, value]) => `${key}: ${typeof value === 'object' ? JSON.stringify(value) : value}`)
            .join(', ');
        } else {
          formatted = typeof obj === 'string' ? obj : JSON.stringify(obj);
        }

        return <div key={i}>{formatted}</div>;
      })}
    </div>
  );
} 