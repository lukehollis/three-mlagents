import React, { useEffect, useState } from 'react';
import config from '../config.js';

const DEFAULT_STATUS = 'Needs a standardized PettingZoo-compatible training wrapper before policy training is enabled.';

export default function RoadmapStatusPanel({ taskId, fallbackStatus = DEFAULT_STATUS }) {
  const [task, setTask] = useState(null);

  useEffect(() => {
    const controller = new AbortController();

    fetch(`${config.API_BASE_URL}/tasks/${encodeURIComponent(taskId)}`, {
      signal: controller.signal,
    })
      .then((response) => (response.ok ? response.json() : null))
      .then((data) => {
        if (data) setTask(data);
      })
      .catch((error) => {
        if (error.name !== 'AbortError') setTask(null);
      });

    return () => controller.abort();
  }, [taskId]);

  const status = task?.status || fallbackStatus;
  const interfaceName = task?.interface || 'pettingzoo';

  return (
    <div
      style={{
        maxWidth: 360,
        padding: '8px 10px',
        border: '1px solid rgba(255,255,255,0.7)',
        background: 'rgba(0,0,0,0.65)',
        color: '#fff',
        fontFamily: 'monospace',
        fontSize: '11px',
        lineHeight: 1.45,
        textTransform: 'none',
      }}
    >
      <div style={{ fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.08em' }}>
        Roadmap eval
      </div>
      <div style={{ opacity: 0.82 }}>Interface: {interfaceName}</div>
      <div style={{ opacity: 0.9 }}>{status}</div>
    </div>
  );
}
