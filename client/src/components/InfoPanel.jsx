import React, { useState } from 'react';
import { Button } from '@geist-ui/core';
import { Line } from 'react-chartjs-2';
import DebugConsole from './DebugConsole.jsx';
import { ChevronsDown, ChevronsUp, Activity } from '@geist-ui/icons';
import { useResponsive } from '../hooks/useResponsive';

export default function InfoPanel({ chartState, logs }) {
  const { isMobile } = useResponsive();
  const [isOpen, setIsOpen] = useState(!isMobile);

  if (!isOpen) {
    return (
      <div style={{ position: 'absolute', bottom: 10, right: 10, zIndex: 1 }}>
        <Button auto icon={<Activity />} onClick={() => setIsOpen(true)} type="secondary">
          Info
        </Button>
      </div>
    );
  }

  return (
    <div
      style={{
        position: 'absolute',
        bottom: 10,
        right: 10,
        width: isMobile ? 'calc(100% - 20px)' : '45%',
        display: 'flex',
        flexDirection: 'column',
        gap: '10px',
        zIndex: 10,
        maxWidth: '420px',
      }}
    >
      <div
        style={{
          height: '180px',
          background: 'rgba(0,0,0,0.6)',
          padding: 4,
          borderRadius: '8px',
          border: '1px solid rgba(255,255,255,0.2)',
          position: 'relative',
        }}
      >
        <Button
          auto
          scale={0.5}
          icon={<ChevronsDown />}
          onClick={() => setIsOpen(false)}
          style={{ position: 'absolute', top: 5, right: 5, zIndex: 10 }}
          type="abort"
        />
        <Line
          data={{
            labels: chartState.labels,
            datasets: [
              {
                label: 'Reward',
                data: chartState.rewards,
                borderColor: '#0f0',
                backgroundColor: 'transparent',
                borderWidth: 1,
                pointRadius: 0,
                yAxisID: 'y',
              },
              {
                label: 'Loss',
                data: chartState.losses,
                borderColor: 'orange',
                backgroundColor: 'transparent',
                borderWidth: 1,
                pointRadius: 0,
                yAxisID: 'y1',
              },
            ],
          }}
          options={{
            responsive: true,
            maintainAspectRatio: false,
            scales: {
              x: {
                ticks: { color: '#aaa' },
                grid: { color: 'rgba(255,255,255,0.1)' },
                title: { display: true, text: 'Timestep', color: '#aaa' },
              },
              y: {
                ticks: { color: '#aaa' },
                grid: { color: 'rgba(255,255,255,0.1)' },
                title: { display: true, text: 'Reward', color: '#aaa' },
              },
              y1: {
                position: 'right',
                ticks: { color: 'orange' },
                grid: { drawOnChartArea: false },
                title: { display: true, text: 'Loss', color: 'orange' },
              },
            },
            plugins: { legend: { labels: { color: '#ddd' } } },
          }}
        />
      </div>
      <div style={{ width: 'calc(100% - 15px)' }}>
        <DebugConsole logs={logs} />
      </div>
    </div>
  );
} 