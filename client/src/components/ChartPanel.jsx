import React, { memo } from 'react';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  LineElement,
  PointElement,
  LinearScale,
  CategoryScale,
} from 'chart.js';

// Ensure required elements are registered once
if (!ChartJS.registry.getElement('line')) {
  ChartJS.register(LineElement, PointElement, LinearScale, CategoryScale);
}

function ChartPanel({ labels, rewards, losses, style = {} }) {
  return (
    <div style={{ width: '100%', height: '100%', ...style }}>
      <Line
        data={{
          labels,
          datasets: [
            { label: 'Reward', data: rewards, borderColor: '#0f0', backgroundColor: 'transparent', borderWidth: 1, pointRadius: 0, yAxisID: 'y' },
            { label: 'Loss', data: losses, borderColor: 'orange', backgroundColor: 'transparent', borderWidth: 1, pointRadius: 0, yAxisID: 'y1' },
          ],
        }}
        options={{
          responsive: true,
          maintainAspectRatio: false,
          scales: {
            x: { ticks: { color: '#aaa' }, grid: { color: 'rgba(255,255,255,0.1)' }, title: { display: true, text: 'Training Batch', color: '#aaa' } },
            y: { ticks: { color: '#aaa' }, grid: { color: 'rgba(255,255,255,0.1)' }, title: { display: true, text: 'Reward', color: '#aaa' } },
            y1: { position: 'right', ticks: { color: 'orange' }, grid: { drawOnChartArea: false }, title: { display: true, text: 'Loss', color: 'orange' } },
          },
          plugins: { legend: { labels: { color: '#ddd' } } },
        }}
      />
    </div>
  );
}

export default memo(ChartPanel); 