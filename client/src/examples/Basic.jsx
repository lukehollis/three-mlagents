import React, { useState, useCallback, useEffect, useRef } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  LineElement,
  PointElement,
  LinearScale,
  CategoryScale,
} from 'chart.js';
import DebugConsole from '../components/DebugConsole.jsx';
import 'katex/dist/katex.min.css';
import { BlockMath } from 'react-katex';

ChartJS.register(LineElement, PointElement, LinearScale, CategoryScale);

const SMALL_GOAL_POS = 7;
const LARGE_GOAL_POS = 17;
const MIN_POS = 0;
const MAX_POS = 20;
const WS_URL = 'ws://localhost:8000/ws/basic';

function Agent({ position }) {
  return (
    <mesh position={[position - 10, 0, 0]}>
      <boxGeometry args={[0.9, 0.9, 0.9]} />
      <meshStandardMaterial color="orange" />
    </mesh>
  );
}

function Goal({ position, color }) {
  return (
    <mesh position={[position - 10, 0, 0]}>
      <boxGeometry args={[0.9, 0.9, 0.9]} />
      <meshStandardMaterial color={color} />
    </mesh>
  );
}

export default function BasicExample() {
  const [pos, setPos] = useState(10);
  const posRef = useRef(10);
  const [rewardAccum, setRewardAccum] = useState(0);

  const [logs, setLogs] = useState([]);
  const wsRef = useRef(null);

  const [training, setTraining] = useState(false);
  const [trained, setTrained] = useState(false);
  const [visualTraining, setVisualTraining] = useState(false);
  const [autoRun, setAutoRun] = useState(false);
  const intervalRef = useRef(null);

  const [chartState, setChartState] = useState({ labels: [], rewards: [], losses: [] });

  const step = useCallback((direction) => {
    setPos((prev) => {
      const next = Math.min(MAX_POS, Math.max(MIN_POS, prev + direction));
      let reward = -0.01;
      let done = false;

      if (next === SMALL_GOAL_POS) {
        reward += 0.1;
        done = true;
      }
      if (next === LARGE_GOAL_POS) {
        reward += 1;
        done = true;
      }

      setRewardAccum((r) => r + reward);

      if (done) {
        setTimeout(() => {
          setPos(10);
          setRewardAccum(0);
        }, 500);
        return prev; // keep current until reset
      }

      return next;
    });
  }, []);

  const handleKey = useCallback(
    (e) => {
      if (autoRun) return;
      if (e.key === 'ArrowLeft') step(-1);
      if (e.key === 'ArrowRight') step(1);
    },
    [step, autoRun]
  );

  // WebSocket connection
  useEffect(() => {
    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;
    ws.onopen = () => {
      addLog('WS opened');
    };
    ws.onmessage = (ev) => {
      addLog(ev.data);
      let parsed;
      try {
        parsed = JSON.parse(ev.data);
      } catch {
        return;
      }
      if (parsed.type === 'train_step') {
        setVisualTraining(true);
        setPos(parsed.pos);
      }
      if (parsed.type === 'progress') {
        setChartState((prev) => ({
          labels: [...prev.labels, parsed.episode],
          rewards: [...prev.rewards, parsed.reward],
          losses: [...prev.losses, parsed.loss ?? null],
        }));
      }
      if (parsed.type === 'trained') {
        setTraining(false);
        setTrained(true);
        setVisualTraining(false);
      }
      if (parsed.type === 'action') {
        const delta = [-1, 0, 1][parsed.action];
        step(delta);
      }
    };
    ws.onclose = () => addLog('WS closed');

    return () => ws.close();
  }, []);

  const addLog = (txt) => setLogs((l) => {
    const updated = [...l, txt];
    return updated.length > 200 ? updated.slice(updated.length - 200) : updated;
  });

  const send = (obj) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(obj));
    }
  };

  const startTraining = () => {
    if (training || trained) return;
    setTraining(true);
    send({ cmd: 'train' });
  };

  const startRun = () => {
    if (!trained || autoRun) return;
    setAutoRun(true);
    intervalRef.current = setInterval(() => {
      send({ cmd: 'inference', obs: posRef.current });
    }, 200);
  };

  useEffect(() => {
    if (!autoRun) return;
    return () => clearInterval(intervalRef.current);
  }, [autoRun]);

  // keep posRef synced
  useEffect(() => {
    posRef.current = pos;
  }, [pos]);

  return (
    <div
      tabIndex={0}
      onKeyDown={handleKey}
      style={{ width: '100%', height: '100%', outline: 'none', background: '#202020' }}
    >
      <Canvas
        orthographic
        camera={{ zoom: 50, position: [0, 0, 50] }}
        style={{ background: '#202020' }}
      >
        <ambientLight intensity={0.6} />
        <pointLight position={[10, 10, 10]} intensity={1} />
        <Goal position={SMALL_GOAL_POS} color="green" />
        <Goal position={LARGE_GOAL_POS} color="blue" />
        <Agent position={pos} />
        <OrbitControls enableRotate={false} enableZoom={false} enablePan={false} />
      </Canvas>
      <div
        style={{
          position: 'absolute',
          top: 10,
          left: 10,
          color: '#fff',
          textShadow: '0 0 4px #000',
        }}
      >
        <h3 style={{ margin: '0 0 12px 0' }}>Reward: {rewardAccum.toFixed(2)}</h3>
        <button disabled={training || trained} onClick={startTraining} style={{ marginRight: 8 }}>
          Train
        </button>
        <button disabled={!trained || autoRun} onClick={startRun}>
          Run
        </button>
      </div>

      <DebugConsole logs={logs} />

      {/* Chart above debug console */}
      <div
        style={{
          position: 'absolute',
          bottom: 160,
          right: 10,
          width: '30%',
          height: '180px',
          background: 'rgba(255,255,255,0.05)',
          padding: 4,
        }}
      >
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
                title: {
                  display: true,
                  text: 'Episode',
                  color: '#aaa',
                },
              },
              y: {
                ticks: { color: '#aaa' },
                grid: { color: 'rgba(255,255,255,0.1)' },
                title: {
                  display: true,
                  text: 'Reward',
                  color: '#aaa',
                },
              },
              y1: {
                position: 'right',
                ticks: { color: 'orange' },
                grid: { drawOnChartArea: false },
                title: {
                  display: true,
                  text: 'Loss',
                  color: 'orange',
                },
              },
            },
            plugins: { legend: { labels: { color: '#ddd' } } },
          }}
        />
      </div>

      {/* Q-learning update equation (bottom-left) */}
      <div
        style={{
          position: 'absolute',
          bottom: 10,
          left: 10,
          width: '30%',
          background: 'rgba(0,0,0,0.95)',
          color: '#fff',
          padding: '6px 8px',
          fontSize: 14,
        }}
      >
        <BlockMath math={
          'Q(s,a) \\leftarrow Q(s,a) + \\alpha \\left[\\, r + \\gamma \\max_{a\'} Q(s\',a\') - Q(s,a)\\,\\right]'
        } />
        <div style={{ fontSize: 10, fontFamily: 'monospace', marginTop: 4 }}>
          Q-learning update:&nbsp;Q(s,a) is the action-value,&nbsp;α the learning rate,&nbsp;γ the discount&nbsp;factor,&nbsp;r the reward,&nbsp;s′ the next&nbsp;state.
        </div>
      </div>

    </div>
  );
} 