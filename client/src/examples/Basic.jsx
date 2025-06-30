import React, { useState, useCallback, useEffect, useRef } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Grid, Stars } from '@react-three/drei';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  LineElement,
  PointElement,
  LinearScale,
  CategoryScale,
} from 'chart.js';
import DebugConsole from '../components/DebugConsole.jsx';
import config from '../config.js';
import 'katex/dist/katex.min.css';
import { BlockMath } from 'react-katex';
import { Text, Button } from '@geist-ui/core';

ChartJS.register(LineElement, PointElement, LinearScale, CategoryScale);

const SMALL_GOAL_POS = 7;
const LARGE_GOAL_POS = 17;
const MIN_POS = 0;
const MAX_POS = 20;
const WS_URL = `${config.WS_BASE_URL}/ws/basic`;

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
  const [modelInfo, setModelInfo] = useState(null);
  const [visualTraining, setVisualTraining] = useState(false);
  const [autoRun, setAutoRun] = useState(false);
  const intervalRef = useRef(null);

  const [chartState, setChartState] = useState({ labels: [], rewards: [], losses: [] });

  const step = useCallback((direction) => {
    setPos((prev) => {
      const next = Math.min(MAX_POS, Math.max(MIN_POS, prev + direction));
      // Base per-step penalty encourages shorter paths
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

      // Treat hitting the extrema of the grid as a terminal (failure) state so
      // the agent cannot get stuck endlessly accumulating negative reward.
      // We also apply an extra penalty to make this outcome clearly undesirable.
      if (next === MIN_POS || next === MAX_POS) {
        reward -= 0.5; // extra penalty for falling off the playable area
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
        setModelInfo({
          filename: parsed.model_filename,
          timestamp: parsed.timestamp,
          sessionUuid: parsed.session_uuid,
          fileUrl: parsed.file_url
        });
      }
      if (parsed.type === 'action') {
        // Action mapping: 0=left, 1=no move, 2=right
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

  const resetTraining = () => {
    setTraining(false);
    setTrained(false);
    setModelInfo(null);
    setVisualTraining(false);
    setAutoRun(false);
    setChartState({ labels: [], rewards: [], losses: [] });
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
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
      style={{ width: '100%', height: '100%', outline: 'none', background: 'linear-gradient(to bottom, #1a1a2e, #16213e)' }}
    >
      <Canvas
        camera={{ position: [0, 15, 25], fov: 50 }}
        style={{ background: 'linear-gradient(to bottom, #1a1a2e, #16213e)' }}
      >
        <ambientLight intensity={0.3} />
        <directionalLight 
          position={[0, 20, 10]} 
          intensity={1.2} 
          castShadow
          shadow-mapSize-width={2048}
          shadow-mapSize-height={2048}
        />
        
        {/* Professional background */}
        <Stars 
          radius={100} 
          depth={50} 
          count={5000} 
          factor={4} 
          saturation={0} 
          fade 
        />
        
        {/* Grid underneath the cubes */}
        <Grid 
          position={[0, -1, 0]}
          args={[30, 30]}
          cellSize={1}
          cellThickness={0.5}
          cellColor="#444"
          sectionSize={5}
          sectionThickness={1}
          sectionColor="#666"
          fadeDistance={25}
          fadeStrength={1}
        />
        
        <Goal position={SMALL_GOAL_POS} color="green" />
        <Goal position={LARGE_GOAL_POS} color="blue" />
        <Agent position={pos} />
        <OrbitControls 
          enableRotate={true} 
          enableZoom={true} 
          enablePan={true}
          target={[0, 0, 0]}
        />
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
        <Text h1 style={{ margin: '0 0 12px 0', color: '#fff' }}>
          Basic Example - 1-D Move-To-Goal
        </Text>
        <Text h3 style={{ margin: '0 0 12px 0', color: '#fff' }}>
          Reward: {rewardAccum.toFixed(2)}
        </Text>

        <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
          <Button
            auto
            type="secondary"
            disabled={training || trained}
            onClick={startTraining}
          >
            Train
          </Button>

          <Button
            auto
            type="success"
            disabled={!trained || autoRun}
            onClick={startRun}
          >
            Run
          </Button>

          {trained && (
            <Button auto type="error" onClick={resetTraining}>
              Reset
            </Button>
          )}
        </div>

        {modelInfo && (
          <div style={{ 
            marginTop: '8px', 
            fontSize: '12px', 
            background: 'rgba(0,255,0,0.1)', 
            padding: '4px 8px', 
            borderRadius: '4px',
            border: '1px solid rgba(0,255,0,0.3)'
          }}>
            <div><strong>Model:</strong> {modelInfo.filename}</div>
            <div><strong>Trained:</strong> {
              modelInfo.timestamp.replace(/(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})/, 
                '$1-$2-$3 $4:$5:$6')
            }</div>
            <div><strong>Session:</strong> {modelInfo.sessionUuid}</div>
            <a 
              href={`${config.API_BASE_URL}${modelInfo.fileUrl}`} 
              download
              style={{ color: '#4CAF50', textDecoration: 'underline' }}
            >
              Download Policy
            </a>
          </div>
        )}
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