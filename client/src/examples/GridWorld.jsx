import React, { useState, useEffect, useRef } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Grid, Stars } from '@react-three/drei';
import config from '../config.js';
import { Text, Button } from '@geist-ui/core';
import 'katex/dist/katex.min.css';
import { BlockMath } from 'react-katex';
import DebugConsole from '../components/DebugConsole.jsx';
import ChartPanel from '../components/ChartPanel.jsx';
import ButtonForkOnGithub from '../components/ButtonForkOnGithub.jsx';

const WS_URL = `${config.WS_BASE_URL}/ws/gridworld`;
const CELL_SIZE = 1;

function CellFloor({ position }) {
  return (
    <mesh position={position} rotation={[-Math.PI / 2, 0, 0]}>
      <planeGeometry args={[CELL_SIZE, CELL_SIZE]} />
      <meshStandardMaterial color="#222" transparent opacity={0.3} depthWrite={false} />
    </mesh>
  );
}

function Goal({ type, position }) {
  const color = type === 'green' ? '#00ff00' : '#ff0000';
  return (
    <mesh position={[position[0], 0.25, position[1]]}>
      <boxGeometry args={[0.6, 0.5, 0.6]} />
      <meshStandardMaterial color={color} />
    </mesh>
  );
}

function Agent({ position, currentGoalType }) {
  const color = currentGoalType === 0 ? '#00ffff' : '#ff00ff';
  return (
    <mesh position={[position[0], 0.35, position[1]]}>
      <sphereGeometry args={[0.3, 16, 16]} />
      <meshStandardMaterial color={color} />
    </mesh>
  );
}

export default function GridWorldExample() {
  const [state, setState] = useState({
    gridSize: 5,
    agentX: 0,
    agentY: 0,
    greenGoals: [],
    redGoals: [],
    currentGoalType: 0,
  });
  const [logs, setLogs] = useState([]);
  const [training, setTraining] = useState(false);
  const [trained, setTrained] = useState(false);
  const [modelInfo, setModelInfo] = useState(null);
  const [chartState, setChartState] = useState({ labels: [], rewards: [], losses: [] });
  const wsRef = useRef(null);
  const intervalRef = useRef(null);

  const addLog = (txt) => {
    setLogs((l) => {
      const upd = [...l, txt];
      return upd.length > 200 ? upd.slice(upd.length - 200) : upd;
    });
  };

  useEffect(() => {
    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;

    ws.onopen = () => addLog('WS opened');
    ws.onmessage = (ev) => {
      addLog(ev.data);
      let parsed;
      try {
        parsed = JSON.parse(ev.data);
      } catch {
        return;
      }
      if (parsed.type === 'train_step') {
        setState(parsed.state);
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
        setModelInfo({
          filename: parsed.model_filename,
          timestamp: parsed.timestamp,
          sessionUuid: parsed.session_uuid,
          fileUrl: parsed.file_url,
        });
      }
      if (parsed.type === 'action') {
        // handle inference step – send action to physics, but here env simulated server-side so ignore
      }
    };
    ws.onclose = () => addLog('WS closed');
    return () => ws.close();
  }, []);

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
    if (!trained) return;
    if (intervalRef.current) return;
    intervalRef.current = setInterval(() => {
      const { agentX, agentY, gridSize, currentGoalType, greenGoals, redGoals } = state;
      const goalPos = currentGoalType === 0 ? (greenGoals[0] || [0, 0]) : (redGoals[0] || [0, 0]);
      const dx = (goalPos[0] - agentX) / Math.max(1, gridSize - 1);
      const dy = (goalPos[1] - agentY) / Math.max(1, gridSize - 1);
      const goalOneHot = currentGoalType === 0 ? [1, 0] : [0, 1];
      send({ cmd: 'inference', obs: [dx, dy, ...goalOneHot] });
    }, 200);
  };

  const resetTraining = () => {
    setTraining(false);
    setTrained(false);
    setModelInfo(null);
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
  };

  const { gridSize, agentX, agentY, greenGoals, redGoals, currentGoalType } = state;

  const half = (gridSize - 1) / 2;

  return (
    <div style={{ width: '100%', height: '100%', background: 'linear-gradient(to bottom, #0c0c28, #060614)' }}>
      <Canvas camera={{ position: [0, 10, 10], fov: 50 }} style={{ background: 'transparent' }}>
        <ambientLight intensity={0.5} />
        <directionalLight position={[5, 10, 5]} intensity={1} />

        {/* Floor grid */}
        <Grid args={[gridSize, gridSize]} cellSize={1} position={[0, 0, 0]}  />

        {/* Render cells (invisible floor for each) */}
        {Array.from({ length: gridSize }).map((_, x) =>
          Array.from({ length: gridSize }).map((_, y) => (
            <CellFloor key={`${x}-${y}`} position={[x - half, 0, y - half]} />
          ))
        )}

        {/* Goals */}
        {greenGoals.map((g, idx) => (
          <Goal key={`g${idx}`} type="green" position={[g[0] - half, g[1] - half]} />
        ))}
        {redGoals.map((g, idx) => (
          <Goal key={`r${idx}`} type="red" position={[g[0] - half, g[1] - half]} />
        ))}

        {/* Agent */}
        <Agent position={[agentX - half, agentY - half]} currentGoalType={currentGoalType} />

        <OrbitControls target={[0, 0, 0]} enablePan enableRotate enableZoom />

        <Stars radius={100} depth={50} count={4000} factor={4} saturation={0} fade />
      </Canvas>

      {/* Overlay UI */}
      <div style={{ position: 'absolute', top: 10, left: 10, color: '#fff' }}>
        <Text h1 style={{ margin: '0 0 12px 0', color: '#fff' }}>
          GridWorld Example
        </Text>
        <div style={{ display: 'flex', gap: '8px' }}>
          <Button auto type="secondary" disabled={training || trained} onClick={startTraining}>Train</Button>
          <Button auto type="success" disabled={!trained} onClick={startRun}>Run</Button>
          {trained && <Button auto type="error" onClick={resetTraining}>Reset</Button>}
        </div>
      </div>

      {/* Q-learning update equation (bottom-left) */}
      <div style={{ position: 'absolute', bottom: 10, left: 10, width: 'auto', maxWidth: '420px', background: 'rgba(0,0,0,0.95)', color: '#fff', padding: '6px 8px', fontSize: 14, textAlign: 'left', justifyContent: 'flex-start' }}>
        <BlockMath math={"Q(s,a) \\leftarrow Q(s,a) + \\alpha\\, \\bigl(\\, r + \\gamma \\max_{a'} Q(s',a') - Q(s,a) \\bigr)"} style={{ textAlign: 'left' }} />
        <div style={{ fontSize: 10, fontFamily: 'monospace', marginTop: 4 }}>
          Q-learning update: α learning rate, γ discount factor, r reward, a′ next action, s′ next state.
        </div>
      </div>

      {/* Chart */}
      <div style={{ position: 'absolute', bottom: 160, right: 10, width: '30%', height: '180px', background: 'rgba(255,255,255,0.05)', padding: 4 }}>
        <ChartPanel labels={chartState.labels} rewards={chartState.rewards} losses={chartState.losses} />
      </div>

      <DebugConsole logs={logs} />

      {/* Fork link (top-right) */}
      <ButtonForkOnGithub position={{ top: '20px', right: '20px' }} />
    </div>
  );
} 