import React, { useState, useEffect, useRef } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Grid, Stars } from '@react-three/drei';
import config from '../config.js';
import { Text, Button, useMediaQuery } from '@geist-ui/core';
import 'katex/dist/katex.min.css';
import ButtonForkOnGithub from '../components/ButtonForkOnGithub.jsx';
import { Link } from 'react-router-dom';
import EquationPanel from '../components/EquationPanel.jsx';
import InfoPanel from '../components/InfoPanel.jsx';
import ModelInfoPanel from '../components/ModelInfoPanel.jsx';

const WS_URL = `${config.WS_BASE_URL}/ws/gridworld`;
const CELL_SIZE = 1;

const ACTION_DELTAS = [
  [0, 0],   // stay
  [0, 1],   // up
  [0, -1],  // down
  [-1, 0],  // left
  [1, 0],   // right
];

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
  const [homeHover, setHomeHover] = useState(false);
  const isMobile = useMediaQuery('sm') || useMediaQuery('xs');
  // Local environment replica for inference – keeps client-side state in sync across steps
  const envRef = useRef({
    gridSize: 5,
    agentX: 0,
    agentY: 0,
    greenGoals: [[0, 0]],
    redGoals: [[0, 0]],
    currentGoalType: 0,
    steps: 0,
  });
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
        // Prefer the scoped handler when the user is in run mode.
        if (wsRef.current._gwActionHandler) {
          wsRef.current._gwActionHandler(parsed.action);
        } else {
          // Fallback: basic positional update (training visualisation)
          setState((prev) => {
            const { agentX, agentY, gridSize } = prev;
            const [dx, dy] = ACTION_DELTAS[parsed.action] ?? [0, 0];
            const newX = Math.min(Math.max(agentX + dx, 0), gridSize - 1);
            const newY = Math.min(Math.max(agentY + dy, 0), gridSize - 1);
            return { ...prev, agentX: newX, agentY: newY };
          });
        }
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
    // Initialise a fresh random episode that mirrors training resets
    const resetLocalEnv = () => {
      const gSize = 5;
      const cells = [];
      for (let x = 0; x < gSize; x += 1) {
        for (let y = 0; y < gSize; y += 1) {
          cells.push([x, y]);
        }
      }
      // shuffle cells in-place (Fisher-Yates)
      for (let i = cells.length - 1; i > 0; i -= 1) {
        const j = Math.floor(Math.random() * (i + 1));
        [cells[i], cells[j]] = [cells[j], cells[i]];
      }
      const agentPos = cells[0];
      const green = [cells[1]];
      const red = [cells[2]];
      const goalType = Math.random() < 0.5 ? 0 : 1;

      envRef.current = {
        gridSize: gSize,
        agentX: agentPos[0],
        agentY: agentPos[1],
        greenGoals: green,
        redGoals: red,
        currentGoalType: goalType,
        steps: 0,
      };

      // Push to React state so UI shows the newly initialised grid
      setState({
        gridSize: gSize,
        agentX: agentPos[0],
        agentY: agentPos[1],
        greenGoals: green,
        redGoals: red,
        currentGoalType: goalType,
      });
    };

    const stepLocalEnv = (actionIdx) => {
      const st = { ...envRef.current };
      const [dxA, dyA] = ACTION_DELTAS[actionIdx] ?? [0, 0];
      st.agentX = Math.min(Math.max(st.agentX + dxA, 0), st.gridSize - 1);
      st.agentY = Math.min(Math.max(st.agentY + dyA, 0), st.gridSize - 1);
      st.steps += 1;

      // Goal collision / episode termination
      const atGreen = st.greenGoals.some(([gx, gy]) => gx === st.agentX && gy === st.agentY);
      const atRed = st.redGoals.some(([rx, ry]) => rx === st.agentX && ry === st.agentY);
      let done = false;
      if (atGreen || atRed) done = true;
      if (st.steps >= 100) done = true; // aligns with MAX_STEPS_PER_EP

      envRef.current = st;
      setState({
        gridSize: st.gridSize,
        agentX: st.agentX,
        agentY: st.agentY,
        greenGoals: st.greenGoals,
        redGoals: st.redGoals,
        currentGoalType: st.currentGoalType,
      });

      if (done) {
        resetLocalEnv();
      }
    };

    // Kick-off first episode
    resetLocalEnv();

    // Fixed-rate control loop
    intervalRef.current = setInterval(() => {
      const st = envRef.current;
      const goalPos = st.currentGoalType === 0 ? (st.greenGoals[0] || [0, 0]) : (st.redGoals[0] || [0, 0]);
      const dx = (goalPos[0] - st.agentX) / Math.max(1, st.gridSize - 1);
      const dy = (goalPos[1] - st.agentY) / Math.max(1, st.gridSize - 1);
      const goalOneHot = st.currentGoalType === 0 ? [1, 0] : [0, 1];
      send({ cmd: 'inference', obs: [dx, dy, ...goalOneHot] });
    }, 200);

    // Handle every action the backend sends
    const onAction = (actionIdx) => stepLocalEnv(actionIdx);

    // Attach a listener scoped to this run
    wsRef.current._gwActionHandler = onAction;
  };

  const resetTraining = () => {
    setTraining(false);
    setTrained(false);
    setModelInfo(null);
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    if (wsRef.current && wsRef.current._gwActionHandler) {
      delete wsRef.current._gwActionHandler;
    }
  };

  const { gridSize, agentX, agentY, greenGoals, redGoals, currentGoalType } = state;

  const half = (gridSize - 1) / 2;

  return (
    <div
      style={{
        width: '100vw',
        height: '100vh',
        overflow: 'hidden',
        outline: 'none',
        background: 'linear-gradient(to bottom, #1a1a2e, #16213e)',
        display: 'flex',
        flexDirection: 'column',
      }}
    >
      <div style={{ flex: 1, position: 'relative' }}>
        <Canvas camera={{ position: [0, 10, 10], fov: 50 }} style={{ background: 'transparent', width: '100vw', height: '100vh', overflow: 'hidden' }}>
          <ambientLight intensity={0.5} />
          <directionalLight position={[5, 10, 5]} intensity={1} />

          {/* Floor grid */}
          <Grid args={[gridSize, gridSize]} cellSize={1} position={[0, 0, 0]} />

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
        <div
          style={{
            position: 'absolute',
            top: 10,
            left: 10,
            color: '#fff',
            textShadow: '0 0 4px #000',
            zIndex: 1,
          }}
        >
          {/* Home link */}
          <Link
            to="/"
            style={{
              fontFamily: 'monospace',
              color: '#fff',
              textDecoration: homeHover ? 'none' : 'underline',
              display: 'inline-block',
              fontSize: isMobile ? '12px' : '14px',
            }}
            onMouseEnter={() => setHomeHover(true)}
            onMouseLeave={() => setHomeHover(false)}
          >
            Home
          </Link>
          <Text h1 style={{ margin: '12px 0 12px 0', color: '#fff', fontSize: isMobile ? '1.2rem' : 'inherit' }}>
            GridWorld Example
          </Text>
          <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
            <Button auto type="secondary" disabled={training || trained} onClick={startTraining}>Train</Button>
            <Button auto type="success" disabled={!trained} onClick={startRun}>Run</Button>
            {trained && <Button auto type="error" onClick={resetTraining}>Reset</Button>}
          </div>
          <ModelInfoPanel modelInfo={modelInfo} />
        </div>

        <EquationPanel equation="Q(s, a) \leftarrow Q(s, a) + \alpha [r + \gamma \max_{a'} Q(s', a') - Q(s, a)]" description="Q-learning update:&nbsp;Q(s,a) is the action-value,&nbsp;α the learning rate,&nbsp;γ the discount factor,&nbsp;r the reward,&nbsp;s' the next state." />
        <InfoPanel logs={logs} chartState={chartState} />
        <ButtonForkOnGithub position={{ top: '10px', right: '10px' }} />
      </div>
    </div>
  );
} 