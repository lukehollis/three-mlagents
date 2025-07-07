import React, { useState, useEffect, useRef } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Grid, Stars } from '@react-three/drei';
import config from '../config.js';
import { Text, Button } from '@geist-ui/core';
import 'katex/dist/katex.min.css';
import ButtonForkOnGithub from '../components/ButtonForkOnGithub.jsx';
import { Link } from 'react-router-dom';
import EquationPanel from '../components/EquationPanel.jsx';
import InfoPanel from '../components/InfoPanel.jsx';
import ModelInfoPanel from '../components/ModelInfoPanel.jsx';
import { useResponsive } from '../hooks/useResponsive.js';

const WS_URL = `${config.WS_BASE_URL}/ws/push`;
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

function GoalStrip({ gridSize, half }) {
  return (
    <mesh position={[0, 0.01, (gridSize - 1) - half]} rotation={[-Math.PI / 2, 0, 0]}>
      <planeGeometry args={[gridSize, CELL_SIZE]} />
      <meshStandardMaterial color="#00ff00" transparent opacity={0.3} depthWrite={false} />
    </mesh>
  );
}

function Box({ position }) {
  return (
    <mesh position={[position[0], 0.3, position[1]]}>
      <boxGeometry args={[0.6, 0.6, 0.6]} />
      <meshStandardMaterial color="#ffffff" />
    </mesh>
  );
}

function Agent({ position }) {
  return (
    <mesh position={[position[0], 0.35, position[1]]}>
      <boxGeometry args={[0.4, 0.4, 0.4]} />
      <meshStandardMaterial color="#00aaff" />
    </mesh>
  );
}

export default function PushExample() {
  const [state, setState] = useState({
    gridSize: 6,
    agentX: 0,
    agentY: 0,
    boxX: 1,
    boxY: 1,
    goalX: 0,
    goalY: 5,
  });
  const [logs, setLogs] = useState([]);
  const [training, setTraining] = useState(false);
  const [trained, setTrained] = useState(false);
  const [modelInfo, setModelInfo] = useState(null);
  const [chartState, setChartState] = useState({ labels: [], rewards: [], losses: [] });
  const [homeHover, setHomeHover] = useState(false);
  const { isMobile } = useResponsive();

  const envRef = useRef({ ...state, steps: 0 });
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
        if (wsRef.current._pushActionHandler) {
          wsRef.current._pushActionHandler(parsed.action);
        } else {
          // Basic positional update during training visualisation
          setState((prev) => {
            const [dx, dy] = ACTION_DELTAS[parsed.action] ?? [0, 0];
            let { agentX, agentY, boxX, boxY, gridSize } = prev;
            let newAgentX = Math.min(Math.max(agentX + dx, 0), gridSize - 1);
            let newAgentY = Math.min(Math.max(agentY + dy, 0), gridSize - 1);
            let newBoxX = boxX;
            let newBoxY = boxY;
            if (newAgentX === boxX && newAgentY === boxY) {
              const tentativeBoxX = boxX + dx;
              const tentativeBoxY = boxY + dy;
              if (
                tentativeBoxX >= 0 && tentativeBoxX < gridSize &&
                tentativeBoxY >= 0 && tentativeBoxY < gridSize
              ) {
                newBoxX = tentativeBoxX;
                newBoxY = tentativeBoxY;
              } else {
                newAgentX = agentX;
                newAgentY = agentY;
              }
            }
            return { ...prev, agentX: newAgentX, agentY: newAgentY, boxX: newBoxX, boxY: newBoxY };
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

    const resetLocalEnv = () => {
      const gSize = 6;
      const cells = [];
      for (let x = 0; x < gSize; x += 1) {
        for (let y = 0; y < gSize; y += 1) {
          cells.push([x, y]);
        }
      }
      // shuffle
      for (let i = cells.length - 1; i > 0; i -= 1) {
        const j = Math.floor(Math.random() * (i + 1));
        [cells[i], cells[j]] = [cells[j], cells[i]];
      }
      const agentPos = cells[0];
      const boxPos = cells[1];
      const goalPos = [Math.floor(Math.random() * gSize), gSize - 1];

      envRef.current = {
        gridSize: gSize,
        agentX: agentPos[0],
        agentY: agentPos[1],
        boxX: boxPos[0],
        boxY: boxPos[1],
        goalX: goalPos[0],
        goalY: goalPos[1],
        steps: 0,
      };

      setState({
        gridSize: gSize,
        agentX: agentPos[0],
        agentY: agentPos[1],
        boxX: boxPos[0],
        boxY: boxPos[1],
        goalX: goalPos[0],
        goalY: goalPos[1],
      });
    };

    const stepLocalEnv = (actionIdx) => {
      const st = { ...envRef.current };
      const [dxA, dyA] = ACTION_DELTAS[actionIdx] ?? [0, 0];

      let newAgentX = Math.min(Math.max(st.agentX + dxA, 0), st.gridSize - 1);
      let newAgentY = Math.min(Math.max(st.agentY + dyA, 0), st.gridSize - 1);
      let newBoxX = st.boxX;
      let newBoxY = st.boxY;

      if (newAgentX === st.boxX && newAgentY === st.boxY) {
        const tentativeBoxX = st.boxX + dxA;
        const tentativeBoxY = st.boxY + dyA;
        if (
          tentativeBoxX >= 0 && tentativeBoxX < st.gridSize &&
          tentativeBoxY >= 0 && tentativeBoxY < st.gridSize
        ) {
          newBoxX = tentativeBoxX;
          newBoxY = tentativeBoxY;
        } else {
          newAgentX = st.agentX;
          newAgentY = st.agentY;
        }
      }

      const steps = st.steps + 1;
      const done = (newBoxX === st.goalX && newBoxY === st.goalY) || steps >= 100;

      envRef.current = {
        ...st,
        agentX: newAgentX,
        agentY: newAgentY,
        boxX: newBoxX,
        boxY: newBoxY,
        steps,
      };

      setState({
        gridSize: st.gridSize,
        agentX: newAgentX,
        agentY: newAgentY,
        boxX: newBoxX,
        boxY: newBoxY,
        goalX: st.goalX,
        goalY: st.goalY,
      });

      if (done) {
        resetLocalEnv();
      }
    };

    resetLocalEnv();

    intervalRef.current = setInterval(() => {
      const st = envRef.current;
      const dx_ab = (st.boxX - st.agentX) / Math.max(1, st.gridSize - 1);
      const dy_ab = (st.boxY - st.agentY) / Math.max(1, st.gridSize - 1);
      const dx_bg = (st.goalX - st.boxX) / Math.max(1, st.gridSize - 1);
      const dy_bg = (st.goalY - st.boxY) / Math.max(1, st.gridSize - 1);
      send({ cmd: 'inference', obs: [dx_ab, dy_ab, dx_bg, dy_bg] });
    }, 200);

    wsRef.current._pushActionHandler = (actionIdx) => stepLocalEnv(actionIdx);
  };

  const resetTraining = () => {
    setTraining(false);
    setTrained(false);
    setModelInfo(null);
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    if (wsRef.current && wsRef.current._pushActionHandler) {
      delete wsRef.current._pushActionHandler;
    }
  };

  const { gridSize, agentX, agentY, boxX, boxY, goalX, goalY } = state;
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

          <Grid args={[gridSize, gridSize]} cellSize={1} position={[0, 0, 0]} />

          {Array.from({ length: gridSize }).map((_, x) =>
            Array.from({ length: gridSize }).map((_, y) => (
              <CellFloor key={`${x}-${y}`} position={[x - half, 0, y - half]} />
            ))
          )}

          <GoalStrip gridSize={gridSize} half={half} />

          <Box position={[boxX - half, boxY - half]} />
          <Agent position={[agentX - half, agentY - half]} />

          <OrbitControls target={[0, 0, 0]} enablePan enableRotate enableZoom />
          <Stars radius={100} depth={50} count={4000} factor={4} saturation={0} fade />
        </Canvas>

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
          <Text h1 style={{ margin: '12px 0 12px 0', color: '#fff', fontSize: isMobile ? '1.2rem' : '2rem' }}>
            Push-Block
          </Text>
          <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
            <Button auto type="secondary" disabled={training || trained} onClick={startTraining}>
              Train
            </Button>
            <Button auto type="success" disabled={!trained} onClick={startRun}>
              Run
            </Button>
            {trained && (
              <Button auto type="error" onClick={resetTraining}>
                Reset
              </Button>
            )}
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