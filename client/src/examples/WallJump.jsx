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
import { Link } from 'react-router-dom';

const WS_URL = `${config.WS_BASE_URL}/ws/walljump`;
const ACTION_DELTAS = [0, 1, -1, 0]; // stay, forward(+x), backward(-x), jump (handled separately)

export default function WallJumpExample() {
  const [state, setState] = useState({
    gridSize: 20,
    agentX: 0,
    goalX: 19,
    wallX: 10,
    wallPresent: 1,
    inAir: 0,
  });
  const [logs, setLogs] = useState([]);
  const [training, setTraining] = useState(false);
  const [trained, setTrained] = useState(false);
  const [modelInfo, setModelInfo] = useState(null);
  const [chartState, setChartState] = useState({ labels: [], rewards: [], losses: [] });
  const [homeHover, setHomeHover] = useState(false);

  const envRef = useRef({ ...state, inAir: 0, steps: 0 });
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
        setState((prev) => ({ ...prev, ...parsed.state }));
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
        if (wsRef.current._walljumpActionHandler) {
          wsRef.current._walljumpActionHandler(parsed.action);
        } else {
          // Basic positional update during training visualisation
          setState((prev) => {
            const actIdx = parsed.action;
            let { agentX, gridSize } = prev;
            let inAir = prev.inAir;
            // Jump action sets air timer
            if (actIdx === 3 && inAir === 0) {
              inAir = 2;
            }
            if (actIdx === 1) agentX = Math.min(agentX + 1, gridSize - 1);
            if (actIdx === 2) agentX = Math.max(agentX - 1, 0);
            // decrement air timer each basic update step
            if (inAir > 0 && actIdx !== 3) {
              inAir -= 1;
            }
            const crossingWall =
              (agentX < state.wallX && agentX >= state.wallX) || (agentX <= state.wallX && agentX > state.wallX);
            if (crossingWall && state.wallPresent && inAir === 0) {
              agentX = agentX; // blocked
            }
            const steps = prev.steps + 1;
            const done = agentX === state.goalX || steps >= 100;
            envRef.current = { ...prev, agentX, steps, inAir };
            setState((prev) => ({ ...prev, agentX, inAir }));
            if (done) {
              envRef.current = { gridSize: 20, agentX: 0, goalX: 19, wallX: 10, wallPresent: Math.random() < 0.7 ? 1 : 0, inAir: 0, steps: 0 };
              setState({ gridSize: 20, agentX: 0, goalX: 19, wallX: 10, wallPresent: Math.random() < 0.7 ? 1 : 0, inAir: 0 });
            }
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
      const gSize = 20;
      const agentX = 0;
      const goalX = gSize - 1;
      const wallX = 10;
      const wallPresent = Math.random() < 0.7 ? 1 : 0;
      envRef.current = { gridSize: gSize, agentX, goalX, wallX, wallPresent, inAir: 0, steps: 0 };
      setState({ gridSize: gSize, agentX, goalX, wallX, wallPresent, inAir: 0 });
    };

    const stepLocalEnv = (actionIdx) => {
      const st = envRef.current;
      let newAgentX = st.agentX;
      let inAir = st.inAir;
      if (actionIdx === 3 && inAir === 0) {
        inAir = 2; // jump duration
      }
      if (actionIdx === 1) newAgentX = Math.min(newAgentX + 1, st.gridSize - 1);
      if (actionIdx === 2) newAgentX = Math.max(newAgentX - 1, 0);
      const crossingWall =
        (st.agentX < st.wallX && newAgentX >= st.wallX) || (newAgentX <= st.wallX && st.agentX > st.wallX);
      if (crossingWall && st.wallPresent && inAir === 0) {
        newAgentX = st.agentX; // blocked
      }
      if (inAir > 0) inAir -= 1;
      const steps = st.steps + 1;
      const done = newAgentX === st.goalX || steps >= 100;
      envRef.current = { ...st, agentX: newAgentX, steps, inAir };
      setState((prev) => ({ ...prev, agentX: newAgentX, inAir }));
      if (done) resetLocalEnv();
    };

    resetLocalEnv();

    intervalRef.current = setInterval(() => {
      const st = envRef.current;
      const dx_goal = (st.goalX - st.agentX) / Math.max(1, st.gridSize - 1);
      const dx_wall = (st.wallX - st.agentX) / Math.max(1, st.gridSize - 1);
      const wall_h = st.wallPresent;
      const on_ground = st.inAir === 0 ? 1 : 0;
      send({ cmd: 'inference', obs: [dx_goal, dx_wall, wall_h, on_ground] });
    }, 200);

    wsRef.current._walljumpActionHandler = (actionIdx) => stepLocalEnv(actionIdx);
  };

  const resetTraining = () => {
    setTraining(false);
    setTrained(false);
    setModelInfo(null);
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    if (wsRef.current && wsRef.current._walljumpActionHandler) {
      delete wsRef.current._walljumpActionHandler;
    }
  };

  const { gridSize, agentX, goalX, wallX, wallPresent, inAir } = state;
  const half = (gridSize - 1) / 2;

  return (
    <div style={{ width: '100%', height: '100%', background: 'linear-gradient(to bottom, #0c0c28, #060614)' }}>
      <Canvas camera={{ position: [0, 10, 12], fov: 50 }} style={{ background: 'transparent' }}>
        <ambientLight intensity={0.5} />
        <directionalLight position={[5, 10, 5]} intensity={1} />

        <Grid args={[gridSize, 1]} cellSize={1} position={[0, 0, 0]} />

        {/* Ground cells */}
        {Array.from({ length: gridSize }).map((_, x) => (
          <mesh key={x} position={[x - half, 0, 0]} rotation={[-Math.PI / 2, 0, 0]}>
            <planeGeometry args={[1, 1]} />
            <meshStandardMaterial color="#222" transparent opacity={0.3} depthWrite={false} />
          </mesh>
        ))}

        {/* Goal strip */}
        <mesh position={[goalX - half, 0.01, 0]} rotation={[-Math.PI / 2, 0, 0]}>
          <planeGeometry args={[1, 1]} />
          <meshStandardMaterial color="#00ff00" transparent opacity={0.3} depthWrite={false} />
        </mesh>

        {/* Wall */}
        {wallPresent === 1 && (
          <mesh position={[wallX - half, 0.5, 0]}>
            <boxGeometry args={[1, 1, 1]} />
            <meshStandardMaterial color="#8888ff" transparent opacity={0.5} />
          </mesh>
        )}

        {/* Agent (raise height when jumping) */}
        <mesh position={[agentX - half, inAir > 0 ? 0.8 : 0.35, 0]}>
          <boxGeometry args={[0.4, 0.4, 0.4]} />
          <meshStandardMaterial color="#00aaff" />
        </mesh>

        <OrbitControls target={[0, 0, 0]} enablePan enableRotate enableZoom />
        <Stars radius={100} depth={50} count={4000} factor={4} saturation={0} fade />
      </Canvas>

      <div style={{ position: 'absolute', top: 10, left: 10, color: '#fff' }}>
        <Link
          to="/"
          style={{
            fontFamily: 'monospace',
            color: '#fff',
            textDecoration: homeHover ? 'none' : 'underline',
            display: 'inline-block',
          }}
          onMouseEnter={() => setHomeHover(true)}
          onMouseLeave={() => setHomeHover(false)}
        >
          Home
        </Link>
        <Text h1 style={{ margin: '0 0 12px 0', color: '#fff' }}>
          Wall Jump Example
        </Text>
        <div style={{ display: 'flex', gap: '8px' }}>
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
      </div>

      <div
        style={{
          position: 'absolute',
          bottom: 10,
          left: 10,
          width: 'auto',
          maxWidth: '420px',
          background: 'rgba(0,0,0,0.95)',
          color: '#fff',
          padding: '6px 8px',
          fontSize: 14,
          textAlign: 'left',
          justifyContent: 'flex-start',
        }}
      >
        <BlockMath
          math={"Q(s,a) \\leftarrow Q(s,a) + \\alpha\\, \\bigl( r + \\gamma (1 - d) \\max_{a'} Q(s',a') - Q(s,a) \\bigr)"}
          style={{ textAlign: 'left' }}
        />
        <div style={{ fontSize: 10, fontFamily: 'monospace', marginTop: 4 }}>
          Q-learning update: α learning rate, γ discount factor, d done flag (1 if terminal), r reward, a′ next action, s′ next state.
        </div>
      </div>

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
        <ChartPanel labels={chartState.labels} rewards={chartState.rewards} losses={chartState.losses} />
      </div>

      <DebugConsole logs={logs} />

      <ButtonForkOnGithub position={{ top: '20px', right: '20px' }} />
    </div>
  );
} 