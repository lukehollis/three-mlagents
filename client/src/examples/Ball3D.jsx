import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Stars, Grid } from '@react-three/drei';
import config from '../config.js';
import 'katex/dist/katex.min.css';
import { Text, Button } from '@geist-ui/core';
import ButtonForkOnGithub from '../components/ButtonForkOnGithub.jsx';
import EquationPanel from '../components/EquationPanel.jsx';
import InfoPanel from '../components/InfoPanel.jsx';
import ModelInfoPanel from '../components/ModelInfoPanel.jsx';
import { Link } from 'react-router-dom';
import { useResponsive } from '../hooks/useResponsive.js';

const ROWS = 3;
const COLS = 4;
const WS_URL = `${config.WS_BASE_URL}/ws/ball3d`;

function PlatformAndBall({ state, position }) {
  const { rotX, rotZ, ballX, ballZ } = state;
  return (
    <group position={position} rotation={[rotX, 0, rotZ]}>
      {/* Platform cuboid */}
      <mesh position={[0, 0, 0]}>
        <boxGeometry args={[6, 0.5, 6]} />
        <meshStandardMaterial color="#3da8ff" />
      </mesh>
      {/* Ball */}
      <mesh position={[ballX, 0.75, ballZ]}>
        <sphereGeometry args={[0.5, 32, 32]} />
        <meshStandardMaterial color="#dddddd" metalness={0.6} roughness={0.3} />
      </mesh>
    </group>
  );
}

export default function Ball3DExample() {
  // Each state item: {rotX, rotZ, ballX, ballZ}
  const [states, setStates] = useState(Array.from({ length: ROWS * COLS }, () => ({ rotX: 0, rotZ: 0, ballX: 0, ballZ: 0 })));
  const [logs, setLogs] = useState([]);
  const wsRef = useRef(null);

  const [training, setTraining] = useState(false);
  const [trained, setTrained] = useState(false);
  const [modelInfo, setModelInfo] = useState(null);
  const [autoRun, setAutoRun] = useState(false);
  const intervalRef = useRef(null);

  const [chartState, setChartState] = useState({ labels: [], rewards: [], losses: [] });
  const [homeHover, setHomeHover] = useState(false);
  const { isMobile } = useResponsive();

  const envRef = useRef({ rotX: 0, rotZ: 0, ballX: 0, ballZ: 0, velX: 0, velZ: 0 });

  const addLog = (txt) => setLogs((l) => {
    const updated = [...l, txt];
    return updated.length > 200 ? updated.slice(updated.length - 200) : updated;
  });

  // WebSocket connection
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
        const s = parsed.state;
        // replicate state across all platforms for now
        setStates(Array.from({ length: ROWS * COLS }, () => s));
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
        physicsStep(parsed.action);
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
    if (!trained || autoRun) return;
    setAutoRun(true);
    // Randomize local env for diversity similar to training resets
    envRef.current = {
      rotX: (Math.random() - 0.5) * (Math.PI / 14), // ±~12°
      rotZ: (Math.random() - 0.5) * (Math.PI / 14),
      ballX: (Math.random() - 0.5) * 3,
      ballZ: (Math.random() - 0.5) * 3,
      velX: (Math.random() - 0.5) * 1,
      velZ: (Math.random() - 0.5) * 1,
    };

    setStates(Array.from({ length: ROWS * COLS }, () => ({
      rotX: envRef.current.rotX,
      rotZ: envRef.current.rotZ,
      ballX: envRef.current.ballX,
      ballZ: envRef.current.ballZ,
    })));

    intervalRef.current = setInterval(() => {
      // send current observation from first platform
      const obs = states[0];
      // Include velocity to match training observation (velX, velZ)
      send({ cmd: 'inference', obs: [obs.rotX, obs.rotZ, obs.ballX, obs.ballZ, envRef.current.velX, envRef.current.velZ] });
    }, 100);
  };

  const resetTraining = () => {
    setTraining(false);
    setTrained(false);
    setModelInfo(null);
    setAutoRun(false);
    setChartState({ labels: [], rewards: [], losses: [] });
    envRef.current = { rotX: 0, rotZ: 0, ballX: 0, ballZ: 0, velX: 0, velZ: 0 };
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
  };

  useEffect(() => () => clearInterval(intervalRef.current), []);

  // Precompute positions for each platform
  const positions = React.useMemo(() => {
    const posArr = [];
    const spacing = 8;
    for (let r = 0; r < ROWS; r += 1) {
      for (let c = 0; c < COLS; c += 1) {
        const x = (c - (COLS - 1) / 2) * spacing;
        const z = (r - (ROWS - 1) / 2) * spacing;
        posArr.push([x, 0, z]);
      }
    }
    return posArr;
  }, []);

  const physicsStep = (actionIdx) => {
    const MAX_TILT = Math.PI / 7; // ≈25°
    const TILT_DELTA = Math.PI / 60; // small increment
    const DT = 0.02;
    const G = 9.81;

    // action mapping
    const deltas = [
      [ TILT_DELTA, 0 ],
      [ -TILT_DELTA, 0 ],
      [ 0, TILT_DELTA ],
      [ 0, -TILT_DELTA ],
      [ 0, 0 ],
    ];

    const st = envRef.current;
    // update rotation
    st.rotX = Math.max(-MAX_TILT, Math.min(MAX_TILT, st.rotX + deltas[actionIdx][0]));
    st.rotZ = Math.max(-MAX_TILT, Math.min(MAX_TILT, st.rotZ + deltas[actionIdx][1]));

    // physics for ball
    st.velX += Math.sin(st.rotX) * G * DT;
    st.velZ += Math.sin(st.rotZ) * G * DT;
    st.velX *= 0.98;
    st.velZ *= 0.98;
    st.ballX += st.velX * DT;
    st.ballZ += st.velZ * DT;

    envRef.current = { ...st };
    setStates(Array.from({ length: ROWS * COLS }, () => ({ rotX: st.rotX, rotZ: st.rotZ, ballX: st.ballX, ballZ: st.ballZ })));
  };

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
        <Canvas camera={{ position: [0, 25, 35], fov: 50 }} style={{ background: 'transparent', width: '100vw', height: '100vh', overflow: 'hidden' }}>
          <ambientLight intensity={0.4} />
          <directionalLight position={[10, 20, 10]} intensity={1.1} />

          <Stars radius={100} depth={50} count={4000} factor={4} saturation={0} fade />

          {/* Grid helper underneath platforms */}
          <Grid
            position={[0, -0.26, 0]}
            args={[60, 60]}
            cellSize={1}
            cellThickness={0.4}
            cellColor="#444"
            sectionSize={5}
            sectionThickness={1}
            sectionColor="#666"
            fadeDistance={40}
            fadeStrength={1}
          />

          {states.map((s, idx) => (
            <PlatformAndBall key={idx} state={s} position={positions[idx]} />
          ))}

          <OrbitControls target={[0, 0, 0]} enablePan enableRotate enableZoom />
        </Canvas>

        {/* UI overlay */}
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
            Ball 3D Example
          </Text>
          <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
            <Button auto type="secondary" disabled={training || trained} onClick={startTraining}>Train</Button>
            <Button auto type="success" disabled={!trained || autoRun} onClick={startRun}>Run</Button>
            {trained && <Button auto type="error" onClick={resetTraining}>Reset</Button>}
          </div>

          <ModelInfoPanel modelInfo={modelInfo} />
        </div>
        <EquationPanel equation="L^{CLIP}(\theta) = \hat{\mathbb{E}}_t[\min(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t)]" description="PPO-clip objective: encourages the policy ratio to stay within a small interval around 1." />
        <InfoPanel logs={logs} chartState={chartState} />
        <ButtonForkOnGithub position={{ top: '10px', right: '10px' }} />
      </div>
    </div>
  );
} 