import React, { useState, useEffect, useRef } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Box, Sphere, Stars } from '@react-three/drei';
import { Button, Text } from '@geist-ui/core';
import { Link } from 'react-router-dom';
import * as THREE from 'three';
import { EffectComposer, Bloom } from '@react-three/postprocessing';
import config from '../config.js';
import ButtonForkOnGithub from '../components/ButtonForkOnGithub.jsx';
import 'katex/dist/katex.min.css';
import EquationPanel from '../components/EquationPanel.jsx';
import InfoPanel from '../components/InfoPanel.jsx';
import ModelInfoPanel from '../components/ModelInfoPanel.jsx';
import { useResponsive } from '../hooks/useResponsive.js';

const WS_URL = `${config.WS_BASE_URL}/ws/brickbreak`;

// Helper to map 2D environment coords to 3D plane
const to3D = (pos, y = 0) => [pos[0], y, pos[1]];

const Brick = ({ pos, size }) => (
  <Box args={[size[0], 2, size[1]]} position={to3D(pos, 1)}>
    <meshStandardMaterial color="#00ffaa" emissive="#00ffaa" emissiveIntensity={0.5} />
  </Box>
);

const Paddle = ({ pos, size }) => (
  <Box args={[size[0], 2, size[1]]} position={to3D(pos, 1)}>
    <meshStandardMaterial color="#ffffff" emissive="#ffffff" emissiveIntensity={0.5} />
  </Box>
);

const Ball = ({ pos, radius }) => (
  <Sphere args={[radius, 32, 32]} position={to3D(pos, 1)}>
    <meshStandardMaterial color="#ffffff" emissive="#00ffff" emissiveIntensity={2} />
  </Sphere>
);

export default function BrickBreakExample() {
  const [state, setState] = useState({ ball: null, paddle: null, bricks: [], bounds: [40, 40] });
  const [training, setTraining] = useState(false);
  const [trained, setTrained] = useState(false);
  const [modelInfo, setModelInfo] = useState(null);
  const [logs, setLogs] = useState([]);
  const [chartState, setChartState] = useState({ labels: [], rewards: [], losses: [] });
  const wsRef = useRef(null);
  const [homeHover, setHomeHover] = useState(false);
  const { isMobile } = useResponsive();

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
      if (parsed.type === 'train_step' || parsed.type === 'run_step' || parsed.type === 'state') {
        setState(prev => ({ ...prev, ...parsed.state }));
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
        setModelInfo({ filename: parsed.model_filename, timestamp: parsed.timestamp, sessionUuid: parsed.session_uuid, fileUrl: parsed.file_url });
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
    send({ cmd: 'run' });
  };

  const resetTraining = () => {
    setTraining(false);
    setTrained(false);
    setModelInfo(null);
    setChartState({ labels: [], rewards: [], losses: [] });
    setState({ ball: null, paddle: null, bricks: [], bounds: [40, 40] });
  };

  const [width, height] = state.bounds;

  return (
    <div
      style={{
        width: '100vw',
        height: '100vh',
        overflow: 'hidden',
        outline: 'none',
        background: '#000011',
        display: 'flex',
        flexDirection: 'column',
      }}
    >
      <div style={{ flex: 1, position: 'relative' }}>
        <Canvas camera={{ position: [width / 2, 80, height / 2], fov: 50 }} style={{ background: 'transparent', width: '100vw', height: '100vh', overflow: 'hidden' }}>
          <ambientLight intensity={0.2} />
          <directionalLight position={[0, 40, 20]} intensity={0.5} />
          <Stars radius={100} depth={50} count={5000} factor={4} saturation={0} fade speed={1} />
          
          <group position={[0, 0, height]} scale={[1, 1, -1]}>
            {state.ball && <Ball {...state.ball} />}
            {state.paddle && <Paddle {...state.paddle} />}
            {state.bricks.map((brick, i) => (
              <Brick key={i} {...brick} />
            ))}

            {/* Walls */}
            <Box position={[width / 2, 1, -5]} args={[width, 2, 2]}><meshStandardMaterial color="#00ffff" emissive="#00ffff" emissiveIntensity={0.1} /></Box>
            <Box position={[width / 2, 1, height + 5]} args={[width, 2, 2]}><meshStandardMaterial color="#00ffff" emissive="#00ffff" emissiveIntensity={0.1} /></Box>
            <Box position={[-5, 1, height/2]} args={[2, 2, height]}><meshStandardMaterial color="#00ffff" emissive="#00ffff" emissiveIntensity={0.1} /></Box>
            <Box position={[width+5, 1, height/2]} args={[2, 2, height]}><meshStandardMaterial color="#00ffff" emissive="#00ffff" emissiveIntensity={0.1} /></Box>

          </group>
          <EffectComposer>
            <Bloom intensity={0.6} luminanceThreshold={0.1} luminanceSmoothing={0.9} />
          </EffectComposer>
          <OrbitControls target={[width / 2, 0, height / 2]} enablePan={false} />
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
            BrickBreak
          </Text>
          <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
            <Button auto type="secondary" disabled={training || trained} onClick={startTraining}>Train</Button>
            <Button auto type="success" disabled={!trained} onClick={startRun}>Run</Button>
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