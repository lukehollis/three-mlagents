import React, { useState, useEffect, useRef } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Box, Cylinder, Stars, Plane } from '@react-three/drei';
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

const WS_URL = `${config.WS_BASE_URL}/ws/bicycle`;

const Bicycle = ({ state }) => {
  const { pos, theta, phi, delta, wheelbase } = state;
  const groupRef = useRef();

  useFrame(() => {
    if (groupRef.current) {
      groupRef.current.position.set(pos[0], 0, pos[1]);
      groupRef.current.rotation.set(0, -theta, 0);
    }
  });

  const wheelRadius = 0.35;
  const frameColor = "#cccccc";
  const wheelColor = "#555555";

  return (
    <group ref={groupRef}>
      <group rotation={[0,0,phi]}>
        {/* Rear Wheel */}
        <Cylinder args={[wheelRadius, wheelRadius, 0.1, 32]} rotation={[0, 0, Math.PI / 2]} position={[0, wheelRadius, 0]}>
          <meshStandardMaterial color={wheelColor} />
        </Cylinder>
        {/* Frame */}
        <Box args={[wheelbase, 0.1, 0.1]} position={[wheelbase / 2, wheelRadius + 0.2, 0]}>
          <meshStandardMaterial color={frameColor} />
        </Box>
        <Box args={[0.1, 0.4, 0.1]} position={[wheelbase, wheelRadius, 0]} rotation={[0,0,-Math.PI/8]}>
           <meshStandardMaterial color={frameColor} />
        </Box>
         <Box args={[0.1, 0.4, 0.1]} position={[0, wheelRadius, 0]} rotation={[0,0,Math.PI/8]}>
           <meshStandardMaterial color={frameColor} />
        </Box>
        
        {/* Front wheel assembly */}
        <group position={[wheelbase, wheelRadius, 0]} rotation={[0, delta, 0]}>
          <Cylinder args={[wheelRadius, wheelRadius, 0.1, 32]} rotation={[0, 0, Math.PI / 2]} position={[0, 0, 0]}>
            <meshStandardMaterial color={wheelColor} />
          </Cylinder>
          {/* Handlebars */}
          <Box args={[0.1, 0.5, 0.1]} position={[0, 0.2, 0]}>
            <meshStandardMaterial color={frameColor} />
          </Box>
          <Box args={[0.6, 0.1, 0.1]} position={[0, 0.7, 0]}>
            <meshStandardMaterial color={frameColor} />
          </Box>
        </group>
      </group>
    </group>
  );
};


export default function BicycleExample() {
  const [state, setState] = useState(null);
  const [training, setTraining] = useState(false);
  const [trained, setTrained] = useState(false);
  const [modelInfo, setModelInfo] = useState(null);
  const [logs, setLogs] = useState([]);
  const [chartState, setChartState] = useState({ labels: [], rewards: [], losses: [] });
  const wsRef = useRef(null);
  const [homeHover, setHomeHover] = useState(false);
  const { isMobile } = useResponsive();
  const cameraRef = useRef();

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
    setState(null);
  };
  
  const [width, height] = state?.bounds ?? [100, 100];

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
        <Canvas camera={{ position: [0, 10, 15], fov: 50 }} ref={cameraRef} style={{ background: 'transparent', width: '100vw', height: '100vh', overflow: 'hidden' }}>
          <ambientLight intensity={0.5} />
          <directionalLight position={[0, 40, 20]} intensity={0.8} />
          <Stars radius={200} depth={50} count={5000} factor={6} saturation={0} fade speed={1} />
          
          {state && <Bicycle state={state} />}
          <Plane args={[width, height]} rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.01, 0]}>
             <meshStandardMaterial color="#222222" />
          </Plane>

          <EffectComposer>
            <Bloom intensity={0.4} luminanceThreshold={0.1} luminanceSmoothing={0.9} />
          </EffectComposer>
          <OrbitControls target={state ? [state.pos[0], 0.5, state.pos[1]] : [0,0,0]} />
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
            Bicycle
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