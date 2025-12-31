import React, { useState, useEffect, useRef } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Grid, Capsule, Stars } from '@react-three/drei';
import { EffectComposer, Bloom } from '@react-three/postprocessing';
import { Button, Text } from '@geist-ui/core';
import HomeButton from '../components/HomeButton.jsx';
import * as THREE from 'three';
import config from '../config.js';
import ButtonForkOnGithub from '../components/ButtonForkOnGithub.jsx';
import 'katex/dist/katex.min.css';
import EquationPanel from '../components/EquationPanel.jsx';
import InfoPanel from '../components/InfoPanel.jsx';
import ModelInfoPanel from '../components/ModelInfoPanel.jsx';
import { useResponsive } from '../hooks/useResponsive.js';

// WebSocket endpoint – exposed by worm.py routes
const WS_URL = `${config.WS_BASE_URL}/ws/worm`;

// 3-D helper conversions to map MuJoCo's Z-up coordinate system to Three.js's Y-up.
const mujocoToThreePos = (p) => (p ? [p[0], p[2], -p[1]] : [0, 0, 0]);
const mujocoToThreeQuat = (q) => {
  if (!q) return [0, 0, 0, 1];
  // For Swimmer, rotation is only around the Z-axis in MuJoCo's world,
  // which corresponds to the Y-axis in Three.js's world.
  // MuJoCo quat is [w, x, y, z]. A pure Z-rotation has x=0, y=0.
  // The resulting Three.js quat is [0, sin(angle/2), 0, cos(angle/2)],
  // where sin comes from MuJoCo's z and cos from w.
  return [0, q[3], 0, q[0]]; // [0, z, 0, w]
};

const WormSegment = ({ segment, isHead }) => {
  const threePos = mujocoToThreePos(segment.pos);
  const threeQuat = mujocoToThreeQuat(segment.quat);
  
  // MuJoCo size is [radius, half-height].
  const radius = segment.size ? segment.size[0] : 0.15;
  // This is half of the segment's total length (from joint to end-cap).
  const halfLength = segment.size ? segment.size[1] : 0.2;
  // The cylinder part of the Drei capsule is the total length minus the two hemi-spherical caps.
  const cylinderLength = 2 * halfLength - 2 * radius;

  // The head/torso's position is its center; child segments' positions are their parent joints.
  // Therefore, only child segments need to be offset from their origin.
  const positionOffset = isHead ? [0, 0, 0] : [0, halfLength, 0];

  return (
    <group position={threePos} quaternion={threeQuat}>
      {/* Rotate geometry to align capsule's Y-axis with MuJoCo's X-axis. */}
      <group rotation={[0, 0, Math.PI / 2]}>
        {/* Apply the conditional offset. */}
        <group position={positionOffset}>
          <Capsule args={[radius, cylinderLength, 16]}>
            <meshStandardMaterial color="#00aaff" emissive="#00aaff" emissiveIntensity={1.5} toneMapped={false} />
          </Capsule>
      
          {/* Head decorations are positioned relative to the capsule's center. */}
          {isHead && (
            // The capsule's tip is at y = halfLength from its center.
            <group position={[0, halfLength -0.3, 0]}>
              <mesh rotation={[Math.PI / 2, 0, 0]}>
                <torusGeometry args={[radius + 0.01, 0.03, 8, 32]} />
                <meshStandardMaterial color="#ffaa00" emissive="#ffaa00" emissiveIntensity={2} toneMapped={false} />
              </mesh>
            </group>
          )}
        </group>
      </group>
    </group>
  );
};

export default function WormExample() {
  const [state, setState] = useState({ segments: [] });
  const [training, setTraining] = useState(false);
  const [trained, setTrained] = useState(false);
  const [modelInfo, setModelInfo] = useState(null);
  const [logs, setLogs] = useState([]);
  const [chartState, setChartState] = useState({ labels: [], rewards: [], losses: [] });
  const wsRef = useRef(null);

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
      if (parsed.type === 'train_step' || parsed.type === 'run_step') {
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
      if (parsed.type === 'state') {
        setState(prev => ({ ...prev, ...parsed.state }));
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
    setState({ segments: [] });
  };

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
        <Canvas camera={{ position: [0, 4, 8], fov: 50 }} style={{ background: 'transparent', width: '100vw', height: '100vh', overflow: 'hidden' }}>
          <ambientLight intensity={0.5} />
          <directionalLight position={[5, 10, 5]} intensity={1.5} />
          <Grid args={[48, 48]} cellSize={1} fadeDistance={25} cellColor="#202020" sectionColor="#4488ff" sectionThickness={1} />

          {/* Worm body visualisation */}
          <group>
            {state.segments && state.segments.map((seg, i) => (
              <WormSegment key={seg.name} segment={seg} isHead={i === 0} />
            ))}
          </group>
          
          
          <OrbitControls target={[0, 0.5, 0]} enablePan enableRotate enableZoom />
          <Stars radius={100} depth={50} count={4000} factor={4} saturation={0} fade />
          <EffectComposer disableNormalPass>
            <Bloom luminanceThreshold={1} mipmapBlur intensity={1.5} radius={0.6} />
          </EffectComposer>
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
          <HomeButton />
          <Text h1 style={{ margin: '12px 0 12px 0', color: '#fff', fontSize: isMobile ? '1.2rem' : '2rem', fontFamily: 'monospace', textTransform: 'uppercase', letterSpacing: '0.15em' }}>
            Worm
          </Text>
          <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
            <Button auto type="secondary" style={{ borderRadius: 0, textTransform: 'uppercase', letterSpacing: '0.1em', border: '1px solid #fff' }} disabled={training || trained} onClick={startTraining}>Train</Button>
            <Button auto type="success" style={{ borderRadius: 0, textTransform: 'uppercase', letterSpacing: '0.1em', border: '1px solid #fff' }} disabled={!trained} onClick={startRun}>Run</Button>
            {trained && <Button auto type="error" style={{ borderRadius: 0, textTransform: 'uppercase', letterSpacing: '0.1em', border: '1px solid #fff' }} onClick={resetTraining}>Reset</Button>}
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