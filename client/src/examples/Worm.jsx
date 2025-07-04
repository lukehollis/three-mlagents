import React, { useState, useEffect, useRef } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Grid, Capsule } from '@react-three/drei';
import { Button, Text } from '@geist-ui/core';
import { Link } from 'react-router-dom';
import * as THREE from 'three';
import config from '../config.js';
import DebugConsole from '../components/DebugConsole.jsx';
import ChartPanel from '../components/ChartPanel.jsx';
import ButtonForkOnGithub from '../components/ButtonForkOnGithub.jsx';
import 'katex/dist/katex.min.css';
import { BlockMath } from 'react-katex';

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
  
  // MuJoCo size is [radius, half-height]. Drei Capsule length is the cylinder part.
  const radius = segment.size ? segment.size[0] : 0.15;
  const length = segment.size ? segment.size[1] : 0.2;

  return (
    <group position={threePos} quaternion={threeQuat}>
      {/*
        Rotate the geometry so the capsule's default Y-axis aligns with
        the MuJoCo body's X-axis, which is the primary axis for length.
      */}
      <group rotation={[0, 0, Math.PI / 2]}>
        {/* Main capsule for the segment */}
        <Capsule args={[radius, length, 16]}>
          <meshStandardMaterial color="#00aaff" />
        </Capsule>
      
        {/* Head decorations (in the new rotated & shifted coordinate system) */}
        {isHead && (
          <group position={[-length / 2, 0, 0]}>
            <mesh rotation={[Math.PI / 2, 0, 0]} position={[length / 2, 0, 0]}>
              <torusGeometry args={[radius + 0.01, 0.03, 8, 32]} />
              <meshStandardMaterial color="#ffaa00" emissive="#331100" />
            </mesh>
            <mesh position={[length / 2 + radius * 0.7, 0, radius * 0.7]}>
              <sphereGeometry args={[0.05, 12, 8]} />
              <meshStandardMaterial color="white" />
            </mesh>
            <mesh position={[length / 2 + radius * 0.7, 0, -radius * 0.7]}>
              <sphereGeometry args={[0.05, 12, 8]} />
              <meshStandardMaterial color="white" />
            </mesh>
          </group>
        )}
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

  return (
    <div style={{ width: '100%', height: '100%', background: 'linear-gradient(to bottom, #08081c, #03030a)' }}>
      <Canvas camera={{ position: [0, 4, 8], fov: 50 }} style={{ background: 'transparent' }}>
        <ambientLight intensity={0.5} />
        <directionalLight position={[5, 10, 5]} intensity={1.5} />
        <Grid args={[48, 48]} cellSize={1} fadeDistance={25} />

        {/* Worm body visualisation */}
        <group>
          {state.segments && state.segments.map((seg, i) => (
            <WormSegment key={seg.name} segment={seg} isHead={i === 0} />
          ))}
        </group>
        
        <OrbitControls target={[0, 0.5, 0]} enablePan enableRotate enableZoom />
      </Canvas>

      {/* UI overlay */}
      <div style={{ position: 'absolute', top: 10, left: 10, color: '#fff' }}>
        <Link to="/" style={{ fontFamily: 'monospace', color: '#fff', textDecoration: 'underline' }}>Home</Link>
        <Text h1 style={{ margin: '0 0 12px 0', color: '#fff' }}>Worm</Text>
        <div style={{ display: 'flex', gap: '8px' }}>
          <Button auto type="secondary" disabled={training || trained} onClick={startTraining}>Train</Button>
          <Button auto type="success" disabled={!trained} onClick={startRun}>Run</Button>
        </div>
      </div>

      {/* PPO loss & description */}
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
          math={
            '\\displaystyle L^{\\text{CLIP}}(\\theta)=\\mathbb{E}_t\\bigl[\\min\\bigl(r_t(\\theta)\\hat{A}_t,\\;\\text{clip}(r_t(\\theta),1\\! -\\!\\varepsilon,1\\! +\\!\\varepsilon)\\hat{A}_t\\bigr)\\bigr]'
          }
          style={{ textAlign: 'left' }}
        />
        <div style={{ fontSize: 10, fontFamily: 'monospace', marginTop: 4 }}>
          PPO clipped-objective: rₜ is the probability ratio
          π<sub>θ</sub>(aₜ|sₜ)/π<sub>θ<sup>old</sup></sub>(aₜ|sₜ), 
          ĤAₜ advantage, ε clip range.
        </div>
      </div>

      <div style={{ position: 'absolute', bottom: 160, right: 10, width: '30%', height: '180px', background: 'rgba(255,255,255,0.05)', padding: 4 }}>
        <ChartPanel labels={chartState.labels} rewards={chartState.rewards} losses={chartState.losses} />
      </div>

      <DebugConsole logs={logs} />
      <ButtonForkOnGithub position={{ top: '20px', right: '20px' }} />
    </div>
  );
} 