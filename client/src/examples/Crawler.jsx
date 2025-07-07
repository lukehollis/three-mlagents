import React, { useState, useEffect, useRef } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Grid } from '@react-three/drei';
import { Button, Text, useMediaQuery } from '@geist-ui/core';
import { Link } from 'react-router-dom';
import config from '../config.js';
import ButtonForkOnGithub from '../components/ButtonForkOnGithub.jsx';
import 'katex/dist/katex.min.css';
import EquationPanel from '../components/EquationPanel.jsx';
import InfoPanel from '../components/InfoPanel.jsx';
import ModelInfoPanel from '../components/ModelInfoPanel.jsx';

// WebSocket endpoint – exposed by crawler2.py routes (assumed to be /ws/ant)
const WS_URL = `${config.WS_BASE_URL}/ws/ant`;

export default function Crawler2Example() {
  const [state, setState] = useState({
    basePos: [0, 0, 0.45],
    baseOri: [0, 0, 0, 1],
    jointAngles: [0, 0.4, 0, 0.4, 0, 0.4, 0, 0.4],
  });
  const [training, setTraining] = useState(false);
  const [trained, setTrained] = useState(false);
  const [modelInfo, setModelInfo] = useState(null);
  const [logs, setLogs] = useState([]);
  const [chartState, setChartState] = useState({ labels: [], rewards: [], losses: [] });
  const wsRef = useRef(null);
  const [homeHover, setHomeHover] = useState(false);
  const isMobile = useMediaQuery('sm') || useMediaQuery('xs');

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
    setState({
      basePos: [0, 0, 0.45],
      baseOri: [0, 0, 0, 1],
      jointAngles: [0, 0.4, 0, 0.4, 0, 0.4, 0, 0.4],
    });
  };

  // 3-D helper conversions to map MuJoCo's Z-up coordinate system to Three.js's Y-up.
  // MuJoCo quaternion from server is [w, x, y, z]. Three.js expects [x, y, z, w].
  // The mapping from MuJoCo's frame (x-forward, y-left, z-up) to
  // Three.js's frame (x-right, y-up, z-forward) requires a coordinate swap.
  // Position: (x, y, z)_mujoco -> (x, z, -y)_three
  // Quaternion: (w, x, y, z)_mujoco -> (x, z, -y, w)_three
  const mujocoToThreeQuat = (q) => [q[1], q[3], -q[2], q[0]];
  const threePos = state.basePos ? [state.basePos[0], state.basePos[2], -state.basePos[1]] : [0, 0, 0];
  const threeQuat = state.baseOri ? mujocoToThreeQuat(state.baseOri) : [0, 0, 0, 1];

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
        <Canvas camera={{ position: [0, 6, 10], fov: 50 }} style={{ background: 'transparent', width: '100vw', height: '100vh', overflow: 'hidden' }}>
          <ambientLight intensity={0.5} />
          <directionalLight position={[5, 10, 5]} intensity={1} />
          <Grid args={[24, 24]} cellSize={1} />

          {/* Ant body visualisation */}
          <group position={threePos} quaternion={threeQuat}>
            {/* Torso */}
            <mesh>
              <sphereGeometry args={[0.25, 16, 12]} />
              <meshStandardMaterial color="#00aaff" />
            </mesh>
            {/* Four legs (simple two-link rendering) */}
            {Array.from({ length: 4 }).map((_, i) => {
              const hipAngle = state.jointAngles[i * 2] || 0; // hip
              const kneeAngle = state.jointAngles[i * 2 + 1] || 0; // knee
              const isLeft = i % 2 === 0; // 0,2 are left
              const isFront = i < 2; // 0,1 front
              const side = isLeft ? 1 : -1;
              const frontBack = isFront ? 1 : -1;

              const hipPos = [0.15 * side, -0.1, 0.173 * frontBack];
              const upperLen = 0.3;
              const lowerLen = 0.3;
              return (
                <group key={i} position={hipPos}>
                  <group rotation={[0, hipAngle, 0]}>
                    <group rotation={[-Math.PI / 4 * frontBack, 0, Math.PI / 4 * side]}>
                      {/* upper */}
                      <mesh position={[0, -upperLen / 2, 0]}>
                        <cylinderGeometry args={[0.05, 0.05, upperLen, 8]} />
                        <meshStandardMaterial color="#ffaa00" />
                      </mesh>
                      <group position={[0, -upperLen, 0]} rotation={[kneeAngle, 0, 0]}>
                        <mesh position={[0, -lowerLen / 2, 0]}>
                          <cylinderGeometry args={[0.05, 0.05, lowerLen, 8]} />
                          <meshStandardMaterial color="#ffdd55" />
                        </mesh>
                      </group>
                    </group>
                  </group>
                </group>
              );
            })}
          </group>

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
            Ant (Crawler)
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