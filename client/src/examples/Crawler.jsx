import React, { useState, useEffect, useRef } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Grid } from '@react-three/drei';
import { Button, Text } from '@geist-ui/core';
import { Link } from 'react-router-dom';
import config from '../config.js';
import DebugConsole from '../components/DebugConsole.jsx';
import ChartPanel from '../components/ChartPanel.jsx';
import ButtonForkOnGithub from '../components/ButtonForkOnGithub.jsx';
import 'katex/dist/katex.min.css';
import { BlockMath } from 'react-katex';

// WebSocket endpoint – exposed by crawler2.py routes (assumed to be /ws/ant)
const WS_URL = `${config.WS_BASE_URL}/ws/ant`;

export default function Crawler2Example() {
  const [state, setState] = useState({
    basePos: [0, 0, 0.75],
    baseOri: [1, 0, 0, 0],
    jointAngles: Array(8).fill(0),
  });
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

  // 3-D helper conversions (same bullet→three as before)
  const bulletToThreeQuat = (q) => [q[0], q[2], -q[1], q[3]];
  const threePos = state.basePos ? [state.basePos[0], state.basePos[2], -state.basePos[1]] : [0, 0, 0];
  const threeQuat = state.baseOri ? bulletToThreeQuat(state.baseOri) : [0, 0, 0, 1];

  return (
    <div style={{ width: '100%', height: '100%', background: 'linear-gradient(to bottom, #08081c, #03030a)' }}>
      <Canvas camera={{ position: [0, 6, 10], fov: 50 }} style={{ background: 'transparent' }}>
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
                <group rotation={[hipAngle, 0, 0]}>
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
            );
          })}
        </group>

        <OrbitControls target={[0, 0, 0]} enablePan enableRotate enableZoom />
      </Canvas>

      {/* UI overlay */}
      <div style={{ position: 'absolute', top: 10, left: 10, color: '#fff' }}>
        <Link to="/" style={{ fontFamily: 'monospace', color: '#fff', textDecoration: 'underline' }}>Home</Link>
        <Text h1 style={{ margin: '0 0 12px 0', color: '#fff' }}>Ant (Crawler)</Text>
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