import React, { useState, useCallback, useEffect, useRef } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Grid, Stars } from '@react-three/drei';
import {
  Chart as ChartJS,
  LineElement,
  PointElement,
  LinearScale,
  CategoryScale,
} from 'chart.js';
import * as THREE from 'three';
import { EffectComposer, Bloom } from '@react-three/postprocessing';
import config from '../config.js';
import 'katex/dist/katex.min.css';
import { Text, Button } from '@geist-ui/core';
import ButtonForkOnGithub from '../components/ButtonForkOnGithub.jsx';
import EquationPanel from '../components/EquationPanel.jsx';
import InfoPanel from '../components/InfoPanel.jsx';
import ModelInfoPanel from '../components/ModelInfoPanel.jsx';
import HomeButton from '../components/HomeButton.jsx';
import { useResponsive } from '../hooks/useResponsive.js';

ChartJS.register(LineElement, PointElement, LinearScale, CategoryScale);

const SMALL_GOAL_POS = 7;
const LARGE_GOAL_POS = 17;
const MIN_POS = 0;
const MAX_POS = 20;
const WS_URL = `${config.WS_BASE_URL}/ws/basic`;

function Agent({ position }) {
  return (
    <mesh position={[position - 10, 0, 0]}>
      <boxGeometry args={[0.9, 0.9, 0.9]} />
      <meshStandardMaterial color="orange" emissive="orange" emissiveIntensity={2} toneMapped={false} />
    </mesh>
  );
}

function Goal({ position, color }) {
  return (
    <mesh position={[position - 10, 0, 0]}>
      <boxGeometry args={[0.9, 0.9, 0.9]} />
      <meshStandardMaterial color={color} emissive={color} emissiveIntensity={2} toneMapped={false} />
    </mesh>
  );
}

export default function BasicExample() {
  const [pos, setPos] = useState(10);
  const posRef = useRef(10);
  const [rewardAccum, setRewardAccum] = useState(0);

  const [logs, setLogs] = useState([]);
  const wsRef = useRef(null);

  const [training, setTraining] = useState(false);
  const [trained, setTrained] = useState(false);
  const [modelInfo, setModelInfo] = useState(null);
  const [visualTraining, setVisualTraining] = useState(false);
  const [autoRun, setAutoRun] = useState(false);
  const intervalRef = useRef(null);

  const [chartState, setChartState] = useState({ labels: [], rewards: [], losses: [] });

  // Add hover state for Home link

  const { isMobile } = useResponsive();

  const step = useCallback((direction) => {
    setPos((prev) => {
      const next = Math.min(MAX_POS, Math.max(MIN_POS, prev + direction));
      // Base per-step penalty encourages shorter paths
      let reward = -0.01;
      let done = false;

      if (next === SMALL_GOAL_POS) {
        reward += 0.1;
        done = true;
      }
      if (next === LARGE_GOAL_POS) {
        reward += 1;
        done = true;
      }

      // Treat hitting the extrema of the grid as a terminal (failure) state so
      // the agent cannot get stuck endlessly accumulating negative reward.
      // We also apply an extra penalty to make this outcome clearly undesirable.
      if (next === MIN_POS || next === MAX_POS) {
        reward -= 0.5; // extra penalty for falling off the playable area
        done = true;
      }

      setRewardAccum((r) => r + reward);

      if (done) {
        setTimeout(() => {
          setPos(10);
          setRewardAccum(0);
        }, 500);
        return prev; // keep current until reset
      }

      return next;
    });
  }, []);

  const handleKey = useCallback(
    (e) => {
      if (autoRun) return;
      if (e.key === 'ArrowLeft') step(-1);
      if (e.key === 'ArrowRight') step(1);
    },
    [step, autoRun]
  );

  // WebSocket connection
  useEffect(() => {
    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;
    ws.onopen = () => {
      addLog('WS opened');
    };
    ws.onmessage = (ev) => {
      addLog(ev.data);
      let parsed;
      try {
        parsed = JSON.parse(ev.data);
      } catch {
        return;
      }
      if (parsed.type === 'train_step') {
        setVisualTraining(true);
        setPos(parsed.pos);
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
        setVisualTraining(false);
        setModelInfo({
          filename: parsed.model_filename,
          timestamp: parsed.timestamp,
          sessionUuid: parsed.session_uuid,
          fileUrl: parsed.file_url
        });
      }
      if (parsed.type === 'action') {
        // Action mapping: 0=left, 1=no move, 2=right
        const delta = [-1, 0, 1][parsed.action];
        step(delta);
      }
    };
    ws.onclose = () => addLog('WS closed');

    return () => ws.close();
  }, []);

  const addLog = (txt) => setLogs((l) => {
    const updated = [...l, txt];
    return updated.length > 200 ? updated.slice(updated.length - 200) : updated;
  });

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
    intervalRef.current = setInterval(() => {
      send({ cmd: 'inference', obs: posRef.current });
    }, 200);
  };

  const resetTraining = () => {
    setTraining(false);
    setTrained(false);
    setModelInfo(null);
    setVisualTraining(false);
    setAutoRun(false);
    setChartState({ labels: [], rewards: [], losses: [] });
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
  };

  useEffect(() => {
    if (!autoRun) return;
    return () => clearInterval(intervalRef.current);
  }, [autoRun]);

  // keep posRef synced
  useEffect(() => {
    posRef.current = pos;
  }, [pos]);

  return (
    <div
      tabIndex={0}
      onKeyDown={handleKey}
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
        <Canvas
          camera={{ position: [0, 15, 25], fov: 50 }}
          style={{ 
            background: 'transparent',
            width: '100vw',
            height: '100vh',
            overflow: 'hidden',
          }}
        >
          <ambientLight intensity={0.3} />
          <directionalLight
            position={[0, 20, 10]}
            intensity={1.2}
            castShadow
            shadow-mapSize-width={2048}
            shadow-mapSize-height={2048}
          />

          {/* Professional background */}
          <Stars radius={100} depth={50} count={5000} factor={4} saturation={0} fade />

          {/* Grid underneath the cubes */}
          <Grid
            position={[0, -1, 0]}
            args={[30, 30]}
            cellSize={1}
            cellThickness={0.5}
            cellColor="#202020"
            sectionSize={5}
            sectionThickness={1}
            sectionColor="#4488ff"
            fadeDistance={25}
            fadeStrength={1}
          />

          <Goal position={SMALL_GOAL_POS} color="green" />
          <Goal position={LARGE_GOAL_POS} color="blue" />
          <Agent position={pos} />
          <OrbitControls enableRotate={true} enableZoom={true} enablePan={true} target={[0, 0, 0]} />
          <Stars radius={100} depth={50} count={4000} factor={4} saturation={0} fade />
          <EffectComposer disableNormalPass>
            <Bloom luminanceThreshold={1} mipmapBlur intensity={1.5} radius={0.6} />
          </EffectComposer>
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
          {/* Home link */}
          <HomeButton />
          <Text h1 style={{ margin: '12px 0 12px 0', color: '#fff', fontSize: isMobile ? '1.2rem' : '2rem', fontFamily: 'monospace', textTransform: 'uppercase', letterSpacing: '0.15em' }}>
            Basic Example - 1-D Move-To-Goal
          </Text>
          <Text h3 style={{ margin: '0 0 12px 0', color: '#fff', fontSize: isMobile ? '12px' : 'inherit', fontFamily: 'monospace' }}>
            Reward: {rewardAccum.toFixed(2)}
          </Text>

          <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
            <Button auto type="secondary" style={{ borderRadius: 0, textTransform: 'uppercase', letterSpacing: '0.1em', border: '1px solid #fff' }} disabled={training || trained} onClick={startTraining}>
              Train
            </Button>

            <Button auto type="success" style={{ borderRadius: 0, textTransform: 'uppercase', letterSpacing: '0.1em', border: '1px solid #fff' }} disabled={!trained || autoRun} onClick={startRun}>
              Run
            </Button>

            {trained && (
              <Button auto type="error" style={{ borderRadius: 0, textTransform: 'uppercase', letterSpacing: '0.1em', border: '1px solid #fff' }} onClick={resetTraining}>
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