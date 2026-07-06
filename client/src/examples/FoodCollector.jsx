import React, { useState, useEffect, useRef } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Box, Sphere, Stars, Cylinder } from '@react-three/drei';
import { Button, Text } from '@geist-ui/core';
import HomeButton from '../components/HomeButton.jsx';
import * as THREE from 'three';
import { EffectComposer, Bloom } from '@react-three/postprocessing';
import config from '../config.js';
import ButtonForkOnGithub from '../components/ButtonForkOnGithub.jsx';
import 'katex/dist/katex.min.css';
import InfoPanel from '../components/InfoPanel.jsx';
import RoadmapStatusPanel from '../components/RoadmapStatusPanel.jsx';
import { useResponsive } from '../hooks/useResponsive.js';

const WS_URL = `${config.WS_BASE_URL}/ws/foodcollector`;

const to3D = (pos, y = 0) => [pos[0], y, pos[1]];

const Agent = ({ pos, rot, frozen }) => (
  <group position={to3D(pos, 1)} rotation={[0, -rot, 0]}>
    <Box args={[1.5, 1.5, 1.5]}>
      <meshStandardMaterial color={frozen ? '#5555ff' : '#ffffff'} emissive={frozen ? '#5555ff' : '#ffffff'} emissiveIntensity={0.5} />
    </Box>
    <Box args={[0.5, 0.5, 2]} position={[0, 0, -1]}>
       <meshStandardMaterial color={'#ff0000'} />
    </Box>
  </group>
);

const Food = ({ pos, color }) => (
  <Sphere args={[0.7, 16, 16]} position={to3D(pos, 0.7)}>
    <meshStandardMaterial color={color} emissive={color} emissiveIntensity={1.5} />
  </Sphere>
);


export default function FoodCollectorExample() {
  const [state, setState] = useState({ agents: [], good_food: [], bad_food: [], bounds: [40, 40] });
  const [logs, setLogs] = useState([]);
  const [chartState, setChartState] = useState({ labels: [], rewards: [], losses: [] });

  const { isMobile } = useResponsive();

  const addLog = (txt) => {
    setLogs((l) => {
      const upd = [...l, txt];
      if (upd.length > 200) {
        return upd.slice(upd.length - 200);
      }
      return upd;
    });
  };

  useEffect(() => {
    const ws = new WebSocket(WS_URL);
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
    };
    ws.onclose = () => addLog('WS closed');
    return () => ws.close();
  }, []);

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
        <Canvas camera={{ position: [0, 80, 50], fov: 50 }} style={{ background: 'transparent' }}>
          <ambientLight intensity={0.2} />
          <directionalLight position={[0, 40, 20]} intensity={0.5} />
          <Stars radius={100} depth={50} count={5000} factor={4} saturation={0} fade speed={1} />
          
          <group position={[-width/2, 0, -height/2]}>
            {state.agents.map((agent, i) => (
              <Agent key={i} {...agent} />
            ))}
            {state.good_food.map((pos, i) => (
              <Food key={`g${i}`} pos={pos} color="#00ffaa" />
            ))}
            {state.bad_food.map((pos, i) => (
                <Food key={`b${i}`} pos={pos} color="#ff0055" />
            ))}

            {/* Ground */}
            <Box position={[width / 2, -1, height / 2]} args={[width, 2, height]}>
                <meshStandardMaterial color="#222233" />
            </Box>
          </group>

          <EffectComposer>
            <Bloom intensity={0.6} luminanceThreshold={0.1} luminanceSmoothing={0.9} />
          </EffectComposer>
          <OrbitControls target={[0, 0, 0]} />
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
            Food Collector
          </Text>
          <RoadmapStatusPanel taskId="foodcollector" />
        </div>
        <InfoPanel logs={logs} chartState={chartState} />
        <ButtonForkOnGithub position={{ top: '10px', right: '10px' }} />
      </div>
    </div>
  );
} 
