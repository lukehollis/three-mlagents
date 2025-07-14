import React, { useState, useEffect, useRef, useMemo } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Grid, Text as DreiText } from '@react-three/drei';
import { Button, Text, Card, Code } from '@geist-ui/core';
import { Link } from 'react-router-dom';
import * as THREE from 'three';
import config from '../config.js';
import { useResponsive } from '../hooks/useResponsive.js';
import { EffectComposer, Bloom } from '@react-three/postprocessing';
import InfoPanel from '../components/InfoPanel.jsx';
import ModelInfoPanel from '../components/ModelInfoPanel.jsx';

const WS_URL = `${config.WS_BASE_URL}/ws/intersection`;

const Vehicle = ({ agent, gridSize }) => {
  const { pos, id, energy, velocity } = agent;
  const groupRef = useRef();

  const offsetX = gridSize ? gridSize[0] / 2 : 0;
  const offsetZ = gridSize ? gridSize[2] / 2 : 0;

  const energyColor = useMemo(() => {
    const green = new THREE.Color("#00ff88");
    const white = new THREE.Color("#ffffff");
    const alpha = Math.max(0, Math.min(1, energy / 100));
    return green.clone().lerp(white, alpha);
  }, [energy]);

  useFrame(() => {
    if (groupRef.current && velocity && new THREE.Vector3(...velocity).lengthSq() > 0.001) {
      const targetPosition = new THREE.Vector3().addVectors(
        groupRef.current.position,
        new THREE.Vector3(velocity[0], velocity[1], velocity[2])
      );
      groupRef.current.lookAt(targetPosition);
    }
  });

  return (
    <group ref={groupRef} position={[pos[0] - offsetX, pos[1], pos[2] - offsetZ]}>
      <mesh>
        <boxGeometry args={[1.5, 0.8, 0.8]} />
        <meshPhongMaterial color={energyColor} emissive={energyColor} emissiveIntensity={energy / 100} wireframe={true} />
      </mesh>
    </group>
  );
};

export default function IntersectionExample() {
  const [state, setState] = useState(null);
  const [running, setRunning] = useState(false);
  const [training, setTraining] = useState(false);
  const [trained, setTrained] = useState(false);
  const [modelInfo, setModelInfo] = useState(null);
  const [logs, setLogs] = useState([]);
  const [chartState, setChartState] = useState({ labels: [], rewards: [], losses: [] });
  const wsRef = useRef(null);
  const { isMobile } = useResponsive();
  
  const gridSize = useMemo(() => state?.grid_size, [state]);
  
  const addLog = (txt) => {
    setLogs((l) => {
      const upd = [...l, txt];
      return upd.length > 200 ? upd.slice(upd.length - 200) : upd;
    });
  };

  useEffect(() => {
    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;
    ws.onopen = () => addLog('Intersection WS opened');
    ws.onmessage = (ev) => {
      addLog(`Received data: ${ev.data.substring(0, 100)}...`);
      const parsed = JSON.parse(ev.data);
      
      if (parsed.type === 'train_step' || parsed.type === 'run_step' || parsed.type === 'state' || parsed.type === 'init') {
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
        setModelInfo(parsed.model_info);
        addLog('Training complete! Vehicles are now using the trained policy.');
      }
      if (parsed.type === 'info') {
          addLog(`INFO: ${parsed.message}`);
      }
    };
    ws.onclose = () => addLog('Intersection WS closed');
    return () => ws.close();
  }, []);

  const send = (obj) => {
    addLog(`Sending: ${JSON.stringify(obj)}`);
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(obj));
    } else {
      addLog('WebSocket not open');
    }
  };

  const startRun = () => {
    if (running) return;
    setRunning(true);
    send({ cmd: 'run' });
  };

  const startTraining = () => {
    if (training || running) return;
    setTraining(true);
    addLog('Starting training run...');
    send({ cmd: 'train' });
  };

  const reset = () => {
    window.location.reload();
  }

  const resetTraining = () => {
    setTrained(false);
    setTraining(false);
    setModelInfo(null);
    addLog("Training state reset. Ready to train a new model.");
  }


  return (
    <div style={{ width: '100vw', height: '100vh', overflow: 'hidden', background: '#000011' }}>
      <Canvas camera={{ position: [-50, 50, 50], fov: 60 }}>
        <fog attach="fog" args={['#000011', 50, 250]} />
        <ambientLight intensity={0.4} />
        <directionalLight 
          color="#00ffff"
          position={[0, 100, 0]} 
          intensity={1.0} 
        />
        <pointLight color="#ff00ff" position={[-40, 20, -40]} intensity={2.0} distance={150} />

        <Grid infiniteGrid cellSize={1} sectionSize={10} sectionColor={"#0044cc"} fadeDistance={250} fadeStrength={1.5} />
        
        {state && state.agents && state.agents.map(agent => <Vehicle key={agent.id} agent={agent} gridSize={gridSize} />)}
        
        <EffectComposer>
          <Bloom intensity={0.9} luminanceThreshold={0.2} luminanceSmoothing={0.8} />
        </EffectComposer>
        <OrbitControls maxDistance={100} minDistance={10} target={[0, 32, 0]} />
      </Canvas>

      <div style={{ position: 'absolute', top: 10, left: 10, zIndex: 1, color: '#fff' }}>
        <Link to="/" style={{ fontFamily: 'monospace', color: '#fff', textDecoration: 'underline' }}>
          Home
        </Link>
          <Text h1 style={{ margin: '12px 0 12px 0', color: '#fff', fontSize: isMobile ? '1.2rem' : '2rem' }}>
            Intersection
          </Text>
        <div style={{ display: 'flex', gap: '8px' }}>
          <Button auto type="secondary" disabled={training || trained} onClick={startTraining}>Train</Button>
          <Button auto type="success" disabled={!trained || running} onClick={startRun}>Run</Button>
          <Button auto type="error" onClick={reset}>Reset </Button>
        </div>
      </div>
      
      <InfoPanel logs={logs} chartState={chartState} />
      <ModelInfoPanel modelInfo={modelInfo} />

    </div>
  );
} 