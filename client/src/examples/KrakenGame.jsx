import React, { useState, useEffect, useRef } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Text as DreiText, Stars, Line } from '@react-three/drei';
import { Button, Text, Card } from '@geist-ui/core';
import { Link } from 'react-router-dom';
import InfoPanel from '../components/InfoPanel.jsx';
import * as THREE from 'three';
import config from '../config.js';
import { useResponsive } from '../hooks/useResponsive.js';
import { EffectComposer, Bloom } from '@react-three/postprocessing';

const WS_URL = `${config.WS_BASE_URL}/ws/kraken`;
const GRID_SIZE = 200;

// Water surface component (adapted from example)
const WaterSurface = ({ waterSize }) => {
  const meshRef = useRef();

  useFrame(({ clock }) => {
    if (meshRef.current) {
      const time = clock.getElapsedTime();
      const positions = meshRef.current.geometry.attributes.position;
      for (let i = 0; i < positions.count; i++) {
        const x = positions.getX(i);
        const y = positions.getY(i);
        const wave1 = Math.sin(x * 0.5 + time * 1.5) * 0.2;
        const wave2 = Math.sin(y * 0.8 + time * 1.0) * 0.2;
        positions.setZ(i, wave1 + wave2);
      }
      positions.needsUpdate = true;
    }
  });

  if (!waterSize) return null;
  return (
    <mesh ref={meshRef} position={[0, 0, 0]} rotation={[-Math.PI / 2, 0, 0]}>
      <planeGeometry args={[waterSize[0], waterSize[1], 100, 100]} />
      <meshPhongMaterial 
        color={new THREE.Color(0.1, 0.3, 0.8)} 
        transparent 
        opacity={0.7}
        shininess={100}
        side={THREE.DoubleSide}
        wireframe={true}
      />
    </mesh>
  );
};

// Simple Ship component
const Ship = ({ position, health }) => {
  return (
    <mesh position={[position[0] - GRID_SIZE/2, 1, position[1] - GRID_SIZE/2]}>
      <boxGeometry args={[2, 1, 4]} />
      <meshStandardMaterial color={health > 0 ? 'brown' : 'gray'} />
      <DreiText position={[0, 2, 0]} fontSize={0.5} color="white">
        Health: {health}
      </DreiText>
    </mesh>
  );
};

// Kraken component
const Kraken = ({ position, health }) => {
  return (
    <mesh position={[position[0] - GRID_SIZE/2, 1, position[1] - GRID_SIZE/2]}>
      <sphereGeometry args={[5, 32, 32]} />
      <meshStandardMaterial color="purple" />
      <DreiText position={[0, 6, 0]} fontSize={1} color="white">
        Kraken: {health}
      </DreiText>
    </mesh>
  );
};

const generateTentacleGeometry = (pathPoints) => {
  const segments = 20;
  const radialSegments = 8;
  const baseRadius = 1;
  const tipRadius = 0.1;

  const positions = [];
  const normals = [];
  const uvs = [];
  const indices = [];

  const path = new THREE.CatmullRomCurve3(pathPoints.map(p => new THREE.Vector3(...p)));

  const frames = path.computeFrenetFrames(segments, false);

  for (let i = 0; i <= segments; i++) {
    const t = i / segments;
    const radius = baseRadius * (1 - t) + tipRadius * t;
    const normal = frames.normals[i];
    const binormal = frames.binormals[i];
    const position = path.getPointAt(t);

    for (let j = 0; j < radialSegments; j++) {
      const theta = j / radialSegments * Math.PI * 2;
      const sin = Math.sin(theta);
      const cos = -Math.cos(theta);

      const x = normal.x * cos + binormal.x * sin;
      const y = normal.y * cos + binormal.y * sin;
      const z = normal.z * cos + binormal.z * sin;

      positions.push(position.x + x * radius, position.y + y * radius, position.z + z * radius);
      normals.push(x, y, z);
      uvs.push(t, j / radialSegments);
    }
  }

  for (let i = 0; i < segments; i++) {
    for (let j = 0; j < radialSegments; j++) {
      const a = i * radialSegments + j;
      const b = (i + 1) * radialSegments + j;
      const c = (i + 1) * radialSegments + (j + 1) % radialSegments;
      const d = i * radialSegments + (j + 1) % radialSegments;

      indices.push(a, b, d);
      indices.push(b, c, d);
    }
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geometry.setAttribute('normal', new THREE.Float32BufferAttribute(normals, 3));
  geometry.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
  geometry.setIndex(indices);

  return geometry;
};

// Tentacle component - Tentacles that come DOWN from the kraken and then UP out of the water!
const Tentacle = ({ start, end }) => {
  if (!end || !Array.isArray(end) || end.length < 2) {
    return null;
  }

  const base = [start[0] - GRID_SIZE/2, 1, start[1] - GRID_SIZE/2]; // Start at Kraken body
  const tip = [end[0] - GRID_SIZE/2, 5, end[1] - GRID_SIZE/2]; // Emerge from water at target
  
  const mid1Ref = useRef([base[0], -5, base[2]]); // Control point deep under kraken
  const mid2Ref = useRef([tip[0], -5, tip[2]]);   // Control point deep under target
  
  const meshRef = useRef();

  useFrame(({ clock }) => {
    const time = clock.getElapsedTime();
    const wave = Math.sin(time * 2 + (start[0] + start[1])) * 2;
    const sway = Math.cos(time * 1.5 + (start[0] + start[1])) * 3;

    // Animate control points for writhing motion
    mid1Ref.current[0] = base[0] + sway * 0.5;
    mid1Ref.current[2] = base[2] + sway * 0.3;
    mid1Ref.current[1] = -5 + wave;

    mid2Ref.current[0] = tip[0] - sway;
    mid2Ref.current[2] = tip[2] - sway * 0.7;
    mid2Ref.current[1] = -5 + wave * 0.5;

    const pathPoints = [base, mid1Ref.current, mid2Ref.current, tip];
    const geom = generateTentacleGeometry(pathPoints);
    if (meshRef.current.geometry) meshRef.current.geometry.dispose();
    meshRef.current.geometry = geom;
  });

  return (
    <mesh ref={meshRef}>
      <meshStandardMaterial color="purple" />
    </mesh>
  );
};

const StatusPanel = ({ ships, kraken }) => {
  const aliveShips = ships.filter(s => s.health > 0).length;
  return (
    <Card>
      <Text h4>Pirate Ships: {aliveShips}/{ships.length}</Text>
      <Text h4>Kraken Health: {kraken.health}</Text>
    </Card>
  );
};

export default function KrakenGame() {
  const initialState = {
    ships: Array(4).fill().map(() => ({ pos: [Math.random() * 200, Math.random() * 200], health: 100 })),
    kraken: { pos: [100, 100], health: 500 },
    tentacles: Array(6).fill().map(() => [Math.random() * 200, Math.random() * 200]),
    grid_size: 200
  };
  const [state, setState] = useState(initialState);
  const [running, setRunning] = useState(false);
  const [training, setTraining] = useState(false);
  const [trained, setTrained] = useState(false);
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

  const resetTraining = () => {
    setTraining(false);
    setTrained(false);
    setChartState({ labels: [], rewards: [], losses: [] });
    setState(initialState);
    addLog('Training has been reset.');
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
        setState(parsed.state);
      } else if (parsed.type === 'progress') {
        setChartState((prev) => ({
          labels: [...prev.labels, parsed.episode],
          rewards: [...prev.rewards, parsed.reward],
          losses: [...prev.losses, parsed.loss ?? null],
        }));
      } else if (parsed.type === 'trained') {
        setTraining(false);
        setTrained(true);
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
    addLog('Starting training...');
    send({ cmd: 'train' });
  };

  const startRun = () => {
    if (!trained) return;
    setRunning(true);
    send({ cmd: 'run' });
  };

  return (
    <div style={{ width: '100vw', height: '100vh', background: '#000011' }}>
      <Canvas camera={{ position: [0, 50, 50], fov: 60 }}>
        <Stars radius={400} depth={50} count={5000} factor={8} saturation={0} fade speed={1} />
        <ambientLight intensity={0.6} />
        <directionalLight position={[100, 100, 50]} intensity={1.2} color="#ffffff" />
        
        {state && <WaterSurface waterSize={[state.grid_size, state.grid_size]} />}
        
        {state && state.ships.map((ship, i) => (
          <Ship key={i} position={ship.pos} health={ship.health} />
        ))}
        
        {state && <Kraken position={state.kraken.pos} health={state.kraken.health} />}
        
        {state && state.tentacles.map((tentacle, i) => (
          <Tentacle key={i} start={state.kraken.pos} end={tentacle} />
        ))}

        <EffectComposer>
          <Bloom intensity={2} luminanceThreshold={0.2} luminanceSmoothing={0.9} toneMapped={false} />
        </EffectComposer>
        <OrbitControls maxDistance={200} minDistance={20} maxPolarAngle={Math.PI / 2} />
      </Canvas>

      <div style={{ position: 'absolute', top: 10, left: 10, zIndex: 1, color: '#fff' }}>
        <Link to="/">Home</Link>
        <Text h1>The Kraken</Text>
        <div style={{ display: "flex", gap: "4px" }}>
          <Button type="secondary" disabled={training || trained} onClick={startTraining}>Train</Button>
          <Button type="success" disabled={!trained || running} onClick={startRun}>Run</Button>
          {trained && <Button type="error" onClick={resetTraining}>Reset</Button>}
        </div>
      </div>
      <InfoPanel logs={logs} chartState={chartState} />
      
      {state && (
        <div style={{ position: 'absolute', top: 10, right: 10, zIndex: 1 }}>
          <StatusPanel ships={state.ships} kraken={state.kraken} />
        </div>
      )}
    </div>
  );
}
