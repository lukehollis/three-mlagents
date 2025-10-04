import React, { useState, useEffect, useRef } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Text as DreiText, Stars, Line } from '@react-three/drei';
import { Button, Text, Card } from '@geist-ui/core';
import { Link } from 'react-router-dom';
import * as THREE from 'three';
import config from '../config.js';
import { useResponsive } from '../hooks/useResponsive.js';
import { EffectComposer, Bloom } from '@react-three/postprocessing';

const WS_URL = `${config.WS_BASE_URL}/ws/pirate-ship`;

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
        Kraken Health: {health}
      </DreiText>
    </mesh>
  );
};

// Tentacle component
const Tentacle = ({ position }) => {
  return (
    <mesh position={[position[0] - GRID_SIZE/2, 1, position[1] - GRID_SIZE/2]}>
      <cylinderGeometry args={[0.5, 0.5, 3]} />
      <meshStandardMaterial color="darkgreen" />
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

export default function PirateShip() {
  const [state, setState] = useState(null);
  const [running, setRunning] = useState(false);
  const [training, setTraining] = useState(false);
  const [trained, setTrained] = useState(false);
  const wsRef = useRef(null);
  const { isMobile } = useResponsive();

  useEffect(() => {
    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;
    ws.onmessage = (ev) => {
      const parsed = JSON.parse(ev.data);
      if (parsed.type === 'train_step' || parsed.type === 'run_step') {
        setState(parsed.state);
      } else if (parsed.type === 'trained') {
        setTraining(false);
        setTrained(true);
      }
    };
    return () => ws.close();
  }, []);

  const send = (obj) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(obj));
    }
  };

  const startTraining = () => {
    setTraining(true);
    send({ cmd: 'train' });
  };

  const startRun = () => {
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
          <Tentacle key={i} position={tentacle} />
        ))}

        <EffectComposer>
          <Bloom intensity={2} luminanceThreshold={0.2} luminanceSmoothing={0.9} toneMapped={false} />
        </EffectComposer>
        <OrbitControls maxDistance={200} minDistance={20} maxPolarAngle={Math.PI / 2} />
      </Canvas>

      <div style={{ position: 'absolute', top: 10, left: 10, zIndex: 1, color: '#fff' }}>
        <Link to="/">Home</Link>
        <Text h1>Pirate Ship vs Kraken</Text>
        <Button type="secondary" disabled={training || trained} onClick={startTraining}>Train</Button>
        <Button type="success" disabled={!trained || running} onClick={startRun}>Run</Button>
      </div>
      
      {state && (
        <div style={{ position: 'absolute', top: 10, right: 10, zIndex: 1 }}>
          <StatusPanel ships={state.ships} kraken={state.kraken} />
        </div>
      )}
    </div>
  );
}
