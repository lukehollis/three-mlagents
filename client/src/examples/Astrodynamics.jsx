import React, { useState, useEffect, useRef, useMemo } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls, Box, Cylinder, Stars, Plane, Grid, Line, Cone, Circle, Sphere, Torus } from '@react-three/drei';
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

const WS_URL = `${config.WS_BASE_URL}/ws/astrodynamics`;

// Spacecraft component
const Spacecraft = ({ state }) => {
  const { spacecraft_pos } = state;
  const groupRef = useRef();

  useFrame(() => {
    if (groupRef.current && spacecraft_pos) {
      // Scale positions for better visualization (meters to scene units)
      const scale = 0.01;
      groupRef.current.position.set(
        spacecraft_pos[0] * scale,
        spacecraft_pos[2] * scale,
        -spacecraft_pos[1] * scale
      );
    }
  });

  return (
    <group ref={groupRef}>
      {/* Main body */}
      <Box args={[2, 0.8, 0.8]}>
        <meshStandardMaterial color="#00ffff" emissive="#00ffff" emissiveIntensity={0.5} toneMapped={false} />
      </Box>
      {/* Solar panels */}
      <Box args={[0.2, 3, 0.1]} position={[0, 1.5, 0]}>
        <meshStandardMaterial color="#0088ff" emissive="#0088ff" emissiveIntensity={0.3} toneMapped={false} />
      </Box>
      <Box args={[0.2, 3, 0.1]} position={[0, -1.5, 0]}>
        <meshStandardMaterial color="#0088ff" emissive="#0088ff" emissiveIntensity={0.3} toneMapped={false} />
      </Box>
      {/* Thruster */}
      <Cone args={[0.3, 0.6, 8]} position={[-1.5, 0, 0]} rotation={[0, 0, Math.PI / 2]}>
        <meshStandardMaterial color="#ff4444" emissive="#ff4444" emissiveIntensity={0.4} toneMapped={false} />
      </Cone>
    </group>
  );
};

// Space station (target)
const SpaceStation = ({ position = [0, 0, 0] }) => {
  const groupRef = useRef();

  useFrame(() => {
    if (groupRef.current) {
      groupRef.current.rotation.y += 0.005;
    }
  });

  return (
    <group ref={groupRef} position={position}>
      {/* Central hub */}
      <Sphere args={[1.5]}>
        <meshStandardMaterial color="#ffffff" emissive="#ffffff" emissiveIntensity={0.3} toneMapped={false} />
      </Sphere>
      {/* Docking ring */}
      <Torus args={[2.5, 0.3, 8, 16]}>
        <meshStandardMaterial color="#ffff00" emissive="#ffff00" emissiveIntensity={0.5} toneMapped={false} />
      </Torus>
      {/* Communication array */}
      <Box args={[0.1, 4, 0.1]} position={[0, 3, 0]}>
        <meshStandardMaterial color="#ff8800" emissive="#ff8800" emissiveIntensity={0.4} toneMapped={false} />
      </Box>
      {/* Solar arrays */}
      <Box args={[8, 0.1, 2]} position={[0, 0, 3]}>
        <meshStandardMaterial color="#0088ff" emissive="#0088ff" emissiveIntensity={0.3} toneMapped={false} />
      </Box>
      <Box args={[8, 0.1, 2]} position={[0, 0, -3]}>
        <meshStandardMaterial color="#0088ff" emissive="#0088ff" emissiveIntensity={0.3} toneMapped={false} />
      </Box>
    </group>
  );
};

// Orbital trail visualization
const SpacecraftTrail = ({ trail }) => {
  if (!trail || trail.length < 2) return null;

  const scale = 0.01;
  const points = trail.map(pos => new THREE.Vector3(
    pos[0] * scale,
    pos[2] * scale,
    -pos[1] * scale
  ));

  return (
    <Line
      points={points}
      color="#00ffff"
      lineWidth={2}
      transparent
      opacity={0.6}
    />
  );
};

// Thrust visualization
const ThrustIndicator = ({ state }) => {
  const { spacecraft_pos, spacecraft_vel } = state;
  if (!spacecraft_pos || !spacecraft_vel) return null;

  const scale = 0.01;
  const velocity = new THREE.Vector3(
    spacecraft_vel[0],
    spacecraft_vel[2],
    -spacecraft_vel[1]
  ).normalize().multiplyScalar(5);

  const position = new THREE.Vector3(
    spacecraft_pos[0] * scale,
    spacecraft_pos[2] * scale,
    -spacecraft_pos[1] * scale
  );

  return (
    <Line
      points={[position, position.clone().add(velocity)]}
      color="#ff00ff"
      lineWidth={3}
      transparent
      opacity={0.8}
    />
  );
};

// Earth visualization
const Earth = () => {
  const earthRef = useRef();

  useFrame(() => {
    if (earthRef.current) {
      earthRef.current.rotation.y += 0.001;
    }
  });

  return (
    <group position={[0, -70, 0]}>
      <Sphere ref={earthRef} args={[12]}>
        <meshStandardMaterial color="#004488" emissive="#004488" emissiveIntensity={0.2} toneMapped={false} />
      </Sphere>
      {/* Atmosphere glow */}
      <Sphere args={[13]}>
        <meshStandardMaterial 
          color="#0088ff" 
          transparent 
          opacity={0.1} 
          emissive="#0088ff" 
          emissiveIntensity={0.1} 
          toneMapped={false} 
        />
      </Sphere>
    </group>
  );
};

// Orbital reference grid
const OrbitalGrid = () => {
  return (
    <group>
      <Grid 
        infiniteGrid 
        cellSize={2} 
        sectionSize={10} 
        sectionColor={"#4488ff"} 
        sectionThickness={1} 
        fadeDistance={100} 
        position={[0, 0, 0]}
      />
      {/* Orbital plane indicator */}
      <Circle args={[50]} rotation={[-Math.PI / 2, 0, 0]} position={[0, 0, 0]}>
        <meshStandardMaterial 
          color="#ffff00" 
          transparent 
          opacity={0.1} 
          wireframe 
          emissive="#ffff00" 
          emissiveIntensity={0.2} 
          toneMapped={false} 
        />
      </Circle>
    </group>
  );
};

// Camera controller for smooth following
const CameraController = ({ state, controlsRef }) => {
  const { camera } = useThree();

  useFrame(() => {
    if (state && state.spacecraft_pos && controlsRef.current) {
      const controls = controlsRef.current;
      const scale = 0.01;
      
      // Spacecraft position in scene coordinates
      const spacecraftPosition = new THREE.Vector3(
        state.spacecraft_pos[0] * scale,
        state.spacecraft_pos[2] * scale,
        -state.spacecraft_pos[1] * scale
      );

      // Smoothly move the camera target to follow spacecraft
      controls.target.lerp(spacecraftPosition, 0.05);

      // Maintain camera distance but follow the target
      const idealOffset = camera.position.clone().sub(controls.target);
      const idealPosition = new THREE.Vector3().copy(spacecraftPosition).add(idealOffset);
      camera.position.lerp(idealPosition, 0.05);

      controls.update();
    }
  });

  return null;
};

// Status display component
const StatusDisplay = ({ state }) => {
  if (!state) return null;

  const { fuel_ratio, distance_to_target, velocity_magnitude } = state;

  return (
    <div style={{
      position: 'absolute',
      bottom: '20px',
      right: '20px',
      color: '#fff',
      background: 'rgba(0, 0, 20, 0.8)',
      padding: '15px',
      borderRadius: '8px',
      fontFamily: 'monospace',
      fontSize: '12px',
      border: '1px solid #4488ff',
      textShadow: '0 0 4px #000',
    }}>
      <div>Fuel: {(fuel_ratio * 100).toFixed(1)}%</div>
      <div style={{ color: distance_to_target < 50 ? '#00ff00' : '#ffffff' }}>
        Distance: {distance_to_target.toFixed(1)}m
      </div>
      <div>Velocity: {velocity_magnitude.toFixed(2)}m/s</div>
      <div style={{ 
        marginTop: '8px', 
        color: distance_to_target < 10 && velocity_magnitude < 0.5 ? '#00ff00' : '#ffffff' 
      }}>
        {distance_to_target < 10 && velocity_magnitude < 0.5 ? 'DOCKING READY' : 'APPROACHING'}
      </div>
    </div>
  );
};

export default function AstrodynamicsExample() {
  const [state, setState] = useState(null);
  const [training, setTraining] = useState(false);
  const [trained, setTrained] = useState(false);
  const [modelInfo, setModelInfo] = useState(null);
  const [logs, setLogs] = useState([]);
  const [chartState, setChartState] = useState({ labels: [], rewards: [], losses: [] });
  const wsRef = useRef(null);
  const [homeHover, setHomeHover] = useState(false);
  const { isMobile } = useResponsive();
  const controlsRef = useRef();

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
        setModelInfo({ 
          filename: parsed.model_filename, 
          timestamp: parsed.timestamp, 
          sessionUuid: parsed.session_uuid, 
          fileUrl: parsed.file_url 
        });
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

  return (
    <div
      style={{
        width: '100vw',
        height: '100vh',
        overflow: 'hidden',
        outline: 'none',
        background: 'radial-gradient(ellipse at center, #001122 0%, #000000 100%)',
        display: 'flex',
        flexDirection: 'column',
      }}
    >
      <div style={{ flex: 1, position: 'relative' }}>
        <Canvas 
          camera={{ position: [30, 20, 30], fov: 60 }} 
          style={{ background: 'transparent', width: '100vw', height: '100vh', overflow: 'hidden' }}
        >
          <ambientLight intensity={0.3} />
          <directionalLight position={[10, 10, 5]} intensity={0.8} />
          <pointLight position={[0, 0, 0]} intensity={0.5} color="#ffff00" />
          
          <Stars radius={200} depth={50} count={3000} factor={6} saturation={0} fade speed={0.5} />
          
          <Earth />
          <OrbitalGrid />
          <SpaceStation position={[0, 0, 0]} />
          
          {state && <Spacecraft state={state} />}
          {state && state.trail && <SpacecraftTrail trail={state.trail} />}
          {state && <ThrustIndicator state={state} />}

          <EffectComposer>
            <Bloom 
              intensity={1.2} 
              luminanceThreshold={0.05} 
              luminanceSmoothing={0.9} 
              toneMapped={false} 
            />
          </EffectComposer>
          
          <OrbitControls 
            ref={controlsRef} 
            maxDistance={100} 
            minDistance={5}
            enablePan={true}
            enableZoom={true}
            enableRotate={true}
          />
          <CameraController state={state} controlsRef={controlsRef} />
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
            Astrodynamics
          </Text>
          <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
            <Button auto type="secondary" disabled={training || trained} onClick={startTraining}>
              Train
            </Button>
            <Button auto type="success" disabled={!trained} onClick={startRun}>
              Run
            </Button>
            {trained && (
              <Button auto type="error" onClick={resetTraining}>
                Reset
              </Button>
            )}
          </div>
          <ModelInfoPanel modelInfo={modelInfo} />
        </div>

        <EquationPanel
          equation="\begin{aligned} \text{Observations} &= [\vec{r}_{rel}, \vec{v}_{rel}, \hat{r}_{target}, d_{target}, |\vec{v}|, m_{fuel}, t] \\ \text{Actions} &= [F_x, F_y, F_z] \in \{0, \pm F_{max}\} \\ \text{Hill's Eq} &: \ddot{x} = 3n^2x + 2n\dot{y} + \frac{F_x}{m}, \quad \ddot{y} = -2n\dot{x} + \frac{F_y}{m}, \quad \ddot{z} = -n^2z + \frac{F_z}{m} \end{aligned}"
          description="Orbital rendezvous: spacecraft state observations, thrust actions, and Hill's equations for relative motion dynamics in orbit"
        />

        <StatusDisplay state={state} />
        <InfoPanel logs={logs} chartState={chartState} />
        <ButtonForkOnGithub position={{ top: '10px', right: '10px' }} />
      </div>
    </div>
  );
} 