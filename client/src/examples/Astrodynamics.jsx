import React, { useState, useEffect, useRef, useMemo } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls, Box, Cylinder, Stars, Plane, Grid, Cone, Circle, Sphere, Torus } from '@react-three/drei';
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
const VIZ_SCALE = 0.00001; // Scale down meters to scene units
const MAX_TRAIL_POINTS = 10000; // Must match backend

const Sun = () => (
    <group position={[20000, 0, -15000]}>
      <Sphere args={[1000, 32, 32]}>
        <meshStandardMaterial emissive="#ffffc5" emissiveIntensity={10} color="#ffffc5" toneMapped={false} />
      </Sphere>
      <pointLight intensity={1e9} distance={100000} color="#ffffff" />
    </group>
);

// Spacecraft component
const Spacecraft = ({ position }) => {
  const groupRef = useRef();
  
  useFrame(() => {
    if (groupRef.current && position) {
      groupRef.current.position.set(
        position[0] * VIZ_SCALE,
        position[2] * VIZ_SCALE,
        -position[1] * VIZ_SCALE
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
const SpaceStation = ({ position }) => {
  const groupRef = useRef();

  useFrame(() => {
    if (groupRef.current) {
      groupRef.current.rotation.y += 0.005;
      if (position) {
        groupRef.current.position.set(
            position[0] * VIZ_SCALE,
            position[2] * VIZ_SCALE,
            -position[1] * VIZ_SCALE
        );
      }
    }
  });

  return (
    <group ref={groupRef}>
      {/* Central hub */}
      <Sphere args={[1.5]}>
        <meshStandardMaterial color="#ffffff" emissive="#ffffff" emissiveIntensity={0.3} toneMapped={false} />
      </Sphere>
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

const StationOrbit = () => {
  const stationOrbitRadius = (6.371e6 + 15000e3) * VIZ_SCALE;
  return (
      <Torus args={[stationOrbitRadius, 0.1, 16, 128]} rotation={[Math.PI / 2, 0, 0]}>
          <meshStandardMaterial
              color="white"
              transparent
              opacity={0.2}
          />
      </Torus>
  );
};

const EarthOrbit = () => {
  const sunPosition = [20000, 0, -15000];
  const earthPosition = [0, 0, 0];
  const radius = Math.sqrt(
      Math.pow(sunPosition[0] - earthPosition[0], 2) +
      Math.pow(sunPosition[1] - earthPosition[1], 2) +
      Math.pow(sunPosition[2] - earthPosition[2], 2)
  );

  return (
      <group position={sunPosition}>
          <Torus args={[radius, 0.1, 16, 200]} rotation={[Math.PI / 2, 0, 0]} wireframe={true}>
              <meshStandardMaterial
                  color="white"
                  transparent
                  opacity={0.2}
                  
              />
          </Torus>
      </group>
  );
};

const Trail = ({ trail, color }) => {
  const lineRef = useRef();

  const intensifiedColor = useMemo(() => new THREE.Color(color).multiplyScalar(9), [color]);

  const geometry = useMemo(() => {
    const geom = new THREE.BufferGeometry();
    const positions = new Float32Array(MAX_TRAIL_POINTS * 3);
    geom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geom.setDrawRange(0, 0);
    return geom;
  }, []);

  useFrame(() => {
    if (!lineRef.current) return;

    if (!trail || trail.length < 2) {
      lineRef.current.geometry.setDrawRange(0, 0);
      return;
    }

    const positions = lineRef.current.geometry.attributes.position.array;
    
    let i = 0;
    for (const pos of trail) {
      positions[i++] = pos[0] * VIZ_SCALE;
      positions[i++] = pos[2] * VIZ_SCALE;
      positions[i++] = -pos[1] * VIZ_SCALE;
    }

    lineRef.current.geometry.setDrawRange(0, trail.length);
    lineRef.current.geometry.attributes.position.needsUpdate = true;
  });

  return (
    <line ref={lineRef} geometry={geometry}>
      <lineBasicMaterial color={intensifiedColor} transparent opacity={0.6} lineWidth={1} toneMapped={false} />
    </line>
  );
};
  

// Thrust visualization
const ThrustIndicator = ({ state }) => {
  const lineRef = useRef();

  const geometry = useMemo(() => {
    const geom = new THREE.BufferGeometry();
    geom.setAttribute('position', new THREE.BufferAttribute(new Float32Array(2 * 3), 3));
    return geom;
  }, []);

  useFrame(() => {
    if (!lineRef.current) return;

    if (!state || !state.spacecraft_vel_abs) {
      lineRef.current.visible = false;
      return;
    }
    lineRef.current.visible = true;

    const positions = lineRef.current.geometry.attributes.position.array;

    const velocity = new THREE.Vector3(
      state.spacecraft_vel_abs[0],
      state.spacecraft_vel_abs[2],
      -state.spacecraft_vel_abs[1]
    ).normalize().multiplyScalar(5);

    const position = new THREE.Vector3(
      state.spacecraft_pos_abs[0] * VIZ_SCALE,
      state.spacecraft_pos_abs[2] * VIZ_SCALE,
      -state.spacecraft_pos_abs[1] * VIZ_SCALE
    );
    const endPoint = position.clone().add(velocity);

    positions[0] = position.x;
    positions[1] = position.y;
    positions[2] = position.z;
    positions[3] = endPoint.x;
    positions[4] = endPoint.y;
    positions[5] = endPoint.z;

    lineRef.current.geometry.attributes.position.needsUpdate = true;
    lineRef.current.geometry.computeBoundingSphere();
  });

  return (
    <line ref={lineRef} geometry={geometry}>
      <lineBasicMaterial color="#ff00ff" transparent opacity={0.8} />
    </line>
  );
};

// Earth visualization
const Earth = () => {
  const earthRef = useRef();
  const earthRadius = 6.371e6; // m

  useFrame(() => {
    if (earthRef.current) {
      earthRef.current.rotation.y += 0.001;
    }
  });

  return (
    <group position={[0, 0, 0]}>
      <Sphere ref={earthRef} args={[earthRadius * VIZ_SCALE, 64, 64]} >
      <meshStandardMaterial color="#004488" emissive="#004488" emissiveIntensity={0.2} toneMapped={false} wireframe />
      </Sphere>
      {/* Atmosphere glow */}
      <Sphere args={[earthRadius * VIZ_SCALE * 1.02, 64, 64]}>
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
  const orbitRadius = (6.371e6 + 15000e3) * VIZ_SCALE;
  return (
    <group rotation={[0, 0, 0]}>
      {/* Orbital plane indicator */}
      <Circle args={[orbitRadius, 128]} rotation={[-Math.PI / 2, 0, 0]} position={[0, 0, 0]}>
        <meshStandardMaterial 
          color="#ffffff" 
          transparent 
          opacity={0.05} 
          wireframe 
          toneMapped={false} 
        />
      </Circle>
    </group>
  );
};

const Scene = ({ state }) => {
    return (
      <>
        <Sun />
        <Earth />
        <OrbitalGrid />
        <StationOrbit />
        <EarthOrbit />
  
        {state && (
          <>
            <Spacecraft position={state.spacecraft_pos_abs} />
            <SpaceStation position={state.target_pos_abs} />
            <Trail trail={state.trail} color="#00ffff" />
            <Trail trail={state.target_trail} color="#ffffff" />
            <ThrustIndicator state={state} />
          </>
        )}
  
        <Stars radius={100000} depth={100} count={5000} factor={20} saturation={0} fade speed={0.2} />
        <EffectComposer>
          <Bloom
            intensity={.3}
            mipmapBlur
            luminanceThreshold={0.4}
            luminanceSmoothing={0}
          />
        </EffectComposer>
      </>
    );
  };

// Status display component
const StatusDisplay = ({ state }) => {
  if (!state) return null;

  const { fuel_ratio, distance_to_target, velocity_magnitude } = state;

  // Format distance with appropriate units
  const formatDistance = (distance) => {
    if (distance > 1000000) {
      return `${(distance / 1000000).toFixed(1)}Mm`; // Megameters
    } else if (distance > 1000) {
      return `${(distance / 1000).toFixed(1)}km`;
    } else {
      return `${distance.toFixed(1)}m`;
    }
  };

  const isDockingReady = distance_to_target < 50 && velocity_magnitude < 2.0;
  const isCloseApproach = distance_to_target < 1000;

  return (
    <div style={{
      position: 'fixed',
      top: '10px',
      right: '10px',
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
      <div style={{ color: isCloseApproach ? '#00ff00' : '#ffffff' }}>
        Distance: {formatDistance(distance_to_target)}
      </div>
      <div>Velocity: {velocity_magnitude.toFixed(2)}m/s</div>
      <div style={{ 
        marginTop: '8px', 
        color: isDockingReady ? '#00ff00' : isCloseApproach ? '#ffff00' : '#ffffff' 
      }}>
        {isDockingReady ? 'DOCKING READY' : isCloseApproach ? 'CLOSE APPROACH' : 'LAUNCHING TO ORBIT'}
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
      return upd.length > 100 ? upd.slice(upd.length - 100) : upd;
    });
  };

  useEffect(() => {
    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;
    ws.onopen = () => addLog('WS opened');
    ws.onmessage = (ev) => {
      let parsed;
      try {
        parsed = JSON.parse(ev.data);
      } catch {
        addLog(`Received non-JSON: ${ev.data}`);
        return;
      }

      if ((parsed.type === 'train_step' || parsed.type === 'run_step' || parsed.type === 'state') && parsed.state) {
        setState(parsed.state);
      } else if (parsed.type === 'progress') {
        addLog(`Episode ${parsed.episode}: Reward=${parsed.reward.toFixed(3)}, Loss=${(parsed.loss ?? 0).toFixed(3)}`);
        setChartState((prev) => ({
          labels: [...prev.labels, parsed.episode],
          rewards: [...prev.rewards, parsed.reward],
          losses: [...prev.losses, parsed.loss ?? null],
        }));
      } else if (parsed.type === 'trained') {
        addLog(`Training complete. Model: ${parsed.model_filename}`);
        setTraining(false);
        setTrained(true);
        setModelInfo({ 
          filename: parsed.model_filename, 
          timestamp: parsed.timestamp, 
          sessionUuid: parsed.session_uuid, 
          fileUrl: parsed.file_url 
        });
      } else {
        addLog(ev.data);
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
    if (!trained || !modelInfo) return;
    send({ cmd: 'run', model_filename: modelInfo.filename });
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
          camera={{ position: [450, 250, 450], fov: 60, far: 50000000 }} 
          style={{ background: 'transparent', width: '100vw', height: '100vh', overflow: 'hidden' }}
        >
          <ambientLight intensity={0.1} />
          <Scene state={state} />
          <OrbitControls 
            ref={controlsRef} 
            maxDistance={40000} 
            minDistance={10}
            target={[0, 0, 0]}
          />
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
          </div>
          <ModelInfoPanel modelInfo={modelInfo} />
        </div>

        <EquationPanel
          equation="\begin{aligned} \text{Observations} &= [\vec{r}_{rel}, \vec{v}_{rel}, \hat{r}_{target}, d_{target}, |\vec{v}|, m_{fuel}, t] \\ \text{Actions} &= [F_x, F_y, F_z] \in \{0, \pm F_{max}\} \\ \text{Hill's Eq} &: \ddot{x} = 3n^2x + 2n\dot{y} + \frac{F_x}{m}, \quad \ddot{y} = -2n\dot{x} + \frac{F_y}{m}, \quad \ddot{z} = -n^2z + \frac{F_z}{m} \end{aligned}"
          description="Orbital rendezvous: spacecraft state observations, thrust actions, and Hill's equations for relative motion dynamics in orbit"
          collapsed={true}
        />

        <StatusDisplay state={state} />
        <InfoPanel logs={logs} chartState={chartState} />
      </div>
    </div>
  );
} 