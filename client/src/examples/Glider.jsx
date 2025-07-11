import React, { useState, useEffect, useRef, useMemo } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls, Box, Cylinder, Stars, Plane, Grid, Line, Cone } from '@react-three/drei';
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

const WS_URL = `${config.WS_BASE_URL}/ws/glider`;

const Arrow = ({ origin, direction }) => {
  const ref = useRef();
  React.useLayoutEffect(() => {
    if (ref.current) {
      // Point the arrow by looking from its origin to a point along the direction vector
      ref.current.lookAt(new THREE.Vector3().addVectors(origin, direction));
    }
  }, [origin, direction]);

  return (
    <group ref={ref} position={origin}>
      <Cone args={[1.5, 5, 8]} rotation={[Math.PI / 2, 0, 0]}>
        <meshStandardMaterial color="#ffff00" emissive="#ffff00" emissiveIntensity={0.8} toneMapped={false} />
      </Cone>
    </group>
  );
};

const WindCurves = ({ windParams }) => {
  if (!windParams || windParams.length < 7) return null;
  const [C1, C2, C3, waveFreq, waveMag, waveFreq2, waveMag2] = windParams;

  const { streamlines, arrows } = useMemo(() => {
    const streamlinesData = [];
    const arrowData = [];
    const height = 50;
    const numStreamlines = 20;
    const arrowInterval = 25;

    for (let i = 0; i < numStreamlines; i++) {
      const points = [];
      // Start streamlines at different lateral positions
      const startZ = -800 + (i / (numStreamlines - 1)) * 1600;
      
      let currentX = -800;
      let currentZ = startZ;

      // Trace the path of a particle to create a streamline
      for (let step = 0; step < 200; step++) {
        const currentPoint = new THREE.Vector3(currentX, height, currentZ);
        points.push(currentPoint);

        const world_y = -currentZ;
        const world_z = height;

        const baseWindSpeed = C1 / (1 + Math.exp(-C2 * (world_z - C3)));
        const angle1 = Math.sin(world_y * waveFreq * 2 * Math.PI) * waveMag;
        const angle2 = Math.sin(world_y * waveFreq2 * 2 * Math.PI) * waveMag2;
        const totalAngle = angle1 + angle2;
        
        const wind_x = baseWindSpeed * Math.cos(totalAngle);
        const wind_y = baseWindSpeed * Math.sin(totalAngle);
        
        const dt = 4; // Step size for the simulation
        currentX += wind_x * dt;
        currentZ -= wind_y * dt;

        if (step > 0 && step % arrowInterval === 0) {
            const prevPoint = points[step -1];
            const direction = new THREE.Vector3().subVectors(currentPoint, prevPoint).normalize();
            arrowData.push({ key: `${i}-${step}`, origin: currentPoint, direction });
        }
      }
      streamlinesData.push(points);
    }
    return { streamlines: streamlinesData, arrows: arrowData };
  }, [windParams]);

  return (
    <group>
      {streamlines.map((points, i) => (
         <Line key={i} points={points} color="white" lineWidth={0.5} opacity={0.7} transparent />
      ))}
      {arrows.map(arrow => (
        <Arrow key={arrow.key} origin={arrow.origin} direction={arrow.direction} />
      ))}
    </group>
  );
};

const Waypoint = ({ position, isTarget }) => {
  const ref = useRef();
  useFrame(() => {
    if (ref.current && isTarget) {
      ref.current.rotation.y += 0.01;
      ref.current.rotation.x += 0.01;
    }
  });
  return (
    <mesh ref={ref} position={position}>
      <boxGeometry args={isTarget? [4,4,4] : [2, 2, 2]} />
      <meshStandardMaterial
        color={isTarget ? '#ff00ff' : '#ffffff'}
        emissive={isTarget ? '#ff00ff' : '#ffffff'}
        emissiveIntensity={isTarget ? 0.8 : 0.2}
        wireframe
        toneMapped={false}
      />
    </mesh>
  );
};

const Waypoints = ({ waypoints, currentWaypointIndex }) => {
  if (!waypoints) return null;
  return (
    <group>
      {waypoints.map((wp, i) => (
        <Waypoint
          key={i}
          position={[wp[0], wp[2], -wp[1]]} // Same coordinate transform as glider
          isTarget={i === currentWaypointIndex}
        />
      ))}
    </group>
  );
};

const Glider = ({ state }) => {
  const { pos, rot } = state;
  const groupRef = useRef();

  useFrame(() => {
    if (groupRef.current) {
      groupRef.current.position.set(pos[0], pos[2], -pos[1]);
      // Note: R3F uses extrinsic 'XYZ' rotation order by default.
      // My state is [roll, pitch, yaw] which corresponds to rotation around x, y, z axes in body frame.
      // I'm applying it as y, x, z to get a more intuitive flight model visualization.
      groupRef.current.rotation.set(rot[1], rot[2], -rot[0]);
    }
  });

  const fuselageColor = "#cccccc";
  const wingColor = "#00ffff";

  return (
    <group ref={groupRef}>
      {/* Fuselage */}
      <Box args={[3, 0.2, 0.2]} position={[0, 0, 0]}>
        <meshStandardMaterial color={fuselageColor} emissive={fuselageColor} emissiveIntensity={0.2} toneMapped={false} />
      </Box>
      {/* Main Wing */}
      <Box args={[0.5, 0.1, 4]} position={[0, 0, 0]}>
         <meshStandardMaterial color={wingColor} emissive={wingColor} emissiveIntensity={0.4} toneMapped={false} />
      </Box>
      {/* Tail */}
      <Box args={[0.5, 0.05, 1]} position={[-1.4, 0.2, 0]}>
         <meshStandardMaterial color={wingColor} emissive={wingColor} emissiveIntensity={0.4} toneMapped={false} />
      </Box>
    </group>
  );
};

const Wind = ({ windParams }) => {
  if (!windParams || windParams.length < 7) return null;
  const [C1, C2, C3, waveFreq, waveMag, waveFreq2, waveMag2] = windParams;
  const particlesRef = useRef();
  const lifetimesRef = useRef([]);

  const PARTICLE_COUNT = 8000;
  const MAX_LIFE = 600; // frames

  const particles = useMemo(() => {
    const temp = [];
    lifetimesRef.current = [];
    for (let i = 0; i < PARTICLE_COUNT; i++) {
      const x = THREE.MathUtils.randFloatSpread(1600);
      const y = THREE.MathUtils.randFloat(0, 150);
      const z = THREE.MathUtils.randFloatSpread(800);
      temp.push(x, y, z);
      lifetimesRef.current.push(Math.random() * MAX_LIFE);
    }
    return new Float32Array(temp);
  }, []);

  useFrame(() => {
    if (!particlesRef.current) return;

    const positions = particlesRef.current.geometry.attributes.position.array;
    for (let i = 0; i < PARTICLE_COUNT; i++) {
      const i3 = i * 3;

      // Increment lifetime
      lifetimesRef.current[i] += 1;

      const three_x = positions[i3];
      const three_y = positions[i3 + 1];
      const three_z = positions[i3 + 2];

      const world_y = -three_z;
      const world_z = three_y;

      const baseWindSpeed = C1 / (1 + Math.exp(-C2 * (world_z - C3)));
      const angle1 = Math.sin(world_y * waveFreq * 2 * Math.PI) * waveMag;
      const angle2 = Math.sin(world_y * waveFreq2 * 2 * Math.PI) * waveMag2;
      const totalAngle = angle1 + angle2;

      const wind_x = baseWindSpeed * Math.cos(totalAngle);
      const wind_y = baseWindSpeed * Math.sin(totalAngle);

      positions[i3] += wind_x * 0.02;
      positions[i3 + 2] -= wind_y * 0.02;

      const outOfBounds =
        positions[i3] > 800 || positions[i3] < -800 ||
        positions[i3 + 2] > 400 || positions[i3 + 2] < -400;

      if (outOfBounds || lifetimesRef.current[i] > MAX_LIFE) {
        // Reset the particle at a completely random location within the volume
        positions[i3] = THREE.MathUtils.randFloatSpread(1600);
        positions[i3 + 1] = THREE.MathUtils.randFloat(0, 150);
        positions[i3 + 2] = THREE.MathUtils.randFloatSpread(800);
        lifetimesRef.current[i] = 0;
      }
    }

    particlesRef.current.geometry.attributes.position.needsUpdate = true;
  });

  return (
    <points ref={particlesRef}>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          count={particles.length / 3}
          array={particles}
          itemSize={3}
        />
      </bufferGeometry>
      <pointsMaterial size={0.15} color="#44ffff" transparent opacity={0.6} />
    </points>
  );
};

const CameraController = ({ state, controlsRef }) => {
  const { camera } = useThree();

  useFrame(() => {
    if (state && controlsRef.current) {
      const controls = controlsRef.current;
      // Current glider position in the 3D scene
      const gliderPosition = new THREE.Vector3(state.pos[0], state.pos[2], -state.pos[1]);

      // The point the camera should be looking at
      const targetPosition = new THREE.Vector3().copy(gliderPosition);

      // Smoothly move the OrbitControls target to the glider's position
      controls.target.lerp(targetPosition, 0.1);

      // We don't want to just teleport the camera. We want it to follow smoothly.
      // We calculate where the camera *should* be based on its current offset from the target.
      const idealOffset = camera.position.clone().sub(controls.target);
      const idealPosition = new THREE.Vector3().copy(targetPosition).add(idealOffset);

      // And smoothly move the camera to that ideal position.
      camera.position.lerp(idealPosition, 0.1);

      // We must call update after manually changing the camera.
      controls.update();
    }
  });

  return null;
};


export default function GliderExample() {
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
    setState(null);
  };
  
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
        <Canvas camera={{ position: [0, 30, 40], fov: 60 }} style={{ background: 'transparent', width: '100vw', height: '100vh', overflow: 'hidden' }}>
          <ambientLight intensity={0.5} />
          <directionalLight position={[0, 40, 20]} intensity={1.0} />
          <Stars radius={400} depth={50} count={5000} factor={8} saturation={0} fade speed={1} />
          
          {state && <Glider state={state} />}
          {state && state.wind_params && <Wind windParams={state.wind_params} />}
          {state && state.wind_params && <WindCurves windParams={state.wind_params} />}
          {state && <Waypoints waypoints={state.waypoints} currentWaypointIndex={state.current_waypoint_index} />}
          
          <Grid infiniteGrid cellSize={5} sectionSize={20} sectionColor={"#4488ff"} sectionThickness={1.5} fadeDistance={250} />
          <Plane args={[100000, 100000]} rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.5, 0]}>
              <meshStandardMaterial color="#001122" transparent opacity={0.11} />
          </Plane>

          <EffectComposer>
            <Bloom intensity={0.8} luminanceThreshold={0.1} luminanceSmoothing={0.9} toneMapped={false} />
          </EffectComposer>
          <OrbitControls ref={controlsRef} maxDistance={250} />
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
            Glider
          </Text>
          <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
            <Button auto type="secondary" disabled={training || trained} onClick={startTraining}>Train</Button>
            <Button auto type="success" disabled={!trained} onClick={startRun}>Run</Button>
            {trained && <Button auto type="error" onClick={resetTraining}>Reset</Button>}
          </div>
          <ModelInfoPanel modelInfo={modelInfo} />
        </div>
        <EquationPanel equation="e_{max} = \sqrt{1 - \frac{2\pi}{L/D}}" description="Max turn efficiency (coeff. of restitution) is related to the lift-to-drag ratio." />
        <InfoPanel logs={logs} chartState={chartState} />
        <ButtonForkOnGithub position={{ top: '10px', right: '10px' }} />
      </div>
    </div>
  );
} 