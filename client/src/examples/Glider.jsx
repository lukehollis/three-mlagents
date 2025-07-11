import React, { useState, useEffect, useRef, useMemo } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls, Box, Cylinder, Stars, Plane, Grid, Line, Cone, Circle } from '@react-three/drei';
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

const Arrow = ({ origin, direction, color = "#ffff00" }) => {
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
        <meshStandardMaterial color={color} emissive={color} emissiveIntensity={0.8} toneMapped={false} />
      </Cone>
    </group>
  );
};

const WindCurves = ({ windParams }) => {
  if (!windParams || windParams.length < 7) return null;
  const [C1, C2, C3, waveFreq, waveMag, waveFreq2, waveMag2] = windParams;

  const { curves, arrows } = useMemo(() => {
    const curveData = [];
    const arrowData = [];
    const num_curves_x = 12;
    const num_curves_z = 12;
    const bounds_x = 800;
    const bounds_z = 400;

    for (let i = 0; i < num_curves_x; i++) {
      for (let j = 0; j < num_curves_z; j++) {
        
        const startX = -bounds_x + (i / (num_curves_x -1)) * bounds_x * 2;
        const startZ = -bounds_z + (j / (num_curves_z -1)) * bounds_z * 2;

        const points = [];
        let currentX = startX;
        let currentY = 0;
        let currentZ = startZ;

        const num_steps = 40;
        const dt = 1.0; 

        for (let step = 0; step < num_steps; step++) {
            const currentPoint = new THREE.Vector3(currentX, currentY, currentZ);
            points.push(currentPoint);

            const world_x = currentX;
            const world_y = -currentZ;

            const updraft1 = Math.sin(world_x * waveFreq * 2 * Math.PI) * Math.cos(world_y * waveFreq * 2 * Math.PI) * C1 * waveMag;
            const updraft2 = Math.sin(world_x * waveFreq2 * 2 * Math.PI / 1.5) * Math.cos(world_y * waveFreq * 2 * Math.PI / 1.5) * C1 * waveMag2;
            const total_updraft = updraft1 + updraft2;
            
            const wind_x = 1.0;
            const wind_z = -0.5; // three.js z is inverted from world y
            const wind_y = total_updraft;

            currentX += wind_x * dt;
            currentY += wind_y * dt;
            currentZ += wind_z * dt;

            if (currentY > 150 || currentY < 0) break;

            if (step > 0 && step % 15 === 0) {
                const direction = new THREE.Vector3(wind_x, wind_y, wind_z).normalize();
                const color = wind_y > 0 ? "#ff8888" : "#8888ff";
                arrowData.push({ key: `${i}-${j}-${step}`, origin: currentPoint, direction, color });
            }
        }

        const world_x_avg = startX;
        const world_y_avg = -startZ;
        const updraft1 = Math.sin(world_x_avg * waveFreq * 2 * Math.PI) * Math.cos(world_y_avg * waveFreq * 2 * Math.PI) * C1 * waveMag;
        const updraft2 = Math.sin(world_x_avg * waveFreq2 * 2 * Math.PI / 1.5) * Math.cos(world_y_avg * waveFreq * 2 * Math.PI / 1.5) * C1 * waveMag2;
        const total_updraft = updraft1 + updraft2;
        const updraftRatio = Math.min(Math.max(total_updraft / C1, -1), 1);
        let color;
        if (updraftRatio > 0.1) {
            color = new THREE.Color().setRGB(1, 1 - updraftRatio, 1 - updraftRatio);
        } else if (updraftRatio < -0.1) {
            color = new THREE.Color().setRGB(1 + updraftRatio, 1 + updraftRatio, 1);
        } else {
            color = new THREE.Color().setRGB(0.5, 0.5, 0.5);
        }
        
        if (points.length > 1 && Math.abs(total_updraft) > 1.5) { // Threshold to draw
            curveData.push({ key: `${i}-${j}`, points, color });
        }
      }
    }
    return { curves: curveData, arrows: arrowData };
  }, [windParams]);

  return (
    <group>
        {curves.map((c) => (
            <Line key={c.key} points={c.points} color={c.color} lineWidth={1.5} transparent opacity={0.5} />
        ))}
        {arrows.map((a) => (
            <Arrow key={a.key} origin={a.origin} direction={a.direction} color={a.color} />
        ))}
    </group>
  )
}

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
  const colorsRef = useRef(new Float32Array());

  const PARTICLE_COUNT = 8000;
  const MAX_LIFE = 600; // frames

  const particles = useMemo(() => {
    const temp = [];
    const colors = [];
    lifetimesRef.current = [];
    for (let i = 0; i < PARTICLE_COUNT; i++) {
      const x = THREE.MathUtils.randFloatSpread(1600);
      const y = THREE.MathUtils.randFloat(0, 150);
      const z = THREE.MathUtils.randFloatSpread(800);
      temp.push(x, y, z);
      colors.push(1, 1, 1); // init as white
      lifetimesRef.current.push(Math.random() * MAX_LIFE);
    }
    colorsRef.current = new Float32Array(colors);
    return new Float32Array(temp);
  }, []);

  useFrame(() => {
    if (!particlesRef.current) return;

    const positions = particlesRef.current.geometry.attributes.position.array;
    const colors = particlesRef.current.geometry.attributes.color.array;

    for (let i = 0; i < PARTICLE_COUNT; i++) {
      const i3 = i * 3;

      lifetimesRef.current[i] += 1;

      const three_x = positions[i3];
      const three_y = positions[i3 + 1];
      const three_z = positions[i3 + 2];

      const world_x = three_x;
      const world_y = -three_z;

      const updraft1 = Math.sin(world_x * waveFreq * 2 * Math.PI) * Math.cos(world_y * waveFreq * 2 * Math.PI) * C1 * waveMag;
      const updraft2 = Math.sin(world_x * waveFreq2 * 2 * Math.PI / 1.5) * Math.cos(world_y * waveFreq * 2 * Math.PI / 1.5) * C1 * waveMag2;
      const total_updraft = updraft1 + updraft2;

      // Gentle horizontal drift
      positions[i3] += 1.0 * 0.02;
      positions[i3 + 1] += total_updraft * 0.02; // Vertical movement
      positions[i3 + 2] += 0.5 * 0.02;

      const color = new THREE.Color();
      const updraftRatio = Math.min(Math.max(total_updraft / C1, -1), 1);
      if (updraftRatio > 0) {
        color.setRGB(1, 1 - updraftRatio, 1 - updraftRatio); // White to Red for updrafts
      } else {
        color.setRGB(1 + updraftRatio, 1 + updraftRatio, 1); // White to Blue for downdrafts
      }
      colors[i3] = color.r;
      colors[i3 + 1] = color.g;
      colors[i3 + 2] = color.b;


      const outOfBounds =
        positions[i3] > 800 || positions[i3] < -800 ||
        positions[i3 + 2] > 400 || positions[i3 + 2] < -400 ||
        positions[i3 + 1] > 150 || positions[i3+1] < 0; // check y bounds too

      if (outOfBounds || lifetimesRef.current[i] > MAX_LIFE) {
        // Reset the particle at a completely random location within the volume
        positions[i3] = THREE.MathUtils.randFloatSpread(1600);
        positions[i3 + 1] = THREE.MathUtils.randFloat(0, 150);
        positions[i3 + 2] = THREE.MathUtils.randFloatSpread(800);
        lifetimesRef.current[i] = 0;
      }
    }

    particlesRef.current.geometry.attributes.position.needsUpdate = true;
    particlesRef.current.geometry.attributes.color.needsUpdate = true;
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
        <bufferAttribute
            attach="attributes-color"
            count={colorsRef.current.length / 3}
            array={colorsRef.current}
            itemSize={3}
        />
      </bufferGeometry>
      <pointsMaterial size={0.2} transparent opacity={0.7} vertexColors />
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
        <EquationPanel
          equation="\begin{aligned} \text{Observations} &= [uvw_{\text{global}}, \text{airspeed}, \text{height}, \alpha\beta\gamma, D_{\text{target}}, \text{Vec}_{\text{target}}, \omega_{\text{Rotor}}, E_{\text{Battery}}] \\ \text{Actions} &= [\text{roll}_{\text{input}}, \text{pitch}_{\text{input}}, \text{throttle}_{\text{input}}, \text{pitch}_{\text{Rotor,input}}] \\ R_{\text{mixed}} &= E \cdot (1 - E) + E \cdot H = E \cdot (H - E + 1) \end{aligned}"
          description="Dynamic soarding: state and action spaces for the glider and the mixed reward function for heading and energy"
        />
        <InfoPanel logs={logs} chartState={chartState} />
        <ButtonForkOnGithub position={{ top: '10px', right: '10px' }} />
      </div>
    </div>
  );
} 


