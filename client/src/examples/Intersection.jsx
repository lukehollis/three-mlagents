import React, { useState, useEffect, useRef, useMemo } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Grid, Text as DreiText, Plane, Line } from '@react-three/drei';
import { Button, Text, Card, Code } from '@geist-ui/core';
import { Link } from 'react-router-dom';
import * as THREE from 'three';
import config from '../config.js';
import { useResponsive } from '../hooks/useResponsive.js';
import { EffectComposer, Bloom } from '@react-three/postprocessing';
import InfoPanel from '../components/InfoPanel.jsx';
import ModelInfoPanel from '../components/ModelInfoPanel.jsx';

const WS_URL = `${config.WS_BASE_URL}/ws/intersection`;

const Vehicle = ({ agent }) => {
  const { pos, id, energy, velocity } = agent;
  const groupRef = useRef();

  const energyColor = useMemo(() => {
    const blue = new THREE.Color("#00ffff");    
    const white = new THREE.Color("#ffffff");
    const alpha = Math.max(0, Math.min(1, energy / 100));
    return white.clone().lerp(blue, alpha);
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
    <group ref={groupRef} position={pos}>
      <mesh>
        <boxGeometry args={[0.8, 0.8, 1.5]} />
        <meshPhongMaterial color={energyColor} emissive={energyColor} emissiveIntensity={energy / 100} wireframe={true} />
      </mesh>
      {/* Front lights */}
      <mesh position={[0.3, 0, 0.75]}>
        <sphereGeometry args={[0.1, 8, 8]} />
        <meshStandardMaterial color={"white"} emissive={"white"} emissiveIntensity={5} />
      </mesh>
      <mesh position={[-0.3, 0, 0.75]}>
        <sphereGeometry args={[0.1, 8, 8]} />
        <meshStandardMaterial color={"white"} emissive={"white"} emissiveIntensity={5} />
      </mesh>
      {/* Back lights */}
      <mesh position={[0.3, 0, -0.75]}>
        <sphereGeometry args={[0.1, 8, 8]} />
        <meshStandardMaterial color={"red"} emissive={"red"} emissiveIntensity={5} />
      </mesh>
      <mesh position={[-0.3, 0, -0.75]}>
        <sphereGeometry args={[0.1, 8, 8]} />
        <meshStandardMaterial color={"red"} emissive={"red"} emissiveIntensity={5} />
      </mesh>
    </group>
  );
};

const Road = ({ points, width = 3 }) => {
    const roadColor = '#1c1c1c'; // Dark asphalt
    const roadY = 0.01;

    const curve = useMemo(() => new THREE.CatmullRomCurve3(
        points.map(p => new THREE.Vector3(p[0], roadY, p[2]))
    ), [points]);

    const segments = points.length > 2 ? 64 : 1;
    const curvePoints = curve.getPoints(segments);

    return (
        <group>
            {curvePoints.slice(0, -1).map((p1, i) => {
                const p2 = curvePoints[i + 1];
                const length = p1.distanceTo(p2);
                const center = new THREE.Vector3().addVectors(p1, p2).multiplyScalar(0.5);
                const angle = Math.atan2(p2.x - p1.x, p2.z - p1.z);

                return (
                    <Plane
                        key={i}
                        args={[width, length]}
                        position={center}
                        rotation={new THREE.Euler( -Math.PI / 2, angle, 0, 'YXZ' )}
                    >
                        <meshStandardMaterial color={roadColor} side={THREE.DoubleSide} transparent opacity={0.5} />
                    </Plane>
                );
            })}
        </group>
    );
};


const TrafficLight = ({ position, state }) => {
  const red = "#ff0000";
  const green = "#00ff00";

  // state 0: NS Green, EW Red
  // state 1: NS Red, EW Green
  const nsColor = state === 0 ? green : red;
  const ewColor = state === 1 ? green : red;

  return (
    <group position={position}>
        {/* Pole */}
        <mesh position={[0, 2, 0]}>
            <cylinderGeometry args={[0.1, 0.1, 4, 8]} />
            <meshStandardMaterial color="#333" />
        </mesh>
        {/* NS Light */}
        <mesh position={[0, 4.25, 0.25]}>
            <sphereGeometry args={[0.25, 16, 16]} />
            <meshStandardMaterial color={nsColor} emissive={nsColor} emissiveIntensity={2} />
        </mesh>
        {/* EW Light */}
        <mesh position={[0.25, 4.25, 0]}>
            <sphereGeometry args={[0.25, 16, 16]} />
            <meshStandardMaterial color={ewColor} emissive={ewColor} emissiveIntensity={2} />
        </mesh>
    </group>
  );
};

const Intersection = () => {
    const roadColor = '#282828';
    const roadY = 0.05;
    const roadMarkingColor = '#ffffff';

    const paths = useMemo(() => [
        // Main Roads
        { points: [[-40, 0, 0], [40, 0, 0]], width: 3 }, // E-W
        { points: [[0, 0, 0], [0, 0, 20]], width: 3 }, // N-S
        
        // Feeder roads
        { points: [[20, 0, 20], [20, 0, 0]], width: 3 },
        { points: [[-25, 0, -20], [-25, 0, 0]], width: 3 },

        // Curved road
        { points: [[25, 0, -20], [10, 0, -10], [0, 0, 0]], width: 3 },

        // Connectors to main roads
        { points: [[20, 0, 0], [5, 0, 0]], width: 3 },
        { points: [[-25, 0, 0], [-5, 0, 0]], width: 3 },
    ], []);

    return (
        <group>
            {paths.map((path, i) => {
                const curve = new THREE.CatmullRomCurve3(path.points.map(p => new THREE.Vector3(p[0], roadY, p[2])));
                const curvePoints = curve.getPoints(100);
                const tangents = curvePoints.map((_, i, arr) => curve.getTangent(i / (arr.length - 1)));
                const halfWidth = path.width / 2;
                const up = new THREE.Vector3(0, 1, 0);
                const leftEdgePoints = [];
                const rightEdgePoints = [];

                for (let j = 0; j < curvePoints.length; j++) {
                    const point = curvePoints[j];
                    const tangent = tangents[j];
                    const normal = new THREE.Vector3().crossVectors(tangent, up).normalize();
                    leftEdgePoints.push(new THREE.Vector3().addVectors(point, normal.clone().multiplyScalar(halfWidth)));
                    rightEdgePoints.push(new THREE.Vector3().addVectors(point, normal.clone().multiplyScalar(-halfWidth)));
                }

                return (
                    <React.Fragment key={i}>
                        <Road points={path.points} width={path.width} />
                        <Line
                            points={curvePoints}
                            color={roadMarkingColor}
                            lineWidth={2}
                            dashed
                            dashSize={1}
                            gapSize={1}
                        />
                        <Line
                            points={leftEdgePoints}
                            color={roadMarkingColor}
                            lineWidth={1}
                            transparent
                            opacity={0.1}
                        />
                        <Line
                            points={rightEdgePoints}
                            color={roadMarkingColor}
                            lineWidth={1}
                            transparent
                            opacity={0.1}
                        />
                    </React.Fragment>
                );
            })}
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
  const [lightState, setLightState] = useState(0);
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
        if (parsed.state && parsed.state.lights !== undefined) {
          setLightState(parsed.state.lights);
        }
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
      <Canvas camera={{ position: [0, 80, 20], fov: 60 }}>
        <fog attach="fog" args={['#000011', 50, 250]} />
        <ambientLight intensity={0.4} />
        <directionalLight 
          color="#00ffff"
          position={[0, 100, 0]} 
          intensity={1.0} 
        />
        <pointLight color="#ff00ff" position={[-40, 20, -40]} intensity={2.0} distance={150} />

        <Grid infiniteGrid cellSize={1} sectionSize={10} sectionColor={"#0044cc"} fadeDistance={250} fadeStrength={1.5} />
        
        <Intersection />
        {state && state.agents && state.agents.map(agent => <Vehicle key={agent.id} agent={agent} />)}
        
        <TrafficLight position={[0, 0, 0]} state={lightState} />
        <TrafficLight position={[-25, 0, 0]} state={lightState} />
        <TrafficLight position={[20, 0, 0]} state={lightState} />

        <EffectComposer>
          <Bloom intensity={0.9} luminanceThreshold={0.2} luminanceSmoothing={0.8} />
        </EffectComposer>
        <OrbitControls maxDistance={150} minDistance={20} target={[0, 0, 0]} />
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