import React, { useState, useEffect, useRef, useMemo } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Grid, Text as DreiText, Line } from '@react-three/drei';
import { Button, Text, Card, Code } from '@geist-ui/core';
import { Link } from 'react-router-dom';
import * as THREE from 'three';
import config from '../config.js';
import { useResponsive } from '../hooks/useResponsive.js';
import { EffectComposer, Bloom } from '@react-three/postprocessing';
import InfoPanel from '../components/InfoPanel.jsx';
import ModelInfoPanel from '../components/ModelInfoPanel.jsx';
import HomeButton from '../components/HomeButton.jsx';

const WS_URL = `${config.WS_BASE_URL}/ws/fish`;

const Fish = ({ agent, gridSize }) => {
  const { pos, id, energy, velocity } = agent;
  const groupRef = useRef();
  const targetPositionRef = useRef([0, 0, 0]);
  const currentPositionRef = useRef([0, 0, 0]);

  const offsetX = gridSize ? gridSize[0] / 2 : 0;
  const offsetZ = gridSize ? gridSize[2] / 2 : 0;

  // Update target position when agent position changes
  const targetPosition = [pos[0] - offsetX, pos[1], pos[2] - offsetZ];
  targetPositionRef.current = targetPosition;

  // Initialize current position to target position on first render
  if (currentPositionRef.current[0] === 0 && currentPositionRef.current[1] === 0 && currentPositionRef.current[2] === 0) {
    currentPositionRef.current = [...targetPosition];
  }

  const energyColor = useMemo(() => {
    const blue = new THREE.Color("#55aaff");
    const white = new THREE.Color("#ffffff");
    const alpha = Math.max(0, Math.min(1, energy / 100));
    return blue.clone().lerp(white, alpha);
  }, [energy]);

  useFrame((state, delta) => {
    if (groupRef.current) {
      // Lerp position smoothly
      const lerpFactor = Math.min(1, delta * 2); // Slower, smoother lerping
      currentPositionRef.current[0] += (targetPositionRef.current[0] - currentPositionRef.current[0]) * lerpFactor;
      currentPositionRef.current[1] += (targetPositionRef.current[1] - currentPositionRef.current[1]) * lerpFactor;
      currentPositionRef.current[2] += (targetPositionRef.current[2] - currentPositionRef.current[2]) * lerpFactor;
      
      groupRef.current.position.set(...currentPositionRef.current);

      // Handle rotation based on velocity
      if (velocity && new THREE.Vector3(...velocity).lengthSq() > 0.001) {
        const targetRotationPosition = new THREE.Vector3().addVectors(
          groupRef.current.position,
          new THREE.Vector3(velocity[0], velocity[1], velocity[2])
        );
        groupRef.current.lookAt(targetRotationPosition);
      }
    }
  });

  return (
    <group ref={groupRef}>
      {/* Head cone - pointing forward */}
      <mesh position={[0, 0, 0.9]} rotation={[Math.PI / 2, 0, 0]}>
        <coneGeometry args={[0.5, 0.8, 8]} />
        <meshPhongMaterial color={energyColor} emissive={energyColor} emissiveIntensity={energy / 100} />
      </mesh>
      
      {/* Body cone - pointing backward */}
      <mesh position={[0, 0, 0]} rotation={[-Math.PI / 2, 0, 0]}>
        <coneGeometry args={[0.5, 1, 8]} />
        <meshPhongMaterial color={energyColor} emissive={energyColor} emissiveIntensity={energy / 100} wireframe={true} />
      </mesh>
      
      {/* Tail cone - small, at the back */}
      <mesh position={[0, 0, -0.8]} rotation={[Math.PI / 2, 0, 0]}>
        <coneGeometry args={[0.3, 0.4, 6]} wireframe={true} />
        <meshPhongMaterial color={energyColor} emissive={energyColor} emissiveIntensity={energy / 100} wireframe={true} />
      </mesh>
      
      {/* <DreiText position={[0, 0.8, 0]} fontSize={0.5} color="white" anchorX="center" anchorY="middle">
        {id}
      </DreiText> */}
    </group>
  );
};

const Shark = ({ agent, gridSize }) => {
    const { pos, color, velocity } = agent;
    const groupRef = useRef();
    const targetPositionRef = useRef([0, 0, 0]);
    const currentPositionRef = useRef([0, 0, 0]);
    
    const offsetX = gridSize ? gridSize[0] / 2 : 0;
    const offsetZ = gridSize ? gridSize[2] / 2 : 0;

    // Update target position when agent position changes
    const targetPosition = [pos[0] - offsetX, pos[1], pos[2] - offsetZ];
    targetPositionRef.current = targetPosition;

    // Initialize current position to target position on first render
    if (currentPositionRef.current[0] === 0 && currentPositionRef.current[1] === 0 && currentPositionRef.current[2] === 0) {
        currentPositionRef.current = [...targetPosition];
    }

    useFrame((state, delta) => {
        if (groupRef.current) {
            // Lerp position smoothly
            const lerpFactor = Math.min(1, delta * 3); // Slower, smoother lerping (slightly slower than fish)
            currentPositionRef.current[0] += (targetPositionRef.current[0] - currentPositionRef.current[0]) * lerpFactor;
            currentPositionRef.current[1] += (targetPositionRef.current[1] - currentPositionRef.current[1]) * lerpFactor;
            currentPositionRef.current[2] += (targetPositionRef.current[2] - currentPositionRef.current[2]) * lerpFactor;
            
            groupRef.current.position.set(...currentPositionRef.current);

            // Handle rotation based on velocity
            if (velocity && new THREE.Vector3(...velocity).lengthSq() > 0.001) {
                const targetRotationPosition = new THREE.Vector3().addVectors(
                    groupRef.current.position,
                    new THREE.Vector3(velocity[0], velocity[1], velocity[2])
                );
                groupRef.current.lookAt(targetRotationPosition);
                // Adjust for shark's geometry - head points in +X direction, so rotate 90 degrees around Y
                groupRef.current.rotateY(Math.PI / 2);
            }
        }
    });

    return (
        <group ref={groupRef}>
            <mesh>
                <boxGeometry args={[4, 1.5, 2]} />
                <meshPhongMaterial color={"white"} emissive={"white"} wireframe={true} emissiveIntensity={0.2} />
            </mesh>
            <mesh position={[2, 0.5, 0]}>
                <boxGeometry args={[0.5, 0.5, 2.5]} />
                <meshPhongMaterial color={"white"} emissive={"white"} wireframe={true} emissiveIntensity={0.2} />
            </mesh>
        </group>
    )
}

const Scenery = ({ grid, resourceTypes, gridSize }) => {
  const sceneryMeshes = useMemo(() => {
    const meshes = [];
    if (!grid || !resourceTypes || !gridSize) return meshes;
    const resourceKeys = Object.keys(resourceTypes);
    
    const offsetX = gridSize[0] / 2;
    const offsetZ = gridSize[2] / 2;

    grid.forEach((plane, x) => {
      plane.forEach((row, y) => {
        row.forEach((cell, z) => {
          if (cell > 0) {
            const resourceName = resourceKeys[cell - 1]; // cell is 1-based index from python
            if (resourceName === 'sand') return; // Don't render the sand floor
            
            const resource = resourceTypes[resourceName];
            if (resource && resource.color) {
                 const isFood = resourceName === 'food';
                 meshes.push(
                    <mesh key={`${x}-${y}-${z}`} position={[x - offsetX, y, z - offsetZ]}>
                        <boxGeometry args={isFood ? [0.3, 0.3, 0.3] : [1, 1, 1]} />
                        <meshPhongMaterial 
                            color={new THREE.Color(...resource.color)} 
                            emissive={isFood ? new THREE.Color(...resource.color) : new THREE.Color(0,0,0)}
                            emissiveIntensity={1.5}
                        />
                    </mesh>
                 );
             }
          }
        });
      });
    });
    return meshes;
  }, [grid, resourceTypes, gridSize]);

  return <group>{sceneryMeshes}</group>;
};

const StaticScenery = ({ gridSize }) => {
    const staticMeshes = useMemo(() => {
        if (!gridSize) return null;

        const meshes = [];
        const count = 128; // Add number of scenery objects
        const offsetX = gridSize[0] / 2;
        const offsetZ = gridSize[2] / 2;

        for (let i = 0; i < count; i++) {
            const x = Math.random() * gridSize[0];
            const z = Math.random() * gridSize[2];
            const isRock = Math.random() > 0.5;

            if (isRock) {
                const height = Math.random() * 8 + 1;
                meshes.push(
                    <mesh key={`rock-${i}`} position={[x - offsetX, height / 2, z - offsetZ]}>
                        <boxGeometry args={[1, height, 1]} />
                        <meshPhongMaterial color="#8800ff" emissive="#440088" emissiveIntensity={0.5} wireframe={true} />
                    </mesh>
                );
            } else {
                const height = Math.random() * 4 + 1;
                const coralColor = Math.random() > 0.5 ? "#ff33cc" : "#33ccff";
                meshes.push(
                    <mesh key={`coral-${i}`} position={[x - offsetX, height / 2, z - offsetZ]}>
                        <boxGeometry args={[0.5, height, 0.5]} />
                        <meshPhongMaterial color={coralColor} emissive={coralColor} emissiveIntensity={2} wireframe={true} />
                    </mesh>
                );
            }
        }
        return meshes;
    }, [JSON.stringify(gridSize)]);

    return <group>{staticMeshes}</group>;
};

const WaterParticles = ({ gridSize }) => {
  const particlesRef = useRef();
  const lifetimesRef = useRef([]);
  const velocitiesRef = useRef([]);
  const colorsRef = useRef(new Float32Array());

  const PARTICLE_COUNT = 4000;
  const MAX_LIFE = 800; // frames

  // Simple water current parameters (procedural)
  const currentParams = useMemo(() => ({
    C1: 8.0,
    waveFreq: 0.008,
    waveMag: 2.0,
    waveFreq2: 0.012,
    waveMag2: 1.5,
    flowStrengthX: 0.5,
    flowStrengthZ: 0.3,
  }), []);

  const particles = useMemo(() => {
    if (!gridSize) return new Float32Array();
    
    const temp = [];
    const colors = [];
    lifetimesRef.current = [];
    velocitiesRef.current = [];
    
    const offsetX = gridSize[0] / 2;
    const offsetZ = gridSize[2] / 2;
    
    for (let i = 0; i < PARTICLE_COUNT; i++) {
      const x = (Math.random() - 0.5) * gridSize[0] * 1.2;
      const y = Math.random() * gridSize[1] * 0.8;
      const z = (Math.random() - 0.5) * gridSize[2] * 1.2;
      temp.push(x, y, z);
      colors.push(0.2, 0.6, 1); // Blue-ish water particles
      lifetimesRef.current.push(Math.random() * MAX_LIFE);
      velocitiesRef.current.push(0, 0, 0); // Initial velocity [vx, vy, vz]
    }
    colorsRef.current = new Float32Array(colors);
    return new Float32Array(temp);
  }, [gridSize]);

  useFrame((state, delta) => {
    if (!particlesRef.current || !gridSize) return;

    const positions = particlesRef.current.geometry.attributes.position.array;
    const colors = particlesRef.current.geometry.attributes.color.array;
    const { C1, waveFreq, waveMag, waveFreq2, waveMag2, flowStrengthX, flowStrengthZ } = currentParams;

    // Use delta time to make movement frame-rate independent
    const timeStep = Math.min(delta, 1/30); // Cap at 30fps equivalent to prevent huge jumps

    for (let i = 0; i < PARTICLE_COUNT; i++) {
      const i3 = i * 3;

      lifetimesRef.current[i] += 1;

      const three_x = positions[i3];
      const three_y = positions[i3 + 1];
      const three_z = positions[i3 + 2];

      // Convert to world coordinates for current calculation
      const world_x = three_x;
      const world_z = three_z;

      // Calculate water currents using sine waves
      const current1 = Math.sin(world_x * waveFreq * 2 * Math.PI) * Math.cos(world_z * waveFreq * 2 * Math.PI) * waveMag;
      const current2 = Math.sin(world_x * waveFreq2 * 2 * Math.PI / 1.5) * Math.cos(world_z * waveFreq2 * 2 * Math.PI / 1.5) * waveMag2;
      const total_current_x = current1 + current2;
      
      const current_z_1 = Math.cos(world_x * waveFreq * 2 * Math.PI) * Math.sin(world_z * waveFreq * 2 * Math.PI) * waveMag * 0.7;
      const current_z_2 = Math.cos(world_x * waveFreq2 * 2 * Math.PI / 1.5) * Math.sin(world_z * waveFreq2 * 2 * Math.PI / 1.5) * waveMag2 * 0.7;
      const total_current_z = current_z_1 + current_z_2;

      // Calculate target velocity based on currents
      const targetVelX = (flowStrengthX + total_current_x * 0.08) * 0.8;
      const targetVelY = total_current_x * 0.03;
      const targetVelZ = (flowStrengthZ + total_current_z * 0.08) * 0.8;

      // Smoothly interpolate velocity towards target (momentum/inertia)
      const lerpFactor = timeStep * 2.0;
      velocitiesRef.current[i * 3] += (targetVelX - velocitiesRef.current[i * 3]) * lerpFactor;
      velocitiesRef.current[i * 3 + 1] += (targetVelY - velocitiesRef.current[i * 3 + 1]) * lerpFactor;
      velocitiesRef.current[i * 3 + 2] += (targetVelZ - velocitiesRef.current[i * 3 + 2]) * lerpFactor;

      // Apply smoothed velocity to position
      positions[i3] += velocitiesRef.current[i * 3] * timeStep;
      positions[i3 + 1] += velocitiesRef.current[i * 3 + 1] * timeStep;
      positions[i3 + 2] += velocitiesRef.current[i * 3 + 2] * timeStep;

      // Color particles based on current strength (update less frequently)
      if (lifetimesRef.current[i] % 5 === 0) { // Update color every 5 frames
        const currentStrength = Math.sqrt(total_current_x * total_current_x + total_current_z * total_current_z);
        const normalizedStrength = Math.min(currentStrength / (waveMag + waveMag2), 1);
        
        const color = new THREE.Color();
        if (normalizedStrength > 0.3) {
          // Strong currents - more cyan/turquoise
          color.setRGB(0.1, 0.8, 1.0);
        } else {
          // Weak currents - deeper blue
          color.setRGB(0.2, 0.4, 0.8);
        }
        
        colors[i3] = color.r;
        colors[i3 + 1] = color.g;
        colors[i3 + 2] = color.b;
      }

      // Check bounds and reset if needed
      const outOfBounds =
        positions[i3] > gridSize[0] * 0.6 || positions[i3] < -gridSize[0] * 0.6 ||
        positions[i3 + 2] > gridSize[2] * 0.6 || positions[i3 + 2] < -gridSize[2] * 0.6 ||
        positions[i3 + 1] > gridSize[1] || positions[i3 + 1] < 0;

      if (outOfBounds || lifetimesRef.current[i] > MAX_LIFE) {
        // Reset particle to random position
        positions[i3] = (Math.random() - 0.5) * gridSize[0] * 1.2;
        positions[i3 + 1] = Math.random() * gridSize[1] * 0.8;
        positions[i3 + 2] = (Math.random() - 0.5) * gridSize[2] * 1.2;
        lifetimesRef.current[i] = 0;
        // Reset velocity when respawning
        velocitiesRef.current[i * 3] = 0;
        velocitiesRef.current[i * 3 + 1] = 0;
        velocitiesRef.current[i * 3 + 2] = 0;
      }
    }

    particlesRef.current.geometry.attributes.position.needsUpdate = true;
    // Only update colors when they actually change
    if (Math.floor(state.clock.elapsedTime * 60) % 5 === 0) {
      particlesRef.current.geometry.attributes.color.needsUpdate = true;
    }
  });

  if (!gridSize) return null;

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
      <pointsMaterial size={0.3} transparent opacity={0.6} vertexColors />
    </points>
  );
};

const WaterCurrents = ({ gridSize }) => {
  const { curves, arrows } = useMemo(() => {
    if (!gridSize) return { curves: [], arrows: [] };
    
    const curveData = [];
    const arrowData = [];
    const num_curves_x = 8;
    const num_curves_z = 8;
    const bounds_x = gridSize[0] * 0.5;
    const bounds_z = gridSize[2] * 0.5;

    // Simple water current parameters
    const waveFreq = 0.008;
    const waveMag = 2.0;
    const waveFreq2 = 0.012;
    const waveMag2 = 1.5;
    const flowStrengthX = 0.5;
    const flowStrengthZ = 0.3;

    for (let i = 0; i < num_curves_x; i++) {
      for (let j = 0; j < num_curves_z; j++) {
        const startX = -bounds_x + (i / (num_curves_x - 1)) * bounds_x * 2;
        const startZ = -bounds_z + (j / (num_curves_z - 1)) * bounds_z * 2;

        const points = [];
        let currentX = startX;
        let currentY = gridSize[1] * 0.3; // Middle depth
        let currentZ = startZ;

        const num_steps = 30;
        const dt = 1.5;

        for (let step = 0; step < num_steps; step++) {
          const currentPoint = new THREE.Vector3(currentX, currentY, currentZ);
          points.push(currentPoint);

          // Calculate water currents
          const current1 = Math.sin(currentX * waveFreq * 2 * Math.PI) * Math.cos(currentZ * waveFreq * 2 * Math.PI) * waveMag;
          const current2 = Math.sin(currentX * waveFreq2 * 2 * Math.PI / 1.5) * Math.cos(currentZ * waveFreq2 * 2 * Math.PI / 1.5) * waveMag2;
          const total_current_x = current1 + current2;
          
          const current_z_1 = Math.cos(currentX * waveFreq * 2 * Math.PI) * Math.sin(currentZ * waveFreq * 2 * Math.PI) * waveMag * 0.7;
          const current_z_2 = Math.cos(currentX * waveFreq2 * 2 * Math.PI / 1.5) * Math.sin(currentZ * waveFreq2 * 2 * Math.PI / 1.5) * waveMag2 * 0.7;
          const total_current_z = current_z_1 + current_z_2;

          const flow_x = flowStrengthX + total_current_x * 0.2;
          const flow_z = flowStrengthZ + total_current_z * 0.2;
          const flow_y = total_current_x * 0.1; // Slight vertical movement

          currentX += flow_x * dt;
          currentY += flow_y * dt;
          currentZ += flow_z * dt;

          // Keep within reasonable Y bounds
          if (currentY > gridSize[1] * 0.9) currentY = gridSize[1] * 0.9;
          if (currentY < gridSize[1] * 0.1) currentY = gridSize[1] * 0.1;

          // Add arrows periodically
          if (step > 0 && step % 10 === 0) {
            const direction = new THREE.Vector3(flow_x, flow_y, flow_z).normalize().multiplyScalar(3);
            const currentStrength = Math.sqrt(total_current_x * total_current_x + total_current_z * total_current_z);
            const color = currentStrength > 1.5 ? "#00ffff" : "#4488ff";
            arrowData.push({ key: `${i}-${j}-${step}`, origin: currentPoint, direction, color });
          }

          // Break if out of bounds
          if (Math.abs(currentX) > bounds_x * 1.5 || Math.abs(currentZ) > bounds_z * 1.5) break;
        }

        // Only draw curves with significant current strength
        const avgCurrentStrength = Math.sin(startX * waveFreq * 2 * Math.PI) * Math.cos(startZ * waveFreq * 2 * Math.PI) * waveMag;
        if (points.length > 1 && Math.abs(avgCurrentStrength) > 0.8) {
          const currentStrength = Math.abs(avgCurrentStrength);
          const normalizedStrength = Math.min(currentStrength / waveMag, 1);
          
          let color;
          if (normalizedStrength > 0.6) {
            color = new THREE.Color("#00ffff"); // Strong currents - cyan
          } else {
            color = new THREE.Color("#4488ff"); // Weaker currents - blue
          }
          
          curveData.push({ key: `${i}-${j}`, points, color });
        }
      }
    }
    return { curves: curveData, arrows: arrowData };
  }, [gridSize]);

  if (!gridSize) return null;

  return (
    <group>
      {curves.map((c) => (
        <Line key={c.key} points={c.points} color={c.color} lineWidth={1} transparent opacity={0.4} dashSize={0.2} gapSize={1} dashed={true} />
      ))}
      {arrows.map((a) => (
        <Arrow key={a.key} origin={a.origin} direction={a.direction} color={a.color}  />
      ))}
    </group>
  );
};

const Arrow = ({ origin, direction, color = "#00ffff" }) => {
  const ref = useRef();
  React.useLayoutEffect(() => {
    if (ref.current) {
      ref.current.lookAt(new THREE.Vector3().addVectors(origin, direction));
    }
  }, [origin, direction]);

  return (
    <group ref={ref} position={origin}>
      <mesh rotation={[Math.PI / 2, 0, 0]}>
        <coneGeometry args={[0.8, 2, 6]} />
        <meshStandardMaterial color={color} emissive={color} emissiveIntensity={0.6} toneMapped={false} transparent opacity={0.01} wireframe={true} />
      </mesh>
    </group>
  );
};


const EnergyPanel = ({ agents }) => {
    if (!agents) return null;
    return (
        <div style={{
            width: '100%',
            maxHeight: '250px', 
            overflowY: 'scroll', 
            background: 'rgba(0,0,0,0.5)', 
            color: '#fff',
            border: '1px solid #4488ff',
            borderRadius: '0px',
        }}>
            <Text h4 style={{ padding: '4px 8px 0', margin: 0, position: 'sticky', top: 0, background: 'rgba(0,0,0,0.5)', color: '#fff' }}>Fish Energy</Text>
            <div style={{padding: '4px 8px'}}>
              {agents.map(agent => (
                  <div key={agent.id} style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px' }}>
                      <Text span>Fish {agent.id}:</Text>
                      <Text span>
                        <progress value={agent.energy} max="64" style={{width: '100px'}} />
                        <span style={{marginLeft: '8px'}}>{agent.energy.toFixed(0)}</span>
                      </Text>
                  </div>
              ))}
            </div>
        </div>
    );
};


export default function FishExample() {
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
    ws.onopen = () => addLog('Fish WS opened');
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
        addLog('Training complete! Fish are now using the trained policy.');
      }
      if (parsed.type === 'info') {
          addLog(`INFO: ${parsed.message}`);
      }
    };
    ws.onclose = () => addLog('Fish WS closed');
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
      <Canvas camera={{ position: [-100, 100, 100], fov: 60 }}>
        <fog attach="fog" args={['#000011', 50, 250]} />
        <ambientLight intensity={0.4} />
        <directionalLight 
          color="#55aaff"
          position={[0, 100, 0]} 
          intensity={1.0} 
        />
        <pointLight color="#ff88ff" position={[-40, 20, -40]} intensity={2.0} distance={150} />

        <Grid infiniteGrid cellSize={1} sectionSize={10} sectionColor={"#0044cc"} fadeDistance={250} fadeStrength={1.5} />
        
        {state && state.agents && state.agents.map(agent => <Fish key={agent.id} agent={agent} gridSize={gridSize} />)}
        {state && state.shark && <Shark agent={state.shark} gridSize={gridSize} />}
        {state && <Scenery grid={state.grid} resourceTypes={state.resource_types} gridSize={gridSize} />}
        {gridSize && <StaticScenery gridSize={gridSize} />}
        {gridSize && <WaterParticles gridSize={gridSize} />}
        {gridSize && <WaterCurrents gridSize={gridSize} />}
        
        <EffectComposer>
          <Bloom intensity={0.9} luminanceThreshold={0.2} luminanceSmoothing={0.8} />
        </EffectComposer>
        <OrbitControls maxDistance={140} minDistance={10} target={[0, 64, 0]} />
      </Canvas>

      <div style={{ position: 'absolute', top: 10, left: 10, zIndex: 1, color: '#fff' }}>
        <HomeButton />
          <Text h1 style={{ margin: '12px 0 12px 0', color: '#fff', fontSize: isMobile ? '1.2rem' : '2rem', fontFamily: 'monospace', textTransform: 'uppercase', letterSpacing: '0.15em' }}>
            Fish
          </Text>
        <div style={{ display: 'flex', gap: '8px' }}>
          <Button auto type="secondary" style={{ borderRadius: 0, textTransform: 'uppercase', letterSpacing: '0.1em', border: '1px solid #fff' }} disabled={training || trained} onClick={startTraining}>Train</Button>
          <Button auto type="success" style={{ borderRadius: 0, textTransform: 'uppercase', letterSpacing: '0.1em', border: '1px solid #fff' }} disabled={!trained || running} onClick={startRun}>Run</Button>
          {/* <Button auto type="error" onClick={reset}>Reset </Button> */}
        </div>
      </div>
      
      {/* <div style={{
        position: 'absolute',
        top: '10px',
        right: '10px',
        display: 'flex',
        flexDirection: 'column',
        gap: '10px',
        width: isMobile ? 'calc(100% - 20px)' : '45%',
        maxWidth: '420px',
        height: 'calc(100vh - 20px)',
      }}>
        {state && state.agents && <EnergyPanel agents={state.agents} />}
      </div> */}
      
      <InfoPanel logs={logs} chartState={chartState} />
      <ModelInfoPanel modelInfo={modelInfo} />

    </div>
  );
} 