
import { useState, useEffect, useRef } from 'react';
import { Canvas, useThree } from '@react-three/fiber';
import { OrbitControls, Stars, Grid } from '@react-three/drei';
import { EffectComposer, Bloom } from '@react-three/postprocessing';
import { Text } from '@geist-ui/core';

import { Academy } from '../libs/ml-agents/Academy';
import { Ball3DAgent } from '../libs/ml-agents/agents/Ball3DAgent';
import { EnvironmentParametersChannel } from '../libs/ml-agents/side-channels/EnvironmentParametersChannel';
import { EngineConfigurationChannel } from '../libs/ml-agents/side-channels/EngineConfigurationChannel';
import { StatsSideChannel, StatsAggregationMethod } from '../libs/ml-agents/side-channels/StatsSideChannel';
import { CameraSensor } from '../libs/ml-agents/sensors/CameraSensor';

import HomeButton from '../components/HomeButton.jsx';

// Physics constants
const MAX_TILT = Math.PI / 7; // ≈25°
const TILT_DELTA = Math.PI / 60; // small increment
const DT = 0.02;

// Visual Component for a single platform
function PlatformAndBall({ platformObj }) {
  // We render based on the mutable platformObj state
  return (
    <group 
        position={[platformObj.idx * 8, 0, 0]} 
        rotation={[platformObj.rotX, 0, platformObj.rotZ]}
    >
      <mesh>
        <boxGeometry args={[6, 0.5, 6]} />
        <meshStandardMaterial color="#3da8ff" emissive="#3da8ff" emissiveIntensity={0.5} toneMapped={false} />
      </mesh>
      <mesh position={[platformObj.ballX, 0.75, platformObj.ballZ]}>
        <sphereGeometry args={[0.5, 32, 32]} />
        <meshStandardMaterial color="#dddddd" emissive="#ffffff" emissiveIntensity={0.8} />
      </mesh>
    </group>
  );
}

// Controller component inside Canvas to access GL context
function MigrationController({ onSetAgents }) {
    const { gl } = useThree();
    const academyRef = useRef(null);
    const envParamsRef = useRef(new EnvironmentParametersChannel());
    const engineConfigRef = useRef(new EngineConfigurationChannel());
    const statsRef = useRef(new StatsSideChannel());
    const gravityRef = useRef(9.81);
    const timeScaleRef = useRef(1.0); // TimeScale
    const [, setTick] = useState(0);

    // One-time Setup
    useEffect(() => {
        const ac = new Academy("ws://localhost:8000/ws/mlagents");
        academyRef.current = ac;

        // 1. Setup Side Channels
        ac.registerSideChannel(envParamsRef.current);
        envParamsRef.current.registerCallback((key, value) => {
            if (key === "gravity") {
                gravityRef.current = value;
                for (const a of ac.agents.values()) {
                    a.gravity = value;
                }
            }
        });
        
        ac.registerSideChannel(engineConfigRef.current);
        engineConfigRef.current.registerCallback((val) => {
            timeScaleRef.current = val;
        });
        
        ac.registerSideChannel(statsRef.current);

        // 2. Create Agents
        const newAgents = [];
        for (let i = 0; i < 3; i++) {
            const pObj = { 
                idx: i,
                rotX: 0, rotZ: 0, 
                ballX: 0, ballZ: 0, 
                velX: 0, velZ: 0 
            };
            const agent = new Ball3DAgent(pObj);
            ac.addAgent(agent);
            newAgents.push(agent);
            
            // Attach CameraSensor to ALL Agents (Uniform Behavior Spec)
             agent.addVisualSensor(new CameraSensor(128, 128, () => {
                 // Capture Canvas
                 // Note: preserveDrawingBuffer: true might be needed on Canvas
                 // toDataURL returns: "data:image/png;base64,....."
                 const dataUrl = gl.domElement.toDataURL("image/png");
                 // Strip prefix
                 return dataUrl.replace(/^data:image\/(png|jpg);base64,/, "");
             }));
        }
        onSetAgents(newAgents); // update parent state for rendering

        // 3. Connect (Now that sensors are attached)
        // Wait a small tick to ensure GL is ready? existing useEffect is fine.
        ac.connect();

        // 4. Physics Loop
        const interval = setInterval(() => {
            // Physics Update
             newAgents.forEach(agent => {
                 const p = agent.platformObj;
                 const a = agent.latestAction || [0,0];
                  // Move tilt
                 p.rotX += a[0] * TILT_DELTA;
                 p.rotZ += a[1] * TILT_DELTA;
                 
                 // Clamp
                 if (p.rotX > MAX_TILT) p.rotX = MAX_TILT;
                 if (p.rotX < -MAX_TILT) p.rotX = -MAX_TILT;
                 if (p.rotZ > MAX_TILT) p.rotZ = MAX_TILT;
                 if (p.rotZ < -MAX_TILT) p.rotZ = -MAX_TILT;
                 
                 // Physics
                 const ts = timeScaleRef.current;
                 const dt = DT * ts;
                 
                 const G = gravityRef.current; // Dynamic gravity
                 p.velX += G * Math.sin(p.rotZ) * dt;
                 p.velZ -= G * Math.sin(p.rotX) * dt;
                 p.ballX += p.velX * dt;
                 p.ballZ += p.velZ * dt;
                 
                 // Damping
                 p.velX *= 0.99;
                 p.velZ *= 0.99;
                 
                 // Reset
                 if (p.ballX*p.ballX + p.ballZ*p.ballZ > 10) {
                     agent.setReward(-1);
                     agent.done = true;
                     agent.reset(); 
                     // Simple visual reset for demo
                     p.rotX = 0; p.rotZ = 0; p.ballX = 0; p.ballZ = 0; p.velX = 0; p.velZ = 0;
                     agent.onEnvironmentReset();
                     
                     // Send Stat
                     if (statsRef.current) {
                         statsRef.current.addStat("EpisodeLength", agent.stepCount, StatsAggregationMethod.AVERAGE);
                     }
                 } else {
                     agent.addReward(0.1);
                 }
             });
             setTick(t => t+1);
        }, 33); // 30fps

        return () => {
            if (ac.ws) ac.ws.close();
            clearInterval(interval);
        };
    }, [gl, onSetAgents]); // Dependencies

    return null;
}

export default function Ball3DMigration() {
  const [agents, setAgents] = useState([]);

  return (
    <div style={{ width: '100vw', height: '100vh', background: '#000011' }}>
      <Canvas 
        gl={{ preserveDrawingBuffer: true }}
        camera={{ position: [0, 15, 25], fov: 50 }}
      >
        <MigrationController onSetAgents={setAgents} />
        
        <ambientLight intensity={0.5} />
        <directionalLight position={[10, 20, 10]} intensity={1} />
        <Stars />
        <Grid position={[0, -5, 0]} args={[50, 50]} />
        
        {agents.map(agent => (
            <PlatformAndBall key={agent.id} platformObj={agent.platformObj} />
        ))}

        <OrbitControls />
        <EffectComposer>
             <Bloom intensity={1} />
        </EffectComposer>
      </Canvas>
      
      <div style={{ position: 'absolute', top: 20, left: 20, color: 'white' }}>
        <HomeButton />
        <Text h2>ML-Agents Migration: {agents.length} Agents</Text>
        <Text p>Run `python api/train_migration_demo.py` to drive this.</Text>
        <Text p small>Visual Observations active on Agent 0.</Text>
      </div>
    </div>
  );
}
