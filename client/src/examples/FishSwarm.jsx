import React, { useState, useEffect, useRef, useMemo } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Grid, Text as DreiText } from '@react-three/drei';
import { Button, Text, Card, Code } from '@geist-ui/core';
import { Link } from 'react-router-dom';
import * as THREE from 'three';
import config from '../config.js';
import { useResponsive } from '../hooks/useResponsive.js';
import { EffectComposer, Bloom } from '@react-three/postprocessing';
import InfoPanel from '../components/InfoPanel.jsx';
import ModelInfoPanel from '../components/ModelInfoPanel.jsx';

const WS_URL = `${config.WS_BASE_URL}/ws/fish_swarm`;

const Fish = ({ agent, gridSize }) => {
  const { pos, color, id } = agent;
  const groupRef = useRef();

  const offsetX = gridSize ? gridSize[0] / 2 : 0;
  const offsetZ = gridSize ? gridSize[2] / 2 : 0;

  return (
    <group ref={groupRef} position={[pos[0] - offsetX, pos[1], pos[2] - offsetZ]}>
      <mesh rotation={[0, 0, Math.PI / 2]}>
        <coneGeometry args={[0.4, 1.2, 8]} />
        <meshPhongMaterial color={new THREE.Color(...color)} emissive={new THREE.Color(...color).multiplyScalar(0.3)} />
      </mesh>
      <DreiText position={[0, 0.8, 0]} fontSize={0.5} color="white" anchorX="center" anchorY="middle">
        {id}
      </DreiText>
    </group>
  );
};

const Scenery = ({ grid, resourceTypes, gridSize }) => {
  const sceneryMeshes = useMemo(() => {
    const meshes = [];
    if (!grid || !resourceTypes || !gridSize) return meshes;
    const resourceList = Object.values(resourceTypes);
    
    const offsetX = gridSize[0] / 2;
    const offsetZ = gridSize[2] / 2;

    grid.forEach((plane, x) => {
      plane.forEach((row, y) => {
        row.forEach((cell, z) => {
          if (cell > 0) {
            const resource = resourceList[cell]; // Note: cell is already index+1, but we have water at 0. Let's adjust server to send names
             if (resource && resource.color) {
                 const resourceName = Object.keys(resourceTypes).find(key => resourceTypes[key] === resource);
                 const isFood = resourceName === 'food';
                 meshes.push(
                    <mesh key={`${x}-${y}-${z}`} position={[x - offsetX, y, z - offsetZ]}>
                        <boxGeometry args={isFood ? [0.3, 0.3, 0.3] : [1, 1, 1]} />
                        <meshPhongMaterial 
                            color={new THREE.Color(...resource.color)} 
                            emissive={isFood ? new THREE.Color(...resource.color) : new THREE.Color(0,0,0)}
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
            borderRadius: '8px',
        }}>
            <Text h4 style={{ padding: '4px 8px 0', margin: 0, position: 'sticky', top: 0, background: 'rgba(0,0,0,0.5)', color: '#fff' }}>Fish Energy</Text>
            <div style={{padding: '4px 8px'}}>
              {agents.map(agent => (
                  <div key={agent.id} style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px' }}>
                      <Text span>Fish {agent.id}:</Text>
                      <Text span>
                        <progress value={agent.energy} max="100" style={{width: '100px'}} />
                        <span style={{marginLeft: '8px'}}>{agent.energy.toFixed(0)}</span>
                      </Text>
                  </div>
              ))}
            </div>
        </div>
    );
};


const MessagePanel = ({ messages }) => {
    if (!messages) return null;
    return (
        <Card style={{
            width: '100%',
            maxHeight: '200px',
            overflowY: 'auto',
            background: 'rgba(0,0,0,0.6)',
            color: '#fff', border: '1px solid #4488ff',
        }}>
            {messages.slice().reverse().map((msg, i) => {
                let content;
                if (msg.recipient_id !== null && msg.recipient_id !== undefined) {
                    content = <Text p style={{ margin: 0 }}><Code>[DM to {msg.recipient_id}]</Code> {msg.message}</Text>;
                } else {
                    content = <Text p style={{ margin: 0 }}><Code>[Broadcast]</Code> {msg.message}</Text>;
                }
                
                return (
                    <div key={i} style={{ marginBottom: '12px', padding: '8px', background: 'rgba(0,100,255,0.1)', borderRadius: '4px', fontSize: '12px' }}>
                        <Text p style={{ margin: 0, fontWeight: 'bold' }}>
                            <Code>[Step {msg.step}] Fish {msg.sender_id}</Code>
                        </Text>
                        {content}
                    </div>
                );
            })}
        </Card>
    );
};


export default function FishSwarmExample() {
  const [state, setState] = useState(null);
  const [running, setRunning] = useState(false);
  const [training, setTraining] = useState(false);
  const [trained, setTrained] = useState(false);
  const [modelInfo, setModelInfo] = useState(null);
  const [error, setError] = useState(null);
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
    ws.onopen = () => addLog('FishSwarm WS opened');
    ws.onmessage = (ev) => {
      addLog(`Received data: ${ev.data.substring(0, 100)}...`);
      try {
        const parsed = JSON.parse(ev.data);
        
        if (parsed.type === 'error') {
          setError(parsed.message);
          addLog(`ERROR: ${parsed.message}`);
          return;
        }

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
      } catch (e) {
        addLog(`Error processing message: ${e}`);
        console.error("Failed to process message: ", e);
      }
    };
    ws.onclose = () => addLog('FishSwarm WS closed');
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

  if (error) {
    return (
        <div style={{ width: '100vw', height: '100vh', display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center', background: '#220000', color: '#ffaaaa' }}>
            <Text h1>A Server Error Occurred</Text>
            <Text p>Could not load the simulation environment.</Text>
            <Code block width="50vw" style={{textAlign: 'left'}}>{error}</Code>
            <Button auto type="error" onClick={reset} style={{marginTop: '20px'}}>Reload Page</Button>
        </div>
    );
  }

  return (
    <div style={{ width: '100vw', height: '100vh', overflow: 'hidden', background: '#000011' }}>
      <Canvas camera={{ position: [-50, 50, 50], fov: 60 }}>
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
        {state && <Scenery grid={state.grid} resourceTypes={state.resource_types} gridSize={gridSize} />}
        
        <EffectComposer>
          <Bloom intensity={1.5} luminanceThreshold={0.2} luminanceSmoothing={0.8} />
        </EffectComposer>
        <OrbitControls maxDistance={400} minDistance={10} />
      </Canvas>

      <div style={{ position: 'absolute', top: 10, left: 10, zIndex: 1, color: '#fff' }}>
        <Link to="/" style={{ fontFamily: 'monospace', color: '#fff', textDecoration: 'underline' }}>
          Home
        </Link>
          <Text h1 style={{ margin: '12px 0 12px 0', color: '#fff', fontSize: isMobile ? '1.2rem' : '2rem' }}>
            Fish Swarm
          </Text>
        <div style={{ display: 'flex', gap: '8px' }}>
          <Button auto type="secondary" disabled={training || trained} onClick={startTraining}>Train</Button>
          <Button auto type="success" disabled={!trained || running} onClick={startRun}>Run</Button>
          <Button auto type="error" onClick={reset}>Reset </Button>
        </div>
      </div>
      
      <div style={{
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
        {state && state.messages && state.messages.length > 0 && <MessagePanel messages={state.messages} />}
      </div>
      
      <InfoPanel logs={logs} chartState={chartState} />
      <ModelInfoPanel modelInfo={modelInfo} />

    </div>
  );
} 