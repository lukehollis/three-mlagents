import React, { useState, useEffect, useRef, useMemo } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Grid, Text as DreiText } from '@react-three/drei';
import { Button, Text, Card, Code } from '@geist-ui/core';
import { Link } from 'react-router-dom';
import * as THREE from 'three';
import config from '../config.js';
import ButtonForkOnGithub from '../components/ButtonForkOnGithub.jsx';
import { useResponsive } from '../hooks/useResponsive.js';

const WS_URL = `${config.WS_BASE_URL}/ws/minefarm`;

const Agent = ({ agent }) => {
  const { pos, color, id } = agent;
  const groupRef = useRef();

  return (
    <group ref={groupRef} position={[pos[0] - 19.5, 1, pos[1] - 19.5]}>
      <mesh>
        <boxGeometry args={[0.8, 0.8, 0.8]} />
        <meshStandardMaterial color={new THREE.Color(...color)} />
      </mesh>
      <DreiText position={[0, 0.8, 0]} fontSize={0.5} color="white" anchorX="center" anchorY="middle">
        {id}
      </DreiText>
    </group>
  );
};

const Resources = ({ grid, resourceTypes }) => {
  const resourceMeshes = useMemo(() => {
    const meshes = [];
    if (!grid || !resourceTypes) return meshes;
    const resourceList = Object.values(resourceTypes);
    
    grid.forEach((row, x) => {
      row.forEach((cell, y) => {
        if (cell > 0) {
          const resource = resourceList[cell - 1];
          meshes.push(
            <mesh key={`${x}-${y}`} position={[x - 19.5, 0.5, y - 19.5]}>
              <boxGeometry args={[1, 1, 1]} />
              <meshStandardMaterial color={new THREE.Color(...resource.color)} />
            </mesh>
          );
        }
      });
    });
    return meshes;
  }, [grid, resourceTypes]);

  return <group>{resourceMeshes}</group>;
};

const ScorePanel = ({ agents }) => {
    if (!agents) return null;
    return (
        <Card style={{
            position: 'absolute', bottom: '10px', right: '10px', width: '300px',
            maxHeight: '40vh', overflowY: 'auto', background: 'rgba(0,0,0,0.5)', color: '#fff'
        }}>
            <Text h4>Agent Scores</Text>
            {agents.map(agent => (
                <div key={agent.id} style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px' }}>
                    <Text span>Agent {agent.id}:</Text>
                    <Text span>
                        {Object.entries(agent.inventory).map(([res, val]) => `${res.charAt(0).toUpperCase()}: ${val}`).join(', ')}
                    </Text>
                </div>
            ))}
        </Card>
    );
};

const MessagePanel = ({ messages }) => {
    if (!messages) return null;
    return (
        <Card style={{
            position: 'absolute', bottom: '10px', left: '10px', width: '300px',
            maxHeight: '40vh', overflowY: 'auto', background: 'rgba(0,0,0,0.5)', color: '#fff'
        }}>
            <Text h4>Agent Comms</Text>
            {messages.slice().reverse().map((msg, i) => (
                <Text p key={i} style={{ margin: 0, fontSize: '12px' }}>
                    <Code>[Step {msg.step}] Agent {msg.agent_id}</Code>: {msg.message}
                </Text>
            ))}
        </Card>
    );
};

export default function MineFarmExample() {
  const [state, setState] = useState(null);
  const [running, setRunning] = useState(false);
  const wsRef = useRef(null);
  const { isMobile } = useResponsive();
  
  useEffect(() => {
    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;
    ws.onopen = () => console.log('MineFarm WS opened');
    ws.onmessage = (ev) => {
      const parsed = JSON.parse(ev.data);
      if (parsed.type === 'state' || parsed.type === 'init' || parsed.type === 'run_step') {
        setState(parsed.state);
      }
    };
    ws.onclose = () => console.log('MineFarm WS closed');
    return () => ws.close();
  }, []);

  const send = (obj) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(obj));
    }
  };

  const startRun = () => {
    if (running) return;
    setRunning(true);
    send({ cmd: 'run' });
  };

  const reset = () => {
    window.location.reload();
  }

  return (
    <div style={{ width: '100vw', height: '100vh', overflow: 'hidden', background: '#000011' }}>
      <Canvas camera={{ position: [20, 40, 40], fov: 60 }}>
        <ambientLight intensity={0.6} />
        <directionalLight position={[50, 50, 50]} intensity={1.0} />
        <Grid infiniteGrid cellSize={1} sectionSize={10} sectionColor={"#4488ff"} fadeDistance={150} />
        
        {state && state.agents.map(agent => <Agent key={agent.id} agent={agent} />)}
        {state && <Resources grid={state.grid} resourceTypes={state.resource_types} />}
        
        <OrbitControls maxDistance={100} minDistance={10}/>
      </Canvas>

      <div style={{ position: 'absolute', top: 10, left: 10, zIndex: 1, color: '#fff' }}>
        <Link to="/"><Button auto size="small">&larr; Home</Button></Link>
        <Text h1 style={{ margin: '12px 0', color: '#fff', fontSize: isMobile ? '1.2rem' : '2rem' }}>Mine Farm</Text>
        <div style={{ display: 'flex', gap: '8px' }}>
          <Button auto type="secondary" disabled={running} onClick={startRun}>Run</Button>
          <Button auto type="error" onClick={reset}>Reset</Button>
        </div>
      </div>
      
      {state && <ScorePanel agents={state.agents} />}
      {state && <MessagePanel messages={state.messages} />}

      <ButtonForkOnGithub position={{ top: '10px', right: '10px' }} />
    </div>
  );
} 