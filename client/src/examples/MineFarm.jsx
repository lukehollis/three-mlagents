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

const WS_URL = `${config.WS_BASE_URL}/ws/minefarm`;


const Agent = ({ agent, gridSize }) => {
  const { pos, color, id } = agent;
  const groupRef = useRef();

  const offsetX = gridSize ? gridSize[0] / 2 : 0;
  const offsetZ = gridSize ? gridSize[2] / 2 : 0;

  return (
    <group ref={groupRef} position={[pos[0] - offsetX, pos[1], pos[2] - offsetZ]}>
      <mesh >
        <boxGeometry args={[0.8, 0.8, 0.8]} />
        <meshPhongMaterial color={new THREE.Color(...color)} />
      </mesh>
      <DreiText position={[0, 0.8, 0]} fontSize={0.5} color="white" anchorX="center" anchorY="middle">
        {id}
      </DreiText>
    </group>
  );
};

const Resources = ({ grid, resourceTypes, gridSize }) => {
  const resourceMeshes = useMemo(() => {
    const meshes = [];
    if (!grid || !resourceTypes || !gridSize) return meshes;
    const resourceList = Object.values(resourceTypes);
    
    const offsetX = gridSize[0] / 2;
    const offsetZ = gridSize[2] / 2;

    grid.forEach((plane, x) => {
      plane.forEach((row, y) => {
        row.forEach((cell, z) => {
          if (cell > 0) {
            const resource = resourceList[cell - 1];
            meshes.push(
              <mesh key={`${x}-${y}-${z}`} position={[x - offsetX, y, z - offsetZ]}>
                <boxGeometry args={[1, 1, 1]} />
                <meshPhongMaterial color={new THREE.Color(...resource.color)} />
              </mesh>
            );
          }
        });
      });
    });
    return meshes;
  }, [grid, resourceTypes, gridSize]);

  return <group>{resourceMeshes}</group>;
};

const ScorePanel = ({ agents }) => {
    const { isMobile } = useResponsive();
    if (!agents) return null;
    return (
        <div style={{
            position: 'absolute', 
            top: '10px', 
            right: '10px', 
            width: isMobile ? 'calc(100% - 20px)' : '45%',
            maxWidth: '420px',
            maxHeight: '120px', 
            overflowY: 'auto', 
            background: 'rgba(0,0,0,0.5)', 
            color: '#fff',
            border: '1px solid #444',
            borderRadius: '8px',
        }}>
            <Text h4 style={{ padding: '4px 8px 0', margin: 0, position: 'sticky', top: 0, background: 'rgba(0,0,0,0.5)', color: '#fff' }}>Scores</Text>

            <div style={{padding: '4px 8px'}}>
              {agents.map(agent => (
                  <div key={agent.id} style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px' }}>
                      <Text span>Agent {agent.id}:</Text>
                      <Text span>
                          {Object.entries(agent.inventory).map(([res, val]) => `${res.charAt(0).toUpperCase()}: ${val}`).join(', ')}
                      </Text>
                  </div>
              ))}
            </div>
        </div>
    );
};

const AgentCommsPanel = ({ logs }) => {
    if (!logs) return null;
    return (
        <Card style={{
            position: 'absolute', bottom: '10px', left: '10px', width: '450px',
            maxHeight: '40vh', overflowY: 'auto', background: 'rgba(0,0,0,0.6)',
            color: '#fff', border: '1px solid #444',
        }}>
            {logs.slice().reverse().map((log, i) => {
                const action = log.response?.action;
                const data = log.response?.data;
                let content;

                if (log.error) {
                    content = <Text p style={{ margin: '4px 0', color: '#ff7777' }}>Error: {log.error}</Text>;
                } else if (action === 'talk') {
                    content = <Text p style={{ margin: 0 }}>says: "{data}"</Text>;
                } else if (action === 'move') {
                    content = <Text p style={{ margin: 0, opacity: 0.7 }}>moves to [{data.join(', ')}]</Text>;
                } else if (action === 'mine') {
                    content = <Text p style={{ margin: 0, opacity: 0.7 }}>mines at [{data.join(', ')}]</Text>;
                } else {
                    content = <Text p style={{ margin: 0, opacity: 0.7 }}>waits</Text>;
                }
                
                return (
                    <div key={i} style={{ marginBottom: '12px', padding: '8px', background: 'rgba(255,255,255,0.05)', borderRadius: '4px', fontSize: '12px' }}>
                        <Text p style={{ margin: 0, fontWeight: 'bold' }}>
                            <Code>[Step {log.step}] Agent {log.agent_id}</Code>
                        </Text>
                        {content}
                         <details style={{marginTop: '8px', opacity: 0.8}}>
                            <summary style={{fontSize: '11px', cursor: 'pointer'}}>View LLM I/O</summary>
                            <Text p style={{ margin: '4px 0', fontSize: '11px', whiteSpace: 'pre-wrap', wordBreak: 'break-word', opacity: 0.7 }}>
                                <details>
                                    <summary>Prompt</summary>
                                    <Code block style={{fontSize: '10px', maxHeight: '150px', overflow: 'auto'}}>{log.prompt}</Code>
                                </details>
                            </Text>
                            <Text p style={{ margin: '4px 0', fontSize: '11px', whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>
                                <details>
                                    <summary>Response</summary>
                                    <Code block style={{fontSize: '10px'}}>{JSON.stringify(log.response, null, 2)}</Code>
                                </details>
                            </Text>
                        </details>
                    </div>
                );
            })}
        </Card>
    );
};

export default function MineFarmExample() {
  const [state, setState] = useState(null);
  const [running, setRunning] = useState(false);
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
    ws.onopen = () => addLog('MineFarm WS opened');
    ws.onmessage = (ev) => {
      addLog(ev.data);
      const parsed = JSON.parse(ev.data);
      if (parsed.type === 'state' || parsed.type === 'init' || parsed.type === 'run_step') {
        setState(parsed.state);
      }
      if (parsed.type === 'progress') {
        setChartState((prev) => ({
          labels: [...prev.labels, parsed.episode],
          rewards: [...prev.rewards, parsed.reward],
          losses: [...prev.losses, parsed.loss ?? null],
        }));
      }
    };
    ws.onclose = () => addLog('MineFarm WS closed');
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
      <Canvas camera={{ position: [-50, 50, 50], fov: 60 }}>
        <ambientLight intensity={0.6} />
        <directionalLight 
          castShadow 
          position={[100, 100, 100]} 
          intensity={1.6} 
        />
        <directionalLight 
          castShadow 
          position={[-100, 100, -100]} 
          intensity={0.5} 
        />
        <Grid infiniteGrid cellSize={1} sectionSize={10} sectionColor={"#4488ff"} fadeDistance={250} fadeStrength={1.5} />
        
        {state && state.agents.map(agent => <Agent key={agent.id} agent={agent} gridSize={gridSize} />)}
        {state && <Resources grid={state.grid} resourceTypes={state.resource_types} gridSize={gridSize} />}
        
        <EffectComposer>
          <Bloom intensity={1.2} luminanceThreshold={0.1} luminanceSmoothing={0.9} toneMapped={false} />
        </EffectComposer>
        <OrbitControls maxDistance={400} minDistance={10} />
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
      {state && <AgentCommsPanel logs={state.llm_logs} />}
      <InfoPanel logs={logs} chartState={chartState} />

    </div>
  );
} 