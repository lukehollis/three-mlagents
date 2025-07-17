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

const WS_URL = `${config.WS_BASE_URL}/ws/minecraft`;


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
            /* position: 'absolute', 
            top: '10px', 
            right: '10px', */
            width: '100%',
            maxHeight: '120px', 
            overflowY: 'scroll', 
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

const CraftingPanel = ({ recipes }) => {
    const { isMobile } = useResponsive();
    if (!recipes) return null;

    return (
        <div style={{
            /* position: 'absolute',
            top: '140px',
            right: '10px', */
            width: '100%',
            flex: '1 1 auto', // Allow panel to grow and shrink
            maxHeight: '120px', 
            overflowY: 'scroll', 
            background: 'rgba(0,0,0,0.5)',
            color: '#fff',
            border: '1px solid #444',
            borderRadius: '8px',
        }}>
            <Text h4 style={{ padding: '4px 8px 0', margin: 0, position: 'sticky', top: 0, background: 'rgba(0,0,0,0.5)', color: '#fff' }}>Crafting Recipes</Text>
            <div style={{ padding: '4px 8px' }}>
                {Object.entries(recipes).map(([itemName, itemData]) => (
                    <div key={itemName} style={{ margin: '8px 0', fontSize: '12px', borderBottom: '1px solid #333', paddingBottom: '8px' }}>
                        <Text span style={{ fontWeight: 'bold', textTransform: 'capitalize' }}>{itemName.replace(/_/g, ' ')}</Text>
                        <div style={{ paddingLeft: '10px', marginTop: '4px' }}>
                            <Text span style={{ opacity: 0.8 }}>Value: {itemData.value}</Text><br />
                            <Text span style={{ opacity: 0.8 }}>Requires: </Text>
                            <Text span>{Object.entries(itemData.recipe).map(([res, val]) => `${val} ${res}`).join(', ')}</Text>
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
};

const TradePanel = ({ offers }) => {
    const { isMobile } = useResponsive();
    if (!offers || offers.length === 0) return null;

    const openOffers = offers.filter(o => o.status === 'open');
    if (openOffers.length === 0) return null;

    return (
        <div style={{
            /* position: 'absolute',
            top: 'calc(140px + calc(100vh - 300px) + 10px)', // Position below crafting panel
            right: '10px', */
            width: '100%',
            maxHeight: '200px',
            overflowY: 'auto',
            background: 'rgba(0,0,0,0.5)',
            color: '#fff',
            border: '1px solid #444',
            borderRadius: '8px',
        }}>
            <Text h4 style={{ padding: '4px 8px 0', margin: 0, position: 'sticky', top: 0, background: 'rgba(0,0,0,0.5)', color: '#fff' }}>Open Trades</Text>
            <div style={{ padding: '4px 8px' }}>
                {openOffers.map((offer) => (
                    <div key={offer.offer_id} style={{ margin: '8px 0', fontSize: '12px', borderBottom: '1px solid #333', paddingBottom: '8px' }}>
                        <Text span style={{ fontWeight: 'bold' }}>Offer #{offer.offer_id} by Agent {offer.agent_id}</Text>
                        <div style={{ paddingLeft: '10px', marginTop: '4px', opacity: 0.9 }}>
                           <Text span>Gives: {offer.gives.amount} {offer.gives.item}</Text><br />
                           <Text span>Receives: {offer.receives.amount} {offer.receives.item}</Text>
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
}

const MessagePanel = ({ messages }) => {
    if (!messages) return null;
    return (
        <Card style={{
            position: 'absolute', bottom: '10px', left: '10px', width: '450px',
            maxHeight: '40vh', overflowY: 'auto', background: 'rgba(0,0,0,0.6)',
            color: '#fff', border: '1px solid #444',
        }}>
            {messages.slice().reverse().map((msg, i) => {
                let content;
                if (msg.recipient_id !== null && msg.recipient_id !== undefined) {
                    content = <Text p style={{ margin: 0 }}><Code>[DM to {msg.recipient_id}]</Code> {msg.message}</Text>;
                } else {
                    content = <Text p style={{ margin: 0 }}><Code>[Broadcast]</Code> {msg.message}</Text>;
                }
                
                return (
                    <div key={i} style={{ marginBottom: '12px', padding: '8px', background: 'rgba(255,255,255,0.05)', borderRadius: '4px', fontSize: '12px' }}>
                        <Text p style={{ margin: 0, fontWeight: 'bold' }}>
                            <Code>[Step {msg.step}] Agent {msg.sender_id}</Code>
                        </Text>
                        {content}
                    </div>
                );
            })}
        </Card>
    );
};


export default function MineCraftExample() {
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
    ws.onopen = () => addLog('MineCraft WS opened');
    ws.onmessage = (ev) => {
      console.log('[MineCraft] WS message:', ev.data); // DEBUG
      addLog(`Received data: ${ev.data.substring(0, 100)}...`); // Log incoming data
      try {
        const parsed = JSON.parse(ev.data);
        
        if (parsed.type === 'error') {
          setError(parsed.message);
          addLog(`ERROR: ${parsed.message}`);
          return;
        }

        if (parsed.type === 'train_step' || parsed.type === 'run_step' || parsed.type === 'state' || parsed.type === 'init') {
          setState(parsed.state);
        } else if (parsed.type === 'train_step_update') {
          setState(prevState => {
            if (!prevState) return null;
            // Merge the new dynamic state into the existing full state
            return {
              ...prevState,
              ...parsed.state,
            };
          });
        }
        if (parsed.type === 'progress') {
          setChartState((prev) => ({
            labels: [...prev.labels, parsed.episode],
            rewards: [...prev.rewards, parsed.reward],
            losses: [...prev.losses, parsed.loss ?? null],
          }));
        }
        if (parsed.type === 'data_collection_progress') {
          addLog(`Collecting training data... ${parsed.progress.toFixed(0)}% (${parsed.samples} samples)`);
        }
        if (parsed.type === 'training_progress') {
          addLog(`Training Progress: Epoch ${parsed.epoch}, Loss: ${parsed.loss.toFixed(4)}`);
        }
        if (parsed.type === 'trained') {
          setTraining(false);
          setTrained(true);
          setModelInfo(parsed.model_info);
          addLog('Training complete! Agents are now using the trained policy.');
        }
      } catch (e) {
        addLog(`Error processing message: ${e}`);
        console.error("Failed to process message: ", e);
      }
    };
    ws.onclose = () => addLog('MineCraft WS closed');
    return () => ws.close();
  }, []);

  const send = (obj) => {
    console.log('[MineCraft] Sending to WS:', obj); // DEBUG
    addLog(`Sending: ${JSON.stringify(obj)}`);
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(obj));
    } else {
      addLog('WebSocket not open');
      console.warn('WebSocket not open when trying to send', obj);
    }
  };

  const startRun = () => {
    if (running) return;
    setRunning(true);
    send({ cmd: 'run' });
  };

  const startTraining = () => {
    console.log('[MineCraft] startTraining clicked'); // DEBUG
    if (training || running) {
      console.log('[MineCraft] Training already in progress or simulation running.');
      return;
    }
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
        <Link
          to="/"
          style={{
            fontFamily: 'monospace',
            color: '#fff',
            textDecoration: 'underline',
            display: 'inline-block',
            fontSize: isMobile ? '12px' : '14px',
          }}
        >
          Home
        </Link>
        <Text h1 style={{ margin: '12px 0', color: '#fff', fontSize: isMobile ? '1.2rem' : '2rem' }}>Minecraft</Text>
        <div style={{ display: 'flex', gap: '8px' }}>
          <Button auto type="secondary" disabled={training || trained} onClick={startTraining}>Train</Button>
          <Button auto type="success" disabled={!trained || running} onClick={startRun}>Run</Button>
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
        {state && <ScorePanel agents={state.agents} />}
        {state && <CraftingPanel recipes={state.crafting_recipes} />}
        {state && <TradePanel offers={state.trade_offers} />}
      </div>
      
      {state && <MessagePanel messages={state.messages} />}
      <InfoPanel logs={logs} chartState={chartState} />
      <ModelInfoPanel modelInfo={modelInfo} />

    </div>
  );
} 