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
import MessagePanel from '../components/MessagePanel.jsx';
import HomeButton from '../components/HomeButton.jsx';

const WS_URL = `${config.WS_BASE_URL}/ws/minecraft`;


const Agent = ({ agent, gridSize }) => {
  const { pos, color, id } = agent;
  const groupRef = useRef();

  const offsetX = gridSize ? gridSize[0] / 2 : 0;
  const offsetZ = gridSize ? gridSize[2] / 2 : 0;

  // Create a slightly darker color for clothing/accessories
  const baseColor = new THREE.Color(...color);
  const bodyColor = baseColor.clone();
  const headColor = baseColor.clone().multiplyScalar(1.1); // Slightly lighter head
  const limbColor = baseColor.clone().multiplyScalar(0.9); // Slightly darker limbs

  return (
    <group ref={groupRef} position={[pos[0] - offsetX, pos[1], pos[2] - offsetZ]}>
      {/* Head */}
      <mesh position={[0, 0.75, 0]}>
        <boxGeometry args={[0.5, 0.5, 0.5]} />
        <meshPhongMaterial color={headColor} />
      </mesh>
      
      {/* Body */}
      <mesh position={[0, 0.1, 0]}>
        <boxGeometry args={[0.5, 0.75, 0.25]} />
        <meshPhongMaterial color={bodyColor} />
      </mesh>
      
      {/* Left Arm */}
      <mesh position={[-0.45, 0.2, 0]}>
        <boxGeometry args={[0.25, 0.6, 0.25]} />
        <meshPhongMaterial color={limbColor} />
      </mesh>
      
      {/* Right Arm */}
      <mesh position={[0.45, 0.2, 0]}>
        <boxGeometry args={[0.25, 0.6, 0.25]} />
        <meshPhongMaterial color={limbColor} />
      </mesh>
      
      {/* Left Leg */}
      <mesh position={[-0.15, -0.45, 0]}>
        <boxGeometry args={[0.25, 0.6, 0.25]} />
        <meshPhongMaterial color={limbColor} />
      </mesh>
      
      {/* Right Leg */}
      <mesh position={[0.15, -0.45, 0]}>
        <boxGeometry args={[0.25, 0.6, 0.25]} />
        <meshPhongMaterial color={limbColor} />
      </mesh>
      
      {/* Agent ID label above head */}
      <DreiText position={[0, 1.2, 0]} fontSize={0.3} color="white" anchorX="center" anchorY="middle">
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

    console.log('[Resources] Regenerating meshes, grid length:', grid.length); // Debug log

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
    
    console.log('[Resources] Generated', meshes.length, 'resource meshes'); // Debug log
    return meshes;
  }, [grid, resourceTypes, gridSize]);

  return <group>{resourceMeshes}</group>;
};

const ScorePanel = ({ agents }) => {
    const { isMobile } = useResponsive();
    if (!agents) return null;

    // Color mapping for different resource types
    const resourceColors = {
        gold: '#FFD700',
        diamond: '#00BFFF', 
        crystal: '#abdbe3',
        wood: '#8B4513',
        stone: '#708090',
        iron: '#C0C0C0',
        coal: '#36454F',
        food: '#32CD32',
        water: '#1E90FF',
        default: '#FFFFFF',
        grass: '#00FF00',
        obsidian: '#aaaaaa',
        dirt: '#e28743',
        stone_pickaxe: '#708090',
        iron_pickaxe: '#C0C0C0',
        gold_pickaxe: '#FFD700',
        diamond_pickaxe: '#FFFFFF',
        crystal_wand: '#abdbe3',
    };

    const getResourceColor = (resourceName) => {
        const lower = resourceName.toLowerCase();
        return resourceColors[lower] || resourceColors.default;
    };

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
            borderRadius: '0px',

        }}>
            <Text h4 style={{ padding: '4px 8px 0', margin: 0, position: 'sticky', top: 0, background: 'rgba(0,0,0,0.5)', color: '#fff' }}>Scores</Text>

            <div style={{padding: '4px 8px'}}>
              {agents.map(agent => (
                  <div key={agent.id} style={{ marginBottom: '8px', fontSize: '11px' }}>
                      <Text span style={{ fontWeight: 'bold', color: '#FFD700' }}>Agent {agent.id}:</Text>
                      <div style={{ 
                          display: 'flex', 
                          flexWrap: 'wrap', 
                          gap: '4px', 
                          marginTop: '2px',
                          paddingLeft: '8px' 
                      }}>
                          {Object.entries(agent.inventory).map(([res, val]) => (
                              <div key={res} style={{
                                  display: 'inline-flex',
                                  alignItems: 'center',
                                  background: 'rgba(255,255,255,0.1)',
                                  borderRadius: '0px',
                                  padding: '2px 6px',
                                  fontSize: '10px',
                                  border: `1px solid ${getResourceColor(res)}`,
                              }}>
                                  <span style={{ 
                                      color: getResourceColor(res), 
                                      fontWeight: 'bold',
                                      marginRight: '3px' 
                                  }}>
                                      {res.charAt(0).toUpperCase() + res.slice(1)}:
                                  </span>
                                  <span style={{ color: '#fff' }}>{val}</span>
                              </div>
                          ))}
                      </div>
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
            borderRadius: '0px',
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
            borderRadius: '0px',
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
          console.log('[MineCraft] Received state update, type:', parsed.type); // Debug log
          console.log('[MineCraft] Grid dimensions:', parsed.state?.grid?.length, parsed.state?.grid?.[0]?.length, parsed.state?.grid?.[0]?.[0]?.length); // Debug log
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
        <HomeButton />
        <Text h1 style={{ margin: '12px 0', color: '#fff', fontSize: isMobile ? '1.2rem' : '2rem', fontFamily: 'monospace', textTransform: 'uppercase', letterSpacing: '0.15em' }}>Minecraft</Text>
        <div style={{ display: 'flex', gap: '8px' }}>
          <Button auto type="secondary" style={{ borderRadius: 0, textTransform: 'uppercase', letterSpacing: '0.1em', border: '1px solid #fff' }} disabled={training || trained} onClick={startTraining}>Train</Button>
          <Button auto type="success" style={{ borderRadius: 0, textTransform: 'uppercase', letterSpacing: '0.1em', border: '1px solid #fff' }} disabled={!trained || running} onClick={startRun}>Run</Button>
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