import React, { useState, useEffect, useRef, useMemo } from 'react';
import { Canvas } from '@react-three/fiber';
import { MapControls, Grid, Text as DreiText } from '@react-three/drei';
import { Button, Text, Card, Code, Link as GeistLink } from '@geist-ui/core';
import { Link } from 'react-router-dom';
import * as THREE from 'three';
import config from '../config.js';
import { useResponsive } from '../hooks/useResponsive.js';
import InfoPanel from '../components/InfoPanel.jsx';

const WS_URL = `${config.WS_BASE_URL}/ws/simcity`;

const Building = ({ building, buildingTypes, gridSize }) => {
    const { pos, type } = building;
    const buildingInfo = Object.values(buildingTypes).find(b => b.model === type);
    const color = buildingInfo ? buildingInfo.color : [1, 1, 1];

    const offsetX = gridSize ? gridSize[0] / 2 : 0;
    const offsetZ = gridSize ? gridSize[1] / 2 : 0;

    const buildingGeometries = {
        "house": <boxGeometry args={[0.8, 1, 0.8]} />,
        "shop": <boxGeometry args={[0.8, 1.5, 0.8]} />,
        "factory": <boxGeometry args={[1, 2, 1]} />,
        "park": <cylinderGeometry args={[0.4, 0.4, 0.1, 16]} />,
        "road": <boxGeometry args={[1, 0.1, 1]} />
    };

    return (
        <mesh position={[pos[0] - offsetX, 0.5, pos[1] - offsetZ]}>
            {buildingGeometries[type] || <boxGeometry args={[0.8, 1, 0.8]} />}
            <meshStandardMaterial color={new THREE.Color(...color)} />
        </mesh>
    );
};

const Buildings = ({ grid, buildingTypes, gridSize }) => {
    const buildingMeshes = useMemo(() => {
        const meshes = [];
        if (!grid || !buildingTypes || !gridSize) return meshes;
        const buildingList = Object.keys(buildingTypes);

        const offsetX = gridSize[0] / 2;
        const offsetZ = gridSize[1] / 2;

        grid.forEach((row, x) => {
            row.forEach((cell, z) => {
                if (cell > 0) {
                    const type = buildingList[cell - 1];
                    const buildingInfo = buildingTypes[type];
                    meshes.push(
                        <mesh key={`${x}-${z}`} position={[x - offsetX, buildingInfo.model === 'road' ? 0.05 : 0.5, z - offsetZ]}>
                             <boxGeometry args={[1, buildingInfo.model === 'road' ? 0.1 : 1, 1]} />
                             <meshStandardMaterial color={new THREE.Color(...buildingInfo.color)} />
                        </mesh>
                    );
                }
            });
        });
        return meshes;
    }, [grid, buildingTypes, gridSize]);

    return <group>{buildingMeshes}</group>;
};

const CityStatsPanel = ({ stats }) => {
    if (!stats) return null;
    const { budget, population, rci_demand } = stats;

    return (
        <Card style={{ width: '100%', maxHeight: '300px', overflowY: 'auto' }}>
            <Text h4>City Stats</Text>
            <p><strong>Budget:</strong> ${budget.toLocaleString()}</p>
            <p><strong>Population:</strong> {population.toLocaleString()}</p>
            <p><strong>RCI Demand:</strong></p>
            <ul>
                <li>Residential: {rci_demand?.residential.toFixed(1)}</li>
                <li>Commercial: {rci_demand?.commercial.toFixed(1)}</li>
                <li>Industrial: {rci_demand?.industrial.toFixed(1)}</li>
            </ul>
        </Card>
    );
}

const MessagePanel = ({ messages }) => {
    const containerRef = useRef(null);
    
    useEffect(() => {
        const el = containerRef.current;
        if (el) {
            el.scrollTop = el.scrollHeight;
        }
    }, [messages]);

    const codeStyle = { color: '#f81ce5', fontFamily: 'monospace' };
    return (
        <Card 
            ref={containerRef}
            style={{
                position: 'absolute', bottom: '10px', left: '10px', width: '450px',
                maxHeight: '40vh', overflowY: 'auto', background: 'rgba(0,0,0,0.6)',
                color: '#fff', border: '1px solid #444',
            }}
        >
          {messages.length === 0 && <Text p style={{ margin: 0, fontSize: '12px' }}>[No messages from the council yet]</Text>}
          {messages.map((msg, i) => (
              <div key={i} style={{ marginBottom: '12px', padding: '8px', background: 'rgba(255,255,255,0.05)', borderRadius: '4px', fontSize: '12px' }}>
                  <Text p style={{ margin: 0, fontWeight: 'bold' }}>
                      <span style={codeStyle}>[Step {msg.step}] {msg.role} (Agent {msg.sender_id})</span>
                  </Text>
                  <Text p style={{ margin: 0 }}>{msg.message}</Text>
              </div>
          ))}
        </Card>
    );
};


// Placeholder for Google Maps integration
const GoogleMapOverlay = () => {
    return (
        <div style={{
            position: 'absolute',
            top: 0,
            left: 0,
            width: '100%',
            height: '100%',
            zIndex: -1, // Behind the three.js canvas
            background: '#334', // Dark blue-grey background
            display: 'flex',
            justifyContent: 'center',
            alignItems: 'center',
            color: 'white',
            fontSize: '2rem'
        }}>
            <Text h3 style={{color: 'white', opacity: 0.2}}>[ Google Maps View Placeholder ]</Text>
        </div>
    )
}


export default function SimCityExample() {
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
        ws.onopen = () => addLog('SimCity WS opened');
        ws.onmessage = (ev) => {
            addLog(`Received data: ${ev.data.substring(0, 100)}...`);
            try {
                const parsed = JSON.parse(ev.data);
                
                if (parsed.type === 'error') {
                    setError(parsed.message);
                    addLog(`ERROR: ${parsed.message}`);
                    return;
                }

                if (parsed.type === 'run_step' || parsed.type === 'state' || parsed.type === 'init' || parsed.type === 'train_step') {
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
                    addLog('Training complete! The Mayor is now using the trained policy.');
                }
            } catch (e) {
                addLog(`Error processing message: ${e}`);
                console.error("Failed to process message: ", e);
            }
        };
        ws.onclose = () => addLog('SimCity WS closed');
        return () => ws.close();
    }, []);

    const send = (obj) => {
        addLog(`Sending: ${JSON.stringify(obj)}`);
        if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
            wsRef.current.send(JSON.stringify(obj));
        }
    };

    const startTraining = () => {
        if (training || running) return;
        setTraining(true);
        addLog('Starting training run...');
        send({ cmd: 'train' });
    };

    const startRun = () => {
        if (running) return;
        setRunning(true);
        send({ cmd: 'run' });
    };

    const stopRun = () => {
        if (!running) return;
        setRunning(false);
        send({ cmd: 'stop' });
    }

    const reset = () => {
        window.location.reload();
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
            <GoogleMapOverlay />
            <Canvas camera={{ position: [0, 50, 50], fov: 60 }} style={{ background: 'transparent' }}>
                <ambientLight intensity={0.7} />
                <directionalLight position={[100, 100, 100]} intensity={1.0} />
                
                {/* Use a flat grid on the ground plane */}
                <Grid infiniteGrid={false} cellSize={1} sectionSize={10} position={[0, 0.01, 0]} sectionColor={"#4488ff"} fadeDistance={100} fadeStrength={1} />
                
                {state && <Buildings grid={state.grid} buildingTypes={state.building_types} gridSize={gridSize} />}
                
                <MapControls enableRotate={false} maxDistance={100} minDistance={10} />
            </Canvas>

            <div style={{ position: 'absolute', top: 10, left: 10, zIndex: 1, color: '#fff' }}>
                <Link to="/"><GeistLink block>Home</GeistLink></Link>
                <Text h1 style={{ margin: '12px 0', color: '#fff' }}>SimCity (Multi-Agent)</Text>
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
                width: '300px',
            }}>
                {state && <CityStatsPanel stats={state.city_stats} />}
            </div>
            
            {state && <MessagePanel messages={state.messages} />}
            <InfoPanel logs={logs} chartState={chartState} />

        </div>
    );
} 