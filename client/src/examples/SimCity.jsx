import React, { useState, useEffect, useRef, useMemo } from 'react';
import { Canvas, useLoader } from '@react-three/fiber';
import { OrbitControls, Grid, Text as DreiText, Line } from '@react-three/drei';
import { Button, Text, Card, Code, Link as GeistLink } from '@geist-ui/core';
import { Link } from 'react-router-dom';
import * as THREE from 'three';
import config from '../config.js';
import { useResponsive } from '../hooks/useResponsive.js';
import InfoPanel from '../components/InfoPanel.jsx';

const WS_URL = `${config.WS_BASE_URL}/ws/simcity`;

const MapPlane = ({ gridSize, rotationDegrees, mapCenter, mapZoom }) => {
    const apiKey = import.meta.env.VITE_GOOGLE_MAPS_API_KEY;
    
    const styles = [
        'style=element:geometry|color:0x212121',
        'style=feature:road|element:geometry|color:0xdddddd',
        'style=feature:road.highway|element:geometry|color:0xdddddd',
        'style=feature:water|element:geometry|color:0x000000',
        'style=feature:all|element:labels|visibility:off'
    ];
    
    const mapUrl = useMemo(() => {
        if (!mapCenter || !mapZoom) return '/basic_example.jpg';
        const center = `${mapCenter[0]},${mapCenter[1]}`;
        return `https://maps.googleapis.com/maps/api/staticmap?center=${center}&zoom=${mapZoom}&size=1024x1024&maptype=roadmap&${styles.join('&')}&key=${apiKey}`;
    }, [mapCenter, mapZoom, apiKey]);
    
    // Fallback to a generic image if the API key is not provided.
    const texture = useLoader(THREE.TextureLoader, apiKey ? mapUrl : '/basic_example.jpg');

    const planeSize = useMemo(() => [gridSize[0], gridSize[1]], [gridSize]);
    const rotationRadians = useMemo(() => (rotationDegrees || 0) * (Math.PI / 180), [rotationDegrees]);

    return (
        <mesh rotation={[-Math.PI / 2, 0, rotationRadians]} position={[0, -0.01, 0]}>
            <planeGeometry args={planeSize} />
            <meshStandardMaterial map={texture} />
        </mesh>
    );
};

const RoadNetwork = ({ lines }) => {
    if (!lines) return null;
    return (
        <group>
            {lines.map((path, i) => {
                const points = path.map(p => new THREE.Vector3(...p));
                return <Line key={i} points={points} color="teal" lineWidth={1} />;
            })}
        </group>
    );
};

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
        "road": <boxGeometry args={[1, 0.1, 1]} wireframe />
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
                        <Building key={`${x}-${z}`} building={buildingInfo} gridSize={gridSize} />
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
        <Card style={{ 
            width: '100%', 
            maxHeight: '240px', 
            height: '100%',
            overflowY: 'scroll',
            background: 'rgba(0,0,0,0.6)',
            color: '#fff',
            border: '1px solid #444',
        }}>
            <Card.Content>
                <Text h4 style={{ color: '#fff', marginTop: '0' }}>City Stats</Text>
                <Text p><strong>Budget:</strong> ${budget.toLocaleString()}</Text>
                <Text p><strong>Population:</strong> {population.toLocaleString()}</Text>
                <Text p><strong>RCI Demand:</strong></Text>
                <Text p>Residential: {rci_demand?.residential.toFixed(1)}</Text>
                <Text p>Commercial: {rci_demand?.commercial.toFixed(1)}</Text>
                <Text p>Industrial: {rci_demand?.industrial.toFixed(1)}</Text>
            </Card.Content>
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
          {messages.length === 0 && <Text p style={{ margin: 0, fontSize: '12px' }}>[No agent messages yet]</Text>}
          {messages.map((msg, i) => (
              <div key={i} style={{ marginBottom: '12px', padding: '8px', background: 'rgba(255,255,255,0.05)', borderRadius: '4px', fontSize: '12px' }}>
                  <Text p style={{ margin: 0, fontWeight: 'bold' }}>
                      <span style={codeStyle}>[Step {msg.step}] {msg.role} (Agent {msg.sender_id}) says:</span>
                  </Text>
                  <Text p style={{ margin: 0, fontStyle: 'italic', paddingLeft: '8px' }}>"{msg.message}"</Text>
              </div>
          ))}
        </Card>
    );
};


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
            <Canvas camera={{ position: [-50, 50, 50], fov: 60 }}>
                <ambientLight intensity={0.7} />
                <directionalLight position={[100, 100, 100]} intensity={1.0} />
                
                {gridSize && <MapPlane 
                    gridSize={gridSize} 
                    rotationDegrees={state.map_rotation_degrees} 
                    mapCenter={state.map_center}
                    mapZoom={state.map_zoom}
                />}
                {state?.road_network && <RoadNetwork lines={state.road_network} />}
                
                {/* Use a flat grid on the ground plane */}
                <Grid infiniteGrid={true} cellSize={1} sectionSize={10} position={[0, -0.02, 0]} sectionColor={"#4488ff"} fadeDistance={100} fadeStrength={1} />
                
                {/* {state && <Buildings grid={state.grid} buildingTypes={state.building_types} gridSize={gridSize} />} */}
                
                <OrbitControls maxDistance={500} minDistance={10} />
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
                <Text h1 style={{ margin: '12px 0', color: '#fff' }}>SimCity (RL+LLM)</Text>
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
            }}>
                {state && <CityStatsPanel stats={state.city_stats} />}
            </div>
            
            {state && <MessagePanel messages={state.messages} />}
            <InfoPanel logs={logs} chartState={chartState} />

        </div>
    );
} 