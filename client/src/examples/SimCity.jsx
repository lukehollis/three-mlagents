import React, { useState, useEffect, useRef, useMemo } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls, Text as DreiText } from '@react-three/drei';
import * as THREE from 'three';
import { Button, Text, Card, Code } from '@geist-ui/core';
import { Link } from 'react-router-dom';
import config from '../config.js';
import { useResponsive } from '../hooks/useResponsive.js';
import { EffectComposer, Bloom } from '@react-three/postprocessing';
import InfoPanel from '../components/InfoPanel.jsx';
import ModelInfoPanel from '../components/ModelInfoPanel.jsx';
import Map2D from '../components/Map2D.jsx';
import Roads from '../components/Roads.jsx';
import TrafficLight from '../components/TrafficLight.jsx';
import Pedestrian from '../components/Pedestrian.jsx';

const WS_URL = `${config.WS_BASE_URL}/ws/simcity`;

const Business = ({ business, coordinateTransformer }) => {
  const { pos, type, id, customers_served, revenue } = business;
  const [businessPosition, setBusinessPosition] = useState(null);

  useEffect(() => {
    if (coordinateTransformer) {
      const [lat, lng] = pos;
      const vector = coordinateTransformer.latLngToECEF(lat, lng);
      setBusinessPosition(vector);
    }
  }, [pos, coordinateTransformer]);

  const businessColor = useMemo(() => {
    const colorMap = {
      'restaurant': '#ff6b6b',
      'shop': '#4ecdc4',
      'office': '#45b7d1',
      'factory': '#96ceb4',
      'market': '#ffeaa7',
      'bank': '#dda0dd'
    };
    return new THREE.Color(colorMap[type] || '#ffffff');
  }, [type]);

  if (!businessPosition) return null;

  return (
    <group position={businessPosition}>
      {/* Building */}
      <mesh position={[0, 15, 0]}>
        <boxGeometry args={[25, 30, 25]} />
        <meshPhongMaterial
          color={businessColor}
          emissive={businessColor}
          emissiveIntensity={0.3}
        />
      </mesh>

      {/* Sign */}
      <DreiText 
        position={[0, 35, 0]} 
        fontSize={4} 
        color="white" 
        anchorX="center" 
        anchorY="middle"
        maxWidth={50}
      >
        {type.toUpperCase()}
      </DreiText>

      {/* Stats */}
      <DreiText 
        position={[0, 42, 0]} 
        fontSize={2} 
        color="#ffff88" 
        anchorX="center" 
        anchorY="middle"
      >
        Customers: {customers_served}
      </DreiText>
    </group>
  );
};

const EconomicPanel = ({ pedestrians, businesses }) => {
  const avgSatisfaction = useMemo(() => {
    if (!pedestrians || pedestrians.length === 0) return 0;
    return pedestrians.reduce((sum, p) => sum + p.satisfaction, 0) / pedestrians.length;
  }, [pedestrians]);

  const totalRevenue = useMemo(() => {
    if (!businesses || businesses.length === 0) return 0;
    return businesses.reduce((sum, b) => sum + b.revenue, 0);
  }, [businesses]);

  const stateDistribution = useMemo(() => {
    if (!pedestrians || pedestrians.length === 0) return {};
    const distribution = {};
    pedestrians.forEach(p => {
      distribution[p.state] = (distribution[p.state] || 0) + 1;
    });
    return distribution;
  }, [pedestrians]);

  return (
    <Card style={{
      position: 'absolute',
      bottom: '10px',
      right: '10px',
      width: '300px',
      background: 'rgba(0,0,0,0.8)',
      color: '#fff',
      border: '1px solid #555',
      padding: '16px',
      boxSizing: 'border-box'
    }}>
      <Text h4 style={{ margin: '0 0 16px 0', color: '#37F5EB' }}>City Economy</Text>
      
      <div style={{ marginBottom: '12px' }}>
        <Text p style={{ margin: '0 0 4px 0', fontSize: '14px' }}>
          <strong>Avg Satisfaction:</strong> {avgSatisfaction.toFixed(1)}%
        </Text>
        <div style={{
          width: '100%',
          height: '8px',
          background: '#333',
          borderRadius: '4px',
          overflow: 'hidden'
        }}>
          <div style={{
            width: `${avgSatisfaction}%`,
            height: '100%',
            background: `linear-gradient(to right, #ff4757 0%, #ffa502 50%, #2ed573 100%)`,
            transition: 'width 0.3s ease'
          }} />
        </div>
      </div>

      <Text p style={{ margin: '0 0 8px 0', fontSize: '14px' }}>
        <strong>Total Business Revenue:</strong> ${totalRevenue.toFixed(0)}
      </Text>

      <Text p style={{ margin: '0 0 8px 0', fontSize: '14px' }}>
        <strong>Population:</strong> {pedestrians?.length || 0}
      </Text>

      <div style={{ marginTop: '12px' }}>
        <Text p style={{ margin: '0 0 4px 0', fontSize: '12px', color: '#ccc' }}>
          Population by Activity:
        </Text>
        {Object.entries(stateDistribution).map(([state, count]) => (
          <div key={state} style={{ fontSize: '11px', color: '#aaa', marginBottom: '2px' }}>
            {state}: {count}
          </div>
        ))}
      </div>
    </Card>
  );
};

const BusinessPanel = ({ businesses }) => {
  const topBusinesses = useMemo(() => {
    if (!businesses) return [];
    return [...businesses]
      .sort((a, b) => b.customers_served - a.customers_served)
      .slice(0, 5);
  }, [businesses]);

  return (
    <Card style={{
      position: 'absolute',
      bottom: '10px',
      left: '10px',
      width: '320px',
      background: 'rgba(0,0,0,0.8)',
      color: '#fff',
      border: '1px solid #555',
      padding: '16px',
      boxSizing: 'border-box'
    }}>
      <Text h4 style={{ margin: '0 0 16px 0', color: '#37F5EB' }}>Top Businesses</Text>
      
      {topBusinesses.map((business, index) => (
        <div key={business.id} style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          marginBottom: '8px',
          padding: '8px',
          background: index === 0 ? 'rgba(55, 245, 235, 0.1)' : 'rgba(255,255,255,0.05)',
          borderRadius: '4px'
        }}>
          <div>
            <div style={{ fontSize: '12px', fontWeight: 'bold', textTransform: 'capitalize' }}>
              {business.type}
            </div>
            <div style={{ fontSize: '10px', color: '#ccc' }}>
              Revenue: ${business.revenue.toFixed(0)}
            </div>
          </div>
          <div style={{ textAlign: 'right' }}>
            <div style={{ fontSize: '14px', fontWeight: 'bold', color: '#37F5EB' }}>
              {business.customers_served}
            </div>
            <div style={{ fontSize: '10px', color: '#ccc' }}>
              customers
            </div>
          </div>
        </div>
      ))}
      
      {topBusinesses.length === 0 && (
        <Text p style={{ fontSize: '12px', color: '#666', textAlign: 'center' }}>
          No business data yet...
        </Text>
      )}
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
  const [mapLoaded, setMapLoaded] = useState(false);
  const [coordinateTransformer, setCoordinateTransformer] = useState(null);

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

              if (parsed.type === 'train_step' || parsed.type === 'run_step' || parsed.type === 'state' || parsed.type === 'init') {
                  setState(parsed.state);
              } else if (parsed.type === 'train_step_update') {
                  setState(prevState => {
                      if (!prevState) return null;
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
              if (parsed.type === 'training_progress') {
                  addLog(`Training Progress: ${parsed.message}`);
              }
              if (parsed.type === 'trained') {
                  setTraining(false);
                  setTrained(true);
                  setModelInfo(parsed.model_info);
                  addLog('Economic simulation training complete!');
              }
          } catch (e) {
              addLog(`Error processing message: ${e}`);
              console.error("Failed to process message: ", e);
          }
      };
      ws.onclose = () => addLog('SimCity WS closed');

      return () => {
          ws.close();
      };
  }, []);

  const addLog = (txt) => {
    setLogs((l) => {
      const upd = [...l, txt];
      return upd.length > 200 ? upd.slice(upd.length - 200) : upd;
    });
  };

  const send = (obj) => {
    addLog(`Sending: ${JSON.stringify(obj)}`);
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(obj));
    } else {
      addLog('WebSocket not open');
    }
  };

  const startRun = () => {
    if (running || training) return;
    setRunning(true);
    send({ cmd: 'run' });
  };

  const stopRun = () => {
    if (!running) return;
    setRunning(false);
    send({ cmd: 'stop' });
  };

  const startTraining = () => {
    if (training || running) {
      return;
    }
    setTraining(true);
    addLog('Starting economic simulation training...');
    send({ cmd: 'train' });
  };

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
      <Canvas camera={{ fov: 60, position: [0, 500, 500], near: 1, far: 10000 }}>
        <SceneContent
            state={state}
            coordinateTransformer={coordinateTransformer}
            setMapLoaded={setMapLoaded}
            setCoordinateTransformer={setCoordinateTransformer}
        />
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
        <Text h1 style={{ margin: '12px 0', color: '#fff', fontSize: isMobile ? '1.2rem' : '2rem' }}>
          SimCity 
        </Text>
        <div style={{ display: 'flex', gap: '8px' }}>
          <Button auto type="secondary" disabled={training || running} onClick={startTraining}>
            Train
          </Button>
          <Button auto type="success" disabled={training} onClick={running ? stopRun : startRun}>
            {running ? 'Stop' : 'Run'}
          </Button>
        </div>
      </div>
      
      <InfoPanel logs={logs} chartState={chartState} />
      <ModelInfoPanel modelInfo={modelInfo} />
      <EconomicPanel pedestrians={state?.pedestrians} businesses={state?.businesses} />
      <BusinessPanel businesses={state?.businesses} />
    </div>
  );
} 

const SceneContent = ({
    state,
    coordinateTransformer,
    setMapLoaded,
    setCoordinateTransformer,
}) => {
    return (
        <>
            <OrbitControls 
                enablePan={true}
                enableZoom={true}
                enableRotate={true}
                maxPolarAngle={Math.PI * 0.9}
                minPolarAngle={0.1}
                minDistance={50}
                maxDistance={1000}
            />
            
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

            <Map2D onMapLoaded={(transformer) => {
                setMapLoaded(true);
                setCoordinateTransformer(transformer);
            }} />

            {state && coordinateTransformer && <Roads roadNetwork={state.road_network} coordinateTransformer={coordinateTransformer} />}
            
            {state && coordinateTransformer && state.businesses?.map(business => 
                <Business key={business.id} business={business} coordinateTransformer={coordinateTransformer} />
            )}
            
            {state && coordinateTransformer && state.pedestrians?.map(ped => 
                <Pedestrian key={ped.id} pedestrian={ped} coordinateTransformer={coordinateTransformer} />
            )}
            
            {state && coordinateTransformer && state.traffic_lights?.map(light => 
                <TrafficLight key={light.id} light={light} coordinateTransformer={coordinateTransformer} />
            )}

            <EffectComposer>
                <Bloom intensity={1.2} luminanceThreshold={0.1} luminanceSmoothing={0.9} toneMapped={false} />
            </EffectComposer>
        </>
    );
}; 