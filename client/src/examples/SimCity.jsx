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
import MessagePanel from '../components/MessagePanel.jsx';
import Map2D from '../components/Map2D.jsx';
import Roads from '../components/Roads.jsx';
import TrafficLight from '../components/TrafficLight.jsx';
import Pedestrian from '../components/Pedestrian.jsx';

const WS_URL = `${config.WS_BASE_URL}/ws/simcity`;

const Building = ({ building, coordinateTransformer }) => {
  const { pos, type, height, status, progress, build_time, id } = building;
  const [buildingPosition, setBuildingPosition] = useState(null);

  useEffect(() => {
    if (coordinateTransformer) {
      const [lat, lng] = pos;
      const vector = coordinateTransformer.latLngToECEF(lat, lng);
      setBuildingPosition(vector);
    }
  }, [pos, coordinateTransformer]);

  const buildingColor = useMemo(() => {
    const colorMap = {
      'house': '#8b4513',         // Brown
      'apartment': '#4682b4',     // Steel blue
      'office': '#2f4f4f',        // Dark slate gray
      'skyscraper': '#c0c0c0',    // Silver
      'factory': '#8B4513'        // Saddle brown
    };
    const baseColor = new THREE.Color(colorMap[type] || '#606060');
    
    // Adjust color based on construction status
    if (status === 'planning') {
      return baseColor.clone().multiplyScalar(0.3); // Very dim for planning
    } else if (status === 'under_construction') {
      const progressRatio = progress / build_time;
      return baseColor.clone().multiplyScalar(0.4 + progressRatio * 0.6); // Gradually brighten
    }
    return baseColor; // Full brightness for completed
  }, [type, status, progress, build_time]);

  if (!buildingPosition) return null;

  const actualHeight = status === 'completed' ? height * 8 : 
                      status === 'under_construction' ? (progress / build_time) * height * 8 : 
                      height * 2; // Planning phase shows foundation

  return (
    <group position={buildingPosition}>
      {/* Multi-story building - each story is 8 units high */}
      <mesh position={[0, actualHeight / 2, 0]}>
        <boxGeometry args={[30, actualHeight, 30]} />
        <meshPhongMaterial
          color={buildingColor}
          emissive={buildingColor}
          emissiveIntensity={status === 'completed' ? 0.2 : 0.1}
          transparent={status === 'planning'}
          opacity={status === 'planning' ? 0.3 : 1.0}
        />
      </mesh>

      {/* Building type label */}
      <DreiText 
        position={[0, actualHeight + 10, 0]} 
        fontSize={3} 
        color="white" 
        anchorX="center" 
        anchorY="middle"
        maxWidth={60}
      >
        {type.toUpperCase()}
      </DreiText>

      {/* Construction status */}
      <DreiText 
        position={[0, actualHeight + 5, 0]} 
        fontSize={2} 
        color={status === 'completed' ? '#00ff00' : status === 'under_construction' ? '#ffff00' : '#ff6600'} 
        anchorX="center" 
        anchorY="middle"
      >
        {status === 'completed' ? 'COMPLETE' : 
         status === 'under_construction' ? `${Math.floor((progress / build_time) * 100)}%` : 
         'PLANNING'}
      </DreiText>

      {/* Building ID */}
      <DreiText 
        position={[0, actualHeight + 15, 0]} 
        fontSize={2} 
        color="#cccccc" 
        anchorX="center" 
        anchorY="middle"
      >
        #{id}
      </DreiText>
      
      {/* Construction progress indicator for active buildings */}
      {status === 'under_construction' && (
        <mesh position={[0, actualHeight + 20, 0]}>
          <cylinderGeometry args={[2, 2, 1]} />
          <meshPhongMaterial color="#ffaa00" emissive="#ffaa00" emissiveIntensity={0.4} />
        </mesh>
      )}
    </group>
  );
};

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

const ResourcePanel = ({ pedestrians, resources }) => {
  const totalResources = useMemo(() => {
    if (!pedestrians || !resources) return {};
    const totals = {};
    Object.keys(resources).forEach(resource => {
      totals[resource] = pedestrians.reduce((sum, p) => sum + (p.resources[resource] || 0), 0);
    });
    return totals;
  }, [pedestrians, resources]);

  const getResourceColor = (resourceName) => {
    if (!resources || !resources[resourceName]) return '#ffffff';
    const colorArray = resources[resourceName].color;
    return `rgb(${Math.floor(colorArray[0] * 255)}, ${Math.floor(colorArray[1] * 255)}, ${Math.floor(colorArray[2] * 255)})`;
  };

  return (
    <Card style={{
      position: 'absolute',
      top: '10px',
      right: '10px',
      width: '320px',
      background: 'rgba(0,0,0,0.8)',
      color: '#fff',
      border: '1px solid #555',
      padding: '0px',
      boxSizing: 'border-box',
      height: '220px',
      overflowY: 'scroll',
    }}>
      <Text h4 style={{ margin: '0 0 8px 0', color: '#37F5EB', fontSize: '12px', position: 'sticky', top: '0px' }}>City Resources</Text>
      
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px' }}>
        {Object.entries(totalResources).map(([resource, total]) => (
          <div key={resource} style={{
            display: 'flex',
            alignItems: 'center',
            padding: '8px',
            background: 'rgba(255,255,255,0.05)',
            borderRadius: '4px',
            border: `1px solid ${getResourceColor(resource)}`
          }}>
            <div style={{
              width: '12px',
              height: '12px',
              background: getResourceColor(resource),
              borderRadius: '2px',
              marginRight: '8px'
            }} />
            <div>
              <div style={{ fontSize: '12px', fontWeight: 'bold', textTransform: 'capitalize' }}>
                {resource}
              </div>
              <div style={{ fontSize: '14px', color: '#fff' }}>
                {total}
              </div>
            </div>
          </div>
        ))}
      </div>
    </Card>
  );
};

const BuildingPanel = ({ buildings, buildingRecipes, pedestrians }) => {
  const buildingStats = useMemo(() => {
    if (!buildings) return { total: 0, completed: 0, underConstruction: 0, planning: 0 };
    return {
      total: buildings.length,
      completed: buildings.filter(b => b.status === 'completed').length,
      underConstruction: buildings.filter(b => b.status === 'under_construction').length,
      planning: buildings.filter(b => b.status === 'planning').length
    };
  }, [buildings]);

  const activeProjects = useMemo(() => {
    if (!buildings) return [];
    return buildings
      .filter(b => b.status !== 'completed')
      .sort((a, b) => b.progress - a.progress)
      .slice(0, 5);
  }, [buildings]);

  return (
    <Card style={{
      position: 'absolute',
      top: '10px',
      right: '340px',
      width: '320px',
      background: 'rgba(0,0,0,0.8)',
      color: '#fff',
      border: '1px solid #555',
      padding: '0px',
      boxSizing: 'border-box',
      height: '220px',
      overflowY: 'scroll'
    }}>
      <Text h4 style={{ margin: '0 0 8px 0', color: '#37F5EB', fontSize: '12px' }}>Building Projects</Text>
      
      <div style={{ marginBottom: '16px' }}>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px', marginBottom: '12px' }}>
          <div style={{ textAlign: 'center', padding: '8px', background: 'rgba(0,255,0,0.1)', borderRadius: '4px' }}>
            <div style={{ fontSize: '18px', fontWeight: 'bold', color: '#00ff00' }}>{buildingStats.completed}</div>
            <div style={{ fontSize: '10px' }}>Completed</div>
          </div>
          <div style={{ textAlign: 'center', padding: '8px', background: 'rgba(255,255,0,0.1)', borderRadius: '4px' }}>
            <div style={{ fontSize: '18px', fontWeight: 'bold', color: '#ffff00' }}>{buildingStats.underConstruction}</div>
            <div style={{ fontSize: '10px' }}>Building</div>
          </div>
        </div>
      </div>

      <Text h5 style={{ margin: '0 0 8px 0', color: '#fff', fontSize: '12px', position: 'sticky', }}>Active Projects:</Text>
      {activeProjects.map(building => {
        // Find agents currently working on this project
        const focusedAgents = pedestrians?.filter(p => p.current_building_project === building.id) || [];
        
        return (
        <div key={building.id} style={{
          marginBottom: '12px',
          padding: '12px',
          background: building.status === 'under_construction' ? 'rgba(255,255,0,0.1)' : 'rgba(255,165,0,0.1)',
          borderRadius: '4px'
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '4px' }}>
            <div style={{ fontSize: '12px', fontWeight: 'bold', textTransform: 'capitalize' }}>
              {building.type} #{building.id}
            </div>
            <div style={{ fontSize: '12px', color: building.status === 'under_construction' ? '#ffff00' : '#ffa500' }}>
              {building.status === 'under_construction' ? 
                `${Math.floor((building.progress / building.build_time) * 100)}%` : 
                'Planning'}
            </div>
          </div>
          
          <div style={{ fontSize: '10px', color: '#ccc', marginBottom: '4px' }}>
            Contributors: {building.contributors.length} | Height: {building.height} stories
            {focusedAgents.length > 0 && (
              <span style={{ color: '#00ff88', marginLeft: '8px' }}>
                • {focusedAgents.length} focused agent{focusedAgents.length !== 1 ? 's' : ''}
              </span>
            )}
          </div>
          
          {/* Progress bar for under construction */}
          {building.status === 'under_construction' && (
            <div style={{
              width: '100%',
              height: '4px',
              background: '#333',
              borderRadius: '2px',
              overflow: 'hidden',
              marginTop: '4px'
            }}>
              <div style={{
                width: `${(building.progress / building.build_time) * 100}%`,
                height: '100%',
                background: 'linear-gradient(to right, #ffa500, #ffff00)',
                transition: 'width 0.3s ease'
              }} />
            </div>
          )}
          
          {/* Resource requirements for planning phase */}
          {building.status === 'planning' && (
            <div style={{ marginTop: '8px' }}>
              <div style={{ fontSize: '10px', color: '#aaa', marginBottom: '4px' }}>Needs:</div>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px' }}>
                {Object.entries(building.resources_needed).map(([resource, needed]) => {
                  const contributed = building.resources_contributed[resource] || 0;
                  const remaining = needed - contributed;
                  return (
                    <div key={resource} style={{
                      fontSize: '10px',
                      padding: '2px 6px',
                      background: remaining > 0 ? 'rgba(255,100,100,0.3)' : 'rgba(100,255,100,0.3)',
                      borderRadius: '2px',
                      border: `1px solid ${remaining > 0 ? '#ff6464' : '#64ff64'}`
                    }}>
                      {resource}: {remaining > 0 ? remaining : '✓'}
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </div>
        );
      })}
      
      {activeProjects.length === 0 && (
        <Text p style={{ fontSize: '12px', color: '#666', textAlign: 'center' }}>
          No active building projects...
        </Text>
      )}
    </Card>
  );
};

const RecipePanel = ({ buildingRecipes }) => {
  return (
    <Card style={{
      position: 'fixed',
      top: '240px',
      right: '10px',
      width: '320px',
      background: 'rgba(0,0,0,0.8)',
      color: '#fff',
      border: '1px solid #555',
      padding: '0px',
      boxSizing: 'border-box',
      maxHeight: '220px',
      overflowY: 'auto'
    }}>
      <Text h4 style={{ margin: '0 0 8px 0', color: '#37F5EB', fontSize: '12px', position: 'sticky', top: '0px' }}>Building Recipes</Text>
      
      {buildingRecipes && Object.entries(buildingRecipes).map(([buildingType, recipe]) => (
        <div key={buildingType} style={{
          marginBottom: '12px',
          padding: '8px',
          background: 'rgba(255,255,255,0.05)',
          borderRadius: '4px'
        }}>
          <div style={{ fontSize: '12px', fontWeight: 'bold', textTransform: 'capitalize', marginBottom: '4px' }}>
            {buildingType.replace(/_/g, ' ')}
          </div>
          <div style={{ fontSize: '10px', color: '#ccc', marginBottom: '4px' }}>
            Height: {recipe.height} stories | Value: ${recipe.base_value} | Time: {recipe.build_time} steps
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px' }}>
            {Object.entries(recipe.recipe).map(([resource, amount]) => (
              <div key={resource} style={{
                fontSize: '10px',
                padding: '2px 6px',
                background: 'rgba(100,150,255,0.3)',
                borderRadius: '2px',
                border: '1px solid #6496ff'
              }}>
                {amount} {resource}
              </div>
            ))}
          </div>
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
      <Canvas camera={{ fov: 60, position: [0, 2000, 2000], near: 1, far: 10000 }}>
        <SceneContent
            state={state}
            coordinateTransformer={coordinateTransformer}
            mapLoaded={mapLoaded}
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
          <Button auto type="success" disabled={training || running || !trained} onClick={startRun}>
            Run
          </Button>
        </div>
        
        {/* Map loading indicator */}
        {!mapLoaded && (
          <div style={{ 
            marginTop: '12px', 
            padding: '8px 12px', 
            background: 'rgba(55, 245, 235, 0.1)', 
            border: '1px solid #37F5EB', 
            borderRadius: '4px',
            fontSize: '12px',
            color: '#37F5EB'
          }}>
            Loading map...
          </div>
        )}
      </div>
      
      <InfoPanel logs={logs} chartState={chartState} />
      <ModelInfoPanel modelInfo={modelInfo} />
      <ResourcePanel pedestrians={state?.pedestrians} resources={state?.resources} />
      <BuildingPanel buildings={state?.buildings} buildingRecipes={state?.building_recipes} pedestrians={state?.pedestrians} />
      <RecipePanel buildingRecipes={state?.building_recipes} />
      <MessagePanel 
        messages={state?.messages} 
      />
    </div>
  );
} 

const SceneContent = ({
    state,
    coordinateTransformer,
    mapLoaded,
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
                maxDistance={2400}
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

            {/* Only render 3D elements after map has loaded and coordinate transformer is available */}
            {mapLoaded && state && coordinateTransformer && (
                <>
                    <Roads roadNetwork={state.road_network} coordinateTransformer={coordinateTransformer} />
                    
                    {state.buildings?.map(building => 
                        <Building key={building.id} building={building} coordinateTransformer={coordinateTransformer} />
                    )}
                    
                    {state.businesses?.map(business => 
                        <Business key={business.id} business={business} coordinateTransformer={coordinateTransformer} />
                    )}
                    
                    {state.pedestrians?.map(ped => 
                        <Pedestrian key={ped.id} pedestrian={ped} coordinateTransformer={coordinateTransformer} />
                    )}
                    
                    {state.traffic_lights?.map(light => 
                        <TrafficLight key={light.id} light={light} coordinateTransformer={coordinateTransformer} />
                    )}
                </>
            )}

            <EffectComposer>
                <Bloom intensity={1.2} luminanceThreshold={0.1} luminanceSmoothing={0.9} toneMapped={false} />
            </EffectComposer>
        </>
    );
}; 