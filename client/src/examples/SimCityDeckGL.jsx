import React, { useState, useEffect, useRef, useMemo } from 'react';
import { DeckGL } from '@deck.gl/react';
import { ColumnLayer, TextLayer, LineLayer, ScatterplotLayer } from '@deck.gl/layers';
import { Map } from 'react-map-gl';
import { Button, Text, Card, Code } from '@geist-ui/core';
import { Link } from 'react-router-dom';
import config from '../config.js';
import { useResponsive } from '../hooks/useResponsive.js';
import InfoPanel from '../components/InfoPanel.jsx';
import ModelInfoPanel from '../components/ModelInfoPanel.jsx';
import MessagePanel from '../components/MessagePanel.jsx';
import 'mapbox-gl/dist/mapbox-gl.css';

const WS_URL = `${config.WS_BASE_URL}/ws/simcity_deckgl`;
const MAPBOX_TOKEN = import.meta.env.VITE_MAPBOX_TOKEN;

const INITIAL_VIEW_STATE = {
    longitude: -122.4194,
    latitude: 37.7749,
    zoom: 13,
    pitch: 45,
    bearing: 0
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

export default function SimCityDeckGL() {
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
    
    const [hoverInfo, setHoverInfo] = useState(null);

    const layers = useMemo(() => {
        if (!state) return [];

        const pedestrians = new ScatterplotLayer({
            id: 'pedestrians',
            data: state.pedestrians,
            getPosition: d => [d.pos[1], d.pos[0]],
            getFillColor: d => d.color.map(c => c * 255),
            getRadius: 10,
            radiusMinPixels: 2,
            radiusMaxPixels: 20,
            pickable: true,
        });
        
        const buildings = new ColumnLayer({
            id: 'buildings',
            data: state.buildings,
            diskResolution: 12,
            radius: 25,
            extruded: true,
            pickable: true,
            getPosition: d => [d.pos[1], d.pos[0]],
            getFillColor: d => {
                const colors = {
                    'house': [139, 69, 19, 255],
                    'apartment': [70, 130, 180, 255],
                    'office': [47, 79, 79, 255],
                    'skyscraper': [192, 192, 192, 255],
                    'factory': [139, 69, 19, 255]
                };
                const color = colors[d.type] || [96, 96, 96, 255];

                if (d.status === 'planning') {
                    return color.map(c => c * 0.3);
                } else if (d.status === 'under_construction') {
                    const progressRatio = d.progress / d.build_time;
                    return color.map(c => c * (0.4 + progressRatio * 0.6));
                }
                return color;
            },
            getElevation: d => {
                if (d.status === 'completed') return d.height * 8;
                if (d.status === 'under_construction') return (d.progress / d.build_time) * d.height * 8;
                return d.height * 2;
            },
        });

        const buildingLabels = new TextLayer({
            id: 'building-labels',
            data: state.buildings,
            getPosition: d => [d.pos[1], d.pos[0]],
            getText: d => `${d.type.toUpperCase()} #${d.id}`,
            getSize: 16,
            getColor: [255, 255, 255, 200],
            getPixelOffset: [0, -60]
        });

        const businesses = new ColumnLayer({
            id: 'businesses',
            data: state.businesses,
            diskResolution: 6,
            radius: 15,
            extruded: true,
            pickable: true,
            getPosition: d => [d.pos[1], d.pos[0]],
            getFillColor: d => {
                const colors = {
                    'restaurant': [255, 107, 107, 255],
                    'shop': [78, 205, 196, 255],
                    'office': [69, 183, 209, 255],
                    'factory': [150, 206, 180, 255],
                    'market': [255, 234, 167, 255],
                    'bank': [221, 160, 221, 255]
                };
                return colors[d.type] || [255, 255, 255, 255];
            },
            getElevation: 30,
        });

        const businessLabels = new TextLayer({
            id: 'business-labels',
            data: state.businesses,
            getPosition: d => [d.pos[1], d.pos[0]],
            getText: d => d.type.toUpperCase(),
            getSize: 14,
            getColor: [255, 255, 255, 200],
            getPixelOffset: [0, -30]
        });

        const roads = new LineLayer({
            id: 'roads',
            data: state.road_network,
            getPath: d => d.map(p => [p[1], p[0]]),
            getColor: [80, 80, 80],
            getWidth: 5,
            widthMinPixels: 1,
        });

        return [roads, buildings, buildingLabels, businesses, businessLabels, pedestrians];
    }, [state]);

    const getTooltip = ({object}) => {
        if (!object) {
          return null;
        }
        if (object.hasOwnProperty('build_time')) { // It's a building
            return `Building #${object.id}
              Type: ${object.type}
              Status: ${object.status}
              Progress: ${Math.floor((object.progress / object.build_time) * 100)}%`;
        }
        if (object.hasOwnProperty('customers_served')) { // It's a business
            return `Business #${object.id}
              Type: ${object.type}
              Customers: ${object.customers_served}
              Revenue: $${object.revenue.toFixed(2)}`;
        }
        if (object.hasOwnProperty('satisfaction')) { // It's a pedestrian
            return `Pedestrian #${object.id}
              State: ${object.state}
              Satisfaction: ${object.satisfaction.toFixed(1)}
              Goal: ${object.building_goal}`;
        }
        return null;
    }

    const onMapLoad = React.useCallback(e => {
        const map = e.target;
        map.getStyle().layers.forEach(layer => {
            if (layer.type === 'symbol') {
                map.removeLayer(layer.id);
            }
        });
    }, []);

    if (error) {
        return (
            <div style={{ width: '100vw', height: '100vh', display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center', background: '#220000', color: '#ffaaaa' }}>
                <Text h1>A Server Error Occurred</Text>
                <Text p>Could not load the simulation environment.</Text>
                <Code block width="50vw" style={{textAlign: 'left'}}>{error}</Code>
                <Button auto type="error" onClick={() => window.location.reload()} style={{marginTop: '20px'}}>Reload Page</Button>
            </div>
        );
    }
    
    return (
        <div style={{ width: '100vw', height: '100vh', overflow: 'hidden', background: '#000011' }}>
            <DeckGL
                initialViewState={INITIAL_VIEW_STATE}
                controller={true}
                layers={layers}
                getTooltip={getTooltip}
            >
                <Map 
                    mapboxAccessToken={MAPBOX_TOKEN}
                    mapStyle="mapbox://styles/mapbox/dark-v11"
                    onLoad={onMapLoad}
                />
            </DeckGL>

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
                  SimCity Deck.gl
                </Text>
                <div style={{ display: 'flex', gap: '8px' }}>
                  <Button auto type="secondary" disabled={training || running} onClick={startTraining}>
                    Train
                  </Button>
                  <Button auto type="success" disabled={training || running || !trained} onClick={startRun}>
                    Run
                  </Button>
                </div>
            </div>

            <InfoPanel logs={logs} chartState={chartState} />
            <ModelInfoPanel modelInfo={modelInfo} />
            <ResourcePanel pedestrians={state?.pedestrians} resources={state?.resources} />
            <BuildingPanel buildings={state?.buildings} buildingRecipes={state?.building_recipes} pedestrians={state?.pedestrians} />
            <RecipePanel buildingRecipes={state?.building_recipes} />
            {state && <MessagePanel messages={state.messages} />}
        </div>
    );
}
