import React, { useState, useEffect, useRef, useMemo } from 'react';
import { Button, Text, Card, Code } from '@geist-ui/core';
import { Link } from 'react-router-dom';
import config from '../config.js';
import { useResponsive } from '../hooks/useResponsive.js';
import InfoPanel from '../components/InfoPanel.jsx';
import ModelInfoPanel from '../components/ModelInfoPanel.jsx';
import MessagePanel from '../components/MessagePanel.jsx';
import KeplerGl from 'kepler.gl/dist/components';
import { addDataToMap } from 'kepler.gl/dist/actions';
import { useDispatch } from 'react-redux';

const WS_URL = `${config.WS_BASE_URL}/ws/simcity_kepler`;

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
      zIndex: 1000,
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
      overflowY: 'scroll',
      zIndex: 1000,
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
      overflowY: 'auto',
      zIndex: 1000,
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

export default function SimCityKeplerExample() {
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
  const mapRef = useRef(null);
  const dispatch = useDispatch();

  // Kepler.gl configuration
  const keplerConfig = useMemo(() => ({
    version: 'v1',
    config: {
      visState: {
        filters: [],
        layers: [
          {
            id: 'agents',
            type: 'point',
            config: {
              dataId: 'agents',
              label: 'Agents',
              color: [255, 255, 255],
              columns: {
                lat: 'lat',
                lng: 'lng'
              },
              isVisible: true,
              visConfig: {
                radius: 5,
                fixedRadius: false,
                opacity: 0.8,
                outline: false,
                thickness: 2,
                strokeColor: [255, 255, 255],
                colorRange: {
                  name: 'Global Warming',
                  type: 'sequential',
                  category: 'Uber',
                  colors: ['#5A1846', '#900C3F', '#C70039', '#E3611C', '#F1920E', '#FFC300']
                },
                strokeColorRange: {
                  name: 'Global Warming',
                  type: 'sequential',
                  category: 'Uber',
                  colors: ['#5A1846', '#900C3F', '#C70039', '#E3611C', '#F1920E', '#FFC300']
                },
                radiusRange: [0, 50],
                filled: true
              }
            }
          },
          {
            id: 'buildings',
            type: 'point',
            config: {
              dataId: 'buildings',
              label: 'Buildings',
              color: [255, 200, 0],
              columns: {
                lat: 'lat',
                lng: 'lng'
              },
              isVisible: true,
              visConfig: {
                radius: 8,
                fixedRadius: false,
                opacity: 0.9,
                outline: true,
                thickness: 2,
                strokeColor: [255, 255, 255],
                colorRange: {
                  name: 'Ice And Fire',
                  type: 'diverging',
                  category: 'Uber',
                  colors: ['#0198BD', '#49E3CE', '#E8FEB5', '#FEEDB1', '#FEAD54', '#D50255']
                },
                strokeColorRange: {
                  name: 'Global Warming',
                  type: 'sequential',
                  category: 'Uber',
                  colors: ['#5A1846', '#900C3F', '#C70039', '#E3611C', '#F1920E', '#FFC300']
                },
                radiusRange: [0, 100],
                filled: true
              }
            }
          },
          {
            id: 'businesses',
            type: 'point',
            config: {
              dataId: 'businesses',
              label: 'Businesses',
              color: [0, 255, 255],
              columns: {
                lat: 'lat',
                lng: 'lng'
              },
              isVisible: true,
              visConfig: {
                radius: 6,
                fixedRadius: false,
                opacity: 0.8,
                outline: true,
                thickness: 2,
                strokeColor: [255, 255, 255],
                colorRange: {
                  name: 'ColorBrewer Set1',
                  type: 'qualitative',
                  category: 'ColorBrewer',
                  colors: ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf']
                },
                radiusRange: [0, 80],
                filled: true
              }
            }
          },
          {
            id: 'roads',
            type: 'line',
            config: {
              dataId: 'roads',
              label: 'Roads',
              color: [100, 100, 100],
              columns: {
                lat0: 'start_lat',
                lng0: 'start_lng',
                lat1: 'end_lat',
                lng1: 'end_lng'
              },
              isVisible: true,
              visConfig: {
                opacity: 0.4,
                thickness: 1,
                colorRange: {
                  name: 'Global Warming',
                  type: 'sequential',
                  category: 'Uber',
                  colors: ['#5A1846', '#900C3F', '#C70039', '#E3611C', '#F1920E', '#FFC300']
                },
                sizeRange: [0, 10],
                targetColor: null
              }
            }
          },
          {
            id: 'traffic_lights',
            type: 'point',
            config: {
              dataId: 'traffic_lights',
              label: 'Traffic Lights',
              color: [255, 0, 0],
              columns: {
                lat: 'lat',
                lng: 'lng'
              },
              isVisible: true,
              visConfig: {
                radius: 4,
                fixedRadius: true,
                opacity: 0.9,
                outline: true,
                thickness: 2,
                strokeColor: [255, 255, 255],
                colorRange: {
                  name: 'Global Warming',
                  type: 'sequential',
                  category: 'Uber',
                  colors: ['#5A1846', '#900C3F', '#C70039', '#E3611C', '#F1920E', '#FFC300']
                },
                radiusRange: [0, 20],
                filled: true
              }
            }
          }
        ],
        interactionConfig: {
          tooltip: {
            fieldsToShow: {
              agents: ['id', 'state', 'satisfaction', 'building_goal'],
              buildings: ['id', 'type', 'status', 'height', 'progress'],
              businesses: ['id', 'type', 'customers_served', 'revenue'],
              roads: ['type'],
              traffic_lights: ['id', 'state']
            },
            enabled: true
          },
          brush: {
            size: 0.5,
            enabled: false
          }
        },
        layerBlending: 'normal',
        splitMaps: [],
        animationConfig: {
          currentTime: null,
          speed: 1
        }
      },
      mapState: {
        bearing: 0,
        dragRotate: true,
        latitude: 37.7749,
        longitude: -122.4194,
        pitch: 45,
        zoom: 13,
        isSplit: false
      },
      mapStyle: {
        styleType: 'dark',
        topLayerGroups: {},
        visibleLayerGroups: {
          label: true,
          road: true,
          border: false,
          building: true,
          water: true,
          land: true,
          '3d building': false
        },
        threeDBuildingColor: [9.665468314072013, 17.18305478057247, 31.1442867897876],
        mapStyles: {}
      }
    }
  }), []);

  useEffect(() => {
      const ws = new WebSocket(WS_URL);
      wsRef.current = ws;
      ws.onopen = () => addLog('SimCity Kepler WS opened');
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
                  updateKeplerData(parsed.state);
              } else if (parsed.type === 'train_step_update') {
                  setState(prevState => {
                      if (!prevState) return null;
                      const newState = {
                          ...prevState,
                          ...parsed.state,
                      };
                      updateKeplerData(newState);
                      return newState;
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
                  addLog('SimCity Kepler training complete!');
              }
          } catch (e) {
              addLog(`Error processing message: ${e}`);
              console.error("Failed to process message: ", e);
          }
      };
      ws.onclose = () => addLog('SimCity Kepler WS closed');

      return () => {
          ws.close();
      };
  }, []);

  const updateKeplerData = (simulationState) => {
    if (!simulationState || !dispatch) return;

    // Format agents data
    const agentsData = simulationState.pedestrians?.map(agent => ({
      id: agent.id,
      lat: agent.lat,
      lng: agent.lng,
      state: agent.state,
      satisfaction: agent.satisfaction,
      building_goal: agent.building_goal,
      current_project: agent.current_building_project,
      total_resources: Object.values(agent.resources || {}).reduce((sum, val) => sum + val, 0)
    })) || [];

    // Format buildings data
    const buildingsData = simulationState.buildings?.map(building => ({
      id: building.id,
      lat: building.lat,
      lng: building.lng,
      type: building.type,
      status: building.status,
      height: building.height,
      progress: building.progress,
      build_time: building.build_time,
      completion_percent: building.status === 'under_construction' ? 
        Math.floor((building.progress / building.build_time) * 100) : 
        (building.status === 'completed' ? 100 : 0)
    })) || [];

    // Format businesses data
    const businessesData = simulationState.businesses?.map(business => ({
      id: business.id,
      lat: business.lat,
      lng: business.lng,
      type: business.type,
      customers_served: business.customers_served,
      revenue: business.revenue
    })) || [];

    // Format roads data
    const roadsData = simulationState.roads?.map(road => ({
      id: road.id,
      start_lat: road.start_lat,
      start_lng: road.start_lng,
      end_lat: road.end_lat,
      end_lng: road.end_lng,
      type: road.type
    })) || [];

    // Format traffic lights data
    const trafficLightsData = simulationState.traffic_lights?.map(light => ({
      id: light.id,
      lat: light.lat,
      lng: light.lng,
      state: light.state
    })) || [];

    // Add data to Kepler.gl
    dispatch(addDataToMap({
      datasets: [
        {
          info: { id: 'agents', label: 'Agents' },
          data: { fields: [
            { name: 'id', type: 'integer' },
            { name: 'lat', type: 'real' },
            { name: 'lng', type: 'real' },
            { name: 'state', type: 'string' },
            { name: 'satisfaction', type: 'real' },
            { name: 'building_goal', type: 'string' },
            { name: 'current_project', type: 'integer' },
            { name: 'total_resources', type: 'integer' }
          ], rows: agentsData.map(agent => [
            agent.id, agent.lat, agent.lng, agent.state, agent.satisfaction,
            agent.building_goal, agent.current_project, agent.total_resources
          ])}
        },
        {
          info: { id: 'buildings', label: 'Buildings' },
          data: { fields: [
            { name: 'id', type: 'integer' },
            { name: 'lat', type: 'real' },
            { name: 'lng', type: 'real' },
            { name: 'type', type: 'string' },
            { name: 'status', type: 'string' },
            { name: 'height', type: 'integer' },
            { name: 'progress', type: 'integer' },
            { name: 'build_time', type: 'integer' },
            { name: 'completion_percent', type: 'integer' }
          ], rows: buildingsData.map(building => [
            building.id, building.lat, building.lng, building.type, building.status,
            building.height, building.progress, building.build_time, building.completion_percent
          ])}
        },
        {
          info: { id: 'businesses', label: 'Businesses' },
          data: { fields: [
            { name: 'id', type: 'integer' },
            { name: 'lat', type: 'real' },
            { name: 'lng', type: 'real' },
            { name: 'type', type: 'string' },
            { name: 'customers_served', type: 'integer' },
            { name: 'revenue', type: 'real' }
          ], rows: businessesData.map(business => [
            business.id, business.lat, business.lng, business.type,
            business.customers_served, business.revenue
          ])}
        },
        {
          info: { id: 'roads', label: 'Roads' },
          data: { fields: [
            { name: 'id', type: 'integer' },
            { name: 'start_lat', type: 'real' },
            { name: 'start_lng', type: 'real' },
            { name: 'end_lat', type: 'real' },
            { name: 'end_lng', type: 'real' },
            { name: 'type', type: 'string' }
          ], rows: roadsData.map(road => [
            road.id, road.start_lat, road.start_lng, road.end_lat, road.end_lng, road.type
          ])}
        },
        {
          info: { id: 'traffic_lights', label: 'Traffic Lights' },
          data: { fields: [
            { name: 'id', type: 'integer' },
            { name: 'lat', type: 'real' },
            { name: 'lng', type: 'real' },
            { name: 'state', type: 'string' }
          ], rows: trafficLightsData.map(light => [
            light.id, light.lat, light.lng, light.state
          ])}
        }
      ],
      option: {
        centerMap: false,
        readOnly: false
      },
      config: keplerConfig.config
    }));
  };

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
    addLog('Starting SimCity Kepler training...');
    send({ cmd: 'train' });
  };

  const reset = () => {
    window.location.reload();
  }

  if (error) {
    return (
        <div style={{ width: '100vw', height: '100vh', display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center', background: '#220000', color: '#ffaaaa' }}>
            <Text h1>A Server Error Occurred</Text>
            <Text p>Could not load the SimCity Kepler simulation environment.</Text>
            <Code block width="50vw" style={{textAlign: 'left'}}>{error}</Code>
            <Button auto type="error" onClick={reset} style={{marginTop: '20px'}}>Reload Page</Button>
        </div>
    );
  }

  return (
    <div style={{ width: '100vw', height: '100vh', overflow: 'hidden', background: '#000011', position: 'relative' }}>
      <div style={{ position: 'absolute', top: 10, left: 10, zIndex: 1000, color: '#fff' }}>
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
          SimCity Kepler 
        </Text>
        <div style={{ display: 'flex', gap: '8px' }}>
          <Button auto type="secondary" disabled={training || running} onClick={startTraining}>
            Train
          </Button>
          <Button auto type="success" disabled={training || running || !trained} onClick={startRun}>
            Run
          </Button>
        </div>
        
        <div style={{ 
          marginTop: '12px', 
          padding: '8px 12px', 
          background: 'rgba(55, 245, 235, 0.1)', 
          border: '1px solid #37F5EB', 
          borderRadius: '4px',
          fontSize: '12px',
          color: '#37F5EB',
          maxWidth: '300px'
        }}>
          Collaborative city building with Kepler.gl visualization. Agents work together to construct buildings using LLM coordination and RL policies.
        </div>
      </div>
      
      <div style={{ width: '100%', height: '100%' }}>
        <KeplerGl
          id="simcity-kepler"
          width={window.innerWidth}
          height={window.innerHeight}
          mapboxApiAccessToken={process.env.REACT_APP_MAPBOX_TOKEN}
          ref={mapRef}
        />
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