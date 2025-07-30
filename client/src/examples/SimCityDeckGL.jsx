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

const WS_URL = `${config.WS_BASE_URL}/ws/simcity`;
const MAPBOX_TOKEN = import.meta.env.VITE_MAPBOX_TOKEN;

const INITIAL_VIEW_STATE = {
    longitude: -122.4194,
    latitude: 37.7749,
    zoom: 13,
    pitch: 45,
    bearing: 0
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
                <Map mapboxAccessToken={MAPBOX_TOKEN} mapStyle="mapbox://styles/mapbox/dark-v9" />
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
            {state && <MessagePanel messages={state.messages} />}
        </div>
    );
}
