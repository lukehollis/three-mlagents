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
import CameraFeeds from '../components/CameraFeeds.jsx';
import Map from '../components/Map.jsx';
import Roads from '../components/Roads.jsx';
import Lidar, { LIDAR_LAYER } from '../components/Lidar.jsx';
import TrafficLight from '../components/TrafficLight.jsx';
import Pedestrian from '../components/Pedestrian.jsx';

const WS_URL = `${config.WS_BASE_URL}/ws/self_driving_car`;

const TRAINING_FRAME_RENDER_INTERVAL = 100;

const Agent = ({ agent, coordinateTransformer, onCamerasCreated, training }) => {
  const { pos, color, id, heading, pitch } = agent;
  const groupRef = useRef();
  const [carPosition, setCarPosition] = useState(null);
  const [orientation, setOrientation] = useState(new THREE.Quaternion());
  const camerasRef = useRef({});

  useEffect(() => {
      const cameras = {
          'Front (Wide)': new THREE.PerspectiveCamera(90, 4 / 3, 1, 1000),
          'Front (Narrow)': new THREE.PerspectiveCamera(45, 4 / 3, 1, 1000),
          'Left': new THREE.PerspectiveCamera(75, 4 / 3, 1, 1000),
          'Right': new THREE.PerspectiveCamera(75, 4 / 3, 1, 1000),
          'Rear': new THREE.PerspectiveCamera(120, 4 / 3, 1, 1000),
          'Lidar': new THREE.OrthographicCamera(-500, 500, 500, -500, 1, 2000)
      };

      const y = 15;
      cameras['Front (Wide)'].position.set(0, y, 21);
      cameras['Front (Wide)'].rotateY(Math.PI);
      cameras['Front (Narrow)'].position.set(0, y, 21);
      cameras['Front (Narrow)'].rotateY(Math.PI);
      cameras['Left'].position.set(-11, y, 0);
      cameras['Left'].rotateY(Math.PI / 2);
      cameras['Right'].position.set(11, y, 0);
      cameras['Right'].rotateY(-Math.PI / 2);
      cameras['Rear'].position.set(0, y, -21);
      cameras['Lidar'].position.set(0, 1000, 0);
      cameras['Lidar'].lookAt(0, 0, 0);
      cameras['Lidar'].layers.enable(LIDAR_LAYER);

      camerasRef.current = cameras;
      onCamerasCreated(id, cameras);

      const group = groupRef.current;
      if (group) {
        Object.values(cameras).forEach(cam => group.add(cam));
      }

      return () => {
          Object.values(cameras).forEach(cam => {
              if (cam.parent) {
                  cam.parent.remove(cam);
              }
          });
          if (group) {
            Object.values(cameras).forEach(cam => group.remove(cam));
          }
      };

  }, [id, onCamerasCreated, training]);


  useEffect(() => {
    if (coordinateTransformer) {
      const [lat, lng] = pos;
      const vector = coordinateTransformer.latLngToECEF(lat, lng);
      setCarPosition(vector);

      const newOrientation = calculateOrientation(lat, lng, heading, pitch || 0, coordinateTransformer);
      newOrientation.multiply(new THREE.Quaternion().setFromEuler(new THREE.Euler(Math.PI / -2, Math.PI / 2, 0)));
      setOrientation(newOrientation);
    }
  }, [pos, heading, pitch, coordinateTransformer]);

  const agentColor = useMemo(() => new THREE.Color(...color), [color]);

  if (!carPosition) return null;

  return (
    <group ref={groupRef} position={carPosition} quaternion={orientation}>
      <mesh>
        <boxGeometry args={[20, 10, 40]} />
        <meshPhongMaterial
          color={agentColor}
          emissive={agentColor}
          emissiveIntensity={0.5}
        />
      </mesh>

      {/* Front lights */}
      <mesh position={[7.5, 0, 20]}>
        <sphereGeometry args={[2.5, 8, 8]} />
        <meshStandardMaterial color={"white"} emissive={"white"} emissiveIntensity={5} />
      </mesh>
      <mesh position={[-7.5, 0, 20]}>
        <sphereGeometry args={[2.5, 8, 8]} />
        <meshStandardMaterial color={"white"} emissive={"white"} emissiveIntensity={5} />
      </mesh>

      {/* Back lights */}
      <mesh position={[7.5, 0, -20]}>
        <sphereGeometry args={[2.5, 8, 8]} />
        <meshStandardMaterial color={"red"} emissive={"red"} emissiveIntensity={5} />
      </mesh>
      <mesh position={[-7.5, 0, -20]}>
        <sphereGeometry args={[2.5, 8, 8]} />
        <meshStandardMaterial color={"red"} emissive={"red"} emissiveIntensity={5} />
      </mesh>

      <DreiText position={[0, 15, 0]} fontSize={5} color="white" anchorX="center" anchorY="middle">
        {id}
      </DreiText>
      <Lidar />
    </group>
  );
};

const calculateOrientation = (lat, lon, heading, pitch, coordinateTransformer) => {
  const ecefPosition = coordinateTransformer.latLngToECEF(lat, lon);
  const up = ecefPosition.clone().normalize();

  // Calculate a point slightly ahead of the car for the "lookAt" target
  const lookAtLat = lat + 0.0001 * Math.cos(THREE.MathUtils.degToRad(heading));
  const lookAtLon = lon + 0.0001 * Math.sin(THREE.MathUtils.degToRad(heading));
  const lookAtPosition = coordinateTransformer.latLngToECEF(lookAtLat, lookAtLon);

  // Create a rotation matrix that makes the car look at the target point
  const lookAtMatrix = new THREE.Matrix4();
  lookAtMatrix.lookAt(ecefPosition, lookAtPosition, up);

  // Convert the rotation matrix to a quaternion
  const finalOrientation = new THREE.Quaternion().setFromRotationMatrix(lookAtMatrix);
  
  // Apply a model correction if necessary (this aligns the model's forward vector)
  const modelCorrection = new THREE.Quaternion().setFromEuler(new THREE.Euler(Math.PI / 2, Math.PI, 0));
  finalOrientation.multiply(modelCorrection);

  return finalOrientation;
};

const FeatureImportancePanel = ({ chartData }) => {
    if (!chartData || !chartData.data || chartData.data.length === 0) {
        return (<Card style={{
            position: 'absolute',
            bottom: '10px', 
            left: '10px',
            width: '450px',
            background: 'rgba(0,0,0,0.7)',
            color: '#fff',
            border: '1px solid #555',
            padding: '6px',
            boxSizing: 'border-box'
        }}></Card>);
    }

    const { data, step, agentId, action } = chartData;
    const codeStyle = { color: '#37F5EB', fontFamily: 'monospace' };

    return (
        <Card style={{
            position: 'absolute',
            bottom: '10px', 
            left: '10px',
            width: '450px',
            background: 'rgba(0,0,0,0.7)',
            color: '#fff',
            border: '1px solid #555',
            padding: '6px',
            boxSizing: 'border-box'
        }}>
            <Text p style={{ margin: 0, fontWeight: 'bold', fontSize: '12px', paddingBottom: '8px' }}>
                <span style={codeStyle}>[Step {step}] Agent {agentId}</span>
                {action && <span style={{ color: '#fff', marginLeft: '8px', background: '#333', padding: '2px 6px', borderRadius: '4px' }}>{action}</span>}
            </Text>
            <FeatureImportanceChart data={data} />
        </Card>
    );
};

const FeatureImportanceChart = ({ data }) => {
    const maxLabelWidth = '140px';
    const barColor = '#37F5EB';

    return (
        <div style={{ marginTop: '4px', fontFamily: 'monospace', fontSize: '12px' }}>
            {data.map(({ label, percentage }, index) => (
                <div key={index} style={{ display: 'flex', alignItems: 'center', marginBottom: '4px' }}>
                    <div style={{
                        width: maxLabelWidth,
                        whiteSpace: 'nowrap',
                        overflow: 'hidden',
                        textOverflow: 'ellipsis',
                        color: '#ccc',
                        textAlign: 'right',
                        paddingRight: '10px'
                    }}>
                        {label}
                    </div>
                    <div style={{ flex: 1, display: 'flex', alignItems: 'center' }}>
                        <div style={{
                            width: `${percentage}%`,
                            background: barColor,
                            height: '14px',
                            borderRadius: '2px',
                            transition: 'width 0.3s ease-in-out',
                            minWidth: '1px'
                        }} />
                        <div style={{ color: '#fff', fontWeight: 'bold', paddingLeft: '5px' }}>
                            {percentage.toFixed(0)}%
                        </div>
                    </div>
                </div>
            ))}
        </div>
    );
};


export default function SelfDrivingCarExample() {
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
  const cameraTargetsRef = useRef({});
  const [cameraFeedData, setCameraFeedData] = useState({});
  const agentCamerasRef = useRef({});
  const [latestFeatureImportance, setLatestFeatureImportance] = useState(null);

  useEffect(() => {
      const ws = new WebSocket(WS_URL);
      wsRef.current = ws;
      ws.onopen = () => addLog('SelfDrivingCar WS opened');
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
                  if (parsed.state && parsed.state.messages && parsed.state.messages.length > 0) {
                      const latestMsg = parsed.state.messages[parsed.state.messages.length - 1];
                      if (latestMsg.message.startsWith("Action:")) {
                          const parts = latestMsg.message.split(", Causes: ");
                          const actionText = parts[0].replace("Action: ", "");
                          
                          if (parts.length > 1 && parts[1]) {
                              const causesString = parts[1];
                              try {
                                  const featureImportanceData = causesString.split(', ').map(s => {
                                      const lastParen = s.lastIndexOf('(');
                                      if (lastParen === -1) return null;
                                      const label = s.substring(0, lastParen).trim();
                                      const percentageStr = s.substring(lastParen + 1, s.length - 2);
                                      const percentage = parseInt(percentageStr, 10);
                                      if (isNaN(percentage)) return null;
                                      return { label, percentage };
                                  }).filter(Boolean);

                                  setLatestFeatureImportance({
                                      data: featureImportanceData,
                                      step: latestMsg.step,
                                      agentId: latestMsg.sender_id,
                                      action: actionText,
                                  });

                              } catch(e) { /* ignore parse errors */ }
                          }
                      }
                  }
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
      ws.onclose = () => addLog('SelfDrivingCar WS closed');

      return () => {
          ws.close();
      };
  }, []);

  useEffect(() => {
      return () => {
          Object.values(cameraTargetsRef.current).forEach(target => target.dispose());
      }
  }, []);

  const handleCamerasCreated = (agentId, cameras) => {
      agentCamerasRef.current[agentId] = cameras;
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
    addLog('Starting training run...');
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
      <Canvas camera={{ fov: 60 }}>
        <SceneContent
            state={state}
            coordinateTransformer={coordinateTransformer}
            handleCamerasCreated={handleCamerasCreated}
            setMapLoaded={setMapLoaded}
            setCoordinateTransformer={setCoordinateTransformer}
            agentCamerasRef={agentCamerasRef}
            cameraTargetsRef={cameraTargetsRef}
            setCameraFeedData={setCameraFeedData}
            training={training}
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
        <Text h1 style={{ margin: '12px 0', color: '#fff', fontSize: isMobile ? '1.2rem' : '2rem' }}>Self-Driving Car (Interpretability)</Text>
        <div style={{ display: 'flex', gap: '8px' }}>
          <Button auto type="secondary" disabled={training || running} onClick={startTraining}>Train</Button>
          <Button auto type="success" disabled={training || running || !trained} onClick={startRun}>Run</Button>
        </div>
      </div>
      
      <CameraFeeds cameraFeedData={cameraFeedData} />
      <InfoPanel logs={logs} chartState={chartState} />
      <ModelInfoPanel modelInfo={modelInfo} />
      <FeatureImportancePanel chartData={latestFeatureImportance} />

    </div>
  );
} 

const SceneContent = ({
    state,
    coordinateTransformer,
    handleCamerasCreated,
    setMapLoaded,
    setCoordinateTransformer,
    agentCamerasRef,
    cameraTargetsRef,
    setCameraFeedData,
    training,
}) => {
    const { scene } = useThree();
    const frameCountRef = useRef(0);

    useFrame(({ gl, scene }) => {
        gl.autoClear = true;

        frameCountRef.current++;

        const newFeedData = {};
        const renderWidth = 400;
        const renderHeight = 300;

        Object.entries(agentCamerasRef.current).forEach(([agentId, cameras]) => {
            Object.entries(cameras).forEach(([cameraName, camera]) => {
                const targetName = `${agentId}-${cameraName}`;
                let renderTarget = cameraTargetsRef.current[targetName];

                if (!renderTarget) {
                    renderTarget = new THREE.WebGLRenderTarget(renderWidth, renderHeight);
                    cameraTargetsRef.current[targetName] = renderTarget;
                }

                gl.setRenderTarget(renderTarget);
                gl.render(scene, camera);

                const buffer = new Uint8Array(renderWidth * renderHeight * 4);
                gl.readRenderTargetPixels(renderTarget, 0, 0, renderWidth, renderHeight, buffer);

                newFeedData[targetName] = { buffer, width: renderWidth, height: renderHeight };
            });
        });

        gl.setRenderTarget(null);
        setCameraFeedData(newFeedData);
    });

    return (
        <>
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

            <Map onMapLoaded={(transformer) => {
                setMapLoaded(true);
                setCoordinateTransformer(transformer);
            }} />

            {state && coordinateTransformer && <Roads roadNetwork={state.road_network} coordinateTransformer={coordinateTransformer} />}
            {state && coordinateTransformer && state.agents.map(agent => <Agent key={agent.id} agent={agent} coordinateTransformer={coordinateTransformer} onCamerasCreated={handleCamerasCreated} training={training} />)}
            {state && coordinateTransformer && state.pedestrians.map(ped => <Pedestrian key={ped.id} pedestrian={ped} coordinateTransformer={coordinateTransformer} />)}
            {state && coordinateTransformer && state.traffic_lights.map(light => <TrafficLight key={light.id} light={light} coordinateTransformer={coordinateTransformer} />)}

            <EffectComposer>
                <Bloom intensity={1.2} luminanceThreshold={0.1} luminanceSmoothing={0.9} toneMapped={false} />
            </EffectComposer>
        </>
    );
}; 