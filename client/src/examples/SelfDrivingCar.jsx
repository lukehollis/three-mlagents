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

const WS_URL = `${config.WS_BASE_URL}/ws/self_driving_car`;

const TRAINING_FRAME_RENDER_INTERVAL = 100;

const Agent = ({ agent, coordinateTransformer, onCamerasCreated, training }) => {
  const { pos, color, id, heading, pitch } = agent;
  const groupRef = useRef();
  const [carPosition, setCarPosition] = useState(null);
  const [orientation, setOrientation] = useState(new THREE.Quaternion());
  const camerasRef = useRef({});

  useEffect(() => {
      if (training) {
          onCamerasCreated(id, {});
          return;
      }

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
      {!training && <Lidar />}
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

const TrafficLight = ({ light, coordinateTransformer }) => {
    const { pos, state } = light;
    const [ecefPosition, setEcefPosition] = useState(null);
    const [orientation, setOrientation] = useState(new THREE.Quaternion());

    useEffect(() => {
        if (coordinateTransformer) {
            const [lat, lng] = pos;
            const vector = coordinateTransformer.latLngToECEF(lat, lng, 1);
            setEcefPosition(vector);
            
            const up = vector.clone().normalize();
            const newOrientation = new THREE.Quaternion().setFromUnitVectors(
                new THREE.Vector3(0, 1, 0), // Default cylinder's 'up'
                up                          // Target 'up' on the globe
            );
            setOrientation(newOrientation);
        }
    }, [pos, coordinateTransformer]);

    const color = state === 'green' ? '#00ff00' : '#ff0000';

    if (!ecefPosition) return null;

    return (
        <group position={ecefPosition} quaternion={orientation}>
            <mesh position={[0, 15, 0]}>
                <cylinderGeometry args={[1, 1, 30, 12]} />
                <meshStandardMaterial color="#333333" />
            </mesh>
            <mesh position={[0, 31, 0]}>
                <sphereGeometry args={[3, 16, 16]} />
                <meshStandardMaterial color={color} emissive={color} emissiveIntensity={3} />
            </mesh>
        </group>
    );
};

const Pedestrian = ({ pedestrian, coordinateTransformer }) => {
    const { pos, state } = pedestrian;
    const [pedPosition, setPedPosition] = useState(null);
    const [orientation, setOrientation] = useState(new THREE.Quaternion());

    useEffect(() => {
        if (coordinateTransformer) {
            const [lat, lng] = pos;
            const vector = coordinateTransformer.latLngToECEF(lat, lng, 1); // Elevate slightly
            setPedPosition(vector);

            const up = vector.clone().normalize();
            const newOrientation = new THREE.Quaternion().setFromUnitVectors(
                new THREE.Vector3(0, 1, 0), // Default cylinder's 'up'
                up                          // Target 'up' on the globe
            );
            setOrientation(newOrientation);
        }
    }, [pos, coordinateTransformer]);

    const color = state === 'jaywalking' ? 'red' : 'white';

    if (!pedPosition) return null;
    
    return (
        <mesh position={pedPosition} quaternion={orientation}>
            <cylinderGeometry args={[5, 5, 20, 8]} />
            <meshStandardMaterial color={color} emissive={color} emissiveIntensity={1} />
        </mesh>
    );
};


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
                maxHeight: '30vh', overflowY: 'auto', background: 'rgba(0,0,0,0.6)',
                color: '#fff', border: '1px solid #444',
            }}
        >
          {messages.length === 0 && <Text p style={{ margin: 0, fontSize: '12px' }}>[No messages yet.]</Text>}
          {messages.map((msg, i) => {
              const isSimpleAction = msg.message.startsWith("Action:");
              let actionText = "";
              let explanationText = msg.message;

              if (isSimpleAction) {
                  const parts = msg.message.split(", Causes: ");
                  actionText = parts[0];
                  explanationText = `Causes: ${parts[1]}`;
              }

              return (
                  <div key={i} style={{ marginBottom: '12px', padding: '8px', background: 'rgba(255,255,255,0.05)', borderRadius: '4px', fontSize: '12px' }}>
                      <Text p style={{ margin: 0, fontWeight: 'bold' }}>
                          <span style={codeStyle}>[Step {msg.step}] Agent {msg.sender_id}</span>
                          {actionText && <span style={{ color: '#888', marginLeft: '8px' }}>{actionText}</span>}
                      </Text>
                      <Text p style={{ margin: 0, marginTop: '4px' }}>{explanationText}</Text>
                  </div>
              );
          })}
        </Card>
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
    if (running) return;
    setRunning(true);
    send({ cmd: 'run' });
  };

  const startTraining = () => {
    if (training || running) {
      return;
    }
    setTraining(true);
    setCameraFeedData({});
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
          <Button auto type="secondary" disabled={training || trained} onClick={startTraining}>Train</Button>
          <Button auto type="success" disabled={!trained || running} onClick={startRun}>Run</Button>
        </div>
      </div>
      
      {state && <MessagePanel messages={state.messages} />}
      <CameraFeeds cameraFeedData={cameraFeedData} />
      <InfoPanel logs={logs} chartState={chartState} />
      <ModelInfoPanel modelInfo={modelInfo} />

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
        if (training) {
            return;
        }

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