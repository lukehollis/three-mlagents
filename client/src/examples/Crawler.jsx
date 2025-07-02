import React, { useState, useEffect, useRef } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Grid } from '@react-three/drei';
import config from '../config.js';
import { Text, Button } from '@geist-ui/core';
import 'katex/dist/katex.min.css';
import { BlockMath } from 'react-katex';
import DebugConsole from '../components/DebugConsole.jsx';
import ChartPanel from '../components/ChartPanel.jsx';
import ButtonForkOnGithub from '../components/ButtonForkOnGithub.jsx';
import { Link } from 'react-router-dom';

const WS_URL = `${config.WS_BASE_URL}/ws/crawler`;

// 3-D arrow helper for heading visualisation
function HeadingArrow({ heading }) {
  const ref = useRef();
  useFrame(() => {
    if (ref.current) {
      ref.current.rotation.y = -heading;
    }
  });
  return (
    <mesh ref={ref} position={[0, 0.3, 0]}>
      <coneGeometry args={[0.2, 0.5, 8]} />
      <meshStandardMaterial color="#ff8800" />
    </mesh>
  );
}

export default function CrawlerExample() {
  const [state, setState] = useState({
    basePos: [0, 0, 0.5],
    baseOri: [0, 0, 0, 1],
    jointAngles: Array(12).fill(0), // 12 joints total: hip_x, hip_y, knee for each leg
    targetPos: [10, 0, 0.5],
    targetDistance: 10,
    orientationForward: [1, 0, 0]
  });
  const [training, setTraining] = useState(false);
  const [trained, setTrained] = useState(false);
  const [modelInfo, setModelInfo] = useState(null);
  const [logs, setLogs] = useState([]);
  const [chartState, setChartState] = useState({ labels: [], rewards: [], losses: [] });
  const [homeHover, setHomeHover] = useState(false);

  const wsRef = useRef(null);

  const addLog = (txt) => {
    setLogs((l) => {
      const upd = [...l, txt];
      return upd.length > 200 ? upd.slice(upd.length - 200) : upd;
    });
  };

  useEffect(() => {
    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;

    ws.onopen = () => addLog('WS opened');
    ws.onmessage = (ev) => {
      addLog(ev.data);
      let parsed;
      try {
        parsed = JSON.parse(ev.data);
      } catch {
        return;
      }
      if (parsed.type === 'train_step') {
        setState((prev) => ({ ...prev, ...parsed.state }));
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
        setModelInfo({ filename: parsed.model_filename, timestamp: parsed.timestamp, sessionUuid: parsed.session_uuid, fileUrl: parsed.file_url });
      }
      if (parsed.type === 'run_step') {
        setState((prev) => ({ ...prev, ...parsed.state }));
      }
    };
    ws.onclose = () => addLog('WS closed');
    return () => ws.close();
  }, []);

  const send = (obj) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(obj));
    }
  };

  const startTraining = () => {
    if (training || trained) return;
    setTraining(true);
    send({ cmd: 'train' });
  };

  const startRun = () => {
    if (!trained) return;
    send({ cmd: 'run' });
  };

  const resetTraining = () => {
    setTraining(false);
    setTrained(false);
    setModelInfo(null);
    if (wsRef.current && wsRef.current._crawlerActionHandler) {
      delete wsRef.current._crawlerActionHandler;
    }
  };

  const { basePos, baseOri, jointAngles } = state;
  const bulletToThreeQuat = (q) => [q[0], q[2], -q[1], q[3]];
  const threePos = basePos ? [basePos[0], basePos[2], -basePos[1]] : [0, 0, 0];
  const threeQuat = baseOri ? bulletToThreeQuat(baseOri) : [0, 0, 0, 1];
  
  // Convert target position to Three.js coordinates
  const targetThreePos = state.targetPos ? [state.targetPos[0], state.targetPos[2], -state.targetPos[1]] : [10, 0.5, 0];
  const orientationForward = state.orientationForward || [1, 0, 0];
  
  // Convert MuJoCo direction [x, y, z] to Three.js direction [x, z, -y]
  // Then calculate Y-axis rotation: atan2(threeZ, threeX) = atan2(-mujocoY, mujocoX)
  const targetDirection = Math.atan2(-orientationForward[1], orientationForward[0]);

  return (
    <div style={{ width: '100%', height: '100%', background: 'linear-gradient(to bottom, #08081c, #03030a)' }}>
      <Canvas camera={{ position: [0, 8, 10], fov: 50 }} style={{ background: 'transparent' }}>
        <ambientLight intensity={0.5} />
        <directionalLight position={[5, 10, 5]} intensity={1} />

        {/* Wider floor grid for better spatial reference */}
        <Grid args={[24, 24]} cellSize={1} position={[0, 0, 0]} />

        {/* Target position indicator */}
        <mesh position={targetThreePos}>
          <boxGeometry args={[0.3, 0.3, 0.3]} />
          <meshStandardMaterial color="#00ff00" transparent opacity={0.7} />
        </mesh>
        
        {/* Direction arrow from agent to target */}
        <group position={threePos}>
          <group rotation={[0, targetDirection - Math.PI/2, 0]}>
            {/* Arrow shaft pointing forward (+Z direction) */}
            <mesh position={[0, 0.4, 0.2]} rotation={[Math.PI/2, 0, 0]}>
              <cylinderGeometry args={[0.02, 0.02, 0.3, 8]} />
              <meshStandardMaterial color="#ffff00" />
            </mesh>
            {/* Arrow head pointing forward */}
            <mesh position={[0, 0.4, 0.4]} rotation={[Math.PI/2, 0, 0]}>
              <coneGeometry args={[0.05, 0.15, 8]} />
              <meshStandardMaterial color="#ffff00" />
            </mesh>
          </group>
        </group>

        <group position={threePos} quaternion={threeQuat}>
          <mesh>
            <boxGeometry args={[0.6, 0.16, 0.4]} />
            <meshStandardMaterial color="#00aaff" />
          </mesh>
          {Array.from({ length: 4 }).map((_, i) => {
            // Joint mapping matches MuJoCo order: FL=0, FR=1, BL=2, BR=3
            // Hip joints: indices 0-7 (hip_x, hip_y for each leg)
            // Knee joints: indices 8-11 (one per leg)
            const hipXAngle = jointAngles[i * 2] || 0;        // FL=0, FR=2, BL=4, BR=6
            const hipYAngle = jointAngles[i * 2 + 1] || 0;    // FL=1, FR=3, BL=5, BR=7
            const kneeAngle = jointAngles[8 + i] || 0;        // FL=8, FR=9, BL=10, BR=11
            
            // Leg positioning: FL, FR, BL, BR
            const side = i < 2 ? 1 : -1; // left/right (FL/BL vs FR/BR)
            const frontBack = i % 2 === 0 ? 1 : -1; // front/back (FL/FR vs BL/BR)
            
            // Updated leg attachment points (wider spacing to match MuJoCo)
            const upperAnchor = [0.25 * side, -0.04, 0.15 * frontBack];
            
            return (
              <group key={i} position={upperAnchor}>
                {/* Hip Y rotation (side-to-side) */}
                <group rotation={[0, hipYAngle, 0]}>
                  {/* Hip X rotation (up-down) */}
                  <group rotation={[hipXAngle, 0, 0]}>
                    {/* Upper leg segment */}
                    <mesh position={[0, -0.12, 0]} rotation={[0, 0, 0]}>
                      <cylinderGeometry args={[0.04, 0.04, 0.24, 8]} />
                      <meshStandardMaterial color="#ffaa00" />
                    </mesh>
                    
                    {/* Lower leg segment with knee rotation */}
                    <group position={[0, -0.24, 0]} rotation={[kneeAngle, 0, 0]}>
                      <mesh position={[0, -0.12, 0]} rotation={[0, 0, 0]}>
                        <cylinderGeometry args={[0.04, 0.04, 0.24, 8]} />
                        <meshStandardMaterial color="#ffdd55" />
                      </mesh>
                      
                      {/* Foot */}
                      <mesh position={[0, -0.24, 0]}>
                        <sphereGeometry args={[0.05, 8, 8]} />
                        <meshStandardMaterial color="#ff6600" />
                      </mesh>
                    </group>
                  </group>
                </group>
              </group>
            );
          })}
        </group>

        <OrbitControls target={[0, 0, 0]} enablePan enableRotate enableZoom />
      </Canvas>

      <div style={{ position: 'absolute', top: 10, left: 10, color: '#fff' }}>
        <Link
          to="/"
          style={{ fontFamily: 'monospace', color: '#fff', textDecoration: homeHover ? 'none' : 'underline', display: 'inline-block' }}
          onMouseEnter={() => setHomeHover(true)}
          onMouseLeave={() => setHomeHover(false)}
        >
          Home
        </Link>
        <Text h1 style={{ margin: '0 0 12px 0', color: '#fff' }}>
          Crawler Example
        </Text>
        <div style={{ display: 'flex', gap: '8px' }}>
          <Button auto type="secondary" disabled={training || trained} onClick={startTraining}>
            Train
          </Button>
          <Button auto type="success" disabled={!trained} onClick={startRun}>
            Run
          </Button>
          {trained && (
            <Button auto type="error" onClick={resetTraining}>
              Reset
            </Button>
          )}
        </div>
      </div>

      <div style={{ position: 'absolute', bottom: 10, left: 10, width: 'auto', maxWidth: '420px', background: 'rgba(0,0,0,0.95)', color: '#fff', padding: '6px 8px', fontSize: 14, textAlign: 'left', justifyContent: 'flex-start' }}>
        <BlockMath
          math={"r_{step} = r_{speed} \\times r_{heading} \\times r_{upright} \\times r_{yaw}"}
          style={{ textAlign: 'left' }}
        />
        <div style={{ fontSize: 10, fontFamily: 'monospace', marginTop: 4 }}>
          Geometric reward combining speed alignment, heading alignment, uprightness, and yaw stability.
        </div>
        <div style={{ fontSize: 10, fontFamily: 'monospace', marginTop: 8, borderTop: '1px solid #333', paddingTop: 4 }}>
          Target Distance: {state.targetDistance ? state.targetDistance.toFixed(2) : 'N/A'}m<br/>
          Target Position: ({state.targetPos ? state.targetPos.map(x => x.toFixed(1)).join(', ') : 'N/A'})<br/>
          Direction: ({state.orientationForward ? state.orientationForward.map(x => x.toFixed(2)).join(', ') : 'N/A'})<br/>
          Forward Velocity: {state.forwardVelocity ? state.forwardVelocity.toFixed(2) : 'N/A'} m/s<br/>
          Heading Alignment: {state.headingAlignment ? state.headingAlignment.toFixed(2) : 'N/A'} (-1=opposite, +1=aligned)<br/>
          <span style={{ color: state.uprightScore > 0.8 ? '#00ff00' : state.uprightScore > 0.6 ? '#ffaa00' : '#ff0000' }}>
            Upright Score: {state.uprightScore ? state.uprightScore.toFixed(2) : 'N/A'} (1.0=upright, 0.0=sideways, -1.0=upside-down)
          </span><br/>
          Torso Height: {state.torsoHeight ? state.torsoHeight.toFixed(2) : 'N/A'}m (target: 0.5m)<br/>
          <div style={{ fontSize: 9, marginTop: 4, borderTop: '1px solid #444', paddingTop: 2 }}>
            DEBUG - Body Axes vs Up: X={state.uprightX ? state.uprightX.toFixed(2) : 'N/A'}, Y={state.uprightY ? state.uprightY.toFixed(2) : 'N/A'}, Z={state.uprightZ ? state.uprightZ.toFixed(2) : 'N/A'}<br/>
            Target Direction: {orientationForward.map(x => x.toFixed(2)).join(', ')} → Angle: {(targetDirection * 180 / Math.PI).toFixed(1)}°
          </div>
        </div>
      </div>

      <div style={{ position: 'absolute', bottom: 160, right: 10, width: '30%', height: '180px', background: 'rgba(255,255,255,0.05)', padding: 4 }}>
        <ChartPanel labels={chartState.labels} rewards={chartState.rewards} losses={chartState.losses} />
      </div>

      <DebugConsole logs={logs} />

      <ButtonForkOnGithub position={{ top: '20px', right: '20px' }} />
    </div>
  );
} 