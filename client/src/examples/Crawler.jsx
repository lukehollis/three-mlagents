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
    basePos: [0, 0, 0.35],
    baseOri: [0, 0, 0, 1],
    jointAngles: Array(8).fill(0),
  });
  const [training, setTraining] = useState(false);
  const [trained, setTrained] = useState(false);
  const [modelInfo, setModelInfo] = useState(null);
  const [logs, setLogs] = useState([]);
  const [chartState, setChartState] = useState({ labels: [], rewards: [], losses: [] });
  const [homeHover, setHomeHover] = useState(false);

  const envRef = useRef({ ...state, steps: 0 });
  const wsRef = useRef(null);
  const intervalRef = useRef(null);

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
      if (parsed.type === 'action') {
        if (wsRef.current._crawlerActionHandler) {
          wsRef.current._crawlerActionHandler(parsed.action);
        }
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
    if (intervalRef.current) return;

    const resetLocalEnv = () => {
      const gridSize = 30;
      const agentX = 0;
      const heading = 0;
      const goalX = gridSize - 1;
      envRef.current = { gridSize, agentX, heading, goalX, steps: 0 };
      setState((prev) => ({ ...prev, gridSize, agentX, heading, goalX }));
    };

    const stepLocalEnv = (actionIdx) => {
      const st = envRef.current;
      let heading = st.heading;
      let x = st.agentX;

      if (actionIdx === 1) heading += 0.25; // left
      if (actionIdx === 2) heading -= 0.25; // right
      if (actionIdx === 3) x = Math.min(st.gridSize - 1, x + Math.cos(heading));

      // wrap heading to -π..π
      if (heading >= Math.PI) heading -= 2 * Math.PI;
      if (heading < -Math.PI) heading += 2 * Math.PI;

      const steps = st.steps + 1;
      const done = x >= st.goalX || steps >= 200;

      envRef.current = { ...st, agentX: x, heading, steps };
      setState((prev) => ({ ...prev, agentX: x, heading }));

      if (done) resetLocalEnv();
    };

    resetLocalEnv();

    intervalRef.current = setInterval(() => {
      const st = envRef.current;
      const dx_goal = (st.goalX - st.agentX) / Math.max(1, st.gridSize - 1);
      const vel_norm = 0; // local env does not track velocity for inference
      const tgt_speed_norm = 0.5; // dummy
      const obs = [dx_goal, Math.cos(st.heading), vel_norm, tgt_speed_norm, Math.sin(st.heading)];
      send({ cmd: 'inference', obs });
    }, 200);

    wsRef.current._crawlerActionHandler = (aIdx) => stepLocalEnv(aIdx);
  };

  const resetTraining = () => {
    setTraining(false);
    setTrained(false);
    setModelInfo(null);
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    if (wsRef.current && wsRef.current._crawlerActionHandler) {
      delete wsRef.current._crawlerActionHandler;
    }
  };

  const { basePos, baseOri, jointAngles } = state;

  return (
    <div style={{ width: '100%', height: '100%', background: 'linear-gradient(to bottom, #08081c, #03030a)' }}>
      <Canvas camera={{ position: [0, 8, 10], fov: 50 }} style={{ background: 'transparent' }}>
        <ambientLight intensity={0.5} />
        <directionalLight position={[5, 10, 5]} intensity={1} />

        <Grid args={[20, 1]} cellSize={1} position={[0, 0, 0]} />

        <group position={basePos} quaternion={baseOri}>
          <mesh>
            <boxGeometry args={[0.4, 0.2, 0.2]} />
            <meshStandardMaterial color="#00aaff" />
          </mesh>
          {Array.from({ length: 4 }).map((_, i) => {
            const upperAngle = jointAngles[i] || 0;
            const lowerAngle = jointAngles[4 + i] || 0;
            const side = i < 2 ? 1 : -1; // left/right
            const upDown = i % 2 === 0 ? 1 : -1;
            const upperAnchor = [0.2 * side, 0.05 * upDown, 0];
            return (
              <group key={i} position={upperAnchor} rotation={[0, upperAngle, 0]}>
                {/* upper */}
                <mesh position={[0, 0, -0.15]} rotation={[Math.PI / 2, 0, 0]}>
                  <cylinderGeometry args={[0.05, 0.05, 0.3, 8]} />
                  <meshStandardMaterial color="#ffaa00" />
                </mesh>
                {/* lower */}
                <group position={[0, 0, -0.3]} rotation={[0, lowerAngle, 0]}>
                  <mesh position={[0, 0, -0.15]} rotation={[Math.PI / 2, 0, 0]}>
                    <cylinderGeometry args={[0.05, 0.05, 0.3, 8]} />
                    <meshStandardMaterial color="#ffdd55" />
                  </mesh>
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
          math={"r_{step} = r_{speed} \\times r_{heading}"}
          style={{ textAlign: 'left' }}
        />
        <div style={{ fontSize: 10, fontFamily: 'monospace', marginTop: 4 }}>
          Geometric reward combining speed alignment and heading alignment.
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