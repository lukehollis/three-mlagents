import React, { useState, useEffect, useRef } from 'react';
import { Button, Text } from '@geist-ui/core';
import { Link } from 'react-router-dom';
import config from '../config.js';
import ButtonForkOnGithub from '../components/ButtonForkOnGithub.jsx';
import 'katex/dist/katex.min.css';
import EquationPanel from '../components/EquationPanel.jsx';
import InfoPanel from '../components/InfoPanel.jsx';
import ModelInfoPanel from '../components/ModelInfoPanel.jsx';
import { useResponsive } from '../hooks/useResponsive.js';

const WS_URL = `${config.WS_BASE_URL}/ws/labyrinth`;

// Component to render the labyrinth
const LabyrinthGrid = ({ grid, fontSize }) => {
  if (!grid) return null;

  const getCellDisplay = (cell) => {
    let style = { color: '#fff' };
    let displayChar = cell;

    switch (cell) {
      case '#':
        style.color = '#888'; // Wall
        break;
      case 'E':
        style.color = '#4caf50'; // Exit
        break;
      case ' ':
        displayChar = '·';
        style.color = '#444'; // Path
        break;
      case 'T':
        style.color = '#2196f3'; // Theseus
        break;
      case 'M':
        style.color = '#f44336'; // Minotaur
        break;
      default:
        break;
    }
    return { style, displayChar };
  };

  return (
    <pre style={{
      fontFamily: 'monospace',
      lineHeight: '1.0',
      fontSize: `${fontSize}px`,
      margin: 'auto',
      textAlign: 'center'
    }}>
      {grid.map((row, y) => (
        <div key={y}>
          {row.map((cell, x) => {
            const { style, displayChar } = getCellDisplay(cell);
            return <span key={x} style={style}>{displayChar}</span>;
          })}
        </div>
      ))}
    </pre>
  );
};

export default function LabyrinthExample() {
  const [state, setState] = useState(null);
  const [training, setTraining] = useState(false);
  const [trained, setTrained] = useState(false);
  const [modelInfo, setModelInfo] = useState(null);
  const [logs, setLogs] = useState([]);
  const [chartState, setChartState] = useState({ labels: [], rewards: [], losses: [] });
  const wsRef = useRef(null);
  const [homeHover, setHomeHover] = useState(false);
  const { isMobile } = useResponsive();
  const [fontSize, setFontSize] = useState(20);

  const addLog = (txt) => {
    setLogs((l) => {
      const upd = [...l, txt];
      return upd.length > 100 ? upd.slice(upd.length - 100) : upd;
    });
  };
  
  useEffect(() => {
    const calculateFontSize = () => {
        if (state && state.grid) {
            const gridWidth = state.grid[0].length;
            const gridHeight = state.grid.length;
            const availableWidth = window.innerWidth * (isMobile ? 0.9 : 0.9);
            const availableHeight = window.innerHeight * 0.9;
            
            // Approximate character width as 0.6 times font size
            const sizeFromWidth = Math.floor(availableWidth / (gridWidth * 0.6));
            const sizeFromHeight = Math.floor(availableHeight / gridHeight);
            
            setFontSize(Math.max(8, Math.min(sizeFromWidth, sizeFromHeight)));
        }
    };

    calculateFontSize();
    window.addEventListener('resize', calculateFontSize);
    return () => window.removeEventListener('resize', calculateFontSize);
}, [state, isMobile]);


  useEffect(() => {
    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;
    ws.onopen = () => addLog('WS opened');
    ws.onmessage = (ev) => {
      let parsed;
      try {
        parsed = JSON.parse(ev.data);
      } catch {
        addLog(`Received non-JSON: ${ev.data}`);
        return;
      }

      if ((parsed.type === 'train_step' || parsed.type === 'run_step' || parsed.type === 'state') && parsed.state) {
        setState(parsed.state);
      } else if (parsed.type === 'progress') {
        addLog(`Episode ${parsed.episode}: Reward=${parsed.reward.toFixed(3)}, Loss=${(parsed.loss ?? 0).toFixed(3)}`);
        setChartState((prev) => ({
          labels: [...prev.labels, parsed.episode],
          rewards: [...prev.rewards, parsed.reward],
          losses: [...prev.losses, parsed.loss ?? null],
        }));
      } else if (parsed.type === 'trained') {
        addLog(`Training complete. Model: ${parsed.model_filename}`);
        setTraining(false);
        setTrained(true);
        setModelInfo({ 
          filename: parsed.model_filename, 
          timestamp: parsed.timestamp, 
          sessionUuid: parsed.session_uuid, 
          fileUrl: parsed.file_url 
        });
      } else {
        addLog(ev.data);
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
    addLog('Starting training...');
    send({ cmd: 'train' });
  };

  const startRun = () => {
    if (!trained || !modelInfo) return;
    send({ cmd: 'run', model_filename: modelInfo.filename });
  };

  const resetTraining = () => {
    setTraining(false);
    setTrained(false);
    setModelInfo(null);
    setChartState({ labels: [], rewards: [], losses: [] });
    setState(null);
    addLog('Training has been reset.');
    // After resetting, we might need to ask the server for a fresh state.
    // A simple way is to just reconnect. Or have a 'reset' command.
    // For now, we just clear the state, a reconnect might be needed if it doesn't reappear.
  };

  return (
    <div
      style={{
        width: '100vw',
        height: '100vh',
        overflow: 'hidden',
        background: '#000',
        color: '#fff',
        display: 'flex',
        flexDirection: 'column',
      }}
    >
      <div style={{ flex: 1, position: 'relative' }}>
        <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh', overflow: 'scroll' }}>
          <LabyrinthGrid grid={state?.grid} fontSize={fontSize} />
        </div>

        {/* UI overlay */}
        <div
          style={{
            position: 'absolute',
            top: 10,
            left: 10,
            color: '#fff',
            textShadow: '0 0 4px #000',
            zIndex: 1,
          }}
        >
          <Link
            to="/"
            style={{
              fontFamily: 'monospace',
              color: '#fff',
              textDecoration: homeHover ? 'none' : 'underline',
              display: 'inline-block',
              fontSize: isMobile ? '12px' : '14px',
            }}
            onMouseEnter={() => setHomeHover(true)}
            onMouseLeave={() => setHomeHover(false)}
          >
            Home
          </Link>
          <Text h1 style={{ margin: '12px 0 12px 0', color: '#fff', fontSize: isMobile ? '1.2rem' : '2rem' }}>
            Labyrinth
          </Text>
          <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
            <Button auto type="secondary" disabled={training || trained} onClick={startTraining}>Train</Button>
            <Button auto type="success" disabled={!trained} onClick={startRun}>Run</Button>
            {trained && <Button auto type="error" onClick={resetTraining}>Reset</Button>}
          </div>
          <ModelInfoPanel modelInfo={modelInfo} />
        </div>
        <EquationPanel
          equation="\begin{aligned} \text{Reward} &= f(\Delta d_{exit}, \Delta d_{minotaur}) \\ \text{Obs} &= \text{Flatten}(\text{Grid}[y, x]) \end{aligned}"
          description="A simple reward function based on distance to the exit and the minotaur, with a flattened grid observation space."
        />
        <InfoPanel logs={logs} chartState={chartState} />
        <ButtonForkOnGithub position={{ top: '10px', right: '10px' }} />
      </div>
    </div>
  );
} 