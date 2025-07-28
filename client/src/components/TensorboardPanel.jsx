import React, { useState } from 'react';
import { Button } from '@geist-ui/core';
import TensorBoardViewer from './TensorBoardViewer.jsx';
import DebugConsole from './DebugConsole.jsx';
import { ChevronsDown, ChevronsUp, Activity } from '@geist-ui/icons';
import { useResponsive } from '../hooks/useResponsive';

export default function TensorboardPanel({ logs }) {
  const { isMobile } = useResponsive();
  const [isOpen, setIsOpen] = useState(!isMobile);

  if (!isOpen) {
    return (
      <div style={{ position: 'absolute', bottom: 10, right: 10, zIndex: 1 }}>
        <Button auto icon={<Activity />} onClick={() => setIsOpen(true)} type="secondary">
          TensorBoard
        </Button>
      </div>
    );
  }

  return (
    <div
      style={{
        position: 'absolute',
        bottom: 10,
        right: 10,
        width: isMobile ? 'calc(100% - 20px)' : '45%',
        display: 'flex',
        flexDirection: 'column',
        gap: '10px',
        zIndex: 10,
        maxWidth: '420px',
      }}
    >
      <div
        style={{
          position: 'relative',
        }}
      >
        <Button
          auto
          scale={0.5}
          icon={<ChevronsDown />}
          onClick={() => setIsOpen(false)}
          style={{ position: 'absolute', top: 5, right: 5, zIndex: 10 }}
          type="abort"
        />
        {/* <TensorBoardViewer height="180px" /> */}
      </div>
      <div style={{ width: 'calc(100% - 15px)' }}>
        <DebugConsole logs={logs} />
      </div>
    </div>
  );
} 