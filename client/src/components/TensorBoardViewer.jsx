import React, { useState, useEffect, useRef } from 'react';
import { Button, Loading } from '@geist-ui/core';
import { Activity, RefreshCw } from '@geist-ui/icons';
import config from '../config.js';

export default function TensorBoardViewer({ height = '300px' }) {
  const [isLoading, setIsLoading] = useState(true);
  const [isServerRunning, setIsServerRunning] = useState(false);
  const [error, setError] = useState(null);
  const iframeRef = useRef(null);
  const pollIntervalRef = useRef(null);

  const tensorboardUrl = `http://localhost:6006`;
  const statusUrl = `${config.API_BASE_URL}/tensorboard/status`;
  const startUrl = `${config.API_BASE_URL}/tensorboard/astrodynamics`;

  const checkTensorBoardStatus = async () => {
    try {
      const response = await fetch(statusUrl);
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      const data = await response.json();
      setIsServerRunning(data.running);
      if (data.running) {
        setIsLoading(false);
        setError(null);
      }
    } catch (err) {
      console.error('TensorBoard status check failed:', err);
      setError(`Failed to check TensorBoard status: ${err.message}`);
      setIsLoading(false);
    }
  };

  const startTensorBoard = async () => {
    try {
      setIsLoading(true);
      setError(null);
      
      // Call the start endpoint
      const startResponse = await fetch(startUrl);
      if (!startResponse.ok) {
        throw new Error(`Failed to start TensorBoard: HTTP ${startResponse.status}`);
      }
      
      // Poll for server to be ready
      let attempts = 0;
      const maxAttempts = 20;
      
      const pollForReady = async () => {
        try {
          const response = await fetch(statusUrl);
          if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
          }
          const data = await response.json();
          if (data.running) {
            setIsServerRunning(true);
            setIsLoading(false);
            setError(null);
            // Server is ready, reload iframe
            if (iframeRef.current) {
              iframeRef.current.src = tensorboardUrl;
            }
          } else if (attempts < maxAttempts) {
            attempts++;
            setTimeout(pollForReady, 500);
          } else {
            setError('TensorBoard server failed to start within timeout');
            setIsLoading(false);
          }
        } catch (err) {
          console.error(`TensorBoard poll attempt ${attempts + 1} failed:`, err);
          if (attempts < maxAttempts) {
            attempts++;
            setTimeout(pollForReady, 500);
          } else {
            setError(`TensorBoard server failed to start: ${err.message}`);
            setIsLoading(false);
          }
        }
      };
      
      pollForReady();
    } catch (err) {
      console.error('Failed to start TensorBoard:', err);
      setError(`Failed to start TensorBoard server: ${err.message}`);
      setIsLoading(false);
    }
  };

  const refreshTensorBoard = () => {
    if (iframeRef.current) {
      iframeRef.current.src = iframeRef.current.src;
    }
  };

  useEffect(() => {
    // Automatically start TensorBoard on mount
    startTensorBoard();
    
    // Set up polling every 5 seconds
    pollIntervalRef.current = setInterval(checkTensorBoardStatus, 5000);
    
    return () => {
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current);
      }
    };
  }, []);

  if (error) {
    return (
      <div
        style={{
          height,
          background: 'rgba(0,0,0,0.6)',
          padding: '20px',
          borderRadius: '8px',
          border: '1px solid rgba(255,255,255,0.2)',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          color: '#fff',
          textAlign: 'center',
        }}
      >
        <div style={{ marginBottom: '15px', color: '#ff6b6b' }}>{error}</div>
        <Button auto onClick={startTensorBoard} type="secondary" icon={<RefreshCw />}>
          Retry
        </Button>
      </div>
    );
  }

  if (isLoading || !isServerRunning) {
    return (
      <div
        style={{
          height,
          background: 'rgba(0,0,0,0.6)',
          padding: '20px',
          borderRadius: '8px',
          border: '1px solid rgba(255,255,255,0.2)',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          color: '#fff',
          textAlign: 'center',
        }}
      >
        <Loading size="medium" />
        <div style={{ marginTop: '15px' }}>Starting TensorBoard...</div>
      </div>
    );
  }

  return (
    <div
      style={{
        height,
        background: 'rgba(0,0,0,0.6)',
        borderRadius: '8px',
        border: '1px solid rgba(255,255,255,0.2)',
        position: 'relative',
        overflow: 'hidden',
      }}
    >
      <div
        style={{
          position: 'absolute',
          top: 5,
          right: 5,
          zIndex: 10,
          display: 'flex',
          gap: '5px',
        }}
      >
        <Button auto scale={0.5} onClick={refreshTensorBoard} type="secondary" icon={<RefreshCw />} />
      </div>
      <iframe
        ref={iframeRef}
        src={tensorboardUrl}
        style={{
          width: '100%',
          height: '100%',
          border: 'none',
          borderRadius: '8px',
        }}
        title="TensorBoard"
      />
    </div>
  );
} 