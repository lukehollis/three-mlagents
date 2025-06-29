import React from 'react';
import { createRoot } from 'react-dom/client';
import { GeistProvider, CssBaseline } from '@geist-ui/core';
import '@fontsource-variable/geist';
import App from './App.jsx';

const container = document.getElementById('root');
createRoot(container).render(
  <GeistProvider>
    <CssBaseline />
    <App />
  </GeistProvider>
); 