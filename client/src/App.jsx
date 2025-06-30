import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import ExamplesIndex from './examples/Index.jsx';
import BasicExample from './examples/Basic.jsx';
import Ball3DExample from './examples/Ball3D.jsx';

export default function App() {
  return (
    <BrowserRouter basename="/three-mlagents">
      <Routes>
        <Route path="/" element={<ExamplesIndex />} />
        <Route path="/basic" element={<BasicExample />} />
        <Route path="/ball3d" element={<Ball3DExample />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  );
} 