import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import ExamplesIndex from './examples/Index.jsx';
import BasicExample from './examples/Basic.jsx';
import Ball3DExample from './examples/Ball3D.jsx';
import GridWorldExample from './examples/GridWorld.jsx';
import PushExample from './examples/Push.jsx';
import WallJumpExample from './examples/WallJump.jsx';
import CrawlerExample from './examples/Crawler.jsx';

export default function App() {
  return (
    <BrowserRouter basename="/three-mlagents">
      {/* Global style override for KaTeX display alignment */}
      <style>{`
      .katex-display{ text-align:left !important; }
      .katex-display > .katex{ text-align:left !important; }
      `}</style>
      <Routes>
        <Route path="/" element={<ExamplesIndex />} />
        <Route path="/basic" element={<BasicExample />} />
        <Route path="/ball3d" element={<Ball3DExample />} />
        <Route path="/gridworld" element={<GridWorldExample />} />
        <Route path="/push" element={<PushExample />} />
        <Route path="/walljump" element={<WallJumpExample />} />
        <Route path="/crawler" element={<CrawlerExample />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  );
} 