import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import ExamplesIndex from './examples/Index.jsx';
import BasicExample from './examples/Basic.jsx';
import Ball3DExample from './examples/Ball3D.jsx';
import GridWorldExample from './examples/GridWorld.jsx';
import PushExample from './examples/Push.jsx';
import WallJumpExample from './examples/WallJump.jsx';
import AntExample from './examples/Ant.jsx';
import WormExample from './examples/Worm.jsx';
import BrickBreakExample from './examples/BrickBreak.jsx';
import FoodCollectorExample from './examples/FoodCollector.jsx';
import BicycleExample from './examples/Bicycle.jsx';
import GliderExample from './examples/Glider.jsx';
import AstrodynamicsExample from './examples/Astrodynamics.jsx';
import LabyrinthExample from './examples/Labyrinth.jsx';
import MinecraftExample from './examples/Minecraft.jsx';
import SimCityExample from './examples/SimCity.jsx';
import SimCityKeplerExample from './examples/SimCityKepler.jsx';
import FishExample from './examples/Fish.jsx';
import IntersectionExample from './examples/Intersection.jsx';
import SelfDrivingCarExample from './examples/SelfDrivingCar.jsx';

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
        <Route path="/crawler" element={<AntExample />} />
        <Route path="/worm" element={<WormExample />} />
        <Route path="/brickbreak" element={<BrickBreakExample />} />
        <Route path="/foodcollector" element={<FoodCollectorExample />} />
        <Route path="/bicycle" element={<BicycleExample />} />
        <Route path="/glider" element={<GliderExample />} />
        <Route path="/astrodynamics" element={<AstrodynamicsExample />} />
        <Route path="/labyrinth" element={<LabyrinthExample />} />
        <Route path="/minecraft" element={<MinecraftExample />} />
        <Route path="/simcity" element={<SimCityExample />} />
        <Route path="/simcity-kepler" element={<SimCityKeplerExample />} />
        <Route path="/fish" element={<FishExample />} />
        <Route path="/intersection" element={<IntersectionExample />} />
        <Route path="/self-driving-car" element={<SelfDrivingCarExample />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  );
} 