import React, { useRef } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { Line } from '@react-three/drei';

function InfiniteGrid() {
  const groupRef = useRef();

  // Grid parameters
  const size = 120;
  const divisions = 60;
  const step = size / divisions;
  const speed = 4; // units per second

  useFrame((state) => {
    if (groupRef.current) {
      // Move grid backward along Z; negative direction gives forward motion illusion
      groupRef.current.position.z = -((state.clock.elapsedTime * speed) % step);
    }
  });

  const lines = [];
  for (let i = 0; i <= divisions; i++) {
    const v = (i * step) - size / 2;
    const fade = Math.abs(v) / (size / 2);
    const opacity = Math.max(0.04, (1 - fade) * 0.7);

    // horizontal lines (constant z)
    lines.push(
      <Line key={`h-${i}`} points={[[-size / 2, 0, v], [size / 2, 0, v]]} color="#00ffff" lineWidth={0.4} transparent opacity={opacity} />
    );
    // vertical lines (constant x)
    lines.push(
      <Line key={`v-${i}`} points={[[v, 0, -size / 2], [v, 0, size / 2]]} color="#00ffff" lineWidth={0.4} transparent opacity={opacity} />
    );
  }

  // place grid slightly below camera
  return (
    <group ref={groupRef} position={[0, -2, 0]} rotation={[0, 0, 0]}>
      {lines}
    </group>
  );
}

function Scene() {
  return (
    <>
      <color attach="background" args={['#000008']} />
      <fog attach="fog" args={['#000008', 15, 60]} />
      
      <InfiniteGrid />
      
      <ambientLight intensity={0.05} />
    </>
  );
}

export default function TronBackground() {
  return (
    <div style={{
      position: 'fixed',
      top: 0,
      left: 0,
      right: 0,
      bottom: 0,
      width: '100vw',
      height: '100vh',
      zIndex: 1,
      pointerEvents: 'none'
    }}>
      <Canvas
        camera={{
          position: [0, 4, 8],
          fov: 75,
          far: 100
        }}
      >
        <Scene />
      </Canvas>
    </div>
  );
} 