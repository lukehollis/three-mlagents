import React, { useMemo } from 'react';
import * as THREE from 'three';

export default function Car({ car }) {
  const { pos, color } = car;
  const carColor = useMemo(() => new THREE.Color(...(color || [1, 1, 1])), [color]);

  if (!pos) return null;

  // The car's y-position is lifted by half its height to sit on the ground plane.
  return (
    <group position={[pos[0], pos[1] + 0.2, pos[2]]}>
      <mesh>
        <boxGeometry args={[0.8, 0.4, 0.4]} />
        <meshPhongMaterial
          color={carColor}
          emissive={carColor}
          emissiveIntensity={0.5}
        />
      </mesh>

      {/* Front lights (oriented along +X) */}
      <mesh position={[0.4, 0, 0.15]}>
        <sphereGeometry args={[0.08, 8, 8]} />
        <meshStandardMaterial color={"white"} emissive={"white"} emissiveIntensity={2} />
      </mesh>
      <mesh position={[0.4, 0, -0.15]}>
        <sphereGeometry args={[0.08, 8, 8]} />
        <meshStandardMaterial color={"white"} emissive={"white"} emissiveIntensity={2} />
      </mesh>

      {/* Back lights */}
      <mesh position={[-0.4, 0, 0.15]}>
        <sphereGeometry args={[0.08, 8, 8]} />
        <meshStandardMaterial color={"red"} emissive={"red"} emissiveIntensity={2} />
      </mesh>
      <mesh position={[-0.4, 0, -0.15]}>
        <sphereGeometry args={[0.08, 8, 8]} />
        <meshStandardMaterial color={"red"} emissive={"red"} emissiveIntensity={2} />
      </mesh>
    </group>
  );
} 