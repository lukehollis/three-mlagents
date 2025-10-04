import React, { useRef } from 'react';
import { useFrame } from '@react-three/fiber';

const Explosion = ({ position, onEnded }) => {
  const ref = useRef();
  const timeToLive = 0.5; // seconds
  let time = 0;

  useFrame((state, delta) => {
    time += delta;
    if (time > timeToLive) {
      if (onEnded) onEnded();
      return;
    }
    const progress = time / timeToLive;
    const scale = progress * 20;
    if (ref.current) {
        ref.current.scale.set(scale, scale, scale);
        ref.current.material.opacity = 1 - progress;
    }
  });

  return (
    <mesh ref={ref} position={position}>
      <sphereGeometry args={[1, 16, 16]} />
      <meshStandardMaterial color="orange" emissive="orange" emissiveIntensity={2} transparent toneMapped={false}/>
    </mesh>
  );
};

export default Explosion;

