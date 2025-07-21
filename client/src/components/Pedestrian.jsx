import React, { useState, useEffect, useMemo } from 'react';
import { Text as DreiText } from '@react-three/drei';
import * as THREE from 'three';

const Pedestrian = ({ pedestrian, coordinateTransformer }) => {
    const { pos, state, id } = pedestrian;
    const [pedPosition, setPedPosition] = useState(null);
    const [orientation, setOrientation] = useState(new THREE.Quaternion());

    useEffect(() => {
        if (coordinateTransformer) {
            const [lat, lng] = pos;
            const vector = coordinateTransformer.latLngToECEF(lat, lng, 1); // Elevate slightly
            setPedPosition(vector);

            const up = vector.clone().normalize();
            const newOrientation = new THREE.Quaternion().setFromUnitVectors(
                new THREE.Vector3(0, 1, 0), // Default model's 'up'
                up                          // Target 'up' on the globe
            );
            setOrientation(newOrientation);
        }
    }, [pos, coordinateTransformer]);

    const baseColor = useMemo(() => new THREE.Color(state === 'jaywalking' ? '#ff4444' : '#ffffff'), [state]);
    const bodyColor = baseColor.clone();
    const headColor = baseColor.clone().multiplyScalar(1.1);
    const limbColor = baseColor.clone().multiplyScalar(0.9);

    if (!pedPosition) return null;
    
    const scale = 8;

    return (
        <group position={pedPosition} quaternion={orientation}>
          <group position={[0, 6, 0]}>
            {/* Head */}
            <mesh position={[0, 0.75 * scale, 0]}>
                <boxGeometry args={[0.5 * scale, 0.5 * scale, 0.5 * scale]} />
                <meshPhongMaterial color={headColor} />
            </mesh>
            
            {/* Body */}
            <mesh position={[0, 0.1 * scale, 0]}>
                <boxGeometry args={[0.5 * scale, 0.75 * scale, 0.25 * scale]} />
                <meshPhongMaterial color={bodyColor} />
            </mesh>
            
            {/* Left Arm */}
            <mesh position={[-0.45 * scale, 0.2 * scale, 0]}>
                <boxGeometry args={[0.25 * scale, 0.6 * scale, 0.25 * scale]} />
                <meshPhongMaterial color={limbColor} />
            </mesh>
            
            {/* Right Arm */}
            <mesh position={[0.45 * scale, 0.2 * scale, 0]}>
                <boxGeometry args={[0.25 * scale, 0.6 * scale, 0.25 * scale]} />
                <meshPhongMaterial color={limbColor} />
            </mesh>
            
            {/* Left Leg */}
            <mesh position={[-0.15 * scale, -0.45 * scale, 0]}>
                <boxGeometry args={[0.25 * scale, 0.6 * scale, 0.25 * scale]} />
                <meshPhongMaterial color={limbColor} />
            </mesh>
            
            {/* Right Leg */}
            <mesh position={[0.15 * scale, -0.45 * scale, 0]}>
                <boxGeometry args={[0.25 * scale, 0.6 * scale, 0.25 * scale]} />
                <meshPhongMaterial color={limbColor} />
            </mesh>
            
            {/* Agent ID label above head */}
            <DreiText position={[0, 1.4 * scale, 0]} fontSize={0.3 * scale} color="white" anchorX="center" anchorY="middle">
              {`P${id}`}
            </DreiText>
          </group>
        </group>
    );
};

export default Pedestrian; 