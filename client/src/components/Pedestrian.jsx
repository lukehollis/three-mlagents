import React, { useState, useEffect, useMemo } from 'react';
import { Text as DreiText } from '@react-three/drei';
import * as THREE from 'three';

const Pedestrian = ({ pedestrian, coordinateTransformer }) => {
    const { pos, state, id, color, satisfaction, resources } = pedestrian;
    const [pedPosition, setPedPosition] = useState(null);

    useEffect(() => {
        if (coordinateTransformer) {
            const [lat, lng] = pos;
            const vector = coordinateTransformer.latLngToECEF(lat, lng, 1); // Elevate slightly
            setPedPosition(vector);
        }
    }, [pos, coordinateTransformer]);

    const baseColor = useMemo(() => {
        if (color && Array.isArray(color)) {
            return new THREE.Color(color[0], color[1], color[2]);
        }
        // Fallback to state-based colors
        switch(state) {
            case 'shopping': return new THREE.Color('#4ecdc4');
            case 'working': return new THREE.Color('#45b7d1');
            case 'traveling': return new THREE.Color('#ffeaa7');
            case 'wandering': return new THREE.Color('#ffffff');
            default: return new THREE.Color('#ffffff');
        }
    }, [color, state]);

    const satisfactionColor = useMemo(() => {
        if (!satisfaction) return new THREE.Color('#ffffff');
        // Color based on satisfaction: red -> yellow -> green
        if (satisfaction < 30) return new THREE.Color('#ff4757');
        if (satisfaction < 70) return new THREE.Color('#ffa502');
        return new THREE.Color('#2ed573');
    }, [satisfaction]);

    const bodyColor = baseColor.clone();
    const headColor = satisfactionColor.clone();
    const limbColor = baseColor.clone().multiplyScalar(0.8);

    if (!pedPosition) return null;
    
    const scale = 6; // Made slightly smaller to fit more pedestrians

    return (
        <group position={pedPosition}>
          <group position={[0, 6, 0]}>
            {/* Head - colored by satisfaction */}
            <mesh position={[0, 0.75 * scale, 0]}>
                <boxGeometry args={[0.5 * scale, 0.5 * scale, 0.5 * scale]} />
                <meshPhongMaterial 
                    color={headColor} 
                    emissive={headColor} 
                    emissiveIntensity={0.2}
                />
            </mesh>
            
            {/* Body - colored by agent color */}
            <mesh position={[0, 0.1 * scale, 0]}>
                <boxGeometry args={[0.5 * scale, 0.75 * scale, 0.25 * scale]} />
                <meshPhongMaterial 
                    color={bodyColor}
                    emissive={bodyColor}
                    emissiveIntensity={0.1}
                />
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
            
            {/* Agent ID and state label above head */}
            <DreiText 
                position={[0, 1.4 * scale, 0]} 
                fontSize={0.25 * scale} 
                color="white" 
                anchorX="center" 
                anchorY="middle"
            >
              {`P${id}`}
            </DreiText>
            
            {/* Activity state */}
            <DreiText 
                position={[0, 1.7 * scale, 0]} 
                fontSize={0.2 * scale} 
                color="#ffff88" 
                anchorX="center" 
                anchorY="middle"
            >
              {state}
            </DreiText>
            
            {/* Satisfaction indicator */}
            {satisfaction !== undefined && (
                <DreiText 
                    position={[0, 2.0 * scale, 0]} 
                    fontSize={0.15 * scale} 
                    color={satisfactionColor} 
                    anchorX="center" 
                    anchorY="middle"
                >
                  {`${satisfaction.toFixed(0)}%`}
                </DreiText>
            )}

            {/* Money indicator (if available) */}
            {resources && resources.money !== undefined && (
                <DreiText 
                    position={[0, 2.3 * scale, 0]} 
                    fontSize={0.12 * scale} 
                    color="#90ee90" 
                    anchorX="center" 
                    anchorY="middle"
                >
                  ${resources.money.toFixed(0)}
                </DreiText>
            )}
          </group>
        </group>
    );
};

export default Pedestrian; 