import React, { useState, useEffect } from 'react';
import * as THREE from 'three';

const TrafficLight = ({ light, coordinateTransformer }) => {
    const { pos, state } = light;
    const [ecefPosition, setEcefPosition] = useState(null);
    const [orientation, setOrientation] = useState(new THREE.Quaternion());

    useEffect(() => {
        if (coordinateTransformer) {
            const [lat, lng] = pos;
            const vector = coordinateTransformer.latLngToECEF(lat, lng, 1);
            setEcefPosition(vector);
            
            const up = vector.clone().normalize();
            const newOrientation = new THREE.Quaternion().setFromUnitVectors(
                new THREE.Vector3(0, 1, 0), // Default cylinder's 'up'
                up                          // Target 'up' on the globe
            );
            setOrientation(newOrientation);
        }
    }, [pos, coordinateTransformer]);

    const color = state === 'green' ? '#00ff00' : '#ff0000';

    if (!ecefPosition) return null;

    return (
        <group position={ecefPosition} quaternion={orientation}>
            <mesh position={[0, 15, 0]}>
                <cylinderGeometry args={[1, 1, 30, 12]} />
                <meshStandardMaterial color="#333333" />
            </mesh>
            <mesh position={[0, 31, 0]}>
                <sphereGeometry args={[3, 16, 16]} />
                <meshStandardMaterial color={color} emissive={color} emissiveIntensity={3} />
            </mesh>
        </group>
    );
};

export default TrafficLight; 