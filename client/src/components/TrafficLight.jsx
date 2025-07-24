import React, { useState, useEffect } from 'react';
import * as THREE from 'three';

const TrafficLight = ({ light, coordinateTransformer }) => {
    const { pos, state } = light;
    const [ecefPosition, setEcefPosition] = useState(null);

    useEffect(() => {
        if (coordinateTransformer) {
            const [lat, lng] = pos;
            const vector = coordinateTransformer.latLngToECEF(lat, lng, 1);
            setEcefPosition(vector);
        }
    }, [pos, coordinateTransformer]);

    const color = state === 'green' ? '#00ff00' : '#ff0000';

    if (!ecefPosition) return null;

    return (
        <group position={ecefPosition}>
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