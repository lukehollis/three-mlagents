import React, { useRef, useMemo, useEffect } from 'react';
import { useFrame, useThree } from '@react-three/fiber';
import * as THREE from 'three';

export const LIDAR_LAYER = 1;

const Lidar = () => {
    const pointsRef = useRef();
    const { scene, camera } = useThree();
    const lastScanTime = useRef(0);
    const raycaster = useMemo(() => new THREE.Raycaster(), []);

    useEffect(() => {
        if (pointsRef.current) {
            pointsRef.current.layers.set(LIDAR_LAYER);
        }
    }, []);

    useFrame((state, delta) => {
        if (!pointsRef.current || !scene || !camera) return;

        lastScanTime.current += delta;
        if (lastScanTime.current < 0.1) {
            return;
        }
        lastScanTime.current = 0;

        raycaster.camera = camera;
        raycaster.layers.set(0);

        const origin = pointsRef.current.parent.position;
        const scanPoints = [];
        const objectsToIntersect = scene.children;

        const numRays = 180;
        for (let i = 0; i < numRays; i++) {
            const angle = (i / numRays) * Math.PI * 2;
            const direction = new THREE.Vector3(Math.cos(angle), 0, Math.sin(angle)).normalize();
            
            raycaster.set(origin, direction);
            raycaster.far = 1500;

            const intersects = raycaster.intersectObjects(objectsToIntersect, true);
            if (intersects.length > 0) {
                scanPoints.push(intersects[0].point);
            }
        }

        const geometry = pointsRef.current.geometry;
        const positions = new Float32Array(scanPoints.length * 3);
        scanPoints.forEach((p, i) => {
            positions[i * 3] = p.x;
            positions[i * 3 + 1] = p.y;
            positions[i * 3 + 2] = p.z;
        });

        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        geometry.computeBoundingSphere();
    });

    return (
        <points ref={pointsRef}>
            <bufferGeometry />
            <pointsMaterial 
                color="#00ffff" 
                size={15} 
                blending={THREE.AdditiveBlending}
                sizeAttenuation={true}
                transparent
                opacity={0.7}
            />
        </points>
    );
};

export default Lidar; 