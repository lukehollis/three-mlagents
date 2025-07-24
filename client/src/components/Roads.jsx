import React, { useState, useEffect } from 'react';
import { Line } from '@react-three/drei';
import * as THREE from 'three';

const Roads = ({ roadNetwork, coordinateTransformer }) => {
  const [lines, setLines] = useState([]);

  useEffect(() => {
    if (!roadNetwork || !coordinateTransformer) return;

    const newLines = roadNetwork.map((road, i) => {
      const points = road.map(p => {
        const [lat, lng] = p;
        return coordinateTransformer.latLngToECEF(lat, lng);
      });
      return (
        <Line
          key={i}
          points={points}
          color="white"
          lineWidth={1}
        />
      );
    });
    setLines(newLines);
  }, [roadNetwork, coordinateTransformer]);

  return <group>{lines}</group>;
};

export default Roads; 