import React, { useEffect, useState } from 'react';
import * as THREE from 'three';

const API_KEY = import.meta.env.VITE_GOOGLE_MAPS_API_KEY;

const Map2D = ({ onMapLoaded }) => {
  const [mapTexture, setMapTexture] = useState(null);

  useEffect(() => {
    if (API_KEY && onMapLoaded) {
      // San Francisco bounds
      const SF_CENTER = { lat: 37.7749, lng: -122.4194 };
      const MAP_SIZE = 1000; // Size of the 2D plane in Three.js units
      
      // Load Google Maps Static API as texture with dark theme
      const zoom = 15;
      const size = '640x640';
      const mapType = 'roadmap';
      
      // Dark theme styling parameters
      const darkStyles = [
        'style=element:geometry|color:0x242f3e',
        'style=element:labels.text.stroke|color:0x242f3e',
        'style=element:labels.text.fill|color:0x746855',
        'style=feature:administrative.locality|element:labels.text.fill|color:0xd59563',
        'style=feature:poi|element:labels.text.fill|color:0xd59563',
        'style=feature:poi.park|element:geometry|color:0x263c3f',
        'style=feature:poi.park|element:labels.text.fill|color:0x6b9a76',
        'style=feature:road|element:geometry|color:0x38414e',
        'style=feature:road|element:geometry.stroke|color:0x212a37',
        'style=feature:road|element:labels.text.fill|color:0x9ca5b3',
        'style=feature:road.highway|element:geometry|color:0x746855',
        'style=feature:road.highway|element:geometry.stroke|color:0x1f2835',
        'style=feature:road.highway|element:labels.text.fill|color:0xf3d19c',
        'style=feature:transit|element:geometry|color:0x2f3948',
        'style=feature:transit.station|element:labels.text.fill|color:0xd59563',
        'style=feature:water|element:geometry|color:0x17263c',
        'style=feature:water|element:labels.text.fill|color:0x515c6d',
        'style=feature:water|element:labels.text.stroke|color:0x17263c'
      ].join('&');
      
      const mapUrl = `https://maps.googleapis.com/maps/api/staticmap?center=${SF_CENTER.lat},${SF_CENTER.lng}&zoom=${zoom}&size=${size}&maptype=${mapType}&${darkStyles}&key=${API_KEY}`;
      
      const textureLoader = new THREE.TextureLoader();
      textureLoader.load(mapUrl, (texture) => {
        texture.wrapS = THREE.ClampToEdgeWrapping;
        texture.wrapT = THREE.ClampToEdgeWrapping;
        texture.minFilter = THREE.LinearFilter;
        setMapTexture(texture);
      });
      
      // Simple coordinate transformation for 2D mapping
      const latLngToECEF = (lat, lng, alt = 0) => {
        // Convert lat/lng to local coordinates relative to SF center
        const latDiff = (lat - SF_CENTER.lat) * 111000; // ~111km per degree
        const lngDiff = (lng - SF_CENTER.lng) * 111000 * Math.cos(SF_CENTER.lat * Math.PI / 180);
        
        // Scale to fit our map size
        const scale = MAP_SIZE / 1000; // Adjust scale as needed
        return new THREE.Vector3(
          lngDiff * scale,
          alt,
          -latDiff * scale // Negative to match typical map orientation
        );
      };
      
      const ecefToLatLng = (x, y, z) => {
        const scale = MAP_SIZE / 1000;
        const lngDiff = x / scale;
        const latDiff = -z / scale; // Negative because we flipped it above
        
        const lat = SF_CENTER.lat + (latDiff / 111000);
        const lng = SF_CENTER.lng + (lngDiff / (111000 * Math.cos(SF_CENTER.lat * Math.PI / 180)));
        
        return { lat, lng, alt: y };
      };
      
      onMapLoaded({ latLngToECEF, ecefToLatLng });
    }
  }, [onMapLoaded]);

  return (
    <group>
      {/* Ground plane with Google Map texture */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -1, 0]}>
        <planeGeometry args={[2000, 2000]} />
        {mapTexture ? (
          <meshPhongMaterial 
            map={mapTexture}
            transparent 
            opacity={0.9}
          />
        ) : (
          <meshPhongMaterial 
            color="#1a1a2e" 
            transparent 
            opacity={0.8}
          />
        )}
      </mesh>
      
      {/* Subtle grid lines for reference */}
      <gridHelper 
        args={[2000, 40, '#333366', '#222244']} 
        position={[0, 0.1, 0]}
        visible={!mapTexture}
      />
    </group>
  );
};

export default Map2D; 