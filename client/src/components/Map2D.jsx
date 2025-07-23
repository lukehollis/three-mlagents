import React, { useEffect, useState, useCallback, useRef } from 'react';
import * as THREE from 'three';

const API_KEY = import.meta.env.VITE_GOOGLE_MAPS_API_KEY;

// Debug environment variable loading on initial load only
if (!window._mapDebugLogged) {
  console.log('Environment check:', {
    hasAPIKey: !!API_KEY,
    apiKeyLength: API_KEY ? API_KEY.length : 0
  });
  window._mapDebugLogged = true;
}

const Map2D = ({ onMapLoaded }) => {
  const [mapTexture, setMapTexture] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [hasAttempted, setHasAttempted] = useState(false);
  const onMapLoadedRef = useRef(onMapLoaded);
  
  // Keep the ref updated
  useEffect(() => {
    onMapLoadedRef.current = onMapLoaded;
  }, [onMapLoaded]);

  const loadMapTexture = useCallback(() => {
    // Prevent multiple attempts
    if (hasAttempted || loading || mapTexture) {
      return;
    }
    
    if (!API_KEY) {
      console.error('VITE_GOOGLE_MAPS_API_KEY environment variable not found');
      setError(new Error('Google Maps API key not configured'));
      setHasAttempted(true);
      return;
    }

    console.log('Loading Google Maps texture...');
    setLoading(true);
    setError(null);
    setHasAttempted(true);

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
    
    // Map URL ready for loading
    
    const textureLoader = new THREE.TextureLoader();
    
    textureLoader.load(
      mapUrl,
      // onLoad - Success callback
      (texture) => {
        texture.wrapS = THREE.ClampToEdgeWrapping;
        texture.wrapT = THREE.ClampToEdgeWrapping;
        texture.minFilter = THREE.LinearFilter;
        setMapTexture(texture);
        setLoading(false);

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
        
        // Only call onMapLoaded after texture is successfully loaded
        if (onMapLoadedRef.current) {
          onMapLoadedRef.current({ latLngToECEF, ecefToLatLng });
        }
      },
      // onProgress - Progress callback
      (progress) => {
        // Optional: handle loading progress
      },
             // onError - Error callback
       (error) => {
         console.error('Error loading Google Maps texture:', error);
         console.error('This could be due to:');
         console.error('1. Invalid or missing VITE_GOOGLE_MAPS_API_KEY');
         console.error('2. API key not enabled for Static Maps API');
         console.error('3. Network connectivity issues');
         console.error('4. CORS restrictions');
         console.error('Please check your VITE_GOOGLE_MAPS_API_KEY and ensure Static Maps API is enabled');
         
         setLoading(false);
         setError(error);
       }
    );
      }, []); // Empty dependency array to prevent recreation

  useEffect(() => {
    if (API_KEY && !hasAttempted) {
      loadMapTexture();
    }
  }, [API_KEY, hasAttempted, loadMapTexture]);

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
            color={loading ? "#1a1a2e" : error ? "#2e1a1a" : "#1a1a2e"} 
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
      
      {/* Loading indicator */}
      {loading && (
        <mesh position={[0, 10, 0]}>
          <sphereGeometry args={[5]} />
          <meshPhongMaterial 
            color="#37F5EB" 
            emissive="#37F5EB" 
            emissiveIntensity={0.3}
            transparent
            opacity={0.7}
          />
        </mesh>
      )}
    </group>
  );
};

export default Map2D; 