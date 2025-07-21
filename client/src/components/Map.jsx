import React, { useState, useEffect } from 'react';
import { useThree, useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import { TilesRenderer as TilesRendererImpl, GlobeControls as GlobeControlsImpl } from '3d-tiles-renderer';
import {
  GLTFExtensionsPlugin,
  GoogleCloudAuthPlugin,
  TileCompressionPlugin,
  TilesFadePlugin,
  UpdateOnChangePlugin
} from '3d-tiles-renderer/plugins';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader';
import { DRACOLoader } from 'three/examples/jsm/loaders/DRACOLoader';
import { Geodetic, PointOfView, radians } from '@takram/three-geospatial';

const API_KEY = import.meta.env.VITE_GOOGLE_MAPS_API_KEY;

const dracoLoader = new DRACOLoader();
dracoLoader.setDecoderPath('https://www.gstatic.com/draco/versioned/decoders/1.5.6/');

const Globe = ({ onMapLoaded }) => {
  const [tiles, setTiles] = useState(null);
  const [controls, setControls] = useState(null);
  const { scene, camera, gl } = useThree();

  const setInitialCameraPosition = (camera) => {
    const longitude = -122.4194;
    const latitude = 37.7749;
    const altitude = 2500;
    const heading = 0;
    const pitch = -55;
    
    const pov = new PointOfView(altitude, radians(heading), radians(pitch));
    const geodetic = new Geodetic(radians(longitude), radians(latitude));
    
    pov.decompose(
      geodetic.toECEF(),
      camera.position,
      camera.quaternion,
      camera.up
    );
    
    camera.updateMatrix();
    camera.updateMatrixWorld(true);
  };

  useEffect(() => {
    if (!API_KEY) return;

    const tilesUrl = `https://tile.googleapis.com/v1/3dtiles/root.json?key=${API_KEY}`;
    const tilesRenderer = new TilesRendererImpl(tilesUrl);
    
    tilesRenderer.setCamera(camera);
    tilesRenderer.setResolutionFromRenderer(camera, gl);
    
    setInitialCameraPosition(camera);
    
    const globeControls = new GlobeControlsImpl(scene, camera, gl.domElement, tilesRenderer);
    globeControls.enableDamping = true;
    globeControls.adjustHeight = false;
    globeControls.maxAltitude = Math.PI * 0.55;
    
    const enableAdjustHeight = () => {
      globeControls.adjustHeight = true;
      globeControls.removeEventListener('start', enableAdjustHeight);
    };
    globeControls.addEventListener('start', enableAdjustHeight);
    
    const gltfLoader = new GLTFLoader(tilesRenderer.manager);
    gltfLoader.setDRACOLoader(dracoLoader);
    tilesRenderer.manager.addHandler(/\.gltf$/i, gltfLoader);
    tilesRenderer.manager.addHandler(/\.glb$/i, gltfLoader);
    
    try {
      const authPlugin = new GoogleCloudAuthPlugin({
        apiToken: API_KEY,
        autoRefreshToken: true
      });
      tilesRenderer.registerPlugin(authPlugin);
      tilesRenderer.registerPlugin(new GLTFExtensionsPlugin());
      tilesRenderer.registerPlugin(new TileCompressionPlugin());
      tilesRenderer.registerPlugin(new UpdateOnChangePlugin());
      tilesRenderer.registerPlugin(new TilesFadePlugin());
    } catch (error) {
      console.warn('Could not register plugins:', error);
    }
    
    scene.add(tilesRenderer.group);
    
    setTiles(tilesRenderer);
    setControls(globeControls);
    
    if (onMapLoaded) {
        const ecefToLatLng = (x, y, z) => {
            const a = 6378137.0;
            const f = 1 / 298.257223563;
            const e2 = 2 * f - f * f;
            
            const p = Math.sqrt(x * x + y * y);
            const theta = Math.atan2(z * a, p * (1 - f) * a);
            
            const lat = Math.atan2(
              z + (e2 * (1 - f) / (1 - e2)) * a * Math.pow(Math.sin(theta), 3),
              p - e2 * a * Math.pow(Math.cos(theta), 3)
            );
            
            const lng = Math.atan2(y, x);
            const N = a / Math.sqrt(1 - e2 * Math.sin(lat) * Math.sin(lat));
            const alt = p / Math.cos(lat) - N;
            
            return {
              lat: lat * 180 / Math.PI,
              lng: lng * 180 / Math.PI,
              alt: alt
            };
        };
        const latLngToECEF = (lat, lng, alt = 0) => {
            const a = 6378137.0;
            const f = 1 / 298.257223563;
            const e2 = 2 * f - f * f;
            
            const latRad = lat * Math.PI / 180;
            const lngRad = lng * Math.PI / 180;
            
            const N = a / Math.sqrt(1 - e2 * Math.sin(latRad) * Math.sin(latRad));
            
            const x = (N + alt) * Math.cos(latRad) * Math.cos(lngRad);
            const y = (N + alt) * Math.cos(latRad) * Math.sin(lngRad);
            const z = (N * (1 - e2) + alt) * Math.sin(latRad);
            
            return new THREE.Vector3(x, y, z);
        };
        onMapLoaded({ latLngToECEF, ecefToLatLng });
    }

    return () => {
      scene.remove(tilesRenderer.group);
      tilesRenderer.dispose();
      globeControls.dispose();
    };
  }, [API_KEY, scene, camera, gl]);

  useFrame(() => {
    if (tiles) {
      tiles.update();
    }
    if (controls) {
      controls.update();
    }
  });

  return null;
};

const Map = ({ onMapLoaded }) => {
    return <Globe onMapLoaded={onMapLoaded} />;
}

export default Map; 