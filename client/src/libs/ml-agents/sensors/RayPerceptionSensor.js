
import * as THREE from 'three';

export class RayPerceptionSensor {
    constructor(object3D, rayLength, rayAngles, detectableTags, scene) {
        this.object3D = object3D; // The Three.js Object3D to cast from
        this.rayLength = rayLength || 20;
        this.rayAngles = rayAngles || [0, 90, 180, 270]; // Degrees
        this.detectableTags = detectableTags || []; // ["wall", "goal"]
        this.scene = scene;
        this.raycaster = new THREE.Raycaster();
    }

    getObservation() {
        const obs = [];
        const agentPos = new THREE.Vector3();
        const agentRot = new THREE.Quaternion();
        
        if (this.object3D) {
            this.object3D.getWorldPosition(agentPos);
            this.object3D.getWorldQuaternion(agentRot);
        }

        // For each angle
        // RayPerceptionOutput usually:
        // [hitFraction, 0/1 (tag1), 0/1 (tag2)...] per ray?
        // Or one-hot tag index?
        // ML-Agents default:
        // For each ray:
        //   - n tags (one hot)
        //   - hit fraction (normalized distance)
        //   - (optional) has hit (1 or 0) - implied by tag?
        
        // Let's assume input size = #Rays * (#Tags + 2)
        // (+2 for hit fraction and hasHit)
        
        this.rayAngles.forEach(angleDeg => {
            // Calculate direction
            // Angle 0 is forward (Z+? or X+?)
            // Assume +Z is forward, Y up.
            // Rotate forward vector by angle around Y axis
            const direction = new THREE.Vector3(0, 0, 1);
            direction.applyAxisAngle(new THREE.Vector3(0, 1, 0), angleDeg * Math.PI / 180);
            direction.applyQuaternion(agentRot); // Transform to world space
            direction.normalize();

            this.raycaster.set(agentPos, direction);
            this.raycaster.far = this.rayLength;
            
            // Intersect objects
            // We need to filter objects by tag. Three.js objects don't have tags natively.
            // We assume userData.tag is set.
            const intersects = this.raycaster.intersectObjects(this.scene.children, true);
            
            let hit = null;
            for (const i of intersects) {
                if (i.object === this.object3D) continue; // Ignore self
                // Check tag
                if (i.object.userData && i.object.userData.tag && this.detectableTags.includes(i.object.userData.tag)) {
                    hit = i;
                    break;
                }
            }

            // Encode observation for this ray
            // 1. Hit Fraction (0-1)
            // 2. One-hot tags
            
            const subObs = [];
            const numTags = this.detectableTags.length;
            
            if (hit) {
                const fraction = hit.distance / this.rayLength;
                subObs.push(fraction);
                subObs.push(1.0); // Has Hit

                const tagIdx = this.detectableTags.indexOf(hit.object.userData.tag);
                for (let t = 0; t < numTags; t++) {
                    subObs.push(t === tagIdx ? 1.0 : 0.0);
                }
            } else {
                subObs.push(1.0); // Max distance fraction
                subObs.push(0.0); // No Hit
                for (let t = 0; t < numTags; t++) {
                    subObs.push(0.0);
                }
            }
            obs.push(...subObs);
        });

        return obs;
    }
}
