
import { Agent } from '../Agent';

export class Ball3DAgent extends Agent {
    constructor(platformObj) {
        super();
        this.behaviorName = "Ball3D";
        this.platformObj = platformObj; // { rotX, rotZ, ballX, ballZ, velX, velZ }
        this.maxStep = 1000; // Reset after 1000 steps
        this.gravity = 9.81;
        this.decisionInterval = 5; // Request decision every 5 steps
    }

    getStats() {
        // Core size = 8
        // Plus any added sensors
        let sensorSize = 0;
        for (const s of this.sensors) {
            // efficient way? s.getObservation().length
            // or s.observationSize
            // Let's assume getObservation returns array
             sensorSize += s.getObservation().length;
        }

        return {
            observationSpecs: [
                { shape: [8 + sensorSize], name: "VectorSensor" } // 8 + extra
            ],
            // 2 continuous actions (tilt X, tilt Z)
            actionSpec: { continuous: 2, discrete: [] }
        };
    }

    collectObservations() {
        // [rotX, rotZ, ballX, ballZ, velX, velZ, 0, 0]
        
        const p = this.platformObj;
        const coreObs = [
            p.rotZ,
            p.rotX, 
            p.ballX,
            0, // Relative Y? Ball is on platform surface usually?
            p.ballZ,
            p.velX,
            0, // velY
            p.velZ
        ];
        
        // Append extra sensors
        const extra = super.collectObservations();
        return coreObs.concat(extra);
    }

    onActionReceived(action) {
        // action is [tiltX, tiltZ] continuous
        // Or if discrete: index
        
        // In this migration we test Continuous control (or discrete if matching original)
        // The original `Ball3D.jsx` used Discrete (5 branches).
        // Let's stick to Continuous for this "Migration" as it's cleaner for 3D physics usually.
        // But if I want to match the `train_migration_demo.py` random which generates continuous...
        
        // Let's assume Continuous: action = [x, z], range -1 to 1
        
        const actionX = action[0]; // -1 to 1
        const actionZ = action[1]; // -1 to 1

        const MAX_TILT = Math.PI / 4; // 45 degrees
        const TILT_SPEED = 0.05; 

        const p = this.platformObj;
        
        // Update rotation
        p.rotX += actionX * TILT_SPEED;
        p.rotZ += actionZ * TILT_SPEED;
        
        // Clamp
        p.rotX = Math.max(-MAX_TILT, Math.min(MAX_TILT, p.rotX));
        p.rotZ = Math.max(-MAX_TILT, Math.min(MAX_TILT, p.rotZ));
        
        
        // Physics step (simple Euler)
        const g = this.gravity;
        const dt = 0.02; // 50fps
        
        // Ball acceleration
        p.velX += g * Math.sin(p.rotX) * dt;
        p.velZ += g * Math.sin(p.rotZ) * dt;
        
        p.ballX += p.velX * dt;
        p.ballZ += p.velZ * dt;
        
        // Damping
        p.velX *= 0.99;
        p.velZ *= 0.99;
        
        // Check failure (ball fell off)
        if (Math.abs(p.ballX) > 2.5 || Math.abs(p.ballZ) > 2.5) {
            this.setReward(-1);
            this.endEpisode();
        } else {
            this.addReward(0.1);
        }
    }

    onEnvironmentReset() {
        // Reset physics
        const p = this.platformObj;
        p.rotX = (Math.random() - 0.5) * 0.2;
        p.rotZ = (Math.random() - 0.5) * 0.2;
        p.ballX = (Math.random() - 0.5) * 0.5;
        p.ballZ = (Math.random() - 0.5) * 0.5;
        p.velX = 0;
        p.velZ = 0;
    }
}
