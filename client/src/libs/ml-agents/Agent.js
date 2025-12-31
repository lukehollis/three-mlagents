
import { v4 as uuidv4 } from 'uuid';

/**
 * Base class for an Agent in the Three.js environment.
 * Users should extend this class and implement collectObservations and onActionReceived.
 */
export class Agent {
    constructor() {
        this.id = uuidv4();
        this.academy = null; // Will be set when registered
        this.behaviorName = "Default";

        this.done = false;
        
        this.visualSensors = []; 
        this.sensors = []; // Vector/Ray sensors
        
        // Decision Requester
        this.decisionInterval = 1; // 1 = every step
        this.lastAction = null;
    }

    addVisualSensor(sensor) {
        this.visualSensors.push(sensor);
    }
    
    addSensor(sensor) {
        this.sensors.push(sensor);
    }

    reset() {
        this.cumulativeReward = 0;
        this.stepCount = 0;
        this.maxStep = 0; // 0 = infinite
        this.done = false;
    }

    /**
     * Override this to define the specs of the agent (obs shapes, action space).
     * This is needed for the handshake with Python.
     * @returns {object} { observationSpecs: [{shape: [], name: ""}], actionSpec: { continuous: 0, discrete: [] } }
     */
    getStats() {
        throw new Error("Agent.getStats() must be implemented (return obs/action specs).");
    }

    /**
     * Collect observations from the environment.
     * @returns {Array<number>} An array of numbers representing the observation.
     */
    collectObservations() {
        let obs = [];
        for (const s of this.sensors) {
            obs = obs.concat(s.getObservation());
        }
        return obs;
    }

    /**
     * Apply the received action to the agent.
     * @param {Array<number>|number} action - The action received from the model. 
     * If discrete, might be a single index. If continuous, an array.
     */
    onActionReceived(action) {
        // Implement physics update here
    }

    /**
     * Called when the environment sends a reset signal (e.g. new episode).
     */
    onEnvironmentReset() {
        // Implement physical reset here
    }

    /**
     * Add a reward to the agent.
     * @param {number} val 
     */
    addReward(val) {
        this.cumulativeReward += val;
    }

    /**
     * Set the reward for the agent (clears previous).
     * @param {number} val 
     */
    setReward(val) {
        this.cumulativeReward = val;
    }

    /**
     * Mark the episode as finished for this agent.
     */
    endEpisode() {
        this.done = true;
        if (this.academy) {
            this.academy.agentSignalDone(this);
        }
    }

    /**
     * Request a decision from the brain at the next step.
     */
    requestDecision() {
        // In this simple version, we assume decision is always requested every step if registered to Academy
    }
}
