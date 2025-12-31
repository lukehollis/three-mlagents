
import { SideChannelManager } from './side-channels/SideChannelManager';

/**
 * The Academy manages the connection to the Python backend and orchestrates the agents.
 * It is a Singleton-like orchestrator.
 */
export class Academy {
    constructor(url = "ws://localhost:8000/ws/mlagents") {
        this.agents = new Map(); // id -> Agent
        this.url = url;
        this.ws = null;
        this.connected = false;
        this.onConnectCallbacks = [];
        this.sideChannelManager = new SideChannelManager();
    }

    /**
     * Register an agent with the Academy.
     * @param {Agent} agent 
     */
    addAgent(agent) {
        this.agents.set(agent.id, agent);
        agent.academy = this;
    }

    /**
     * Register a side channel.
     * @param {SideChannel} channel 
     */
    registerSideChannel(channel) {
        this.sideChannelManager.registerChannel(channel);
    }

    /**
     * Connect to the Python training server.
     */
    connect() {
        this.ws = new WebSocket(this.url);

        this.ws.onopen = () => {
            console.log("Academy: Connected to Python backend.");
            this.connected = true;
            this.sendHandshake();
            this.onConnectCallbacks.forEach(cb => { cb(); });
        };

        this.ws.onmessage = (event) => {
            const msg = JSON.parse(event.data);
            this.handleMessage(msg);
        };

        this.ws.onclose = () => {
            console.log("Academy: Disconnected.");
            this.connected = false;
        };
    }

    onConnect(cb) {
        if (this.connected) cb();
        else this.onConnectCallbacks.push(cb);
    }

    sendHandshake() {
        // Gather specs from all registered agents
        // Group by BehaviorName
        const behaviors = {};
        
        for (const agent of this.agents.values()) {
            const name = agent.behaviorName;
            if (!behaviors[name]) {
                behaviors[name] = agent.getStats();
            }
        }

        const msg = {
            type: "handshake",
            behaviors: behaviors,
            version: "1.0.0" // Custom protocol version
        };
        this.send(msg);
    }

    handleMessage(msg) {
        // Handle Side Channels first (if any)
        if (msg.side_channels) {
             try {
                 const binStr = atob(msg.side_channels);
                 const len = binStr.length;
                 const bytes = new Uint8Array(len);
                 for (let i = 0; i < len; i++) {
                     bytes[i] = binStr.charCodeAt(i);
                 }
                 this.sideChannelManager.processSideChannelData(bytes.buffer);
             } catch (e) {
                 console.error("Academy: Failed to decode side_channels", e);
             }
        }

        switch(msg.type) {
            case "reset":
                this.handleReset();
                break;
            case "step":
                this.handleStep(msg);
                break;
            default:
                // console.warn("Academy received unknown message:", msg);
        }
    }

    handleReset() {
        for (const agent of this.agents.values()) {
            agent.reset();
            agent.onEnvironmentReset();
        }
        // Immediately send new initial observations after reset
        this.sendStep();
    }

    handleStep(msg) {
        const actions = msg.actions || {};

        // 1. Process received actions and update cached lastAction
        for (const [agentId, action] of Object.entries(actions)) {
            const agent = this.agents.get(agentId);
            if (agent) {
                agent.lastAction = action;
            }
        }

        // 2. Step all agents (Physics Step)
        for (const agent of this.agents.values()) {
             // If we have a lastAction (or received one), apply it
             if (agent.lastAction) {
                 agent.onActionReceived(agent.lastAction);
             }
             
             agent.stepCount++;
             if (agent.maxStep > 0 && agent.stepCount >= agent.maxStep) {
                 agent.endEpisode();
             }
        }

        // 3. Send next observations
        this.sendStep();
    }

    sendStep() {
        // Collect data from agents requesting decisions
        const stepData = {};

        for (const agent of this.agents.values()) {
            // Check if requesting decision
            // Always send if done (to reset) ? ML-Agents usually sends terminal obs regardless.
            // If done, we force send.
            
            const requestDecision = (agent.stepCount % agent.decisionInterval === 0) || agent.done;
            
            if (requestDecision) {
                // Collect Obs
                const visualObs = [];
                if (agent.visualSensors) {
                    for (const sensor of agent.visualSensors) {
                        visualObs.push(sensor.getObservation());
                    }
                }

                stepData[agent.id] = {
                    vectorObs: agent.collectObservations(),
                    visualObs: visualObs,
                    reward: agent.cumulativeReward,
                    done: agent.done,
                    info: {} 
                };
            }
            
            // Reset state if done
            if (agent.done) {
                agent.cumulativeReward = 0;
                agent.stepCount = 0;
                agent.done = false;
                agent.onEnvironmentReset(); 
                agent.lastAction = null; // Clear action on reset
            } else {
                agent.setReward(0); 
            }
        }
        
        const msg = {
            type: "step",
            agents: stepData
        };
        
        // Attach Side Channels
        const scData = this.sideChannelManager.generateSideChannelData();
        if (scData) {
            // Encode Base64
            let binary = '';
            const bytes = new Uint8Array(scData);
            const len = bytes.byteLength;
            for (let i = 0; i < len; i++) {
                binary += String.fromCharCode(bytes[i]);
            }
            msg.side_channels = btoa(binary);
        }

        this.send(msg);
    }

    agentSignalDone(_agent) {
        // Helper if we wanted to interrupt immediately, but usually we wait for step cycle.
    }

    send(data) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify(data));
        }
    }
}
