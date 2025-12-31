
import { SideChannel } from './SideChannel';

export class EngineConfigurationChannel extends SideChannel {
    constructor() {
        super("a1d8f7b7-d954-4747-96d7-52905df40081");
        this.timeScale = 1.0;
        this.callbacks = [];
    }

    onMessageReceived(view) {
        let offset = 0;
        // The protocol sends key (int) and value (float) pairs? 
        // Or strict ordering?
        // ML-Agents C# impl reads keys and values until end of stream.
        
        while (offset < view.byteLength) {
            const key = view.getInt32(offset, true);
            offset += 4;
            const value = view.getFloat32(offset, true);
            offset += 4;

            if (key === 3) { // TimeScale
                this.timeScale = value;
                this.callbacks.forEach(cb => cb(this.timeScale));
                console.log(`EngineConfig: TimeScale set to ${this.timeScale}`);
            }
        }
    }
    
    registerCallback(cb) {
        this.callbacks.push(cb);
    }
}
