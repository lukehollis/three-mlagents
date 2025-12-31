
import { SideChannel } from './SideChannel';

export class FloatPropertiesChannel extends SideChannel {
    constructor(channelId = "60ccf7d0-4f7e-11ea-b238-784f4387d1f7") {
        super(channelId);
        this.floatProperties = new Map();
        this.callbacks = []; // (key, value) => void
    }

    onMessageReceived(bodyView) {
        // Format: [KeyLen (4)][Key (utf8)][Value (4 float)]
        let offset = 0;
        
        while (offset < bodyView.byteLength) {
            const keyLen = bodyView.getInt32(offset, true);
            offset += 4;
            
            const keyBytes = new Uint8Array(bodyView.buffer, bodyView.byteOffset + offset, keyLen);
            const key = new TextDecoder().decode(keyBytes);
            offset += keyLen;
            
            const value = bodyView.getFloat32(offset, true);
            offset += 4;
            
            this.floatProperties.set(key, value);
            this.dispatchUpdate(key, value);
        }
    }

    setFloat(key, value) {
        this.floatProperties.set(key, value);
        
        // Serialize: [KeyLen][Key][Value]
        const keyBytes = new TextEncoder().encode(key);
        const len = 4 + keyBytes.length + 4;
        const buf = new ArrayBuffer(len);
        const view = new DataView(buf);
        
        view.setInt32(0, keyBytes.length, true);
        new Uint8Array(buf).set(keyBytes, 4);
        view.setFloat32(4 + keyBytes.length, value, true);
        
        this.queueMessage(buf);
    }

    getFloat(key) {
        return this.floatProperties.get(key);
    }

    registerCallback(cb) {
        this.callbacks.push(cb);
    }

    dispatchUpdate(key, value) {
        for (const cb of this.callbacks) {
            cb(key, value);
        }
    }
}
