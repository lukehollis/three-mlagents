
import { SideChannel } from './SideChannel';

export const StatsAggregationMethod = {
    AVERAGE: 0,
    SUM: 1,
    LAST: 2
};

export class StatsSideChannel extends SideChannel {
    constructor() {
        super("621f0a70-4f87-11ea-a6bf-784f4387d1f7");
    }

    addStat(key, value, aggregationMethod = StatsAggregationMethod.AVERAGE) {
        // Prepare message:
        // Key (string), Value (float), Aggregation (int)
        
        // Protocol:
        // string key
        // float value
        // int aggregation
        
        // Construct buffer
        const encoder = new TextEncoder();
        const keyBytes = encoder.encode(key);
        const len = keyBytes.length;
        
        // Size: 4 (len) + len + 4 (value) + 4 (agg) = 12 + len
        const buffer = new ArrayBuffer(12 + len);
        const view = new DataView(buffer);
        
        let offset = 0;
        view.setInt32(offset, len, true);
        offset += 4;
        
        new Uint8Array(buffer).set(keyBytes, offset);
        offset += len;
        
        view.setFloat32(offset, value, true);
        offset += 4;
        
        view.setInt32(offset, aggregationMethod, true);
        
        this.queueMessage(buffer);
    }

    onMessageReceived(view) {
        // Usually Stats channel is send-only from Client to Python
    }
}
