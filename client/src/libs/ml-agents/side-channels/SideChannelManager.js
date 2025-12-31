
import { SideChannel } from './SideChannel';

export class SideChannelManager {
    constructor() {
        this.channels = new Map(); // id (string) -> SideChannel
    }

    registerChannel(channel) {
        this.channels.set(channel.channelId, channel);
    }

    /**
     * Parse a binary blob containing multiple messages.
     * Format: [ChannelId (16 bytes)][Length (4 bytes)][Body (Length bytes)] ...
     * @param {ArrayBuffer} buffer 
     */
    processSideChannelData(buffer) {
        const view = new DataView(buffer);
        let offset = 0;

        while (offset < buffer.byteLength) {
            // Read Channel UUID (16 bytes)
            // UUID bytes are high-endian
            const channelId = this.readUuid(view, offset);
            offset += 16;
            
            // Read Length
            const length = view.getInt32(offset, true); // Little Endian (Unity/C# default for this protocol?)
            // WAIT: ML-Agents uses Little Endian? C# BinaryWriter is Little-Endian.
            // Python struct.pack is user-defined but uses system default or explicit.
            // mlagents_envs.side_channel.outgoing_message writes Little Endian.
            offset += 4;
            
            if (offset + length > buffer.byteLength) {
                console.error("SideChannelManager: Message length exceeds buffer.");
                break;
            }

            const bodyBuffer = buffer.slice(offset, offset + length);
            offset += length;

            const channel = this.channels.get(channelId);
            if (channel) {
                channel.onMessageReceived(new DataView(bodyBuffer));
            } else {
                console.warn(`SideChannelManager: Unknown channel ${channelId}`);
            }
        }
    }

    /**
     * Generate a binary blob of all queued messages.
     * @returns {ArrayBuffer|null}
     */
    generateSideChannelData() {
        // aggregate all messages
        let totalLen = 0;
        const queuedMsgs = []; // { channelId, buffer }

        for (const channel of this.channels.values()) {
            for (const msgBuf of channel.messagesToSend) {
                // Header = 16 (UUID) + 4 (Len) = 20
                totalLen += 20 + msgBuf.byteLength;
                queuedMsgs.push({ channelId: channel.channelId, buffer: msgBuf });
            }
            channel.messagesToSend = []; // clear
        }

        if (totalLen === 0) return null;

        const result = new Uint8Array(totalLen);
        const resultView = new DataView(result.buffer);
        let offset = 0;

        for (const q of queuedMsgs) {
            // Write UUID
            this.writeUuid(resultView, offset, q.channelId);
            offset += 16;
            // Write Length
            resultView.setInt32(offset, q.buffer.byteLength, true); // Little Endian
            offset += 4;
            // Write Body
            result.set(new Uint8Array(q.buffer), offset);
            offset += q.buffer.byteLength;
        }

        return result.buffer;
    }

    // Helper: UUID Parsing (String <-> Bytes)
    // UUID format "60ccf7d0-4f7e-11ea-b238-784f4387d1f7"
    // Python outputs raw bytes in network order (Big Endian usually for UUID).
    // But ML-Agents might just do raw bytes.
    // Let's assume standard UUID string parsing.

    readUuid(view, offset) {
        // Read 16 bytes
        const bytes = [];
        for (let i = 0; i < 16; i++) {
            bytes.push(view.getUint8(offset + i));
        }
        // Convert to hex string
        const hex = bytes.map(b => b.toString(16).padStart(2, '0')).join('');
        // add dashed: 8-4-4-4-12
        return `${hex.substr(0,8)}-${hex.substr(8,4)}-${hex.substr(12,4)}-${hex.substr(16,4)}-${hex.substr(20)}`;
    }

    writeUuid(view, offset, uuidStr) {
        const clean = uuidStr.replace(/-/g, '');
        for (let i = 0; i < 16; i++) {
            const byte = parseInt(clean.substr(i*2, 2), 16);
            view.setUint8(offset + i, byte);
        }
    }
}
