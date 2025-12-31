
/**
 * Base class for Side Channels.
 * Side Channels allow sending binary data alongside the main step/reset payload.
 */
export class SideChannel {
    /**
     * @param {string} channelId - The UUID of the channel.
     */
    constructor(channelId) {
        this.channelId = channelId;
        this.messagesToSend = [];
    }

    /**
     * Process a message received from Python.
     * @param {DataView} _bodyView - The DataView of the message body.
     */
    onMessageReceived(_bodyView) {
        // Implement in subclass
    }

    /**
     * Queue a binary message to be sent to Python.
     * @param {ArrayBuffer} buffer 
     */
    queueMessage(buffer) {
        this.messagesToSend.push(buffer);
    }
}
