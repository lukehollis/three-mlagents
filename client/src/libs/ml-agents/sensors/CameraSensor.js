
export class CameraSensor {
    constructor(width, height, captureFn) {
        this.width = width;
        this.height = height;
        this.captureFn = captureFn; // Returns base64 string or Uint8Array
    }

    getObservation() {
        if (this.captureFn) {
            return this.captureFn();
        }
        return null;
    }
}
