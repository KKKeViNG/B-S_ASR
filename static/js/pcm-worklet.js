/**
 * AudioWorklet processor that forwards raw microphone PCM to the main thread.
 *
 * Each `process()` call receives 128 frames; the block is copied and
 * transferred so the main thread owns a stable buffer.
 */
class PcmFloat32Processor extends AudioWorkletProcessor {
  process(inputs) {
    const input = inputs[0];
    if (input && input.length > 0) {
      const channel = input[0];
      if (channel && channel.length > 0) {
        const copy = new Float32Array(channel.length);
        copy.set(channel);
        this.port.postMessage(copy, [copy.buffer]);
      }
    }
    // Returning true keeps the processor alive even when the input is silent.
    return true;
  }
}

registerProcessor('pcm-float32-processor', PcmFloat32Processor);
