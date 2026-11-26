/**
 * WAV encoding and small audio helpers used by the streaming recorder.
 * Everything runs in the browser; no build step and no dependencies.
 */

export const TARGET_SAMPLE_RATE = 16000;

/**
 * Encode mono float32 samples ([-1, 1]) as a 16-bit PCM WAV blob.
 * The server decodes this with the same PyAV path as uploaded files.
 *
 * @param {Float32Array} samples
 * @param {number} sampleRate
 * @returns {Blob}
 */
export function encodeWav(samples, sampleRate = TARGET_SAMPLE_RATE) {
  const bytesPerSample = 2;
  const buffer = new ArrayBuffer(44 + samples.length * bytesPerSample);
  const view = new DataView(buffer);

  writeAscii(view, 0, 'RIFF');
  view.setUint32(4, 36 + samples.length * bytesPerSample, true);
  writeAscii(view, 8, 'WAVE');
  writeAscii(view, 12, 'fmt ');
  view.setUint32(16, 16, true); // PCM chunk size
  view.setUint16(20, 1, true); // format = PCM
  view.setUint16(22, 1, true); // mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * bytesPerSample, true); // byte rate
  view.setUint16(32, bytesPerSample, true); // block align
  view.setUint16(34, 16, true); // bits per sample
  writeAscii(view, 36, 'data');
  view.setUint32(40, samples.length * bytesPerSample, true);

  let offset = 44;
  for (let i = 0; i < samples.length; i += 1) {
    const value = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, value < 0 ? value * 0x8000 : value * 0x7fff, true);
    offset += bytesPerSample;
  }
  return new Blob([view], { type: 'audio/wav' });
}

function writeAscii(view, offset, text) {
  for (let i = 0; i < text.length; i += 1) {
    view.setUint8(offset + i, text.charCodeAt(i));
  }
}

/** Root-mean-square level of a block of samples. */
export function computeRms(samples) {
  if (!samples || samples.length === 0) return 0;
  let sum = 0;
  for (let i = 0; i < samples.length; i += 1) sum += samples[i] * samples[i];
  return Math.sqrt(sum / samples.length);
}

/** Concatenate a list of Float32Array blocks. */
export function concatFloat32(blocks) {
  if (blocks.length === 0) return new Float32Array(0);
  if (blocks.length === 1) return blocks[0];
  let total = 0;
  for (const block of blocks) total += block.length;
  const output = new Float32Array(total);
  let offset = 0;
  for (const block of blocks) {
    output.set(block, offset);
    offset += block.length;
  }
  return output;
}

/**
 * Stateful linear resampler. AudioContext usually honours the requested 16 kHz
 * rate, but some devices/browsers refuse; in that case each captured block is
 * resampled while keeping the fractional position across block boundaries so
 * the stream does not drift or click.
 */
export class StreamingResampler {
  constructor(fromRate, toRate = TARGET_SAMPLE_RATE) {
    this.fromRate = fromRate;
    this.toRate = toRate;
    this.ratio = fromRate / toRate;
    this.position = 0;
    this.tail = new Float32Array(0);
  }

  process(block) {
    if (this.fromRate === this.toRate || block.length === 0) return block;

    const input = new Float32Array(this.tail.length + block.length);
    input.set(this.tail, 0);
    input.set(block, this.tail.length);

    const available = input.length - 1 - this.position;
    if (available <= 0) {
      this.tail = input;
      return new Float32Array(0);
    }

    const count = Math.floor(available / this.ratio) + 1;
    const output = new Float32Array(count);

    let position = this.position;
    for (let i = 0; i < count; i += 1) {
      const index = Math.floor(position);
      const fraction = position - index;
      const left = input[index] || 0;
      const right = input[index + 1] !== undefined ? input[index + 1] : left;
      output[i] = left * (1 - fraction) + right * fraction;
      position += this.ratio;
    }

    const consumed = Math.min(Math.floor(position), input.length);
    this.tail = input.slice(consumed);
    this.position = position - consumed;
    return output;
  }
}
