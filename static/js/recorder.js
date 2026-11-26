/**
 * Microphone recording for "record, then transcribe" mode.
 *
 * Uses MediaRecorder with Opus/WebM (Chrome/Edge default), reports a live input
 * level and enforces the 60 s recording limit.
 */

const PREFERRED_MIME_TYPES = [
  'audio/webm;codecs=opus',
  'audio/webm',
  'audio/ogg;codecs=opus',
  'audio/ogg',
  'audio/mp4',
];

const EXTENSION_BY_MIME = {
  'audio/webm': 'webm',
  'audio/ogg': 'ogg',
  'audio/opus': 'opus',
  'audio/mp4': 'm4a',
  'audio/wav': 'wav',
  'audio/x-wav': 'wav',
};

export function pickMimeType() {
  if (typeof window.MediaRecorder === 'undefined') return '';
  for (const candidate of PREFERRED_MIME_TYPES) {
    if (MediaRecorder.isTypeSupported(candidate)) return candidate;
  }
  return '';
}

function extensionFor(mimeType) {
  const base = String(mimeType || '').split(';')[0].trim().toLowerCase();
  return EXTENSION_BY_MIME[base] || 'webm';
}

export class MicRecorder {
  constructor({ maxSeconds = 60, onLevel = () => {} } = {}) {
    this.maxSeconds = maxSeconds;
    this.onLevel = onLevel;
    this.onAutoStop = null;

    this._stream = null;
    this._recorder = null;
    this._chunks = [];
    this._recording = false;
    this._startedAt = 0;
    this._elapsed = 0;
    this._mimeType = '';
    this._audioContext = null;
    this._analyser = null;
    this._meterFrame = null;
    this._limitTimer = null;
  }

  static get supported() {
    return Boolean(navigator.mediaDevices && navigator.mediaDevices.getUserMedia)
      && typeof window.MediaRecorder !== 'undefined';
  }

  get isRecording() {
    return this._recording;
  }

  get elapsedSeconds() {
    if (!this._recording) return this._elapsed;
    return (performance.now() - this._startedAt) / 1000;
  }

  async start() {
    if (this._recording) throw new Error('录音已在进行中');
    if (!MicRecorder.supported) {
      throw new Error('当前浏览器不支持录音，请使用最新版 Chrome 或 Edge。');
    }

    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
    });

    this._stream = stream;
    this._chunks = [];
    this._recording = true;
    this._startedAt = performance.now();
    this._elapsed = 0;

    const requestedMime = pickMimeType();
    let recorder;
    try {
      recorder = requestedMime
        ? new MediaRecorder(stream, { mimeType: requestedMime, audioBitsPerSecond: 128000 })
        : new MediaRecorder(stream);
    } catch (error) {
      this._releaseStream();
      this._recording = false;
      throw error;
    }

    this._recorder = recorder;
    this._mimeType = recorder.mimeType || requestedMime || 'audio/webm';

    recorder.addEventListener('dataavailable', (event) => {
      if (event.data && event.data.size > 0) this._chunks.push(event.data);
    });
    recorder.addEventListener('error', (event) => {
      this._fail(event.error || new Error('录音设备出现错误'));
    });

    recorder.start(1000); // gather a chunk per second so a stop never loses audio
    this._startMeter(stream);

    this._limitTimer = setTimeout(() => {
      if (!this._recording) return;
      this._autoStopped = true;
      if (typeof this.onAutoStop === 'function') this.onAutoStop();
    }, this.maxSeconds * 1000);
  }

  /**
   * Stop recording and resolve with the captured clip.
   * @returns {Promise<{blob: Blob, mimeType: string, extension: string, durationSeconds: number}>}
   */
  async stop() {
    if (!this._recording) {
      return null;
    }
    this._recording = false;
    this._elapsed = (performance.now() - this._startedAt) / 1000;

    clearTimeout(this._limitTimer);
    this._limitTimer = null;
    this._teardownMeter();

    const recorder = this._recorder;
    const blob = await new Promise((resolve) => {
      const finish = () => resolve(new Blob(this._chunks, { type: this._mimeType }));
      if (!recorder || recorder.state === 'inactive') {
        finish();
        return;
      }
      recorder.addEventListener('stop', finish, { once: true });
      recorder.addEventListener('error', finish, { once: true });
      try {
        recorder.stop();
      } catch (error) {
        finish();
      }
    });

    this._recorder = null;
    this._chunks = [];
    this._releaseStream();
    this.onLevel(0);

    return {
      blob,
      mimeType: this._mimeType,
      extension: extensionFor(this._mimeType),
      durationSeconds: this._elapsed,
      autoStopped: Boolean(this._autoStopped),
    };
  }

  /** Release the microphone without producing a result (error path). */
  abort() {
    this._recording = false;
    clearTimeout(this._limitTimer);
    this._limitTimer = null;
    this._teardownMeter();
    try {
      if (this._recorder && this._recorder.state !== 'inactive') this._recorder.stop();
    } catch (error) {
      /* ignore */
    }
    this._recorder = null;
    this._chunks = [];
    this._releaseStream();
    this.onLevel(0);
  }

  _fail(error) {
    this._error = error;
    this.abort();
  }

  _releaseStream() {
    if (this._stream) {
      this._stream.getTracks().forEach((track) => track.stop());
      this._stream = null;
    }
  }

  _startMeter(stream) {
    try {
      const AudioContextClass = window.AudioContext || window.webkitAudioContext;
      if (!AudioContextClass) return;
      const context = new AudioContextClass();
      this._audioContext = context;
      const source = context.createMediaStreamSource(stream);
      const analyser = context.createAnalyser();
      analyser.fftSize = 1024;
      analyser.smoothingTimeConstant = 0.65;
      source.connect(analyser);
      this._analyser = analyser;

      const buffer = new Float32Array(analyser.fftSize);
      const tick = () => {
        if (!this._analyser) return;
        this._analyser.getFloatTimeDomainData(buffer);
        let sum = 0;
        for (let i = 0; i < buffer.length; i += 1) sum += buffer[i] * buffer[i];
        const rms = Math.sqrt(sum / buffer.length);
        this.onLevel(Math.min(1, Math.pow(rms * 6, 0.65)));
        this._meterFrame = requestAnimationFrame(tick);
      };
      tick();
    } catch (error) {
      // Metering is cosmetic; never let it break recording.
      console.warn('Audio level meter unavailable:', error);
    }
  }

  _teardownMeter() {
    if (this._meterFrame) cancelAnimationFrame(this._meterFrame);
    this._meterFrame = null;
    this._analyser = null;
    if (this._audioContext) {
      this._audioContext.close().catch(() => {});
      this._audioContext = null;
    }
  }
}
