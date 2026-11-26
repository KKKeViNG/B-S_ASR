/**
 * Bonus feature: near real-time recognition.
 *
 * The browser captures raw PCM through an AudioWorklet (with a ScriptProcessor
 * fallback), resamples to 16 kHz, and sends ~2.5 s WAV slices over WebSocket.
 * Slices are transcribed independently by the backend and concatenated, which
 * gives a live transcript without a natively streaming model.
 *
 * Flow control: only one slice is in flight at a time, so a slow CPU never
 * builds an unbounded backlog; the next slice is sent as soon as the previous
 * partial transcript arrives.
 */

import { computeRms, encodeWav, concatFloat32, StreamingResampler } from './wav.js';

const TARGET_RATE = 16000;
const WORKLET_URL = '/static/js/pcm-worklet.js';
const READY_TIMEOUT_MS = 10000;
const FINAL_TIMEOUT_MS = 90000;
const REFINE_TIMEOUT_MS = 60000;

export class StreamRecorder {
  constructor({
    chunkSeconds = 2.5,
    maxSeconds = 60,
    onLevel = () => {},
    onPartial = () => {},
    onError = () => {},
    onStatus = () => {},
  } = {}) {
    this.chunkSeconds = chunkSeconds;
    this.maxSeconds = maxSeconds;
    this.onLevel = onLevel;
    this.onPartial = onPartial;
    this.onError = onError;
    this.onStatus = onStatus;

    this.onAutoStop = null;

    this._ws = null;
    this._context = null;
    this._stream = null;
    this._source = null;
    this._node = null;
    this._mute = null;
    this._analyser = null;
    this._meterFrame = null;

    this._blocks = [];
    this._allBlocks = [];
    this._bufferedSamples = 0;
    this._inFlight = false;
    this._stopping = false;
    this._stopSent = false;
    this._listening = false;
    this._limitReached = false;
    this._sentSeconds = 0;

    this._assembledFinal = null;
    this._refineTimer = null;
    this.refineEnabled = true;

    this._readyResolve = null;
    this._readyReject = null;
    this._finalResolve = null;
    this._finalReject = null;
  }

  static get supported() {
    return typeof window.WebSocket !== 'undefined'
      && Boolean(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
  }

  get elapsedSeconds() {
    return this._sentSeconds + this._bufferedSamples / TARGET_RATE;
  }

  async start(language) {
    if (this._listening) throw new Error('实时识别已在进行中');

    const readyPromise = new Promise((resolve, reject) => {
      this._readyResolve = resolve;
      this._readyReject = reject;
    });

    const url = `${window.location.protocol === 'https:' ? 'wss' : 'ws'}://${window.location.host}/ws/stream`;
    await new Promise((resolve, reject) => {
      const socket = new WebSocket(url);
      this._ws = socket;
      const timer = setTimeout(() => reject(new Error('连接实时识别服务超时，请确认后端正在运行。')), READY_TIMEOUT_MS);
      socket.addEventListener('open', () => {
        clearTimeout(timer);
        resolve();
      }, { once: true });
      socket.addEventListener('error', () => {
        clearTimeout(timer);
        reject(new Error('无法建立实时识别连接（WebSocket 不可用）。'));
      }, { once: true });
    });

    this._ws.addEventListener('message', (event) => this._onMessage(event));
    this._ws.addEventListener('close', () => {
      if (this._stopping && this._finalReject) {
        this._finalReject(new Error('实时识别连接提前断开，请重试。'));
      }
    });

    await this._startCapture();

    this._ws.send(JSON.stringify({ type: 'start', language }));
    const ready = await Promise.race([
      readyPromise,
      new Promise((_, reject) => setTimeout(() => reject(new Error('实时识别服务未就绪，请重试。')), READY_TIMEOUT_MS)),
    ]);

    this._listening = true;
    this._startMeter();
    return ready;
  }

  /** Stop capture, flush the remaining audio and resolve with the final result. */
  async stop() {
    if (!this._listening && !this._stopping) return null;
    this._listening = false;
    this._stopping = true;
    this._teardownCapture();
    this.onLevel(0);
    this.onStatus('stopping');

    const finalPromise = new Promise((resolve, reject) => {
      this._finalResolve = resolve;
      this._finalReject = reject;
      setTimeout(() => reject(new Error('等待最终识别结果超时，请重试。')), FINAL_TIMEOUT_MS);
    });

    this._pump();
    return finalPromise;
  }

  abort() {
    this._listening = false;
    this._stopping = true;
    if (this._refineTimer) {
      clearTimeout(this._refineTimer);
      this._refineTimer = null;
    }
    this._teardownCapture();
    try {
      if (this._ws && this._ws.readyState === WebSocket.OPEN) this._ws.close();
    } catch (error) {
      /* ignore */
    }
    if (this._finalReject) this._finalReject(new Error('实时识别已取消。'));
    this._finalResolve = null;
    this._finalReject = null;
    this.onLevel(0);
  }

  // ---- capture ---------------------------------------------------------

  async _startCapture() {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
    });
    this._stream = stream;
    this.onStatus('capturing');

    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    if (!AudioContextClass) throw new Error('当前浏览器不支持 Web Audio API。');

    const context = new AudioContextClass({ sampleRate: TARGET_RATE });
    this._context = context;
    if (context.state === 'suspended') await context.resume();

    this._source = context.createMediaStreamSource(stream);
    this._resampler = new StreamingResampler(context.sampleRate, TARGET_RATE);

    const analyser = context.createAnalyser();
    analyser.fftSize = 1024;
    analyser.smoothingTimeConstant = 0.6;
    this._analyser = analyser;
    this._source.connect(analyser);

    const mute = context.createGain();
    mute.gain.value = 0;
    mute.connect(context.destination);
    this._mute = mute;

    if (context.audioWorklet) {
      try {
        await context.audioWorklet.addModule(WORKLET_URL);
        const node = new AudioWorkletNode(context, 'pcm-float32-processor');
        node.port.onmessage = (event) => this._onPcm(event.data);
        this._source.connect(node);
        node.connect(mute);
        this._node = node;
        return;
      } catch (error) {
        console.warn('AudioWorklet unavailable, falling back to ScriptProcessor:', error);
      }
    }
    this._startScriptProcessor();
  }

  _startScriptProcessor() {
    const node = this._context.createScriptProcessor(4096, 1, 1);
    node.onaudioprocess = (event) => {
      const channel = event.inputBuffer.getChannelData(0);
      this._onPcm(new Float32Array(channel));
    };
    this._source.connect(node);
    node.connect(this._mute);
    this._node = node;
  }

  _onPcm(block) {
    if (!block || block.length === 0) return;
    if (!this._listening && !this._stopping) return;
    if (this._stopping) return; // audio captured after stop is not sent

    const samples = this._resampler ? this._resampler.process(block) : block;
    if (samples.length === 0) return;

    this._blocks.push(samples);
    this._allBlocks.push(samples);
    this._bufferedSamples += samples.length;
    this.onLevel(Math.min(1, Math.pow(computeRms(samples) * 6, 0.65)));

    if (!this._limitReached && this.elapsedSeconds >= this.maxSeconds) {
      this._limitReached = true;
      if (typeof this.onAutoStop === 'function') this.onAutoStop();
      return;
    }
    this._pump();
  }

  // ---- websocket -------------------------------------------------------

  _onMessage(event) {
    let payload;
    try {
      payload = JSON.parse(event.data);
    } catch (error) {
      return;
    }

    switch (payload.type) {
      case 'ready':
        if (this._readyResolve) {
          this._readyResolve(payload);
          this._readyResolve = null;
        }
        break;
      case 'partial':
        this._inFlight = false;
        this.onPartial(payload);
        this._pump();
        break;
      case 'final':
        this._inFlight = false;
        if (payload.refined || !this.refineEnabled) {
          this._resolveFinal(payload);
        } else {
          this._assembledFinal = payload;
          this._sendRefine();
        }
        break;
      case 'error':
        this._inFlight = false;
        this.onError(payload);
        if (payload.refine && this._assembledFinal) {
          this._resolveFinal(this._assembledFinal);
          break;
        }
        if (payload.fatal) {
          this.abort();
          return;
        }
        this._pump();
        break;
      default:
        break;
    }
  }

  _pump() {
    const socket = this._ws;
    if (!socket || socket.readyState !== WebSocket.OPEN) return;
    if (this._inFlight) return;

    const chunkSamples = Math.round(this.chunkSeconds * TARGET_RATE);

    if (this._bufferedSamples > 0) {
      if (!this._stopping && this._bufferedSamples < chunkSamples) return;
      const take = this._stopping ? this._bufferedSamples : chunkSamples;
      const samples = this._takeSamples(take);
      if (samples.length === 0) return;

      const blob = encodeWav(samples, TARGET_RATE);
      this._sentSeconds += samples.length / TARGET_RATE;
      this._inFlight = true;
      blob.arrayBuffer().then((buffer) => {
        if (socket.readyState === WebSocket.OPEN) {
          socket.send(buffer);
        } else {
          this._inFlight = false;
        }
      });
      return;
    }

    if (this._stopping && !this._stopSent) {
      this._stopSent = true;
      socket.send(JSON.stringify({ type: 'stop' }));
    }
  }

  _takeSamples(count) {
    const blocks = [];
    let remaining = Math.min(count, this._bufferedSamples);
    while (remaining > 0 && this._blocks.length > 0) {
      const head = this._blocks[0];
      if (head.length <= remaining) {
        blocks.push(head);
        this._blocks.shift();
        remaining -= head.length;
      } else {
        blocks.push(head.slice(0, remaining));
        this._blocks[0] = head.slice(remaining);
        remaining = 0;
      }
    }
    const taken = concatFloat32(blocks);
    this._bufferedSamples -= taken.length;
    if (this._bufferedSamples < 0) this._bufferedSamples = 0;
    return taken;
  }

  _teardownSocket() {
    try {
      if (this._ws) {
        this._ws.onmessage = null;
        if (this._ws.readyState === WebSocket.OPEN || this._ws.readyState === WebSocket.CONNECTING) {
          this._ws.close(1000, 'finished');
        }
      }
    } catch (error) {
      /* ignore */
    }
    this._ws = null;
  }

  /**
   * Ask the backend to re-decode the complete clip. Slice-by-slice recognition
   * keeps the live transcript instant but cuts words and punctuation at chunk
   * borders; this final pass produces a clean transcript for the same audio.
   */
  _sendRefine() {
    const socket = this._ws;
    const samples = concatFloat32(this._allBlocks);
    this._allBlocks = [];

    if (!socket || socket.readyState !== WebSocket.OPEN || samples.length < TARGET_RATE * 0.5) {
      this._resolveFinal(this._assembledFinal);
      return;
    }

    const blob = encodeWav(samples, TARGET_RATE);
    blob
      .arrayBuffer()
      .then((buffer) => {
        if (socket.readyState === WebSocket.OPEN) {
          socket.send(JSON.stringify({ type: 'refine' }));
          socket.send(buffer);
        } else {
          this._resolveFinal(this._assembledFinal);
        }
      })
      .catch(() => this._resolveFinal(this._assembledFinal));

    this._refineTimer = setTimeout(() => this._resolveFinal(this._assembledFinal), REFINE_TIMEOUT_MS);
  }

  _resolveFinal(payload) {
    if (this._refineTimer) {
      clearTimeout(this._refineTimer);
      this._refineTimer = null;
    }
    if (!this._finalResolve) return;
    const resolve = this._finalResolve;
    this._finalResolve = null;
    this._finalReject = null;
    this._teardownSocket();
    resolve(payload);
  }

  // ---- resources -------------------------------------------------------

  _teardownCapture() {
    if (this._meterFrame) cancelAnimationFrame(this._meterFrame);
    this._meterFrame = null;

    try {
      if (this._node) {
        if (this._node.port) this._node.port.onmessage = null;
        if (this._node.onaudioprocess !== undefined) this._node.onaudioprocess = null;
        this._node.disconnect();
      }
    } catch (error) {
      /* ignore */
    }
    this._node = null;

    try {
      if (this._source) this._source.disconnect();
    } catch (error) {
      /* ignore */
    }
    this._source = null;

    if (this._stream) {
      this._stream.getTracks().forEach((track) => track.stop());
      this._stream = null;
    }
    if (this._context) {
      this._context.close().catch(() => {});
      this._context = null;
    }
  }

  _startMeter() {
    if (!this._analyser) return;
    const buffer = new Float32Array(this._analyser.fftSize);
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
  }
}
