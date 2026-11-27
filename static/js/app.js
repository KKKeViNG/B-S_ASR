/**
 * Application controller: wires the microphone recorder, the upload flow and
 * the streaming bonus feature to the DOM.
 */

import {
  fetchConfig,
  fetchModelStatus,
  fetchModels,
  fetchSample,
  fetchSubtitles,
  selectModel,
  streamModelDownload,
  transcribeAudio,
  ApiError,
} from './api.js';
import { MicRecorder } from './recorder.js';
import { StreamRecorder } from './streaming.js';
import {
  ERROR_ADVICE,
  ERROR_TITLES,
  describeMicError,
  formatBytes,
  formatDurationLong,
  formatSeconds,
  formatTimer,
  languageLabel,
} from './messages.js';

const $ = (id) => document.getElementById(id);

const el = {
  serviceBadge: $('service-badge'),
  modelBadge: $('model-badge'),

  modelSelect: $('model-select'),
  modelApply: $('model-apply'),
  modelStatus: $('model-status'),
  modelHint: $('model-hint'),
  modelProgressTrack: $('model-progress-track'),
  modelProgressBar: $('model-progress-bar'),

  languageGroup: $('language-group'),
  wordTimestamps: $('word-timestamps'),
  modeGroup: $('mode-group'),
  recordLimit: $('record-limit'),
  recordButton: $('record-button'),
  stopButton: $('stop-button'),
  recordingDot: $('recording-dot'),
  recordStatus: $('record-status'),
  recordTimer: $('record-timer'),
  levelBar: $('level-bar'),
  micHint: $('mic-hint'),

  dropZone: $('drop-zone'),
  fileInput: $('file-input'),
  fileRow: $('file-row'),
  fileName: $('file-name'),
  fileSize: $('file-size'),
  fileClear: $('file-clear'),
  uploadButton: $('upload-button'),
  sampleChips: Array.from(document.querySelectorAll('[data-sample]')),

  resultHint: $('result-hint'),
  metaLanguage: $('meta-language'),
  metaDuration: $('meta-duration'),
  metaProcessing: $('meta-processing'),
  metaModel: $('meta-model'),
  resultText: $('result-text'),
  audioPlayer: $('audio-player'),
  segmentPanel: $('segment-panel'),
  segmentList: $('segment-list'),
  exportTxt: $('export-txt'),
  exportSrt: $('export-srt'),
  exportVtt: $('export-vtt'),
  copyButton: $('copy-button'),
  clearButton: $('clear-button'),
  copyFeedback: $('copy-feedback'),

  statusState: $('status-state'),
  statusLanguage: $('status-language'),
  statusDuration: $('status-duration'),
  statusProcessing: $('status-processing'),
  statusSource: $('status-source'),
  statusModel: $('status-model'),
  statusDevice: $('status-device'),

  progressTrack: $('progress-track'),
  progressBar: $('progress-bar'),

  alertCard: $('alert-card'),
  alertBox: $('alert-box'),
  alertTitle: $('alert-title'),
  alertDetail: $('alert-detail'),
  alertRaw: $('alert-raw'),

  tipChunk: $('tip-chunk'),
  tipLimit: $('tip-limit'),
  tipFileLimit: $('tip-file-limit'),
  footerVersion: $('footer-version'),
};

const languageButtons = Array.from(el.languageGroup.querySelectorAll('[data-language]'));
const modeButtons = Array.from(el.modeGroup.querySelectorAll('[data-mode]'));

const state = {
  config: null,
  model: null,
  models: [],
  selectedModel: null,
  modelBusy: false,
  language: 'zh',
  wordTimestamps: false,
  mode: 'batch',
  busy: false,
  recording: false,
  streaming: false,
  file: null,
  recorder: null,
  streamRecorder: null,
  timerId: null,
  startedAt: 0,
  streamProcessing: 0,
  segments: [],
  activeSegment: -1,
  audioBlob: null,
  audioUrl: null,
};

/* -------------------------------------------------------------------------- */
/* boot                                                                       */
/* -------------------------------------------------------------------------- */

async function init() {
  bindEvents();
  updateLanguageUI();
  updateModeUI();
  clearResult();
  refreshControls();
  setStatus('等待开始');

  try {
    const [config, model] = await Promise.all([fetchConfig(), fetchModelStatus()]);
    state.config = config;
    state.model = model;
    applyConfig(config);
    applyModelInfo(model);
    setBadge(el.serviceBadge, 'ok', '服务正常');
    if (config.version) el.footerVersion.textContent = `v${config.version}`;
    await loadModels();
  } catch (error) {
    setBadge(el.serviceBadge, 'error', '服务不可用');
    setStatus('服务不可用', 'error');
    showError(error);
  }
}

function applyConfig(config) {
  const maxSeconds = config.max_recording_seconds ?? 60;
  const maxMb = config.max_upload_mb ?? 10;
  const extensions = (config.allowed_extensions || [])
    .map((ext) => ext.replace('.', '').toUpperCase())
    .join(' / ');

  el.recordLimit.textContent = `单次最长 ${maxSeconds} 秒`;
  $('upload-limit').textContent = `支持 ${extensions || 'WAV / WebM / OGG / MP3 / M4A / FLAC'}，最大 ${maxMb} MB`;
  el.tipChunk.textContent = String(config.streaming?.chunk_seconds ?? 2.5);
  el.tipLimit.textContent = String(maxSeconds);
  el.tipFileLimit.textContent = `${maxMb} MB`;
}

function applyModelInfo(model) {
  const name = model.model || '—';
  el.statusModel.textContent = name;
  el.metaModel.textContent = name;
  el.statusDevice.textContent = `${String(model.device || 'cpu').toUpperCase()} · ${model.compute_type || 'int8'}`;
  if (model.loaded) {
    setBadge(el.modelBadge, 'ok', `模型已加载 · ${name}`);
  } else {
    setBadge(el.modelBadge, 'warn', `模型：${name} · 首次识别时加载`);
  }
}

/* -------------------------------------------------------------------------- */
/* model management                                                           */
/* -------------------------------------------------------------------------- */

async function loadModels() {
  try {
    const payload = await fetchModels();
    state.models = Array.isArray(payload.models) ? payload.models : [];
    state.selectedModel = payload.selected || state.selectedModel;
    renderModelOptions();
    updateModelSummary();
    refreshControls();
  } catch (error) {
    el.modelStatus.textContent = '模型列表读取失败，可刷新页面重试。';
    el.modelApply.disabled = true;
    showError(error);
  }
}

function renderModelOptions() {
  el.modelSelect.innerHTML = '';
  state.models.forEach((model) => {
    const option = document.createElement('option');
    option.value = model.id;
    const stateLabel = model.downloaded ? '已下载' : `需下载 ${model.size_mb} MB`;
    option.textContent = `${model.display_name} · ${model.params} · ${stateLabel}${model.recommended ? ' · 推荐' : ''}`;
    el.modelSelect.appendChild(option);
  });
  const preferred = state.selectedModel || (state.models[0] && state.models[0].id);
  if (preferred) el.modelSelect.value = preferred;
}

function updateModelSummary() {
  const model = state.models.find((item) => item.id === state.selectedModel);
  if (!model) {
    el.modelStatus.textContent = '模型信息不可用';
    return;
  }
  const downloadState = model.downloaded
    ? '已下载'
    : `未下载（约 ${model.size_mb} MB，应用后自动下载）`;
  el.modelStatus.textContent = `${model.description} 当前：${model.display_name} · ${downloadState}`;
  el.modelHint.textContent = model.recommended
    ? '推荐档位：准确率与 CPU 速度折中'
    : '切换模型无需重启服务';
}

async function applySelectedModel() {
  const modelId = el.modelSelect.value;
  if (!modelId || state.modelBusy) return;

  clearAlert();
  state.modelBusy = true;
  el.modelProgressTrack.hidden = true;
  refreshControls();
  setStatus(`正在切换模型 ${modelId}…`, 'busy');

  try {
    const response = await selectModel(modelId);
    state.selectedModel = response.selected;
    updateModelSummary();

    if (response.model && response.model.downloaded) {
      el.modelStatus.textContent = `已切换到 ${response.model.display_name}，下一次识别时加载权重。`;
      setBadge(el.modelBadge, 'warn', `模型：${response.model.display_name} · 待加载`);
      setStatus('模型已切换');
    } else {
      await downloadModelWeights(modelId);
    }
    await loadModels();
  } catch (error) {
    showError(error);
    setStatus('模型切换失败', 'error');
  } finally {
    state.modelBusy = false;
    el.modelProgressTrack.hidden = true;
    el.modelProgressBar.style.width = '0%';
    refreshControls();
  }
}

async function downloadModelWeights(modelId) {
  el.modelProgressTrack.hidden = false;
  el.modelProgressBar.style.width = '0%';
  el.modelStatus.textContent = `正在准备模型 ${modelId}…`;
  setStatus(`正在下载/加载模型 ${modelId}…`, 'busy');

  const finalEvent = await streamModelDownload(modelId, (event) => {
    if (typeof event.progress === 'number') {
      el.modelProgressBar.style.width = `${Math.round(event.progress * 100)}%`;
    }
    if (event.stage === 'starting') {
      el.modelStatus.textContent = `开始下载 ${modelId}…`;
    } else if (event.stage === 'downloading') {
      const downloaded = Math.round((event.bytes || 0) / 1048576);
      const total = Math.round((event.total_bytes || 0) / 1048576);
      el.modelStatus.textContent = `正在下载 ${modelId}：${downloaded} / ${total} MB`;
    } else if (event.stage === 'loading') {
      el.modelStatus.textContent = `正在加载 ${modelId} 权重…`;
    } else if (event.stage === 'ready') {
      el.modelStatus.textContent = `模型 ${modelId} 已就绪（加载 ${event.load_seconds}s）`;
      setBadge(el.modelBadge, 'ok', `模型已加载 · ${modelId}`);
    }
  });

  if (finalEvent && finalEvent.stage === 'error') {
    throw new ApiError(finalEvent.detail || '模型下载或加载失败。', {
      code: finalEvent.code || 'model_unavailable',
      status: 0,
      payload: finalEvent,
    });
  }
  setStatus('模型已就绪', 'ok');
}

/* -------------------------------------------------------------------------- */
/* audio player, segments and export                                          */
/* -------------------------------------------------------------------------- */

function formatClock(seconds) {
  const safe = Math.max(0, Number(seconds) || 0);
  const minutes = Math.floor(safe / 60);
  const rest = safe - minutes * 60;
  return `${String(minutes).padStart(2, '0')}:${rest.toFixed(1).padStart(4, '0')}`;
}

function setAudioFromBlob(blob) {
  if (state.audioUrl) URL.revokeObjectURL(state.audioUrl);
  state.audioBlob = blob || null;
  state.audioUrl = blob ? URL.createObjectURL(blob) : null;
  if (blob) {
    el.audioPlayer.src = state.audioUrl;
    el.audioPlayer.hidden = false;
  } else {
    el.audioPlayer.removeAttribute('src');
    el.audioPlayer.hidden = true;
  }
}

function renderSegments(segments) {
  state.segments = Array.isArray(segments) ? segments : [];
  state.activeSegment = -1;
  el.segmentList.innerHTML = '';

  if (state.segments.length === 0) {
    el.segmentPanel.hidden = true;
    setExportEnabled(false);
    return;
  }

  el.segmentPanel.hidden = false;
  state.segments.forEach((segment) => {
    const item = document.createElement('li');
    item.className = 'segment';

    const timeButton = document.createElement('button');
    timeButton.type = 'button';
    timeButton.className = 'segment__time';
    timeButton.textContent = formatClock(segment.start);
    timeButton.title = '跳转到这一段';
    timeButton.addEventListener('click', () => seekTo(segment.start));

    const text = document.createElement('span');
    text.className = 'segment__text';
    text.textContent = String(segment.text || '').trim();

    item.append(timeButton, text);
    el.segmentList.appendChild(item);
  });
  setExportEnabled(true);
}

function setExportEnabled(enabled) {
  el.exportTxt.disabled = !enabled;
  el.exportSrt.disabled = !enabled;
  el.exportVtt.disabled = !enabled;
}

function seekTo(seconds) {
  if (!state.audioUrl) return;
  try {
    el.audioPlayer.currentTime = Math.max(0, Number(seconds) || 0);
    const playback = el.audioPlayer.play();
    if (playback && typeof playback.catch === 'function') playback.catch(() => {});
  } catch (error) {
    /* seeking is best effort */
  }
}

function handleAudioTimeUpdate() {
  if (state.segments.length === 0) return;
  const time = el.audioPlayer.currentTime;
  let index = -1;
  for (let i = 0; i < state.segments.length; i += 1) {
    const segment = state.segments[i];
    if (time >= (segment.start || 0) - 0.05 && time <= (segment.end || 0) + 0.05) {
      index = i;
      break;
    }
  }
  if (index === state.activeSegment) return;
  state.activeSegment = index;
  Array.from(el.segmentList.children).forEach((node, i) => {
    node.classList.toggle('is-active', i === index);
  });
  if (index >= 0 && !el.audioPlayer.paused) {
    el.segmentList.children[index].scrollIntoView({ block: 'nearest' });
  }
}

async function exportTranscript(format) {
  if (state.segments.length === 0) return;
  try {
    const content = await fetchSubtitles({
      segments: state.segments,
      format,
      text: el.resultText.value,
    });
    const mime = format === 'txt' ? 'text/plain' : format === 'vtt' ? 'text/vtt' : 'application/x-subrip';
    downloadTextFile(content, `transcript.${format}`, mime);
    el.copyFeedback.textContent = `已导出 ${format.toUpperCase()}`;
    window.setTimeout(() => {
      el.copyFeedback.textContent = '';
    }, 2000);
  } catch (error) {
    showError(error);
  }
}

function downloadTextFile(content, filename, mimeType) {
  const blob = new Blob([content], { type: `${mimeType};charset=utf-8` });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 2000);
}

/* -------------------------------------------------------------------------- */
/* events                                                                     */
/* -------------------------------------------------------------------------- */

function bindEvents() {
  languageButtons.forEach((button) => {
    button.addEventListener('click', () => {
      if (state.busy || state.recording || state.streaming) return;
      state.language = button.dataset.language;
      updateLanguageUI();
    });
  });

  modeButtons.forEach((button) => {
    button.addEventListener('click', () => {
      if (state.busy || state.recording || state.streaming) return;
      state.mode = button.dataset.mode;
      updateModeUI();
    });
  });

  el.recordButton.addEventListener('click', () => {
    if (state.mode === 'stream') startStreaming();
    else startBatchRecording();
  });

  el.stopButton.addEventListener('click', () => {
    if (state.mode === 'stream') stopStreaming(false);
    else finishBatchRecording(false);
  });

  el.clearButton.addEventListener('click', () => {
    if (state.recording || state.streaming) return;
    clearResult();
    clearAlert();
    el.recordTimer.textContent = formatTimer(0);
    setStatus('等待开始');
  });

  el.copyButton.addEventListener('click', copyResult);

  el.wordTimestamps.addEventListener('change', () => {
    state.wordTimestamps = Boolean(el.wordTimestamps.checked);
  });

  el.modelApply.addEventListener('click', applySelectedModel);
  el.modelSelect.addEventListener('change', () => {
    const model = state.models.find((item) => item.id === el.modelSelect.value);
    if (!model) return;
    const stateLabel = model.downloaded ? '已下载' : `需下载约 ${model.size_mb} MB`;
    el.modelStatus.textContent = `${model.description} ${model.display_name} · ${stateLabel}`;
  });

  el.exportTxt.addEventListener('click', () => exportTranscript('txt'));
  el.exportSrt.addEventListener('click', () => exportTranscript('srt'));
  el.exportVtt.addEventListener('click', () => exportTranscript('vtt'));
  el.audioPlayer.addEventListener('timeupdate', handleAudioTimeUpdate);

  el.dropZone.addEventListener('click', () => el.fileInput.click());
  el.dropZone.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      el.fileInput.click();
    }
  });
  el.dropZone.addEventListener('dragover', (event) => {
    event.preventDefault();
    el.dropZone.classList.add('is-dragover');
  });
  el.dropZone.addEventListener('dragleave', () => el.dropZone.classList.remove('is-dragover'));
  el.dropZone.addEventListener('drop', (event) => {
    event.preventDefault();
    el.dropZone.classList.remove('is-dragover');
    const file = event.dataTransfer && event.dataTransfer.files ? event.dataTransfer.files[0] : null;
    if (file) setFile(file);
  });

  el.fileInput.addEventListener('change', () => {
    const file = el.fileInput.files && el.fileInput.files[0];
    if (file) setFile(file);
  });

  el.fileClear.addEventListener('click', () => {
    setFile(null);
    el.fileInput.value = '';
  });

  el.uploadButton.addEventListener('click', uploadSelected);

  el.sampleChips.forEach((button) => {
    button.addEventListener('click', () => loadSample(button));
  });
}

/* -------------------------------------------------------------------------- */
/* language / mode                                                            */
/* -------------------------------------------------------------------------- */

function updateLanguageUI() {
  languageButtons.forEach((button) => {
    const active = button.dataset.language === state.language;
    button.classList.toggle('is-active', active);
    button.setAttribute('aria-checked', String(active));
  });
  el.statusLanguage.textContent = languageLabel(state.language);
}

function updateModeUI() {
  modeButtons.forEach((button) => {
    const active = button.dataset.mode === state.mode;
    button.classList.toggle('is-active', active);
    button.setAttribute('aria-selected', String(active));
  });

  const streaming = state.mode === 'stream';
  el.recordButton.innerHTML = streaming
    ? '<span class="btn__icon" aria-hidden="true">●</span>开始实时识别'
    : '<span class="btn__icon" aria-hidden="true">●</span>开始录音';
  el.stopButton.innerHTML = streaming
    ? '<span class="btn__icon" aria-hidden="true">■</span>停止并输出结果'
    : '<span class="btn__icon" aria-hidden="true">■</span>结束录音';
  el.micHint.textContent = streaming
    ? '实时模式下，说话过程中识别文字会持续出现在下方结果区。'
    : '首次使用请允许浏览器访问麦克风。建议使用 Chrome / Edge。';
  el.resultHint.textContent = streaming
    ? '点击「开始实时识别」后，文字会边说边出'
    : '停止录音或上传文件后自动开始识别';
}

/* -------------------------------------------------------------------------- */
/* batch recording                                                            */
/* -------------------------------------------------------------------------- */

async function startBatchRecording() {
  if (state.busy || state.recording) return;
  clearAlert();
  clearResult();

  if (!MicRecorder.supported) {
    showAlert(
      '浏览器不支持录音',
      '请使用最新版 Chrome 或 Edge，并通过 http://127.0.0.1:8000 访问本页面（http 下的局域网 IP 不允许使用麦克风）。',
    );
    setStatus('浏览器不支持录音', 'error');
    return;
  }

  const maxSeconds = state.config?.max_recording_seconds ?? 60;
  const recorder = new MicRecorder({ maxSeconds, onLevel: setLevel });
  recorder.onAutoStop = () => {
    void finishBatchRecording(true);
  };

  state.recorder = recorder;
  state.recording = true;
  el.recordTimer.textContent = formatTimer(0);
  el.recordTimer.classList.add('is-recording');
  setProgress(0, true);
  refreshControls();
  setStatus('正在请求麦克风权限…', 'busy');

  try {
    await recorder.start();
  } catch (error) {
    state.recording = false;
    state.recorder = null;
    el.recordTimer.classList.remove('is-recording');
    refreshControls();
    const info = describeMicError(error);
    showAlert(info.title, info.detail);
    setStatus('录音未开始', 'error');
    return;
  }

  setStatus('正在录音…', 'recording');
  startTimer(maxSeconds);
}

async function finishBatchRecording(auto) {
  const recorder = state.recorder;
  if (!state.recording || !recorder) return;

  state.recording = false;
  stopTimer();
  el.recordTimer.classList.remove('is-recording');
  setLevel(0);
  refreshControls();
  setStatus('正在处理录音…', 'busy');

  let clip = null;
  try {
    clip = await recorder.stop();
  } catch (error) {
    state.recorder = null;
    setProgress(0, false);
    showAlert('录音失败', error.message || '录音过程出现错误，请重试。');
    setStatus('录音失败', 'error');
    return;
  }
  state.recorder = null;

  if (!clip || clip.blob.size === 0) {
    setProgress(0, false);
    setStatus('没有采集到音频', 'error');
    showAlert('录音为空', '没有采集到音频数据，请检查麦克风权限与设备后重试。');
    return;
  }

  el.recordTimer.textContent = formatTimer(clip.durationSeconds);

  if (clip.durationSeconds < 0.4) {
    setProgress(0, false);
    setStatus('录音太短', 'error');
    showAlert('录音太短', '请至少录制 0.5 秒语音后再点击「结束录音」。');
    return;
  }

  const limit = state.config?.max_recording_seconds ?? 60;
  if (auto) setStatus(`已达到 ${limit} 秒上限，已自动停止并识别…`, 'busy');

  await runTranscription({
    blob: clip.blob,
    filename: `recording.${clip.extension}`,
    source: `麦克风录音（${clip.extension.toUpperCase()}）`,
  });

  if (auto) {
    el.resultHint.textContent = `单次录音上限 ${limit} 秒，已自动结束并完成识别`;
  }
}

/* -------------------------------------------------------------------------- */
/* upload                                                                     */
/* -------------------------------------------------------------------------- */

function setFile(file) {
  state.file = file || null;
  if (!state.file) {
    el.fileRow.hidden = true;
    el.fileName.textContent = '—';
    el.fileSize.textContent = '';
    refreshControls();
    return;
  }
  el.fileRow.hidden = false;
  el.fileName.textContent = state.file.name;
  el.fileSize.textContent = formatBytes(state.file.size);
  refreshControls();
}

function validateFile(file) {
  if (!file.size) {
    return { title: '文件为空', detail: '所选文件大小为 0 字节，请选择包含音频内容的文件。' };
  }
  const limit = state.config?.max_upload_bytes ?? 10 * 1024 * 1024;
  if (file.size > limit) {
    return {
      title: '文件过大',
      detail: `文件大小为 ${formatBytes(file.size)}，超过上限 ${formatBytes(limit)}。`,
    };
  }
  const allowed = state.config?.allowed_extensions ?? [
    '.wav', '.webm', '.ogg', '.oga', '.opus', '.mp3', '.m4a', '.mp4', '.flac',
  ];
  const dot = file.name.lastIndexOf('.');
  const extension = dot >= 0 ? file.name.slice(dot).toLowerCase() : '';
  if (!extension) {
    return {
      title: '无法判断文件格式',
      detail: `文件名缺少扩展名。请把文件命名为带扩展名的形式，例如 audio.wav。`,
    };
  }
  if (!allowed.includes(extension)) {
    return {
      title: '不支持的音频格式',
      detail: `${extension} 不在支持列表中。支持的格式：${allowed.join('、')}。`,
    };
  }
  return null;
}

async function uploadSelected() {
  const file = state.file;
  if (!file) return;
  clearAlert();

  const problem = validateFile(file);
  if (problem) {
    showAlert(problem.title, problem.detail);
    setStatus('文件校验未通过', 'error');
    return;
  }

  await runTranscription({ blob: file, filename: file.name, source: '文件上传' });
}

async function loadSample(button) {
  if (state.busy || state.recording || state.streaming) return;
  clearAlert();
  clearResult();

  const name = button.dataset.sample;
  const language = button.dataset.sampleLanguage || 'zh';
  state.language = language;
  updateLanguageUI();
  setStatus('正在加载示例音频…', 'busy');

  try {
    const blob = await fetchSample(name);
    const file = new File([blob], `${name}.wav`, { type: 'audio/wav' });
    setFile(file);
    await runTranscription({ blob: file, filename: file.name, source: '内置示例音频' });
  } catch (error) {
    showError(error);
    setStatus('示例音频加载失败', 'error');
  }
}

/* -------------------------------------------------------------------------- */
/* batch transcription                                                        */
/* -------------------------------------------------------------------------- */

async function runTranscription({ blob, filename, source }) {
  if (state.busy) return;

  state.busy = true;
  refreshControls();
  clearAlert();
  setStatus('正在识别…', 'busy');
  el.resultHint.textContent = '正在识别，请稍候…';
  setProgress(0.4, true);

  try {
    const response = await transcribeAudio({
      blob,
      filename,
      language: state.language,
      wordTimestamps: state.wordTimestamps,
    });
    applyTranscription(response, source, blob);
    setProgress(1, true);
    window.setTimeout(() => setProgress(0, false), 700);
    setStatus('识别完成', 'ok');
  } catch (error) {
    setProgress(0, false);
    el.resultHint.textContent = '识别失败，可重试';
    setStatus('识别失败', 'error');
    showError(error);
  } finally {
    state.busy = false;
    refreshControls();
  }
}

function applyTranscription(response, source, blob = null) {
  const text = response.text || '';
  el.resultText.value = text;
  el.resultText.scrollTop = 0;

  setAudioFromBlob(blob);
  renderSegments(response.segments || []);

  const warnings = String(response.warning || '')
    .split(',')
    .map((item) => item.trim())
    .filter(Boolean);

  el.metaLanguage.textContent = languageLabel(response.language);
  el.metaDuration.textContent = formatDurationLong(response.duration_seconds);
  el.metaProcessing.textContent = formatSeconds(response.processing_seconds);
  el.metaModel.textContent = response.model || state.model?.model || '—';

  el.statusLanguage.textContent = languageLabel(response.language);
  el.statusDuration.textContent = formatDurationLong(response.duration_seconds);
  el.statusProcessing.textContent = formatSeconds(response.processing_seconds);
  el.statusSource.textContent = source || '—';
  el.statusModel.textContent = response.model || state.model?.model || '—';
  el.statusState.textContent = '识别完成';
  if (response.model) {
    setBadge(el.modelBadge, 'ok', `模型已加载 · ${response.model}`);
  }

  if (warnings.includes('degraded_model')) {
    showAlert(
      '已降级到备用模型',
      '主模型本次推理失败，结果由备用模型生成，准确率可能略有下降；可稍后重试主模型。',
      'warn',
    );
  } else if (warnings.includes('deadline_exceeded')) {
    showAlert(
      '识别超出时限',
      '已返回部分识别结果。可尝试缩短音频，或在「识别模型」中切换到更小的档位。',
      'warn',
    );
  }

  if (!text) {
    el.resultHint.textContent = '识别完成，但没有检测到可识别的语音';
    showAlert(
      '没有识别到文字',
      '音频中没有检测到清晰的语音内容。请靠近麦克风、放慢语速，或确认所选语言与音频语言一致后重试。',
      'warn',
    );
    return;
  }

  const suffix = warnings.includes('deadline_exceeded') ? '（部分结果）' : '';
  el.resultHint.textContent = `识别完成 · ${text.length} 个字符${suffix}`;
}

/* -------------------------------------------------------------------------- */
/* streaming (bonus)                                                          */
/* -------------------------------------------------------------------------- */

async function startStreaming() {
  if (state.busy || state.streaming) return;
  clearAlert();
  clearResult();

  const streaming = state.config?.streaming;
  if (!streaming || !streaming.enabled) {
    showAlert(
      '实时识别未启用',
      '服务端已关闭实时识别功能（STT_STREAMING_ENABLED=0）。可使用「录音结束后识别」模式。',
      'warn',
    );
    return;
  }
  if (!StreamRecorder.supported) {
    showAlert('浏览器不支持实时识别', '实时识别需要 WebSocket 与麦克风支持，请使用最新版 Chrome 或 Edge。');
    setStatus('浏览器不支持实时识别', 'error');
    return;
  }

  state.streaming = true;
  state.busy = true;
  state.streamProcessing = 0;
  el.resultText.classList.add('is-streaming');
  el.resultHint.textContent = '正在建立实时连接…';
  el.recordTimer.textContent = formatTimer(0);
  el.recordTimer.classList.add('is-recording');
  refreshControls();
  setStatus('正在连接实时识别…', 'busy');

  const recorder = new StreamRecorder({
    chunkSeconds: streaming.chunk_seconds,
    maxSeconds: streaming.max_seconds,
    onLevel: setLevel,
    onPartial: handlePartial,
    onError: handleStreamError,
  });
  recorder.onAutoStop = () => {
    void stopStreaming(true);
  };
  state.streamRecorder = recorder;

  try {
    await recorder.start(state.language);
  } catch (error) {
    state.streaming = false;
    state.busy = false;
    state.streamRecorder = null;
    el.resultText.classList.remove('is-streaming');
    el.recordTimer.classList.remove('is-recording');
    refreshControls();
    setStatus('实时识别未开始', 'error');
    if (error instanceof ApiError) showError(error);
    else {
      const info = describeMicError(error);
      showAlert(info.title, info.detail);
    }
    return;
  }

  el.resultHint.textContent = '实时识别中，文字会持续更新…';
  setProgress(0, true);
  startTimer(streaming.max_seconds);
  setStatus('正在实时识别…', 'recording');
}

function handlePartial(payload) {
  if (typeof payload.text === 'string') {
    el.resultText.value = payload.text;
    el.resultText.scrollTop = el.resultText.scrollHeight;
    el.copyButton.disabled = !payload.text;
  }

  if (typeof payload.duration_seconds === 'number') {
    el.metaDuration.textContent = formatDurationLong(payload.duration_seconds);
    el.statusDuration.textContent = formatDurationLong(payload.duration_seconds);
  }

  if (typeof payload.processing_seconds === 'number') {
    state.streamProcessing += payload.processing_seconds;
    el.metaProcessing.textContent = formatSeconds(state.streamProcessing);
    el.statusProcessing.textContent = formatSeconds(state.streamProcessing);
  }

  el.metaLanguage.textContent = languageLabel(state.language);
  el.statusLanguage.textContent = languageLabel(state.language);
  el.statusSource.textContent = '麦克风（实时片段）';
}

function handleStreamError(payload) {
  const title = ERROR_TITLES[payload.code] || '实时识别出错';
  const advice = ERROR_ADVICE[payload.code] || '';
  showAlert(title, [payload.detail, advice].filter(Boolean).join(' '), 'error', payload.code);
}

async function stopStreaming(auto) {
  const recorder = state.streamRecorder;
  if (!state.streaming || !recorder) return;

  state.streaming = false;
  stopTimer();
  el.resultText.classList.remove('is-streaming');
  el.recordTimer.classList.remove('is-recording');
  setLevel(0);
  setProgress(0, false);
  refreshControls();
  setStatus(auto ? '已达到时长上限，正在生成最终结果…' : '正在生成最终结果…', 'busy');

  try {
    const final = await recorder.stop();
    if (final) {
      el.resultText.value = final.text || '';
      renderSegments(final.segments || []);
      if (recorder.lastAudioBlob) setAudioFromBlob(recorder.lastAudioBlob);
      const language = final.language || state.language;
      el.metaLanguage.textContent = languageLabel(language);
      el.statusLanguage.textContent = languageLabel(language);
      el.metaDuration.textContent = formatDurationLong(final.duration_seconds);
      el.statusDuration.textContent = formatDurationLong(final.duration_seconds);
      el.metaProcessing.textContent = formatSeconds(final.processing_seconds);
      el.statusProcessing.textContent = formatSeconds(final.processing_seconds);
      el.metaModel.textContent = final.model || '—';
      el.statusModel.textContent = final.model || '—';
      el.statusSource.textContent = '麦克风（实时识别）';
      el.copyButton.disabled = !final.text;
      el.recordTimer.textContent = formatTimer(final.duration_seconds || 0);

      if (final.text) {
        el.resultHint.textContent = final.refined
          ? `实时识别完成（已对整段音频复核）· ${final.text.length} 个字符`
          : `实时识别完成 · ${final.text.length} 个字符`;
        setStatus('识别完成', 'ok');
      } else {
        el.resultHint.textContent = '实时识别完成，但没有识别到文字';
        setStatus('没有识别到文字', 'warn');
        showAlert(
          '没有识别到文字',
          '实时识别没有返回文本，请靠近麦克风、清晰说话后重试。',
          'warn',
        );
      }
    }
  } catch (error) {
    if (error instanceof ApiError) showError(error);
    else showAlert('实时识别失败', error.message || '实时识别过程中出现错误，请重试。');
    setStatus('实时识别失败', 'error');
  } finally {
    state.busy = false;
    state.streamRecorder = null;
    refreshControls();
  }
}

/* -------------------------------------------------------------------------- */
/* UI helpers                                                                 */
/* -------------------------------------------------------------------------- */

function refreshControls() {
  const locked = state.busy || state.recording || state.streaming;
  languageButtons.forEach((button) => {
    button.disabled = locked;
  });
  modeButtons.forEach((button) => {
    button.disabled = locked;
  });
  el.recordButton.disabled = locked;
  el.stopButton.disabled = !(state.recording || state.streaming);
  el.uploadButton.disabled = locked || !state.file;
  el.fileClear.disabled = locked || !state.file;
  el.fileInput.disabled = locked;
  el.dropZone.setAttribute('aria-disabled', String(locked));
  el.sampleChips.forEach((button) => {
    button.disabled = locked;
  });
  el.modelSelect.disabled = locked || state.modelBusy;
  el.modelApply.disabled = locked || state.modelBusy || state.models.length === 0;
  el.copyButton.disabled = !el.resultText.value;
  setExportEnabled(state.segments.length > 0 && !locked);
}

function setStatus(text, tone = 'idle') {
  el.recordStatus.textContent = text;
  el.statusState.textContent = text;

  const dot = el.recordingDot;
  dot.classList.remove('is-recording', 'is-busy', 'is-ok', 'is-error');
  if (state.recording || state.streaming) dot.classList.add('is-recording');
  else if (tone === 'busy') dot.classList.add('is-busy');
  else if (tone === 'ok') dot.classList.add('is-ok');
  else if (tone === 'error') dot.classList.add('is-error');
}

function setLevel(level) {
  const percent = Math.max(0, Math.min(1, level || 0)) * 100;
  el.levelBar.style.width = `${percent.toFixed(1)}%`;
}

function setProgress(value, visible) {
  el.progressTrack.hidden = !visible;
  el.progressBar.style.width = `${Math.round(Math.max(0, Math.min(1, value)) * 100)}%`;
}

function startTimer(limitSeconds) {
  stopTimer();
  state.startedAt = performance.now();
  state.timerId = window.setInterval(() => {
    const seconds = (performance.now() - state.startedAt) / 1000;
    el.recordTimer.textContent = formatTimer(seconds);
    if (limitSeconds) {
      el.progressBar.style.width = `${Math.min(100, (seconds / limitSeconds) * 100).toFixed(1)}%`;
    }
  }, 100);
}

function stopTimer() {
  if (state.timerId) window.clearInterval(state.timerId);
  state.timerId = null;
}

function clearResult() {
  el.resultText.value = '';
  el.metaLanguage.textContent = '—';
  el.metaDuration.textContent = '—';
  el.metaProcessing.textContent = '—';
  el.metaModel.textContent = state.model?.model || '—';
  el.statusDuration.textContent = '—';
  el.statusProcessing.textContent = '—';
  el.statusSource.textContent = '—';
  el.copyFeedback.textContent = '';
  el.copyButton.disabled = true;
  el.recordTimer.textContent = formatTimer(0);
  setAudioFromBlob(null);
  renderSegments([]);
}

function setBadge(element, tone, text) {
  element.classList.remove('badge--ok', 'badge--warn', 'badge--error');
  if (tone) element.classList.add(`badge--${tone}`);
  element.textContent = '';
  const dot = document.createElement('span');
  dot.className = 'badge__dot';
  element.appendChild(dot);
  element.appendChild(document.createTextNode(text));
}

function showAlert(title, detail, tone = 'error', raw = '') {
  el.alertCard.hidden = false;
  el.alertBox.classList.toggle('alert--warn', tone === 'warn');
  el.alertTitle.textContent = title;
  el.alertDetail.textContent = detail || '';
  if (raw) {
    el.alertRaw.hidden = false;
    el.alertRaw.textContent = raw;
  } else {
    el.alertRaw.hidden = true;
    el.alertRaw.textContent = '';
  }
}

function clearAlert() {
  el.alertCard.hidden = true;
  el.alertTitle.textContent = '';
  el.alertDetail.textContent = '';
  el.alertRaw.hidden = true;
  el.alertRaw.textContent = '';
}

function showError(error) {
  if (error instanceof ApiError) {
    const title = ERROR_TITLES[error.code] || '识别请求失败';
    const advice = ERROR_ADVICE[error.code] || '';
    const retryHint = error.retryable ? '（该错误可重试）' : '';
    const detail = [error.message, retryHint, advice].filter(Boolean).join(' ');
    const raw = error.status ? `HTTP ${error.status} · ${error.code}` : error.code;
    showAlert(title, detail, 'error', raw);
    return;
  }
  showAlert('出现未知错误', (error && error.message) || '请重试或查看浏览器控制台。');
}

async function copyResult() {
  const text = el.resultText.value;
  if (!text) return;

  try {
    if (navigator.clipboard && window.isSecureContext) {
      await navigator.clipboard.writeText(text);
    } else {
      el.resultText.removeAttribute('readonly');
      el.resultText.select();
      document.execCommand('copy');
      el.resultText.setAttribute('readonly', 'readonly');
    }
    el.copyFeedback.textContent = '已复制到剪贴板';
  } catch (error) {
    el.copyFeedback.textContent = '复制失败，请手动选择文本复制';
  }
  window.setTimeout(() => {
    el.copyFeedback.textContent = '';
  }, 2400);
}

init();
