/** Chinese UI copy: error translations, labels and small formatters. */

export const ERROR_TITLES = {
  unsupported_format: '不支持的音频格式',
  payload_too_large: '文件超过大小限制',
  audio_decode_error: '音频无法解码',
  invalid_language: '识别语言无效',
  model_unavailable: '本地模型加载失败',
  transcription_failed: '识别失败',
  bad_request: '请求无效',
  validation_error: '请求参数不完整',
  no_speech_detected: '没有检测到语音',
  network_error: '无法连接后端服务',
  sample_missing: '示例音频不可用',
  bad_response: '无法解析识别结果',
  http_error: '请求失败',
  config_error: '服务配置读取失败',
  model_status_error: '模型状态读取失败',
  unknown_error: '出现未知错误',
};

export const ERROR_ADVICE = {
  unsupported_format: '请改用 WAV、WebM、OGG、OPUS、MP3、M4A 或 FLAC 格式的音频。',
  payload_too_large: '请裁剪或压缩音频后重试（单文件上限见页面提示）。',
  audio_decode_error: '文件可能已损坏、为空或内容不是真正的音频，请换一个文件重试。',
  invalid_language: '请选择「中文普通话」「English」或「自动检测」。',
  model_unavailable: '请确认模型权重已下载：python scripts/download_model.py，然后重新识别。',
  transcription_failed: '可以稍后重试；若持续失败，请查看后端控制台日志。',
  network_error: '请确认后端已启动：python -m uvicorn app.main:app --host 127.0.0.1 --port 8000',
  sample_missing: '请确认 samples/chinese.wav 与 samples/english.wav 存在。',
};

const LANGUAGE_LABELS = {
  zh: '中文普通话',
  en: 'English',
  auto: '自动检测',
};

export function languageLabel(code) {
  return LANGUAGE_LABELS[code] || code || '—';
}

export function formatSeconds(value) {
  if (typeof value !== 'number' || Number.isNaN(value)) return '—';
  if (value < 1) return `${(value * 1000).toFixed(0)} ms`;
  return `${value.toFixed(2)} 秒`;
}

export function formatDurationLong(value) {
  if (typeof value !== 'number' || Number.isNaN(value)) return '—';
  return `${value.toFixed(2)} 秒`;
}

export function formatBytes(bytes) {
  if (!Number.isFinite(bytes)) return '—';
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

export function formatTimer(seconds) {
  const safe = Math.max(0, seconds || 0);
  const minutes = Math.floor(safe / 60);
  const rest = safe - minutes * 60;
  const whole = Math.floor(rest);
  const tenth = Math.floor((rest - whole) * 10);
  return `${String(minutes).padStart(2, '0')}:${String(whole).padStart(2, '0')}.${tenth}`;
}

/** Translate a MediaRecorder/getUserMedia failure into actionable UI copy. */
export function describeMicError(error) {
  const name = error && error.name ? error.name : '';
  switch (name) {
    case 'NotAllowedError':
    case 'PermissionDeniedError':
      return {
        title: '麦克风权限被拒绝',
        detail:
          '浏览器或系统拒绝了麦克风访问。请点击地址栏左侧的站点图标，把「麦克风」改为允许，然后刷新页面重试。',
      };
    case 'NotFoundError':
    case 'DevicesNotFoundError':
      return {
        title: '未检测到可用麦克风',
        detail: '系统中没有找到录音设备。请连接麦克风或在系统声音设置中启用输入设备后重试。',
      };
    case 'NotReadableError':
    case 'TrackStartError':
      return {
        title: '麦克风不可用',
        detail: '麦克风可能正被其他应用占用（会议软件、录屏工具等），请关闭后重试。',
      };
    case 'OverconstrainedError':
    case 'ConstraintNotSatisfiedError':
      return {
        title: '麦克风不支持所需参数',
        detail: '当前录音设备的参数不被支持，请换一个输入设备后重试。',
      };
    case 'SecurityError':
      return {
        title: '需要安全上下文',
        detail:
          '浏览器只允许在 https 或 localhost/127.0.0.1 下使用麦克风。请通过 http://127.0.0.1:8000 访问本页面。',
      };
    case 'AbortError':
      return { title: '录音被中断', detail: '录音过程被系统中断，请重新开始录音。' };
    default:
      return {
        title: '无法开始录音',
        detail: (error && error.message) || '浏览器未能启动录音，请检查麦克风与权限设置后重试。',
      };
  }
}
