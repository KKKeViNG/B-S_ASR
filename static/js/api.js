/**
 * Thin wrapper around the backend HTTP API.
 * Every non-2xx answer is turned into an ApiError carrying the machine readable
 * `code` and the human readable `detail` produced by the FastAPI layer.
 */

export class ApiError extends Error {
  constructor(message, { code = 'unknown_error', status = 0, payload = null, cause = null, retryable = false } = {}) {
    super(message);
    this.name = 'ApiError';
    this.code = code;
    this.status = status;
    this.payload = payload;
    this.cause = cause;
    this.retryable = Boolean(retryable);
  }
}

async function parseJson(response) {
  const text = await response.text();
  if (!text) return null;
  try {
    return JSON.parse(text);
  } catch (error) {
    return null;
  }
}

function networkError(error) {
  return new ApiError('无法连接后端服务，请确认服务正在运行。', {
    code: 'network_error',
    status: 0,
    cause: error,
  });
}

export async function fetchConfig() {
  let response;
  try {
    response = await fetch('/api/config', { headers: { Accept: 'application/json' } });
  } catch (error) {
    throw networkError(error);
  }
  if (!response.ok) throw new ApiError('无法读取服务配置。', { code: 'config_error', status: response.status });
  return response.json();
}

export async function fetchModelStatus() {
  let response;
  try {
    response = await fetch('/api/model', { headers: { Accept: 'application/json' } });
  } catch (error) {
    throw networkError(error);
  }
  if (!response.ok) throw new ApiError('无法读取模型状态。', { code: 'model_status_error', status: response.status });
  return response.json();
}

/**
 * Upload a clip for recognition.
 *
 * @param {{blob: Blob, filename: string, language: string}} options
 */
export async function transcribeAudio({ blob, filename, language, wordTimestamps = false }) {
  const form = new FormData();
  form.append('file', blob, filename);
  form.append('language', language);
  if (wordTimestamps) form.append('word_timestamps', 'true');

  let response;
  try {
    response = await fetch('/api/transcribe', { method: 'POST', body: form });
  } catch (error) {
    throw networkError(error);
  }

  const payload = await parseJson(response);
  if (!response.ok) {
    const detail = payload && typeof payload.detail === 'string' ? payload.detail : `请求失败（HTTP ${response.status}）`;
    const code = payload && typeof payload.code === 'string' ? payload.code : 'http_error';
    throw new ApiError(detail, {
      code,
      status: response.status,
      payload,
      retryable: Boolean(payload && payload.retryable),
    });
  }
  if (!payload || typeof payload.text !== 'string') {
    throw new ApiError('后端返回了无法解析的识别结果。', { code: 'bad_response', status: response.status });
  }
  return payload;
}

export async function fetchSample(name) {
  let response;
  try {
    response = await fetch(`/samples/${encodeURIComponent(name)}.wav`);
  } catch (error) {
    throw networkError(error);
  }
  if (!response.ok) {
    throw new ApiError('示例音频不可用（samples/ 目录未找到）。', { code: 'sample_missing', status: response.status });
  }
  return response.blob();
}

export async function fetchModels() {
  let response;
  try {
    response = await fetch('/api/models', { headers: { Accept: 'application/json' } });
  } catch (error) {
    throw networkError(error);
  }
  if (!response.ok) {
    throw new ApiError('无法读取模型列表。', { code: 'models_error', status: response.status });
  }
  return response.json();
}

export async function selectModel(modelId) {
  let response;
  try {
    response = await fetch('/api/models/select', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
      body: JSON.stringify({ model: modelId }),
    });
  } catch (error) {
    throw networkError(error);
  }

  const payload = await parseJson(response);
  if (!response.ok) {
    throw new ApiError(payload && payload.detail ? payload.detail : `模型切换失败（HTTP ${response.status}）`, {
      code: payload && payload.code ? payload.code : 'model_select_error',
      status: response.status,
      payload,
      retryable: Boolean(payload && payload.retryable),
    });
  }
  return payload;
}

export async function fetchSubtitles({ segments, format, text }) {
  let response;
  try {
    response = await fetch('/api/subtitles', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Accept: 'text/plain' },
      body: JSON.stringify({ segments, format, text }),
    });
  } catch (error) {
    throw networkError(error);
  }
  if (!response.ok) {
    const payload = await parseJson(response);
    throw new ApiError(payload && payload.detail ? payload.detail : `字幕生成失败（HTTP ${response.status}）`, {
      code: payload && payload.code ? payload.code : 'subtitle_error',
      status: response.status,
      payload,
    });
  }
  return response.text();
}

/**
 * Open the Server-Sent-Events download/warm-up stream for one model.
 * Resolves with the final event (`ready` or `error`).
 */
export async function streamModelDownload(modelId, onEvent = () => {}) {
  let response;
  try {
    response = await fetch(`/api/models/${encodeURIComponent(modelId)}/download`, {
      headers: { Accept: 'text/event-stream' },
    });
  } catch (error) {
    throw networkError(error);
  }

  if (!response.ok) {
    const payload = await parseJson(response);
    throw new ApiError(payload && payload.detail ? payload.detail : `模型加载失败（HTTP ${response.status}）`, {
      code: payload && payload.code ? payload.code : 'model_download_error',
      status: response.status,
      payload,
    });
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let lastEvent = null;

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    let boundary = buffer.indexOf('\n\n');
    while (boundary >= 0) {
      const rawEvent = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      for (const line of rawEvent.split('\n')) {
        if (!line.startsWith('data: ')) continue;
        try {
          lastEvent = JSON.parse(line.slice(6));
        } catch (error) {
          continue;
        }
        onEvent(lastEvent);
      }
      boundary = buffer.indexOf('\n\n');
    }
  }
  return lastEvent;
}
