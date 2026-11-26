/**
 * Thin wrapper around the backend HTTP API.
 * Every non-2xx answer is turned into an ApiError carrying the machine readable
 * `code` and the human readable `detail` produced by the FastAPI layer.
 */

export class ApiError extends Error {
  constructor(message, { code = 'unknown_error', status = 0, payload = null, cause = null } = {}) {
    super(message);
    this.name = 'ApiError';
    this.code = code;
    this.status = status;
    this.payload = payload;
    this.cause = cause;
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
export async function transcribeAudio({ blob, filename, language }) {
  const form = new FormData();
  form.append('file', blob, filename);
  form.append('language', language);

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
    throw new ApiError(detail, { code, status: response.status, payload });
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
