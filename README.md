# 本地语音识别系统 · Local Speech-to-Text

基于 **B/S 架构** 的本地语音识别 Web 系统：浏览器麦克风录音或上传音频文件，由**本机运行的开源 AI 模型**（faster-whisper / Whisper `small`，CTranslate2 int8 量化）将中文普通话或英语转写为文字，并在页面上实时展示识别状态、音频时长与处理耗时。

- **完全本地推理**：不调用任何云端 ASR API，不使用浏览器 Web Speech API；模型准备完成后可在断网状态下识别。
- **CPU 可用**：默认 `int8` + `cpu`，无需 NVIDIA GPU / CUDA。
- **单条安装命令**：`python -m pip install -r requirements.txt`，无 Docker、Node.js、数据库等外部依赖。
- **必做功能 F1–F5 全部实现**，并实现加分项 **F6 实时语音识别**（WebSocket + 音频分片）。
- **模型可热切换**：页面内选择 `tiny / base / small / medium / large-v3-turbo`，带下载进度，无需重启服务。
- **结果可播放、可导出**：识别结果附带分段与时间戳，音频播放器可点击跳转，支持导出 TXT / SRT / VTT。
- **稳健性**：每请求推理超时（504）、可配置备用模型降级、统一错误体带 `stage` / `retryable`。

---

## 1. 环境要求

| 项目 | 要求 |
| --- | --- |
| 操作系统 | Windows 11 x64（开发/验证环境），macOS / Linux 同样适用 |
| Python | 3.12（已验证 3.12.13；3.10+ 应可用） |
| 浏览器 | Chrome / Edge 最新稳定版（麦克风需 `http://127.0.0.1` 或 `localhost` 访问） |
| 硬件 | 普通 CPU 即可（验证机：AMD Ryzen 5 5600U，12 线程，16 GB 内存） |
| 磁盘 | 约 1 GB（Python 依赖约 350 MB + 模型权重约 480 MB） |
| 网络 | **仅首次下载模型权重时需要**，之后可完全离线 |

## 2. 快速开始

```powershell
# 1) 安装依赖
python -m pip install -r requirements.txt

# 2) 首次准备模型权重（只需执行一次，约 480 MB）
python scripts/download_model.py

# 3) 启动服务
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

浏览器打开 <http://127.0.0.1:8000>，首次点击「开始录音」时允许麦克风权限即可。

> 也可以跳过第 2 步直接启动：模型会在第一次识别请求时自动下载（首次请求会明显变慢），
> 下载完成后不再需要网络。

## 3. 模型初始化与离线运行

### 3.1 下载权重

```powershell
python scripts/download_model.py            # 自动选择可用的下载源
python scripts/download_model.py --check    # 顺便跑一次中文识别自检
python scripts/download_model.py --mirror hf-mirror   # 网络受限时使用镜像
```

权重缓存在 `models/`（已加入 `.gitignore`，不会提交到 Git）。
脚本先探测 `huggingface.co`，不可达时自动切换到 `hf-mirror.com`。

### 3.2 确认离线可识别

```powershell
# 该脚本会显式禁用进程内的 socket 连接，再加载模型并识别中英文样本
python scripts/offline_check.py
```

输出示例（真实运行结果）：

```
Network disabled (socket.connect / getaddrinfo blocked).
Model loaded offline in 1.71s
[zh] chinese.wav (10.07s audio) -> 3.21s, 33 chars: 你好,欢迎使用本地语音识别系统。今天天气很好,我们一起出去散步吧。
[en] english.wav (8.34s audio) -> 3.54s, 108 chars: Hello and welcome to the local speech recognition system. The weather is nice today, so let's go for a walk.
Offline recognition verified for Chinese and English.
```

生产/演示时还可以设置 `STT_OFFLINE=1`（对应 `faster-whisper` 的 `local_files_only`），
让服务完全不去访问网络检查模型版本：

```powershell
$env:STT_OFFLINE = "1"
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

## 4. 使用说明

### 4.1 选择识别模型（可选）

「识别模型」卡片列出本地支持的档位（`tiny / base / small / medium / large-v3-turbo / large-v3`），
显示参数量、下载体积、是否已缓存以及推荐档位。选择后点击 **应用并加载**：

- 已下载的档位：立即切换（下一次识别生效），无需重启服务；
- 未下载的档位：自动下载并加载，进度条与已下载/总大小实时显示；
- 选择结果持久化在 `models/state.json`，重启后仍然生效。

准确率与速度的折中：`small` 是默认；想明显提升中文效果可用 `large-v3-turbo`（CPU 上约慢 2–3 倍），
低配机器可用 `base`。

### 4.2 麦克风录音（F1）

1. 选择识别语言（中文普通话 / English / 自动检测）。
2. 点击 **开始录音**，页面显示录音状态、计时器与实时音量条。
3. 点击 **结束录音** 后自动上传并识别，结果、音频时长、处理耗时显示在「识别结果」卡片中。
4. 单次录音最长 60 秒：计时到 60 秒会自动停止并立即识别。
5. 识别完成后可直接 **复制文本**，并可以再次录音。

### 4.3 上传音频（F2）

- 支持格式：`WAV / WebM / OGG / OPUS / MP3 / M4A / FLAC`（覆盖 WAV 必做格式与浏览器录音的 WebM/Opus）。
- 支持拖拽或点击选择，单文件上限 10 MB，与麦克风录音共用同一套后端识别逻辑。
- 校验失败（空文件、损坏文件、超限、格式不支持）会在页面给出明确提示。

### 4.4 内置样本（无需麦克风）

页面「上传音频文件」卡片下有 **中文样本 / English sample** 两个按钮，直接点击即可体验完整识别链路，
样本文件位于 `samples/`。

### 4.5 实时识别（F6，加分项）

切换到「实时识别」，点击 **开始实时识别** 后持续说话：

- 前端按约 2.5 秒切片，编码为 16 kHz 单声道 WAV，通过 WebSocket `/ws/stream` 发送；
- 后端对每个切片独立推理，把增量文本回推，页面边说话边更新；
- 点击 **停止并输出结果**：先返回分片拼接结果，前端随即把整段音频重发一次（`refine`），
  后端对整段音频复核，得到与一次性识别同样干净的最终文本；
- 采用「一次只发一片」的流控，CPU 较慢时不会堆积无界任务队列。

### 4.6 结果播放、分段与导出

识别完成后：

- 结果区出现音频播放器（麦克风录音与上传文件均可用），点击分段列表左侧时间即可跳转播放；
- 播放时当前分段会高亮；
- **导出 TXT / SRT / VTT** 使用服务端格式化接口，不重复推理；
- 勾选「词级时间戳」可让接口返回每个词的时间（约慢 2 倍，适合精细字幕）。

## 5. 接口

### 5.1 `GET /health`

```bash
curl -s http://127.0.0.1:8000/health
```

```json
{ "status": "ok" }
```

### 5.2 `POST /api/transcribe`

`multipart/form-data`，字段 `file`（音频）与 `language`（`zh` / `en`，额外支持 `auto`），
可选 `word_timestamps=true` 请求词级时间戳。

```bash
curl -s -X POST http://127.0.0.1:8000/api/transcribe ^
     -F "file=@samples/chinese.wav" ^
     -F "language=zh"
```

```json
{
  "text": "你好,欢迎使用本地语音识别系统。今天天气很好,我们一起出去散步吧。",
  "language": "zh",
  "duration_seconds": 10.071,
  "processing_seconds": 3.913,
  "detected_language": "zh",
  "language_probability": 1.0,
  "segments": [
    {
      "start": 0.0,
      "end": 4.1,
      "text": "你好,欢迎使用本地语音识别系统。",
      "words": [{ "word": "你好,", "start": 0.0, "end": 0.5, "probability": 0.92 }]
    }
  ],
  "model": "faster-whisper/small",
  "warning": null,
  "truncated": false
}
```

- `text`：识别文本（`language=auto` 时为检测到的语言输出）。
- `language`：本次请求选择的语言；`auto` 模式下返回检测结果。
- `duration_seconds`：音频实际时长（服务端解码后计算，16 kHz 重采样）。
- `processing_seconds`：服务端处理耗时（解码 + 推理），不含首次模型下载。
- `segments[].words`：词级时间戳（仅当开启 `word_timestamps` 时非空）。
- `warning`：`degraded_model` / `deadline_exceeded` / `no_speech_detected` 的组合，正常为 `null`。
- 其余字段为增量信息，不影响必做契约。

### 5.3 其他接口

| Method | Endpoint | 说明 |
| --- | --- | --- |
| GET | `/` | Web 页面 |
| GET | `/api/config` | 前端所需的限制与模型信息 |
| GET | `/api/model` | 当前模型状态（是否已加载、设备、是否已下载） |
| GET | `/api/models` | 模型目录（档位、体积、许可、是否已缓存） |
| POST | `/api/models/select` | 选择并持久化模型（`{"model":"large-v3-turbo"}`） |
| GET | `/api/models/{id}/download` | 下载/预热模型并以 SSE 推送进度 |
| POST | `/api/subtitles` | 用分段生成 SRT / VTT / TXT（不重复推理） |
| WS | `/ws/stream` | 实时识别（加分项） |
| GET | `/docs` | 自动生成的 OpenAPI 文档 |

WebSocket 协议（详见 `app/streaming.py` 文档字符串）：

```
客户端 → {"type":"start","language":"zh"}   <binary WAV chunk>...   {"type":"stop"}
服务端 → {"type":"ready",...}   {"type":"partial","text":...}   {"type":"final","text":...,"refined":false}

# 可选：停止后把整段音频再发一次，得到整段复核结果
客户端 → {"type":"refine"}   <binary 完整 WAV>
服务端 → {"type":"final","text":...,"refined":true}
```

### 5.4 错误格式

所有 4xx / 5xx 都返回统一 JSON：

```json
{
  "code": "unsupported_format",
  "detail": "Unsupported audio format '.txt'. Supported formats: ...",
  "stage": "request",
  "retryable": false
}
```

`stage` 取值 `request / decode / model / inference / server`；`retryable` 表示客户端是否可以原样重试。

| HTTP | `code` | 触发场景 |
| --- | --- | --- |
| 400 | `bad_request` | 空文件（0 字节） |
| 413 | `payload_too_large` | 文件超过 10 MB / 音频超过服务端时长上限 |
| 415 | `unsupported_format` | 扩展名不在支持列表（如 `.txt`、`.pdf`） |
| 422 | `invalid_language` | `language` 不是 `zh`/`en`/`auto` |
| 422 | `validation_error` | 缺少 `file` 字段等请求格式错误 |
| 422 | `audio_decode_error` | 文件损坏、截断、无音频流或时长过短 |
| 500 | `transcription_failed` | 推理失败 |
| 500 | `internal_error` | 未预期的服务端异常 |
| 503 | `model_unavailable` | 模型加载失败（权重缺失、后端不可用） |
| 504 | `timeout` | 超过本请求的推理时限（可重试或换更小模型） |

## 6. 测试

```powershell
# 常规自动化测试：Mock 模型，无需下载模型、无需麦克风，约 3 秒
python -m pytest -q

# 真实本地模型验证（中英文样本 + 端到端 HTTP）
python -m pytest tests/test_real_model.py --run-real-model -q

# 浏览器端到端验证（可选，需要 playwright 与 chromium）
python -m pip install -r requirements-dev.txt
python -m playwright install chromium
python scripts/e2e_browser_check.py
```

测试分层：

| 类型 | 位置 | 说明 |
| --- | --- | --- |
| HTTP API | `tests/test_health.py`、`tests/test_transcribe_api.py` | 健康检查、正常/非法请求、响应字段、重复识别 |
| 音频处理 | `tests/test_audio.py` | WAV/WebM/Opus 解码、重采样、空/损坏文件、时长上限 |
| 语言参数 | `tests/test_language.py` | 中英文、别名、非法语言、`auto` |
| 模型集成 | `tests/test_model_integration.py` | 懒加载、参数传递、结果映射、异常翻译、并发串行化 |
| 模型管理 | `tests/test_models.py` | 目录元数据、下载缓存识别、选择持久化、SSE 下载流、失败事件 |
| 字幕格式 | `tests/test_subtitles.py` | SRT/VTT 时间码与编号、空分段、格式校验、附件下载 |
| 超时/降级 | `tests/test_timeout_fallback.py` | 504、部分结果、备用模型降级、错误 `stage`/`retryable` |
| 错误处理 | `tests/test_errors.py` | 415/413/422/500/503 与错误体一致性 |
| 前后端流程 | `tests/test_frontend_flow.py`、`tests/test_streaming.py` | 页面契约、上传流程、WebSocket 分片识别 |
| 样本校验 | `tests/test_samples.py` | 样本存在、5–15 秒、16 kHz 单声道、参考文本 |
| 真实模型 | `tests/test_real_model.py`（`--run-real-model`） | 用真实权重识别中英文并与人工参考文本计算 CER/WER |

当前结果（详见随源码包单独提交的 `test-report.md`）：

```
102 passed, 5 skipped in 2.54s          # python -m pytest -q（旧版记录）
131 passed, 5 skipped in 3.32s          # 当前版本（新增模型/字幕/超时用例）
5 passed in 35.04s                      # --run-real-model
```
### 测试音频

| 文件 | 时长 | 内容 |
| --- | --- | --- |
| `samples/chinese.wav` | 10.07 s | 你好，欢迎使用本地语音识别系统。今天天气很好，我们一起出去散步吧。 |
| `samples/english.wav` | 8.34 s | Hello and welcome to the local speech recognition system. The weather is nice today, so let's go for a walk. |

两份样本由 `scripts/prepare_samples.py` 通过 Windows 内置语音合成（zh-CN Huihui / en-US Zira）在本地生成，
为 16 kHz 单声道 16-bit PCM，不含任何个人信息，可随仓库分发；人工参考文本见 `samples/reference.txt`。
如需替换为真人录音，请保持 16 kHz（或任意采样率，服务端会重采样）单声道 WAV 与 5–15 秒时长。

## 7. 配置项

全部通过环境变量覆盖（详见 `app/config.py`）：

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `STT_MODEL_SIZE` | `small` | 模型档位或本地路径（`tiny`/`base`/`small`/`medium`…） |
| `STT_MODEL_DIR` | `./models` | 模型缓存目录 |
| `STT_DEVICE` | `cpu` | `cpu` 或 `cuda`（GPU 可选） |
| `STT_COMPUTE_TYPE` | `int8` | `int8` / `float32` / `float16`（后者需 GPU） |
| `STT_BEAM_SIZE` | `5` | 解码 beam 大小（越大越准、越慢） |
| `STT_PRELOAD` | `1` | 权重已缓存时在启动阶段后台预加载模型，首次请求不再等待加载 |
| `STT_WORD_TIMESTAMPS` | `0` | 是否默认返回词级时间戳（约慢 2 倍，也可在页面勾选） |
| `STT_REQUEST_TIMEOUT_SECONDS` | `0` | 单请求推理时限；`0` 表示按音频时长自动计算（4×时长+15 秒，15–600 秒） |
| `STT_FALLBACK_MODEL` | 空 | 推理失败时自动降级一次的备用档位（如 `base`），需该模型已缓存或允许下载 |
| `STT_OFFLINE` | `0` | 设为 `1` 时只使用本地缓存，不访问网络 |
| `STT_MAX_UPLOAD_BYTES` | `10485760` | 上传上限（10 MB） |
| `STT_MAX_AUDIO_SECONDS` | `120` | 服务端允许的最长音频（上传） |
| `STT_MAX_RECORDING_SECONDS` | `60` | 前端录音上限（下发给浏览器） |
| `STT_STREAMING_ENABLED` | `1` | 是否启用 WebSocket 实时识别 |
| `STT_STREAMING_CHUNK_SECONDS` | `2.5` | 实时切片长度 |
| `STT_STREAMING_MAX_SECONDS` | `60` | 实时识别单次最长时长 |
| `STT_DEFAULT_LANGUAGE` | `zh` | 未传 `language` 时的默认语言 |

## 8. 项目结构

```
app/
  main.py            FastAPI 应用工厂、路由（/health、/api/transcribe、/api/models、/api/subtitles、/ws/stream）
  config.py          环境变量配置（Settings）
  audio.py           格式嗅探、上传校验、PyAV 解码 + 16 kHz 重采样、WAV 编码
  models.py          模型目录、缓存识别、选择持久化、SSE 下载进度
  subtitles.py       SRT / VTT / TXT 格式化
  serializers.py     引擎结果 → API 负载（REST 与 WebSocket 共用）
  asr/
    base.py          ASREngine 协议、TranscriptionResult（含词级时间戳）
    whisper_engine.py faster-whisper 懒加载、模型热切换、加锁推理、异常翻译
  streaming.py       WebSocket 实时识别会话
  errors.py          统一错误类型（code/stage/retryable）与 JSON 错误处理
  schemas.py         Pydantic 响应模型
static/
  index.html         单页界面
  css/app.css        样式（无构建步骤）
  js/app.js          页面状态机、模型切换、播放器/分段/导出
  js/recorder.js     MediaRecorder 录音、音量条、60 秒上限
  js/streaming.js    实时识别（AudioWorklet + WebSocket 流控）
  js/wav.js          浏览器侧 WAV 编码与重采样
  js/pcm-worklet.js  AudioWorklet PCM 采集
  js/api.js          后端 API 封装（含模型 SSE 流）
  js/messages.js     中文文案与错误映射
tests/               pytest 自动化测试（Mock 与真实模型分层）
samples/             chinese.wav / english.wav / reference.txt
scripts/             download_model.py / offline_check.py / prepare_samples.py / e2e_browser_check.py
AGENTS.md            AI 编码代理的协作规范与不可违反约束
```

> 文档（`requirements.md`、`design.md`、`test-report.md`、`ai-development.md`）随源码包单独提交，
> 不包含在本公开仓库中。

## 9. 设计要点

### 9.1 为什么选 faster-whisper + Whisper small

- **中英文双语**：Whisper 多语种模型原生支持 zh/en，可显式指定语言，也支持自动检测（加分项）。
- **CPU 友好**：CTranslate2 int8 量化后 `small` 在普通笔记本 CPU 上约 **2.5–3× 实时**（10 秒音频约 3.5–4 秒）。
- **纯 pip 依赖**：`ctranslate2`、`av`、`tokenizers` 均提供 Windows 预编译 wheel，无需自行编译或安装 ffmpeg。
- **体积适中**：`small` 约 480 MB，`base`/`tiny` 更小，可用 `STT_MODEL_SIZE` 一键切换。

选型对比（详见随源码包提交的 `design.md`）：whisper.cpp 需要自行编译或依赖第三方二进制；
sherpa-onnx/Paraformer 中文强但英文与自动语种识别弱；Vosk 体积小但准确率明显不足。

### 9.2 音频处理

浏览器 `MediaRecorder` 产出 WebM/Opus，上传可能是 WAV/MP3/M4A/FLAC。统一走 PyAV（wheel 内置 FFmpeg）
解码为 16 kHz 单声道 `float32`，再由 faster-whisper 推理。这样：

- 无需外部 ffmpeg 可执行文件（满足「不能有需自行编译的特殊原生依赖」）；
- 麦克风录音与文件上传共用同一条识别管线；
- 纯静音片段直接短路为「未检测到语音」，避免 Whisper 在静音上产生幻觉文本。

### 9.3 模型管理与热切换

`app/models.py` 维护一份精选模型目录（参数量、下载体积、语言、许可），并检查本地 HuggingFace
缓存判断哪些档位已经下载。选择结果写入 `models/state.json`，`ModelManager` 在应用启动时把它应用到
`WhisperEngine`，因此切换模型不需要重启服务；下载/预热通过 SSE 推送进度（`GET /api/models/{id}/download`），
前端据此显示百分比。

### 9.4 稳定性

- 模型**懒加载**：导入应用、运行 pytest、访问页面都不会触发下载。
- CTranslate2 模型非线程安全：加载与推理分别加锁，多请求自动串行，避免崩溃。
- **每请求时限**：`deadline = 4 × 音频时长 + 15 s`（可配置），解码在分段边界检查时限；
  超时返回 504 或带 `deadline_exceeded` 的部分结果，避免页面长时间无反馈。
- **降级阶梯**：配置 `STT_FALLBACK_MODEL` 后，主模型推理失败会自动改用备用档位一次，
  响应以 `degraded_model` 标注，并恢复主模型供后续请求使用。
- 所有失败都转换为带 `code` / `stage` / `retryable` 的 JSON 错误，页面给出中文可读提示并可重试。

## 10. 已知限制

- `small` 模型在强噪声、口音较重或远场录音下仍可能出错；中文标点为半角（`,`/`。`），页面展示为模型原始输出。
- 实时识别是「分片识别 + 拼接 + 停止后整段复核」，首字延迟约 6–7 秒；跨切片的词可能被切断。
- 词级时间戳（`word_timestamps`）会额外做一次对齐，推理时间约增加到 2–2.5 倍，默认关闭。
- 备用模型降级要求该档位已缓存或允许联网下载，否则仍返回原始错误。
- 服务端同一时刻只处理一个推理请求（CPU 场景下的取舍），并发请求会排队。
- 仅支持 Chrome / Edge 最新稳定版；Safari / Firefox 未做适配。

## 11. 文档

本仓库只包含源码、测试、样本与脚本。以下文档随源码包单独提交（不进入公开仓库）：

| 文件 | 内容 |
| --- | --- |
| `requirements.md` | 需求理解、功能拆分、假设与边界条件 |
| `design.md` | 架构、模块、音频处理流程、ASR 模型选型 |
| `test-report.md` | 测试用例、真实运行结果、真实模型验证与已知问题 |
| `ai-development.md` | AI 辅助开发过程记录（三阶段与问题定位实录） |

## 12. 许可

代码以 [MIT](LICENSE) 许可发布。Whisper 模型权重来自 OpenAI（MIT），
经 [Systran/faster-whisper-small](https://huggingface.co/Systran/faster-whisper-small) 转换为 CTranslate2 格式；
模型权重不随仓库分发。
