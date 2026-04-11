# EasyOCR Vision API — 開發計畫書

---

## 一、專案概覽

### 目標

建立一個以 **FastAPI + uvicorn** 為基礎的本地 HTTP 服務，對外暴露與 **OpenAI Chat Completions（Vision）** 相容的端點。呼叫方可透過標準 OpenAI SDK 或任何相容客戶端，以 Data URL（`base64`）方式傳入圖片，服務內部使用 **EasyOCR** 執行文字辨識，並將結果包裝成 OpenAI Response 格式回傳。

### 已確認的本地模型

| 檔名 | 用途 | 對應語言代碼 |
|------|------|-------------|
| `craft_mlt_25k.pth` | 文字區域偵測（CRAFT Detector） | *(通用)* |
| `chinese.pth` | 文字辨識（Recognition） | `ch_tra`（繁體中文）|

### 不支援清單（硬性限制，不得實作）

| 功能 | 說明 |
|------|------|
| `stream: true` | 不支援 Streaming |
| 單請求多圖 | 僅處理第一張圖片 |
| API Key 認證 | 無任何 Auth 機制 |
| `model` 欄位路由 | 忽略 model 值，永遠使用 EasyOCR |

---

## 二、專案結構

```
easyocr-vision-api/
├── main.py                  # FastAPI App 建立與 lifespan 管理
├── config.py                # 所有環境設定（Pydantic BaseSettings）
├── requirements.txt         # 相依套件清單
│
├── schemas/
│   └── openai_compat.py     # Request / Response Pydantic 模型
│
├── services/
│   └── ocr_service.py       # EasyOCR 封裝、Singleton 管理
│
├── routers/
│   └── chat.py              # POST /v1/chat/completions 路由
│
└── utils/
    └── image_utils.py       # Data URL 解析、base64 解碼、MIME 驗證
```

---

## 三、開發階段（Phases）

---

### Phase 1 — 環境設定（`config.py`）

**目標**: 集中管理所有可調整參數，避免 magic string 散落各模組。

**`config.py` 須包含的設定欄位**:

| 欄位名稱 | 型別 | 預設值 / 說明 |
|----------|------|--------------|
| `MODEL_STORAGE_DIR` | `str` | `C:\Users\ramune\Desktop\LocalModels\easyocr` |
| `OCR_LANGUAGES` | `List[str]` | `["ch_tra", "en"]` |
| `USE_GPU` | `bool` | `False`（預設 CPU，可改 True）|
| `HOST` | `str` | `"0.0.0.0"` |
| `PORT` | `int` | `8000` |
| `LOG_LEVEL` | `str` | `"info"` |

**實作規範**:
- 使用 **`pydantic-settings`** 的 `BaseSettings`，支援從 `.env` 檔覆寫。
- 在模組層級建立一個 `settings = Settings()` singleton instance，其他模組直接 `from config import settings` 取用。
- `MODEL_STORAGE_DIR` 應使用 `pathlib.Path` 做型別轉換，並在啟動時驗證路徑存在（若不存在，`raise ValueError` 並說明原因）。

---

### Phase 2 — Pydantic Schema 定義（`schemas/openai_compat.py`）

**目標**: 定義符合 OpenAI Vision API 規格的 Request / Response 資料結構，提供嚴格型別驗證。

#### 2-A. Request Schema

OpenAI Vision payload 的 `messages[].content` 可以是純字串，也可以是物件陣列（含 `text` 和 `image_url` 兩種 type）。

**須定義的型別（由內而外）**:

```
ImageUrlContent
  └─ url: str              # 僅接受 data:<mime>;base64,<data> 格式

ContentPartText
  └─ type: Literal["text"]
     text: str

ContentPartImage
  └─ type: Literal["image_url"]
     image_url: ImageUrlContent

ContentPart = Union[ContentPartText, ContentPartImage]

Message
  └─ role: Literal["user", "assistant", "system"]
     content: Union[str, List[ContentPart]]

ChatCompletionRequest
  └─ model: str
     messages: List[Message]
     max_tokens: Optional[int] = None
     temperature: Optional[float] = None
     stream: Optional[bool] = False   # 接收但必須驗證為 False
```

**驗證規則**:
- 若 `stream == True`：回傳 HTTP 400，錯誤訊息：`"Streaming is not supported by this server."`
- `messages` 不得為空陣列：回傳 HTTP 422。
- `image_url.url` 必須以 `data:` 開頭；否則回傳 HTTP 400，錯誤訊息：`"Only data URL (base64) images are supported."`

#### 2-B. Response Schema

依照 OpenAI Chat Completions Response 格式：

```
ChatCompletionChoice
  └─ index: int
     message: AssistantMessage (role="assistant", content=str)
     finish_reason: Literal["stop"]

UsageInfo
  └─ prompt_tokens: int = 0
     completion_tokens: int
     total_tokens: int

ChatCompletionResponse
  └─ id: str                  # 格式：f"chatcmpl-{uuid4().hex[:8]}"
     object: Literal["chat.completion"]
     created: int             # int(time.time())
     model: str               # 回傳 request 中的 model 欄位
     choices: List[ChatCompletionChoice]
     usage: UsageInfo
```

---

### Phase 3 — 圖片工具函式（`utils/image_utils.py`）

**目標**: 將 Data URL 解析成可供 EasyOCR 處理的格式，並集中處理錯誤。

**須實作的函式**（介面定義，不含實作程式碼）:

#### `parse_data_url(data_url: str) -> tuple[str, bytes]`

- **輸入**: `"data:image/png;base64,iVBOR..."`
- **輸出**: `(mime_type, raw_bytes)`，例如 `("image/png", b'\x89PNG...')`
- **驗證規則**:
  - 格式必須符合 `data:<mime>;base64,<data>`，否則 raise `ValueError`。
  - `<data>` 必須可被 `base64.b64decode` 解碼，否則 raise `ValueError`。
- **支援的 MIME 類型**: `image/png`, `image/jpeg`, `image/jpg`, `image/webp`, `image/bmp`。不在清單內者，raise `ValueError("Unsupported image format: ...")`。

#### `bytes_to_numpy(raw_bytes: bytes) -> np.ndarray`

- 使用 `PIL.Image.open(io.BytesIO(raw_bytes))` 讀取影像。
- 轉換為 RGB 模式（確保不是 RGBA / P 等 EasyOCR 不支援的格式）。
- 轉換成 `numpy.ndarray` (uint8) 後回傳。
- 若 PIL 無法讀取，raise `ValueError("Cannot decode image data.")`。

#### `extract_first_image_url(messages: List[Message]) -> str | None`

- 遍歷 `messages`，找到第一個 `content` 中 `type == "image_url"` 的 `ContentPartImage`。
- 回傳其 `image_url.url`；若無任何圖片，回傳 `None`。

---

### Phase 4 — OCR 服務封裝（`services/ocr_service.py`）

**目標**: 將 EasyOCR Reader 封裝為 Singleton，避免每次請求都重新載入模型（模型載入約需 5-30 秒）。

#### `OCRService` Class 介面

**初始化參數**（由 `config.py` 注入）:
- `model_storage_directory: str` — 指向本地模型資料夾
- `languages: List[str]` — 語言列表，預設 `["ch_tra", "en"]`
- `gpu: bool` — 是否使用 GPU

**須實作的方法**（介面定義）:

```
OCRService.load() -> None
```
- 建立 `easyocr.Reader(lang_list, model_storage_directory, gpu, download_enabled=False)`
- `download_enabled=False`：禁止自動下載，確保只用本地模型。
- 若 Reader 建立失敗（模型檔案不存在等），raise `RuntimeError` 並記錄清楚的錯誤訊息。
- 標記 `self._loaded = True`。

```
OCRService.recognize(image: np.ndarray) -> str
```
- 呼叫 `self.reader.readtext(image, detail=0)`。
- `detail=0` 回傳純文字 list，將結果以換行符號 `"\n"` join 成單一字串後回傳。
- 若結果為空 list，回傳空字串 `""`（不拋出例外）。

**Singleton 管理**:
- 在模組層級宣告 `_ocr_service: OCRService | None = None`。
- 提供 `get_ocr_service() -> OCRService` 函式，若尚未建立則初始化，否則直接回傳現有 instance。

---

### Phase 5 — 路由實作（`routers/chat.py`）

**目標**: 實作 `POST /v1/chat/completions` 端點，串接 Schema 驗證 → 圖片解析 → OCR → 回應組裝。

#### 端點處理流程（Pseudocode，非實作程式碼）

```
POST /v1/chat/completions
│
├─ [Schema] 驗證 ChatCompletionRequest
│   └─ stream == True → 回傳 400
│
├─ [Utils] extract_first_image_url(request.messages)
│   └─ 無圖片 → 回傳 400（"No image found in the request."）
│
├─ [Utils] parse_data_url(image_url)
│   └─ 解析失敗 → 回傳 400（附詳細錯誤訊息）
│
├─ [Utils] bytes_to_numpy(raw_bytes)
│   └─ 解碼失敗 → 回傳 400（附詳細錯誤訊息）
│
├─ [Service] ocr_service.recognize(numpy_image)
│   └─ 例外 → 回傳 500（"OCR processing failed."）
│
└─ [Schema] 組裝 ChatCompletionResponse
    └─ choices[0].message.content = OCR 辨識文字
    └─ usage.completion_tokens = len(ocr_text.split())
    └─ 回傳 200 JSON
```

**錯誤回應格式**（統一包裝，符合 OpenAI 錯誤格式）:

```json
{
  "error": {
    "message": "<錯誤說明>",
    "type": "invalid_request_error",
    "code": null
  }
}
```

---

### Phase 6 — 應用程式入口（`main.py`）

**目標**: 建立 FastAPI App，管理啟動/關閉生命週期，掛載路由。

**須包含的元件**:

1. **Lifespan Context Manager** (`@asynccontextmanager`):
   - `startup`：呼叫 `ocr_service.load()`，若失敗則阻止服務啟動（`sys.exit(1)`）。記錄 log 說明模型載入成功與使用的路徑。
   - `shutdown`：記錄 log 說明服務關閉。

2. **FastAPI App 建立**:
   - `title = "EasyOCR Vision API"`
   - `description = "OpenAI Chat Completions (Vision) compatible API powered by EasyOCR"`
   - `version = "1.0.0"`
   - 掛載 lifespan。

3. **路由掛載**:
   - `app.include_router(chat_router, prefix="/v1")`

4. **健康檢查端點**（選用但建議）:
   - `GET /health` → 回傳 `{"status": "ok", "model_loaded": bool}`

5. **uvicorn 啟動入口**（`if __name__ == "__main__":`）:
   ```python
   uvicorn.run("main:app", host=settings.HOST, port=settings.PORT, log_level=settings.LOG_LEVEL)
   ```

---

### Phase 7 — 相依套件（`requirements.txt`）

```
fastapi>=0.111.0
uvicorn[standard]>=0.29.0
pydantic>=2.0.0
pydantic-settings>=2.0.0
easyocr>=1.7.0
Pillow>=10.0.0
numpy>=1.24.0
python-multipart>=0.0.9
```

> **注意**: `easyocr` 相依 `torch`，首次安裝體積較大（約 2-4 GB）。若環境已有 PyTorch，請先確認版本相容性再安裝。

---

## 四、邊界條件與錯誤處理對照表

| 情境 | HTTP 狀態碼 | 處理方式 |
|------|------------|----------|
| `stream: true` | 400 | 立即拒絕，說明不支援 Streaming |
| messages 無圖片 | 400 | 說明找不到 image_url |
| Data URL 格式錯誤 | 400 | 說明格式應為 `data:<mime>;base64,...` |
| base64 解碼失敗 | 400 | 說明 base64 內容無效 |
| 不支援的 MIME 類型 | 400 | 列出支援的格式 |
| PIL 無法讀取影像 | 400 | 說明影像資料損毀 |
| EasyOCR 辨識失敗 | 500 | 記錄 traceback，回傳通用錯誤 |
| 模型檔案不存在 | 啟動失敗 | 服務啟動時即終止，印出明確路徑錯誤 |
| OCR 結果為空 | 200 | content 回傳空字串，finish_reason 仍為 "stop" |
