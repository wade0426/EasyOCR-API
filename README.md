# EasyOCR Vision API（OpenAI Chat Completions 相容）

使用 **FastAPI + EasyOCR** 建立的本地 OCR 服務，提供與 OpenAI `POST /v1/chat/completions` 相容的介面（非 Streaming）。

你可以用 OpenAI SDK（指定 `base_url`）或任何相容客戶端，傳入含 `data:image/...;base64,...` 的請求，服務會執行 OCR，並回傳 OpenAI 風格的回應格式。

---

## 1) 功能總覽

- 提供相容端點：`POST /v1/chat/completions`
- 接受 `messages[].content` 的 Vision 格式（`text` + `image_url`）
- `image_url.url` 僅支援 **Data URL（base64）**
- 內部使用 EasyOCR（本地模型、`download_enabled=False`）
- 回傳 OpenAI Chat Completions 風格 JSON
- 提供健康檢查：`GET /health`

---

## 2) 專案結構

```text
.
├── main.py
├── config.py
├── requirements.txt
├── routers/
│   └── chat.py
├── schemas/
│   └── openai_compat.py
├── services/
│   └── ocr_service.py
└── utils/
    └── image_utils.py
```

---

## 3) 環境需求

- Python 3.10+
- 需安裝相依套件（含 `easyocr`，會依賴 `torch`）
- 已準備好的 EasyOCR 本地模型目錄（本專案不會自動下載模型）

> 注意：`easyocr` + `torch` 安裝體積較大，請預留磁碟空間與安裝時間。

---

## 4) 安裝與啟動

### 4.1 安裝套件

```bash
pip install -r requirements.txt
```

### 4.2 設定 `.env`（可選，但建議）

你可以在專案根目錄建立 `.env` 來覆蓋預設值。

```env
MODEL_STORAGE_DIR=C:\Users\ramune\Desktop\LocalModels\easyocr
OCR_LANGUAGES=["ch_tra","en"]
USE_GPU=false
HOST=0.0.0.0
PORT=8080
LOG_LEVEL=info
```

### 4.3 啟動服務

```bash
python main.py
```

啟動時會做兩件事：

1. 驗證 `MODEL_STORAGE_DIR` 存在且為資料夾
2. 預先載入 EasyOCR Reader

任一步驟失敗，服務會直接終止（fail fast）。

---

## 5) API 使用方式

### 5.1 健康檢查

- `GET /health`

成功回應範例：

```json
{
  "status": "ok",
  "model_loaded": true
}
```

---

### 5.2 OCR 端點（OpenAI Chat Completions 相容）

- `POST /v1/chat/completions`
- `Content-Type: application/json`

#### Request 範例

```json
{
  "model": "easyocr-local",
  "stream": false,
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "請讀取這張圖的文字"
        },
        {
          "type": "image_url",
          "image_url": {
            "url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA..."
          }
        }
      ]
    }
  ]
}
```

#### Response 範例

```json
{
  "id": "chatcmpl-a1b2c3d4",
  "object": "chat.completion",
  "created": 1760000000,
  "model": "easyocr-local",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "第一行文字\n第二行文字"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 0,
    "completion_tokens": 4,
    "total_tokens": 4
  }
}
```

> `usage.completion_tokens` 目前使用 `len(ocr_text.split())` 估算，屬近似值，不是模型 tokenizer 的精確 token 數。

---

## 6) 請求限制（目前行為）

1. **不支援 Streaming**
   - `stream: true` 會回傳 400
   - 訊息：`Streaming is not supported by this server.`

2. **單請求多圖時只取第一張**
   - 會遍歷 `messages`，找到第一個 `type="image_url"` 後即處理

3. **僅支援 Data URL 圖片輸入**
   - `image_url.url` 必須以 `data:` 開頭

4. **支援 MIME 類型**
   - `image/png`
   - `image/jpeg`
   - `image/jpg`
   - `image/webp`
   - `image/bmp`

5. **不含 API Key/Auth 機制**

6. **`model` 不做路由用途**
   - 目前僅在回應中原樣回填 `response.model`

---

## 7) 錯誤格式與狀態碼

業務層錯誤採 OpenAI 風格：

```json
{
  "error": {
    "message": "<錯誤訊息>",
    "type": "invalid_request_error",
    "code": null
  }
}
```

常見情境：

- 400
  - `stream=true`
  - 沒有找到圖片
  - Data URL 格式錯誤
  - base64 無法解碼
  - MIME 不支援
  - 圖片無法解碼

- 500
  - OCR 處理失敗（例如 Reader 異常）

- 422（FastAPI/Pydantic 驗證）
  - `messages` 空陣列
  - 欄位型別不符、必要欄位缺失

---

## 8) 與 OpenAI SDK 搭配使用

```python
from openai import OpenAI

client = OpenAI(
    api_key="dummy-key",
    base_url="http://127.0.0.1:8080/v1",
)

resp = client.chat.completions.create(
    model="easyocr-local",
    stream=False,
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Read this image"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/png;base64,<YOUR_BASE64>"
                    },
                },
            ],
        }
    ],
)

print(resp.choices[0].message.content)
```

---

## 9) 快速除錯清單

1. 啟動即退出
   - 檢查 `MODEL_STORAGE_DIR` 是否正確、存在且可讀
   - 檢查是否已安裝 `easyocr` 與其依賴

2. 回傳 `No image found in the request.`
   - 確認 `messages[].content` 中包含 `{"type":"image_url", ...}`

3. 回傳 `Only data URL (base64) images are supported.`
   - 確認 `image_url.url` 以 `data:` 開頭

4. 回傳 `Invalid base64 image data.`
   - 確認 Base64 字串完整且未被換行/截斷

5. 回傳 `Unsupported image format: ...`
   - MIME 必須在支援清單內（png/jpeg/jpg/webp/bmp）

---

## 10) 開發驗證

```bash
python -m compileall .
```

若你要接到前端或其他服務，建議再補上：

- request body size 限制
- 影像尺寸上限
- 併發控制（避免 OCR 過載）
- API 金鑰驗證與稽核日誌
