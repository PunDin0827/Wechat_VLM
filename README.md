# VLM Chat Screenshot Extraction Benchmark

這是一個公開展示版專案，示範如何把 **LINE / WeChat 聊天截圖**交給本地或遠端 VLM 解析，並建立一套可重現的 **Golden Sample 評估流程**。

專案重點不是單純「呼叫模型」，而是完整呈現一條可維護的 AI 工程管線：

```text
聊天截圖
  ↓
桌面版左右面板切割（可選）
  ↓
Stage 1：VLM 分類器
  ↓
Prompt Router
  ↓
Stage 2：VLM 結構化抽取器
  ↓
JSON Parser / Fallback
  ↓
Normalization Layer
  ↓
Golden Sample Evaluation
  ↓
訊息層級 Precision / Recall / F1 + 欄位級 Accuracy + Error Report
```

## 專案亮點

### 1. Two-stage Prompting

使用兩階段 prompt 設計：

- **Stage 1 Classifier**：只判斷 `app_source`、`screen_type`、`chat_type`
- **Stage 2 Extractor**：依據 Stage 1 結果，route 到對應的抽取 prompt

這樣可以避免所有任務都塞進單一巨大 prompt，讓流程更容易維護，也更適合做 A/B test 與 benchmark。

### 2. Golden Sample 評估框架

專案提供人工標註 schema 與批次評估流程，可輸出：

- `screen_type` / `app_source` / `chat_type` 混淆矩陣
- 訊息層級 Precision / Recall / F1
- 欄位級 Accuracy，例如 `side`、`speaker`、`content_exact`、`content_type`、`time_raw`
- 每張圖片的詳細 alignment diff report

### 3. Needleman-Wunsch 軟對齊

聊天截圖抽取最常見的問題是：模型漏抓一則訊息，後面全部 index shift，導致硬比對全面扣分。

本專案使用 **Needleman-Wunsch global sequence alignment** 對齊模型輸出與人工標註，能將：

- 模型多抓的訊息標成 `extra`
- 模型漏抓的訊息標成 `missed`
- 成功配對的訊息再做欄位級評估

這讓評估結果更接近真實錯誤型態。

### 4. Normalization Layer

VLM 輸出容易不穩定，因此本專案將原始輸出轉成 typed schema：

- `NormalizedMessage`
- `NormalizedFriend`
- `content_type`
- `speaker_canonical`
- `time_parsed`
- `time_confidence`

下游不用直接依賴模型原始 JSON。

### 5. OpenCV 輔助前處理

對桌面版聊天截圖，專案使用 OpenCV + Lab 色彩空間 + 垂直持續性偵測左右面板分割線，將聊天列表與聊天內容分開處理，降低 VLM 對複合 UI 的解析負擔。

## 專案結構

```text
vlm-chat-extraction-benchmark/
├── src/
│   ├── labels/              # Golden Sample schema
│   ├── prompts/             # Stage 1 / Stage 2 prompt registry
│   ├── parsing/             # VLM JSON parser、fallback、dedup
│   ├── normalization/       # 正規化資料結構與時間解析
│   ├── evaluation/          # 軟對齊與 benchmark pipeline
│   ├── preprocessing/       # OpenCV 圖片前處理
│   └── server/              # FastAPI demo server
├── examples/                # 假資料範例，不含真實聊天資料
├── docs/                    # 流程、指標、錯誤分析說明
├── requirements.txt
├── .env.example
└── README.md
```

## 安裝

```bash
python -m venv .venv
source .venv/bin/activate  # Windows 可改用 .venv\Scripts\activate
pip install -r requirements.txt
```

## 啟動 Demo Server

請先複製 `.env.example`：

```bash
cp .env.example .env
```

設定你的 OpenAI-compatible VLM server，例如 vLLM、SGLang、llama.cpp server：

```bash
export VLM_BASE_URL="http://localhost:8000/v1"
export VLM_API_KEY="EMPTY"
export VLM_MODEL="local"
```

啟動：

```bash
uvicorn src.server.demo_server:app --host 0.0.0.0 --port 8080 --reload
```

上傳圖片：

```bash
curl -X POST "http://localhost:8080/api/upload" \
  -F "file=@examples/demo_chat.png" \
  -F "client_id=demo"
```

查詢結果：

```bash
curl http://localhost:8080/api/messages
curl http://localhost:8080/api/raw
curl http://localhost:8080/api/conversation
```

## 執行評估

範例指令：

```bash
python -m src.evaluation.eval_pipeline \
  --labels examples/labels \
  --results examples/sample_prediction.json \
  --output examples/sample_eval_report.json
```

評估輸出包含：

- Stage 1 confusion matrix
- Stage 2 message-level F1
- 欄位級 accuracy
- extra / missed 統計
- per-image alignment details

## 注意事項

這是公開展示版，已移除：

- 內網 IP
- API token
- 公司 DB API
- 真實聊天截圖
- 真實標註資料
- 公司 payload schema

若要接到正式環境，請使用 `.env` 管理設定，並避免將任何真實資料 commit 到 GitHub。

## 技術關鍵字

`VLM`、`LLM Evaluation`、`Golden Sample`、`Needleman-Wunsch Alignment`、`Prompt Routing`、`FastAPI`、`OpenCV`、`Pydantic`、`Information Extraction`、`Benchmark Pipeline`
