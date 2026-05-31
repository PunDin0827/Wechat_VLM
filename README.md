# VLM 聊天截圖解析與 Golden Sample 評測 Prototype

這是一個精簡版作品集專案，展示如何把 VLM 對聊天截圖的輸出轉成結構化資料，並用 Golden Sample 進行評測。

## 專案重點

- 使用兩階段 Prompt：先分類畫面，再依畫面類型抽取內容
- 將 VLM 輸出的 JSON 正規化為穩定 schema
- 建立人工標註格式作為 Golden Sample
- 使用軟對齊演算法處理訊息漏抓、多抓與順序錯位
- 輸出訊息層級 Precision / Recall / F1 與欄位級 Accuracy

## 流程

```text
聊天截圖
  ↓
Stage 1：判斷 app_source / screen_type / chat_type
  ↓
Stage 2：依分類結果選擇 extractor prompt
  ↓
解析 VLM JSON 輸出
  ↓
Normalize 成標準訊息格式
  ↓
與 Golden Sample 軟對齊
  ↓
輸出評測結果
```

## 檔案說明

| 檔案 | 說明 |
|---|---|
| `prompt_registry.py` | 兩階段 prompt 與 routing 邏輯 |
| `output_parser.py` | VLM JSON 輸出解析、欄位清洗與 fallback |
| `normalizer.py` | 訊息正規化、時間解析、speaker normalization |
| `golden_label.py` | Golden Sample 人工標註 schema |
| `message_aligner.py` | Needleman-Wunsch 軟對齊與文字相似度計算 |
| `eval_pipeline.py` | 混淆矩陣、訊息層級 F1、欄位級 accuracy 評測 |
| `sample_label.json` | 範例人工標註 |
| `sample_prediction.json` | 範例模型輸出 |

## 快速執行

```bash
pip install -r requirements.txt

python eval_pipeline.py ^
  --labels sample_label.json ^
  --results sample_prediction.json ^
  --output sample_eval_report.json
```

## 評測指標

### 訊息層級

- Precision：模型抽出的訊息中，有多少能對上 Golden Sample
- Recall：Golden Sample 中有多少訊息被模型成功抽出
- F1：Precision 與 Recall 的調和平均

### 欄位層級

對成功對齊或應被比對的訊息，統計：

- `side`
- `speaker_canonical`
- `content_exact`
- `content_type`
- `time_raw`

## 設計重點

這個專案不是完整產品，而是用來展示 VLM 資料抽取任務中的幾個核心工程問題：

1. 設計可維護的 prompt routing
2. 把不穩定的 VLM output 轉成穩定資料結構
3. 建立 Golden Sample
4. 避免訊息漏抓造成整批 index 對齊錯位
5. 用可重現的指標比較不同模型與 prompt

## 注意

本 repo 使用假資料與公開化 prompt，不包含公司內部截圖、真實聊天內容、內網 IP、API token 或正式系統設定。
