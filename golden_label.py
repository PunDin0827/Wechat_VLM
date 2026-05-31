"""
golden_label.py — Golden Sample 標註格式定義
=====================================
對應 output_parser / normalizer / prompt_registry 的輸出結構，
提供兩層標註：

  Stage 1  →  screen_type / app_source / chat_type
  Stage 2  →  具體內容（messages / friends / chat_entries）

使用方式：
  label = GoldenLabel.from_file("labels/20241025_153000_001.json")
  label.save("labels/20241025_153000_001.json")
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, ConfigDict, model_validator


# ============================================================
# Enum — 與 normalizer 保持同步
# ============================================================

class AppSource(str, Enum):
    LINE    = "LINE"
    WECHAT  = "WeChat"
    UNKNOWN = "unknown"

class ScreenType(str, Enum):
    CHAT_DETAIL  = "chat_detail"
    CHAT_LIST    = "chat_list"
    FRIEND_LIST  = "friend_list"
    OTHER        = "other"
    UNKNOWN      = "unknown"

class ChatType(str, Enum):
    DIRECT  = "direct"
    GROUP   = "group"
    UNKNOWN = "unknown"
    NA      = "n/a"        # 非 chat_detail 時使用

class Side(str, Enum):
    LEFT  = "left"
    RIGHT = "right"

class ContentType(str, Enum):
    TEXT     = "text"
    STICKER  = "sticker"
    IMAGE    = "image"
    AUDIO    = "audio"
    CALL     = "call"
    LOCATION = "location"
    CONTACT  = "contact"
    FILE     = "file"
    UNKNOWN  = "unknown"

class AnnotationConfidence(str, Enum):
    HIGH   = "high"    # 標註者確定
    MEDIUM = "medium"  # 有輕微疑慮
    LOW    = "low"     # 圖片模糊 / 部分截斷


# ============================================================
# Stage 1 — 分類標籤
# ============================================================

class Stage1Label(BaseModel):
    """Stage 1 分類結果的人工標準答案。"""
    model_config = ConfigDict(use_enum_values=True)

    app_source  : AppSource  = Field(..., description="來源 App")
    screen_type : ScreenType = Field(..., description="頁面類型")
    chat_type   : ChatType   = Field(
        default=ChatType.NA,
        description="對話類型；只在 screen_type=chat_detail 時有意義",
    )

    @model_validator(mode="after")
    def _fix_chat_type(self) -> Stage1Label:
        if self.screen_type != ScreenType.CHAT_DETAIL:
            self.chat_type = ChatType.NA
        return self


# ============================================================
# Stage 2 — 內容標籤（依 screen_type 分支）
# ============================================================

class MessageLabel(BaseModel):
    """單一對話泡泡的標準答案。"""
    model_config = ConfigDict(use_enum_values=True)

    seq              : int            = Field(..., ge=0, description="由上到下，0-index")
    side             : Side           = Field(..., description="left=對方 / right=發起人")
    speaker_canonical: Optional[str]  = Field(None, description="正規化後發言者；right 側固定填『發起人』")
    content          : str            = Field(..., min_length=1, description="訊息正文或標籤如 [貼圖]")
    content_type     : ContentType    = Field(default=ContentType.TEXT)
    time_raw         : Optional[str]  = Field(None, description="畫面上原始時間字串，如『下午 3:42』")

class FriendLabel(BaseModel):
    """好友 / 聯絡人列表中單筆的標準答案。"""
    model_config = ConfigDict(use_enum_values=True)

    seq                   : int           = Field(..., ge=0)
    display_name          : str           = Field(..., min_length=1)
    last_message          : Optional[str] = Field(None, description="最後一則訊息摘要")
    last_message_time_raw : Optional[str] = Field(None, description="原始時間字串")

class ChatEntryLabel(BaseModel):
    """聊天清單首頁單筆的標準答案。"""
    model_config = ConfigDict(use_enum_values=True)

    seq           : int           = Field(..., ge=0)
    contact_name  : str           = Field(..., min_length=1)
    last_message  : Optional[str] = Field(None)
    time          : Optional[str] = Field(None)
    unread_count  : Optional[int] = Field(None, ge=0)

class Stage2Label(BaseModel):
    """
    Stage 2 內容標準答案容器。
    依 screen_type 只需填寫對應欄位，其餘留空即可。
    """
    model_config = ConfigDict(use_enum_values=True)

    # chat_detail
    chat_header  : Optional[str]             = Field(None, description="標題列對話名稱")
    messages     : list[MessageLabel]        = Field(default_factory=list)

    # friend_list
    friends      : list[FriendLabel]         = Field(default_factory=list)

    # chat_list
    chat_entries : list[ChatEntryLabel]      = Field(default_factory=list)

    # other / unknown
    description  : Optional[str]            = Field(None)


# ============================================================
# 頂層 GoldenLabel
# ============================================================

class GoldenLabel(BaseModel):
    """
    單張截圖的完整人工標籤。

    欄位分組：
      [meta]    → 標註管理資訊
      [stage1]  → 分類標準答案
      [stage2]  → 內容標準答案
    """
    model_config = ConfigDict(use_enum_values=True)

    # ── meta ──────────────────────────────────────────────
    annotation_id         : str                  = Field(
        default_factory=lambda: str(uuid.uuid4())[:8],
        description="8 碼隨機 ID，用於追蹤",
    )
    source_file           : str                  = Field(..., description="截圖檔名（含路徑）")
    annotated_at          : datetime             = Field(default_factory=datetime.now)
    annotator             : str                  = Field(..., description="標註者識別碼，如 email 前綴")
    annotation_confidence : AnnotationConfidence = Field(default=AnnotationConfidence.HIGH)
    notes                 : Optional[str]        = Field(None, description="標註備註，記錄疑難雜症")

    # ── stage1 ────────────────────────────────────────────
    stage1 : Stage1Label

    # ── stage2 ────────────────────────────────────────────
    stage2 : Stage2Label = Field(default_factory=Stage2Label)

    # ── 便利方法 ──────────────────────────────────────────
    def save(self, path: str | Path) -> None:
        """序列化寫入 JSON，確保中文不被 escape。"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(
            self.model_dump_json(indent=2),
            encoding="utf-8",
        )

    @classmethod
    def from_file(cls, path: str | Path) -> GoldenLabel:
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))

    @classmethod
    def from_dict(cls, data: dict) -> GoldenLabel:
        return cls.model_validate(data)


# ============================================================
# 批次載入工具
# ============================================================

def load_all_labels(label_dir: str | Path) -> list[GoldenLabel]:
    """遞迴讀取目錄下所有 .json 標籤檔。"""
    return [
        GoldenLabel.from_file(p)
        for p in sorted(Path(label_dir).rglob("*.json"))
    ]


# ============================================================
# 範例：產生空白標籤模板（人工填寫用）
# ============================================================

def make_blank_template(source_file: str, annotator: str) -> dict:
    """
    產生一份待填寫的 JSON 模板，輸出給標註人員。
    Stage2 依 screen_type 只需保留對應陣列，其餘可刪除。
    """
    return {
        "source_file": source_file,
        "annotator": annotator,
        "annotation_confidence": "high",
        "notes": None,
        "stage1": {
            "app_source":  "LINE | WeChat | unknown",
            "screen_type": "chat_detail | chat_list | friend_list | other | unknown",
            "chat_type":   "direct | group | unknown  (非 chat_detail 留 n/a)",
        },
        "stage2": {
            "__hint__": "依 screen_type 只填對應欄位",
            "chat_header": None,
            "messages": [
                {
                    "seq": 0,
                    "side": "left | right",
                    "speaker_canonical": "發起人 (right) 或對方名稱 (left) 或 null",
                    "content": "訊息文字 或 [貼圖] [照片] 等標籤",
                    "content_type": "text | sticker | image | audio | call | location | contact | file | unknown",
                    "time_raw": "下午 3:42 或 null",
                }
            ],
            "friends": [
                {
                    "seq": 0,
                    "display_name": "好友名稱",
                    "last_message": None,
                    "last_message_time_raw": None,
                }
            ],
            "chat_entries": [],
            "description": None,
        },
    }


if __name__ == "__main__":
    # 快速示範：產生一份模板並存檔
    import sys
    file_arg = sys.argv[1] if len(sys.argv) > 1 else "sample.jpg"
    tmpl = make_blank_template(file_arg, annotator="demo_user")
    print(json.dumps(tmpl, ensure_ascii=False, indent=2))
