import re
import json
import hashlib
import unicodedata

from typing import Any
from datetime import datetime


# ============================================================
# TASK_PROMPT_V2
# ============================================================

TASK_PROMPT_V2 = """
你是通訊軟體截圖解析器，負責解析 LINE 和 WeChat（微信）的截圖。
 
═══ 第一步：判斷畫面類型 ═══
 
請從以下五類中選一個：
1. chat_detail  = 單一聊天對話視窗（可見聊天泡泡）
2. chat_list    = 聊天列表首頁（多個聊天摘要上下排列）
3. friend_list  = 好友名單 / 聯絡人列表
4. other        = 其他畫面（設定、通話、相簿等）
5. unknown      = 畫面不完整或無法判斷
 
═══ 第二步：判斷來源 App ═══
 
根據畫面風格判斷來源 App，填入 app_source：
LINE 特徵：
- 主題偏亮綠色
- 聊天泡泡較圓
- 右側通常為自己（綠色或白色）
- 標題列偏簡潔

WeChat 特徵：
- 主題偏深色或灰色
- 左右泡泡顏色差異較明顯（白 vs 綠）
- 群聊中名稱通常顯示在訊息上方
- UI 結構較密集，間距較小

若不確定，填 unknown
注意：
不同 App 的 UI 結構不同，
請先判斷 app_source，再依該 App 的常見 UI 習慣解析訊息。

═══ 第三步：依畫面類型填寫對應 JSON ═══
 
【chat_detail — 聊天對話視窗】
 
抽取規則：
1. 只抽清楚可見的聊天泡泡，由上到下排序
2. 不可補全被截斷、模糊的文字
3. 不要把標題列、系統列、輸入框、日期分隔線當成訊息
4. 左側泡泡：若畫面顯示名稱則填入 user_name，否則填 null
5. 右側泡泡：user_name 固定填「發起人」
6. 每個泡泡獨立輸出，不可合併
7. time 只填清楚可見且能對應該泡泡的時間，否則填 null
8. 貼圖/圖片/語音等非文字訊息，message 填可見描述或 null
9. 若聊天泡泡中同時包含圖片與文字，必須優先擷取可見文字。
新增欄位：
- chat_header：畫面頂部標題列顯示的對話名稱（通常是對方名字或群組名稱）
  若標題列不可見或被截斷，填 null
- chat_type：聊天型態，只能填以下三種之一：
  - "direct"  = 1對1聊天視窗
  - "group"   = 群組聊天視窗
  - "unknown" = 無法從畫面判斷
  判斷原則：
  a. 若左側訊息上方出現不同成員名稱，通常是 group
  b. 若左上名稱後面有"(int)"，通常是 group
  c. 若明顯是和單一對象對話，填 direct
  d. 無法確定時填 unknown

若訊息為非文字內容，message 請使用以下固定標籤之一：
[貼圖]
[圖片]
[語音]
[通話]
[位置資訊]
[聯絡人]
[檔案]
若無法判斷，填 [未知非文字訊息]
 
輸出格式：
{
  "screen_type": "chat_detail",
  "app_source": "LINE",
  "chat_type": "direct",
  "chat_header": "王小明",
  "messages": [
    {"user_name": null, "side": "left", "message": "你好", "time": "下午 3:42"},
    {"user_name": "發起人", "side": "right", "message": "嗨！", "time": "下午 3:43"}
  ]
}
 
【chat_list — 聊天列表首頁】
 
抽取每個可見的聊天摘要項目，由上到下排序。
輸出格式：
{
  "screen_type": "chat_list",
  "app_source": "LINE",
  "chat_entries": [
    {"contact_name": "王小明", "last_message": "好的，明天見", "time": "下午 5:30", "unread_count": 2},
    {"contact_name": "工作群組", "last_message": "會議改到三點", "time": "下午 4:15", "unread_count": null}
  ]
}
 
【friend_list — 好友名單】
 
抽取每個可見的聯絡人，由上到下排序。
輸出格式：
{
  "screen_type": "friend_list",
  "app_source": "LINE",
  "friends": [
    {"display_name": "王小明", "status_message": "努力工作中"},
    {"display_name": "李小華", "status_message": null}
  ]
}
 
【other / unknown】
 
{
  "screen_type": "other",
  "app_source": "unknown",
  "description": "設定頁面"
}
 
═══ 嚴格規定 ═══
- 只能輸出 JSON，不可輸出任何說明文字
- 不可捏造畫面中不存在的資訊
"""


# ============================================================
# 畫面解析
# ============================================================

# --- null 判定 ---
_NULL_STRINGS = frozenset({"null" , "none" , "無" , "n/a" , "未知" , ""})

def _is_null(value:Any)-> bool:
    """把輸出 null 統一判定"""
    if value is None:
        return True
    if isinstance(value , str) and value.lower() in _NULL_STRINGS:
        return True
    return False

def _clean(value:Any)-> str|None:
    """ null -> None，其餘去前後空白"""
    if _is_null(value):
        return None
    return str(value).strip() or None

# --- JSON 提取---

def _extract_json_str(text) -> str|None:
    """從 VLM 中提取 JSON """
    m = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
    if m :
        return m.group(1).strip()
    # 或是{}
    m = re.search(r'\{[\s\S]*}' , text)
    return m.group(0) if m else None

# --- 各類螢幕截圖標準化 ---


_VALID_CHAT_TYPES = {"direct", "group", "unknown"}

def _normalize_chat_type(value: Any) -> str:
    """把 chat_type 正規化成 direct / group / unknown。"""
    if value is None:
        return "unknown"
    s = str(value).strip().lower()
    if s in _VALID_CHAT_TYPES:
        return s
    aliases = {
        "1to1": "direct",
        "1v1": "direct",
        "private": "direct",
        "single": "direct",
        "personal": "direct",
        "one_to_one": "direct",
        "one-to-one": "direct",
        "dm": "direct",
        "group_chat": "group",
        "multi": "group",
        "多人": "group",
        "群聊": "group",
        "群組": "group",
    }
    return aliases.get(s, "unknown")

def infer_chat_type(chat_header: str | None, messages: list[dict] | None) -> str:
    """
    當 VLM 沒有明確輸出 chat_type 時，用保守規則推測。
    規則：
    1. 左側若出現兩個以上不同名字，幾乎可判定為 group。
    2. 左側若出現一個名字，且與 chat_header 不同，優先視為 group。
    3. 左側若沒有名字，通常較像 direct，但保留 unknown 的空間。
    """
    msgs = messages or []
    left_names = []
    for msg in msgs:
        if not isinstance(msg, dict):
            continue
        if _clean(msg.get("side")) != "left":
            continue
        name = _clean(msg.get("user_name"))
        if name and name != "發起人":
            left_names.append(name)

    normalized_names = {normalize_text(name) for name in left_names if normalize_text(name)}
    if len(normalized_names) >= 2:
        return "group"

    header_norm = normalize_text(chat_header or "")
    if len(normalized_names) == 1:
        only_name = next(iter(normalized_names))
        if header_norm and only_name != header_norm:
            return "group"
        return "direct"

    if header_norm:
        return "direct"
    return "unknown"


def _norm_chat_detail(data: dict) -> dict:
    """標準化 chat_detail 輸出。"""
    chat_header = _clean(data.get("chat_header"))
    result = {
        "screen_type": "chat_detail",
        "app_source": _clean(data.get("app_source")) or "unknown",
        "chat_type": _normalize_chat_type(data.get("chat_type")),
        "chat_header": chat_header,
        "messages": [],
    }
    for msg in (data.get("messages") or []):
        if not isinstance(msg, dict):
            continue
        message = _clean(msg.get("message"))
        if not message:
            continue
        result["messages"].append({
            "user_name": _clean(msg.get("user_name")),
            "side": _clean(msg.get("side")) or "left",
            "message": message,
            "time": _clean(msg.get("time")),
        })
    if result["chat_type"] == "unknown":
        result["chat_type"] = infer_chat_type(chat_header, result["messages"])
    return result
 
 
def _norm_chat_list(data: dict) -> dict:
    """標準化 chat_list 輸出。"""
    result = {
        "screen_type": "chat_list",
        "app_source": _clean(data.get("app_source")) or "unknown",
        "chat_entries": [],
    }
    for entry in (data.get("chat_entries") or []):
        if not isinstance(entry, dict):
            continue
        name = _clean(entry.get("contact_name"))
        if not name:
            continue
        # unread_count：可能是 int、字串數字、或 null
        unread = entry.get("unread_count")
        if isinstance(unread, str):
            if _is_null(unread):
                unread = None
            else:
                try:
                    unread = int(unread)
                except ValueError:
                    unread = None
        elif not isinstance(unread, (int, float)):
            unread = None
 
        result["chat_entries"].append({
            "contact_name": name,
            "last_message": _clean(entry.get("last_message")),
            "time": _clean(entry.get("time")),
            "unread_count": unread,
        })
    return result
 
 
def _norm_friend_list(data: dict) -> dict:
    """標準化 friend_list 輸出。"""
    result = {
        "screen_type": "friend_list",
        "app_source": _clean(data.get("app_source")) or "unknown",
        "friends": [],
    }
    for f in (data.get("friends") or []):
        if not isinstance(f, dict):
            continue
        name = _clean(f.get("display_name"))
        if not name:
            continue

            # 相容舊 prompt 的錯字欄位
        last_message = (
            _clean(f.get("last_message"))
            or _clean(f.get("last_conversation"))
            or _clean(f.get("last_covation"))
        )

        last_message_time_raw = (
            _clean(f.get("last_message_time_raw"))
            or _clean(f.get("last_message_time"))
            or _clean(f.get("last_conversation_time"))
            or _clean(f.get("last_conversation_date"))
            or _clean(f.get("last_convation_date"))
            or _clean(f.get("time"))
        )

        result["friends"].append({
            "display_name": name,
            "last_message": last_message,
            "last_message_time_raw": last_message_time_raw,
            # 如果你還想保留狀態訊息，也可以留下
            "status_message": _clean(f.get("status_message")),
        })

    return result

 
def _norm_other(data: dict) -> dict:
    """標準化 other / unknown 輸出。"""
    st = data.get("screen_type", "other")
    if st not in ("other", "unknown"):
        st = "other"
    return {
        "screen_type": st,
        "app_source": _clean(data.get("app_source")) or "unknown",
        "description": _clean(data.get("description")),
    }
 
 
_NORMALIZERS = {
    "chat_detail": _norm_chat_detail,
    "chat_list": _norm_chat_list,
    "friend_list": _norm_friend_list,
    "other": _norm_other,
    "unknown": _norm_other,
}
 
 
# ---- regex fallback（僅 chat_detail）----
 
def _regex_fallback(text: str) -> list[dict]:
    """VLM 輸出 JSON 解析失敗時的備援，只能處理 chat_detail 格式。"""
    messages = []
    blocks = re.split(
        r'(?=[-•*\s]*(?:使用者名稱|user_name)\s*[:：])', text
    )
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        sp = re.search(r'(?:使用者名稱|user_name)\s*[:：]\s*(.+?)(?=\n|$)', block)
        mg = re.search(
            r'(?:訊息|message)\s*[:：]\s*([\s\S]+?)'
            r'(?=\s*[-•*\s]*(?:時間|time)\s*[:：]|$)',
            block, re.DOTALL,
        )
        tm = re.search(r'(?:時間|time)\s*[:：]\s*(.+?)(?=\n|$)', block)
        if mg:
            message = _clean(mg.group(1))
            if message:
                messages.append({
                    "user_name": _clean(sp.group(1)) if sp else None,
                    "side": "left",
                    "message": message,
                    "time": _clean(tm.group(1)) if tm else None,
                })
    return messages
 

def infer_message_speaker(
        user_name : str|None,
        side : str|None,
        chat_type : str|None,
        chat_header : str|None,
) -> str|None:
    """
    1. right-side 一律為「發起人」
    2. left-side 有 user_name 就用 user_name
    3. direct + left + user_name 缺失 + chat_header 存在，就用 chat_header
    4. 其他情況回 None
    """
    side = _clean(side) or "left"
    user_name = _clean(user_name)
    chat_type = _normalize_chat_type(chat_type)
    chat_header = _clean(chat_header)

    if side == "right":
        return "發起人"
    
    if user_name:
        return user_name
    
    if chat_type == "direct" and chat_header:
        return chat_header
    
    return None

# ---- 主解析函式 ----
 
def parse_screen_output(output_text: str) -> dict:
    """
    解析 VLM 原始輸出 → 標準化結構。
 
    這是整個解析管線的入口。不管 screen_type 是什麼，
    都回傳統一結構的 dict，下游程式碼不需要分支處理。
    """
    if not output_text or not output_text.strip():
        return {"screen_type": "unknown", "app_source": "unknown",
                "parse_method": "failed", "raw_json": None}
 
    # 第一路：JSON 解析
    json_str = _extract_json_str(output_text)
    if json_str:
        try:
            data = json.loads(json_str)
            if isinstance(data, dict):
                st = data.get("screen_type", "unknown")
                norm = _NORMALIZERS.get(st, _norm_other)
                result = norm(data)
                result["raw_json"] = data
                result["parse_method"] = "json"
                return result
        except json.JSONDecodeError:
            pass
 
    # 第二路：regex fallback（僅 chat_detail）
    msgs = _regex_fallback(output_text)
    if msgs:
        return {
            "screen_type": "chat_detail",
            "app_source": "unknown",
            "chat_type": infer_chat_type(None, msgs),
            "chat_header": None,
            "messages": msgs,
            "raw_json": None,
            "parse_method": "regex_fallback",
        }
 
    return {"screen_type": "unknown", "app_source": "unknown",
            "parse_method": "failed", "raw_json": None}
 
 
# ════════════════════════════════════════════════════════════
# §2.5  對外便利函式（讓 demo_server.py 直接呼叫）
# ════════════════════════════════════════════════════════════
 
def extract_observations(output_text: str, source_file: str = "unknown",
                         app_hint: str | None = None) -> dict:
    """
    Phase 1 核心：提取所有可用的觀測資料（不只 chat_detail）。
 
    app_hint 用於交叉驗證：截圖系統已知是 LINE，
    但 VLM 說 unknown → 用截圖系統的判斷。
    """
    parsed = parse_screen_output(output_text)
 
    app_source = parsed.get("app_source", "unknown")
    if app_source == "unknown" and app_hint:
        app_source = app_hint
 
    obs = {
        "source_file": source_file,
        "screen_type": parsed["screen_type"],
        "app_source": app_source,
        "parse_method": parsed["parse_method"],
    }
 
    st = parsed["screen_type"]
    if st == "chat_detail":
        obs["chat_type"] = parsed.get("chat_type", "unknown")
        obs["chat_header"] = parsed.get("chat_header")
        obs["messages"] = parsed.get("messages", [])
        obs["message_count"] = len(obs["messages"])
    elif st == "chat_list":
        obs["chat_entries"] = parsed.get("chat_entries", [])
        obs["entry_count"] = len(obs["chat_entries"])
    elif st == "friend_list":
        obs["friends"] = parsed.get("friends", [])
        obs["entry_count"] = len(obs["friends"])
    else:
        obs["description"] = parsed.get("description")
 
    return obs
 
 
def extract_bubbles_for_csv(output_text: str , screen_type = "", app_source = "" , chat_header = "") -> list[dict]:
    """
    向後相容：替代原本 早期 demo 的 parse_bubbles()。
    回傳中文欄位名的 dict list，直接對接 append_csv()。
    """
    parsed = parse_screen_output(output_text)
    if parsed["screen_type"] != "chat_detail":
        return []
 
    bubbles = []
    for msg in parsed.get("messages", []):
        speaker = msg["user_name"]
        if speaker is None:
            speaker = "N/A"
        bubbles.append({
            "發話者": speaker,
            "訊息": (msg["message"] or "").replace("\n", " "),
            "訊息時間": msg["time"] or "",
        })
    return bubbles


# ============================================================
# 訊息去重引擎（原 dedup.py）
# ============================================================
 
def normalize_text(text: str) -> str:
    """正規化文字，用於去重比對。"""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r'[\u200b\u200c\u200d\ufeff]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text
 
 
def extract_timestep(file_name: str):
    """從檔名提取時間戳（用於排序）。"""
    match = re.search(r'(\d{8})_(\d{6})_(\d{3})', file_name)
    if not match:
        return None
    try:
        dt = datetime.strptime(
            f"{match.group(1)}_{match.group(2)}", "%Y%m%d_%H%M%S"
        )
        dt = dt.replace(microsecond=int(match.group(3)) * 1000)
        return dt
    except ValueError:
        return None
 
 
def make_message_fingerprint(
    speaker: str | None,
    time_str: str | None,
    content: str,
    chat_type: str | None = None,
    chat_header: str | None = None,
    app_source: str | None = None,
) -> str:
    """建立訊息指紋：SHA-256(app|chat_type|chat_header|speaker|time|content)。"""
    parts = [
        normalize_text(app_source or ""),
        normalize_text(chat_type or ""),
        normalize_text(chat_header or ""),
        normalize_text(speaker or ""),
        normalize_text(time_str or ""),
        normalize_text(content or ""),
    ]
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
 
 
def _parse_messages_from_result(result: dict) -> list[dict]:
    """
    從單筆 inference result 中提取 messages（供去重用）。

    用 parse_screen_output 統一解析，然後轉成 dedup 格式。
    這取代了原本 dedup.py 裡的 parse_output_text()。
    """
    parsed = parse_screen_output(result.get("output_text", ""))

    # 不是聊天對話視窗，就沒有 messages 可以給 dedup 使用
    if parsed.get("screen_type") != "chat_detail":
        return []

    messages = []
    chat_type = parsed.get("chat_type", "unknown")
    chat_header = parsed.get("chat_header")
    app_source = parsed.get("app_source", "unknown")

    for msg in parsed.get("messages", []):
        side = msg.get("side") or "left"

        speaker = infer_message_speaker(
            user_name=msg.get("user_name"),
            side=side,
            chat_type=chat_type,
            chat_header=chat_header,
        )

        messages.append({
            "speaker": speaker,
            "speaker_raw": msg.get("user_name"),
            "side": side,
            "message": msg.get("message"),
            "time": msg.get("time"),
            "chat_type": chat_type,
            "chat_header": chat_header,
            "app_source": app_source,
        })

    return messages
 
class MessageDeduplicator:
    """
    訊息級去重器。
 
    維護「指紋 → 訊息」的映射表，
    新訊息的指紋已存在時跳過。
    """
 
    def __init__(self):
        self._fingerprints: dict[str, dict] = {}
        self._source_map: dict[str, list[str]] = {}
        self._all_messages_count = 0
 
    def add_result(self, result: dict) -> dict:
        """加入一筆 inference 結果，執行訊息級去重。"""
        file_name = result.get("file_name", "unknown")
        messages = _parse_messages_from_result(result)
 
        new_count = 0
        dup_count = 0
 
        for msg in messages:
            self._all_messages_count += 1
            fp = make_message_fingerprint(
                msg["speaker"],
                msg["time"],
                msg["message"],
                chat_type=msg.get("chat_type"),
                chat_header=msg.get("chat_header"),
                app_source=msg.get("app_source"),
            )
            if fp in self._fingerprints:
                dup_count += 1
                self._source_map[fp].append(file_name)
            else:
                new_count += 1
                self._fingerprints[fp] = {
                    **msg,
                    "fingerprint": fp,
                    "first_seen_in": file_name,
                }
                self._source_map[fp] = [file_name]
 
        return {
            "new_count": new_count,
            "dup_count": dup_count,
            "messages": messages,
        }
 
    def get_deduplicated_conversation(self) -> list[dict]:
        """取得去重後的對話序列（按首次出現順序）。"""
        return [
            {
                **msg,
                "appeared_in_files": self._source_map[fp],
                "appearance_count": len(self._source_map[fp]),
            }
            for fp, msg in self._fingerprints.items()
        ]
 
    @property
    def stats(self) -> dict:
        unique = len(self._fingerprints)
        total = self._all_messages_count
        return {
            "total_messages_parsed": total,
            "unique_messages": unique,
            "duplicates_removed": total - unique,
            "dedup_ratio": (
                f"{((total - unique) / total * 100):.1f}%"
                if total > 0 else "0%"
            ),
        }
 
# ============================================================ 
# 對外界面
# ============================================================
def deduplicate_results(results: list[dict]) -> dict[str, Any]:
    """
    對整個 result_sum.json 進行完整去重 — 外部主要入口。
 
    回傳：
      {
        "conversation": 去重後的對話序列,
        "stats": 統計數據,
        "per_file_stats": 每個檔案的去重統計,
      }
    """
    def sort_key(result):
        ts = extract_timestep(result.get("file_name", ""))
        if ts is not None:
            return (0, ts)
        return (1, result.get("file_name", ""))
 
    sorted_results = sorted(results, key=sort_key)
    deduper = MessageDeduplicator()
    per_file_stats = []
 
    for result in sorted_results:
        file_stat = deduper.add_result(result)
        per_file_stats.append({
            "file_name": result.get("file_name", "unknown"),
            "new_messages": file_stat["new_count"],
            "duplicate_messages": file_stat["dup_count"],
        })
 
    return {
        "conversation": deduper.get_deduplicated_conversation(),
        "stats": deduper.stats,
        "per_file_stats": per_file_stats,
    }
