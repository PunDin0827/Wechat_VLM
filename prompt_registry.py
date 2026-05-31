"""
prompt_registry.py — 兩階段 Prompt 管理中心
=============================================
拆成兩階段：
  Stage 1 (Classifier) — 只做分類，輸出極輕量的 JSON
  Stage 2 (Extractor)  — 針對特定 (app, screen_type) 組合，精準提取
---------
1. STAGE1_PROMPT          — 分類 prompt
2. parse_stage1_output()  — 解析 Stage 1 的 VLM 回應
3. EXTRACTORS dict        — (app_source, screen_type) → Stage 2 prompt
4. get_extractor_prompt() — routing 函式，帶 fallback 機制
"""

import re
import json
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# ============================================================
# Stage 1: 分類 Prompt
# ============================================================
# 設計原則：
# - 極簡指令，只要求輸出 3 個欄位
# - max_tokens 可以壓到 128~256，省推論時間
# - 不需要教模型怎麼提取訊息，只要「看」UI 風格

STAGE1_PROMPT = """\
你是截圖分類器，只需要判斷以下三件事。

判斷這張截圖的三個屬性,輸出 JSON:

{"app_source": "LINE | WeChat | unknown",
 "screen_type": "chat_detail | chat_list | friend_list | other | unknown",
 "chat_type": "direct | group | unknown | n/a"}

判斷順序:
1. 畫面中有可見的聊天泡泡 → chat_detail(優先級最高)
2. 畫面是聊天摘要列表 → chat_list
3. 畫面是好友/聯絡人清單 → friend_list
4. 其他 → other

chat_type 僅 chat_detail 時填寫,其他填 "n/a"。
"""

# ============================================================
# Stage 1: 輸出解析器
# ============================================================

@dataclass
class ClassificationResult:
    """
    Stage 1 分類結果的結構化容器。
    
    用 dataclass 而非 dict，好處：
    - IDE 自動補全，不會拼錯 key
    - 下游取值時有型別提示
    - 預設值集中管理
    """
    app_source: str = "unknown"
    screen_type: str = "unknown"
    raw_screen_type: str = "unknown"
    chat_type: str = "unknown"
    raw_output: str = ""          # 保留原始輸出，方便 debug
    parse_success: bool = False   # 標記解析是否成功

# 合法值白名單 — 防止 VLM 輸出奇怪的值穿透到下游
_VALID_APP_SOURCES = {"LINE", "WeChat", "unknown"}
_VALID_SCREEN_TYPES = {"chat_detail", "chat_list", "friend_list", "other", "unknown"}
_VALID_CHAT_TYPES = {"direct", "group", "unknown", "n/a"}


def _extract_json_from_text(text: str) -> Optional[dict]:
    """
    從 VLM 輸出中提取 JSON。
    
    VLM 有時會在 JSON 前後加上多餘的文字或 markdown 格式，
    這個函式嘗試多種方式提取：
    1. 先找 ```json ... ``` 包裹的區塊
    2. 再找第一組 { ... }
    3. 都失敗就回 None
    """
    if not text or not text.strip():
        return None
    
    # 方式 1: markdown code block
    m = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass
    
    # 方式 2: 直接找 { ... }
    m = re.search(r'\{[\s\S]*?\}', text)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    
    return None


def _validate_and_normalize(value: str, valid_set: set, default: str) -> str:
    """
    驗證 VLM 輸出的值是否在白名單內。
    
    為什麼需要這一步？
    VLM 可能輸出大小寫不一致（"line" vs "LINE"）或別名（"wechat" vs "WeChat"），
    這裡統一處理，不讓髒值流進下游。
    """
    if not value:
        return default
    
    # 先試精確匹配
    if value in valid_set:
        return value
    
    # 再試大小寫不敏感匹配
    # 建一個 lower → 原始值 的映射
    lower_map = {v.lower(): v for v in valid_set}
    normalized = value.strip().lower()
    if normalized in lower_map:
        return lower_map[normalized]
    
    # 常見別名處理
    app_aliases = {
        "wechat": "WeChat", "微信": "WeChat", "weixin": "WeChat",
        "line": "LINE",
    }
    if normalized in app_aliases and app_aliases[normalized] in valid_set:
        return app_aliases[normalized]
    
    logger.warning(f"Stage1: 未知值 '{value}'，fallback 到 '{default}'")
    return default


def parse_stage1_output(
    raw_output: str,
    app_hint: Optional[str] = None,
) -> ClassificationResult:
    """
    解析 Stage 1 VLM 的回應 → ClassificationResult。
    
    Parameters
    ----------
    raw_output : str
        VLM 原始回應文字
    app_hint : str, optional
        從檔名推斷的 app 提示（如檔名含 "LINE"），
        當 VLM 輸出 unknown 時用來補救
    
    Returns
    -------
    ClassificationResult
        永遠回傳有效結果，解析失敗時所有欄位都是 "unknown"
    
    設計原則：fail gracefully
    -----------------------
    Stage 1 解析失敗不應該讓整個 pipeline 爆掉。
    最差情況就是所有欄位都是 "unknown"，
    Stage 2 會用 fallback prompt（= 現有的 monolithic prompt）。
    """
    result = ClassificationResult(raw_output=raw_output)
    
    data = _extract_json_from_text(raw_output)
    if data is None:
        logger.warning(f"Stage1: JSON 解析失敗，raw='{raw_output[:200]}'")
        # 即使解析失敗，app_hint 仍可補救
        if app_hint:
            result.app_source = _validate_and_normalize(
                app_hint, _VALID_APP_SOURCES, "unknown"
            )
        return result
    
    # ── 逐欄位驗證 ──
    result.app_source = _validate_and_normalize(
        data.get("app_source", ""), _VALID_APP_SOURCES, "unknown"
    )
    result.screen_type = _validate_and_normalize(
        data.get("screen_type", ""), _VALID_SCREEN_TYPES, "unknown"
    )
    result.chat_type = _validate_and_normalize(
        data.get("chat_type", ""), _VALID_CHAT_TYPES, "unknown"
    )
    raw_screen_type = _validate_and_normalize(
    data.get("screen_type", ""),
    _VALID_SCREEN_TYPES,
    "unknown",
    )
    result.raw_screen_type = raw_screen_type
    # chat_type 只在 chat_detail 時有意義，其他情況強制為 n/a
    if result.screen_type != "chat_detail":
        result.chat_type = "n/a"
    # n/a 在 chat_detail 語境下當作 unknown
    elif result.chat_type == "n/a":
        result.chat_type = "unknown"
    # chat_list 轉 friend_list
    if raw_screen_type == "chat_list":
        result.screen_type = "friend_list"
    else:
        result.screen_type = raw_screen_type
    # ── app_hint 交叉驗證 ──
    # 場景：檔名明確寫了 LINE，但 VLM 說 unknown → 信檔名
    if result.app_source == "unknown" and app_hint:
        result.app_source = _validate_and_normalize(
            app_hint, _VALID_APP_SOURCES, "unknown"
        )
        logger.info(f"Stage1: app_source 由 app_hint 補正為 '{result.app_source}'")
    
    result.parse_success = True
    return result


# ============================================================
# Stage 2: 針對性提取 Prompt
# ============================================================
# 設計原則：
# - 每個 prompt 只描述「一種」輸出格式
# - 包含該 app 專屬的 UI 辨識規則（顏色、位置、時間格式）
# - 篇幅全部集中在「怎麼正確提取」，不浪費在無關的 screen_type

# ------------------------------------------------------------------
# LINE + chat_detail（最核心的組合，你的主要 use case）
# ------------------------------------------------------------------
PROMPT_LINE_CHAT_DETAIL = """\
提取這張 LINE 聊天截圖的所有訊息泡泡,由上到下排序。

核心規則:
- 右側泡泡 = 截圖者本人,user_name 填 "發起人"
- 左側泡泡 = 對方,有顯示名稱就填,否則填 null
- 非文字訊息用標籤:[貼圖] [圖片] [語音] [通話] [位置資訊] [聯絡人] [檔案]


輸出格式(嚴格遵守,不要任何說明文字):
{
  "screen_type": "chat_detail",
  "app_source": "LINE",
  "chat_type": "direct | group | unknown",
  "chat_header": "對話標題或 null",
  "messages": [
    {"user_name": null, "side": "left", "message": "你好", "time": "5月14日 15:15"},
    {"user_name": "發起人", "side": "right", "message": "嗨!", "time": null}
  ]
}
"""

# ------------------------------------------------------------------
# WeChat + chat_detail
# ------------------------------------------------------------------
PROMPT_WECHAT_CHAT_DETAIL = """\
提取這張 WeChat 聊天截圖的所有訊息泡泡,由上到下排序。

核心規則:
- 右側泡泡 = 截圖者本人,user_name 填 "發起人"
- 左側泡泡 = 對方,有顯示名稱就填,否則填 null
- 非文字訊息用標籤:[貼圖] [圖片] [語音] [通話] [位置資訊] [聯絡人] [檔案]


輸出格式(嚴格遵守,不要任何說明文字):
{
  "screen_type": "chat_detail",
  "app_source": "WeChat",
  "chat_type": "direct | group | unknown",
  "chat_header": "對話標題或 null",
  "messages": [
    {"user_name": null, "side": "left", "message": "你好", "time": "5月14日 15:15"},
    {"user_name": "發起人", "side": "right", "message": "嗨!", "time": null}
  ]
}
"""



# ------------------------------------------------------------------
# 通用版 friend_list（LINE/WeChat 差異較小，共用即可）
# ------------------------------------------------------------------
PROMPT_FRIEND_LIST = """\
提取這張聊天列表/好友清單的所有項目,由上到下排序。

每個項目需要:
- display_name: 名稱
- last_message: 該項目顯示的最後訊息預覽,沒有就填 null
- last_message_time_raw: 原始時間文字(如 "下午 3:42"、"昨天"、"05/04"),沒有就填 null

輸出格式:
{
  "screen_type": "friend_list",
  "app_source": "LINE | WeChat | unknown",
  "friends": [
    {"display_name": "王小明", "last_message": "我想下班", "last_message_time_raw": "05/04"}
  ]
}
"""

# ------------------------------------------------------------------
# 通用版 other / unknown
# ------------------------------------------------------------------
PROMPT_OTHER = """\
你是截圖描述器。這張截圖不是聊天對話或列表畫面。
請簡要描述畫面內容。

只輸出 JSON，不可輸出任何說明文字：
{
  "screen_type": "other",
  "app_source": "LINE 或 WeChat 或 unknown",
  "description": "畫面的簡要描述"
}
"""


# ============================================================
# Prompt Routing — 核心 Router
# ============================================================

# 用 (app_source, screen_type) 做 key 的查找表
# 順序：精確匹配 → screen_type 通用版 → 最終 fallback
_EXTRACTORS: dict[tuple[str, str], str] = {
    # ── LINE 系列 ──
    ("LINE", "chat_detail"):   PROMPT_LINE_CHAT_DETAIL,
    ("LINE", "friend_list"):   PROMPT_FRIEND_LIST,
    ("LINE", "other"):         PROMPT_OTHER,
    ("unknown", "chat_detail"): PROMPT_LINE_CHAT_DETAIL,
    
    # ── WeChat 系列 ──
    ("WeChat", "chat_detail"): PROMPT_WECHAT_CHAT_DETAIL,
    ("WeChat", "friend_list"): PROMPT_FRIEND_LIST,
    ("WeChat", "other"):       PROMPT_OTHER,
    
    # ── 通用版（app_source unknown 時的 fallback）──
    ("unknown", "friend_list"): PROMPT_FRIEND_LIST,
    ("unknown", "other"):       PROMPT_OTHER,
    ("unknown", "unknown"):     PROMPT_OTHER,
}


def get_extractor_prompt(
    app_source: str,
    screen_type: str,
    fallback_prompt: Optional[str] = None,
) -> str:
    """
    根據 Stage 1 分類結果，取得對應的 Stage 2 prompt。
    
    查找策略（三層 fallback）：
    ─────────────────────────
    1. 精確匹配：(app_source, screen_type) 直接命中
    2. 通用版：("unknown", screen_type) — 同類型但不分 app
    3. 最終 fallback：傳入的 fallback_prompt（= 你現有的 TASK_PROMPT_V2）
    
    Parameters
    ----------
    app_source : str
        Stage 1 判斷的 app 來源
    screen_type : str
        Stage 1 判斷的畫面類型
    fallback_prompt : str, optional
        所有查找都失敗時的保底 prompt，
        建議傳入現有的 TASK_PROMPT_V2，確保不會「什麼都沒有」
    
    Returns
    -------
    str
        Stage 2 要使用的 prompt
    """
    # Layer 1: 精確匹配
    key = (app_source, screen_type)
    if key in _EXTRACTORS:
        logger.info(f"Stage2 routing: 精確匹配 {key}")
        return _EXTRACTORS[key]
    
    # Layer 2: 通用版 fallback
    generic_key = ("unknown", screen_type)
    if generic_key in _EXTRACTORS:
        logger.info(f"Stage2 routing: 通用版 fallback {generic_key}")
        return _EXTRACTORS[generic_key]
    
    # Layer 3: 最終 fallback — 用原本的 monolithic prompt
    if fallback_prompt:
        logger.warning(
            f"Stage2 routing: 無匹配 {key}，使用 fallback_prompt"
        )
        return fallback_prompt
    
    # 極端情況：連 fallback_prompt 都沒給（不應該發生）
    logger.error(f"Stage2 routing: 無匹配且無 fallback，使用 PROMPT_OTHER")
    return PROMPT_OTHER


def needs_stage2(screen_type: str) -> bool:
    """
    判斷該 screen_type 是否需要跑 Stage 2。
    
    有些畫面類型（other、unknown）在 Stage 1 就已經得到足夠資訊，
    不值得再花一次 VLM 推論。
    
    Returns
    -------
    bool
        True = 需要 Stage 2, False = Stage 1 結果就夠了
    """
    # chat_detail 和 chat_list 有大量結構化資料要提取 → 需要 Stage 2
    # friend_list 也有列表要提取 → 需要 Stage 2
    # other / unknown → Stage 1 已足夠
    return screen_type in ("chat_detail", "friend_list")


# ============================================================
# 便利函式 — 組合 Stage 1 結果供 demo_server 使用
# ============================================================

def build_stage1_observations(
    classification: ClassificationResult,
    source_file: str,
) -> dict:
    """
    當不需要 Stage 2 時（other/unknown），
    直接從 Stage 1 結果建立 observations dict。
    
    這讓下游的存檔邏輯不需要區分「有沒有跑 Stage 2」。
    """
    return {
        "source_file": source_file,
        "screen_type": classification.screen_type,
        "app_source": classification.app_source,
        "parse_method": "stage1_only",
        "description": f"Stage 1 分類結果: {classification.screen_type}",
    }