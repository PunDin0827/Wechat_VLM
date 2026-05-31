"""
normalizer.py — VLM 輸出正規化層
把產出的 observation 正規化輸出

"""
import re

from enum import Enum
from datetime import datetime
from typing import Optional
from pydantic import BaseModel , Field , ConfigDict

#==================================================================
# 定義
#==================================================================

class MessageType(str , Enum):
    TEXT = "text"            # 純文字
    STICKER = "sticker"      # 貼圖
    IMAGE = "image"          # 影像
    AUDIO = "audio"          # 聲音檔案
    CALL = "call"            # 電話 / 視訊電話
    LOCATION = "location"    # 地圖 / 位置訊息
    CONTACT = "contact"      # 聯絡人資訊 
    FILE = "file"            # 檔案
    UNKNOWN = "unknown"      # 其餘都未知


class Side(str , Enum):
    "聊天畫面的位置"
    LEFT = "left"   # 截圖者以外的人
    RIGHT = "right" # 截圖者本人
    UNKNOWN = "unknown" 

class AppSource(str , Enum):
    LINE = "line"
    WECHAT = "wechat"
    UNKNOWN = "unknown"

class ChatType(str , Enum):
    DIRECT = "direct"
    GROUP = "group"
    UNKNOWN = "unknown"

class TimeConfidence(str , Enum):
    "時間解析的信心等級，越可信權重可以拉高"
    EXACT = "exact"         # 畫面上明確標註且解析成功
    INFERRED = "inferred"   # 從其他訊息或截圖推理而來
    MISSING = "missing"     # 畫面上沒有

#==================================================================
# 核心資料結構
#==================================================================

class NormalizedMessage(BaseModel):
    """
    正規化後的訊息
    欄位分群：
    ------------------------
    1.識別：(identity)：message_id , source_file , sequence 
    2.內容：(content)：content , content_type
    3.發話者：(speaker)：speaker_raw , speaker_canonical , side , is_self
    4.時間：(time)：time_raw , time_parsed , time_confidence
    5.情境：(context)：app_source , chat_header , screenshot_taken_at
    ------------------------
    """

    model_config = ConfigDict(use_enum_values = True , extra = "forbid")
    
    message_id : str = Field(... , 
                           description = (
                              "唯一識別碼，格式為{file_hash_short}_{sequence}。" 
                           ),
                           examples = ["a3f2b1c8_0"]
                           )
    
    source_file : str = Field(
        ...,
        description = "來源截圖檔名，追朔與 debug 用"
    )
    
    sequence : int = Field(
        ...,
        ge = 0 , 
        description = "在該截圖內的訊息順序(由上到下 , 0-index)",
    )

    #----- 內容欄位(Context Fields) -----
    content : str = Field(
        ...,
        min_length = 1 , # 確保下拿到不是空字串
        description = (
            "訊息正文，文字訊息填原始文字(去除多餘空白)"
            "非文字訊息填畫面描述(如：貼圖)"
        ),
    )

    content_type : MessageType = Field(
        default = MessageType.TEXT,
        description = "訊息類型"
    )

    #----- 發話者欄位(Speaker Fields) -----
    speaker_raw : Optional[str] = Field(
        default = None , 
        description = (
            " VLM 原始輸出的發話者名稱，用來驗證正規化輸出結果是否正確" \
            "如'小明' 與 '王小明' 會正規化成同一個人" \
            " None 表示畫面上沒有名字且無法從上下文推斷" 
        ),
    )

    speaker_canonical : Optional[str] = Field(
        default = None,
        description = (
            "正規化後的發話者名稱，同一人不同稱呼會被統一" \
            "右側泡泡一律標成 '發起人' (截圖者本人)" \
            "左側泡泡有名字就用,若為 direct 且左側無名字，使用 chat_header 補對方名稱；仍無法判斷則 None"
        ),
    )

    side : Side = Field(
        ...,
        description = "對話框方位，'left'是對方，'right'是截圖者本人"
    )

    is_self : bool = Field(
        ...,
        description = (
            "是否為截圖者本人的訊息" \
            "等價於 side == right " \
            "避免下游每次都要判斷 side "
        )
    )

    #----- 時間欄位(Time Fields) -----

    time_raw : Optional[str] = Field(
        default = None ,
        description = "畫面顯示的原始時間字串"
    )

    time_parsed : Optional[str] = Field(
        default = None , 
        description = (
            "time_raw 正規化後的時間字串；不補截圖日期或年份"
        ),
    )

    time_confidence : TimeConfidence = Field(
        default = TimeConfidence.MISSING , 
        description = "時間解析的信心度，下游可用分數加權計算"
    )

    #----- 情境欄位(Context Fields) -----
    
    app_source : AppSource = Field(
        default = AppSource.UNKNOWN ,
        description = "來源 APP " ,
    )
    
    chat_header : Optional[str] = Field(
        default = None , 
        description = "聊天室窗頂部標題(通常是對方名字或是群組名稱)"
    )

    chat_type : ChatType = Field(
        default = ChatType.UNKNOWN,
        description = "聊天型態：1對1(direct) / 群組(group) / unknown"
    )

    screenshot_taken_at : Optional[datetime] = Field(
        default = None , 
        description = (
            "截圖產生時間，由檔名時間戳提取"
        )
    )
    
    client_id : str = Field(
        default = "default",
        description = "上傳來源 client ID"
    )
    
class NormalizedFriend(BaseModel):
    """
    正規化好友資料
    只輸出好友名單
    """
    model_config = ConfigDict(use_enum_values = True , extra = "forbid")

    friend_id : str = Field(
        ...,
        description = "好友資料唯一 ID ，格式為 {file_hash_short}_{sequence}"
    )

    source_file : str = Field(
        ...,
        description = "來源檔名"
    )

    sequence : int = Field(
        ...,
        ge = 0,
        description = "好友在該節圖順序，由上到下"
    )

    display_name_raw : str = Field(
        ...,
        min_length = 1,
        description = "影像抽取到的好友名稱"
    )

    display_name_canonical : Optional[str] = Field(
        default = None,
        description = "正規化後的名稱"
    )
    last_message_raw: Optional[str] = Field(
        default=None,
        description="好友 / 聊天項目的最後一句可見訊息摘要"
    )

    last_message_time_raw: Optional[str] = Field(
        default=None,
        description="畫面上顯示的原始最後訊息時間，例如 下午 3:42、昨天、05/04"
    )

    last_message_time_parsed: Optional[str] = Field(
        default=None,
        description="最後訊息時間正規化後的字串；無法解析時為 None"
    )

    last_message_time_confidence: TimeConfidence = Field(
        default=TimeConfidence.MISSING,
        description="最後訊息時間解析信心度"
    )
    app_source : AppSource = Field(
        default = AppSource.UNKNOWN,
        description = "來源 APP "
    )

    screenshot_taken_at : Optional[datetime] = Field(
        default = None,
        description = "截圖時間，由檔名時間戳解析"
    )
    client_id : str = Field(
        default = "default",
        description = "上傳來源 client ID"
    )

#==================================================================
# 中間結果
#==================================================================

class ScreenshotContext(BaseModel):
    source_file : str
    screenshot_taken_at : Optional[datetime]
    app_source : AppSource
    chat_header : Optional[str]
    chat_type : ChatType
    client_id : str = "default"


#==================================================================
# 時間正規化工具
#==================================================================

# 把畫面顯示時間字串轉換，產出不補日期/年份的正規化字串
# 每個 regex pattern 代表一種支援的格式，依序嘗試
# 第一個 match 成功就採用，全失敗回傳 ( None , MISSING )
import re

_PATTERN_CHINESE_AMPM = re.compile(
    r'^(?P<period>上午|下午|凌晨|晚上|早上|中午)'
    r'\s*'
    r'(?P<h>\d{1,2})[:：](?P<m>\d{2})$'
)
_PATTERN_24HOUR = re.compile(
    r'^(?P<h>\d{1,2})[:：](?P<m>\d{2})$'
)
# 畫面上同時顯示月日 + 時分，例如 "5月14日 01:36" / "5月14日 下午 3:42"
# 時間部分可選，允許只有日期 "5月14日"（由 _PATTERN_CHINESE_MONTH_DAY 接手）
_PATTERN_CHINESE_FULL_DATETIME = re.compile(
    r'^(?P<month>\d{1,2})\s*月\s*(?P<day>\d{1,2})\s*日'
    r'(?:\s*(?P<period>上午|下午|凌晨|晚上|早上|中午)?'
    r'\s*(?P<h>\d{1,2})[:：](?P<m>\d{2}))?$'
)
 
def _apply_ampm_period(hour : int , period : str):
    """
    修正小時 -> 24 小時制
    重點：
    下午12:XX -> 12:XX (下午)，不是 24:XX
    上午12:XX -> 00:XX (半夜)，不是 12:XX
    下午1:XX -> 13:XX
    上午9:XX -> 09:XX (不變)
    """
    if period in ("上午" , "早上"):
        # AM:12 視為 0 點
        return 0 if hour == 12 else hour
    if period in ("下午" ,"晚上"):
        # PM:12 點不變，其餘 +12
        return hour if hour == 12 else hour +12
    if period == "中午":
        # 中午12:XX -> 12:XX ，中午 1:XX -> 13:XX
        return hour if hour == 12 else hour +12
    if period == "凌晨":
        # 凌晨通常 0-8 點，不變
        return hour
    return hour

def _format_hm(hour: int, minute: int) -> str:
    return f"{hour:02d}:{minute:02d}"

def _format_chinese_month_day(month: int, day: int) -> str:
    return f"{month}月{day}日"

def _format_chinese_month_day_hm(month: int, day: int, hour: int, minute: int) -> str:
    return f"{_format_chinese_month_day(month, day)} {_format_hm(hour, minute)}"

def parse_message_time(
        time_raw :  Optional[str],
        context : "ScreenshotContext"
) -> tuple[Optional[str], TimeConfidence]:
    """
    解析訊息時間 -> (normalized time string , confidence)
    提供下游時間信心度是 明確的 or 推斷的 or 沒有時間
    目前支援格式:
    1. 完整月日時分: "5月14日 01:36" / "5月14日 下午 3:42"
       → 直接從 time_raw 解析，不補年份
    2. 中文時段 + 時間: "下午 3:42" / "上午 10:15" / "晚上 11:30"
       → 只輸出 24 小時制時分
    3. 24 小時制:       "15:42" / "09:15"
       → 只輸出 24 小時制時分
    4. None / 空字串:   回傳 (None, MISSING)
    5. 不認識的格式:     回傳 (None, MISSING) — fail gracefully
    """
    # step.1 前置檢查
    if not time_raw or not time_raw.strip():
        return None , TimeConfidence.MISSING

    s = time_raw.strip()

    # step.2 嘗試完整月日時分格式 "5月14日 01:36" / "5月14日 下午 3:42"
    # 畫面已包含日期時，直接使用 time_raw 內容，截圖時間與畫面時間各自獨立
    m = _PATTERN_CHINESE_FULL_DATETIME.match(s)
    if m and m["h"] is not None:
        month = int(m["month"])
        day = int(m["day"])
        hour = int(m["h"])
        minute = int(m["m"])
        if m["period"]:
            hour = _apply_ampm_period(hour, m["period"])
        if not (1 <= month <= 12 and 1 <= day <= 31 and 0 <= hour <= 23 and 0 <= minute <= 59):
            return None, TimeConfidence.MISSING
        try:
            datetime(2000, month, day, hour, minute)
            return _format_chinese_month_day_hm(month, day, hour, minute), TimeConfidence.EXACT
        except ValueError:
            return None, TimeConfidence.MISSING

    # step.3 嘗試中文時段+時間
    m = _PATTERN_CHINESE_AMPM.match(s)
    if m:
        hour = _apply_ampm_period(int(m["h"]) , m["period"])
        minute = int(m["m"])
        # 防止小時或分鐘超出範圍 -> 視為解析失敗
        if not (0 <= hour <= 23 and 0 <= minute <= 59):
            return None , TimeConfidence.MISSING
        return _format_hm(hour , minute) , TimeConfidence.EXACT

    # step.4 試 24 小時格式
    m = _PATTERN_24HOUR.match(s)
    if m:
        hour = int(m["h"])
        minute = int(m["m"])
        if not (0 <= hour <=23 and 0 <= minute <= 59):
            return None , TimeConfidence.MISSING
        return _format_hm(hour , minute) , TimeConfidence.EXACT

    return None , TimeConfidence.MISSING
_PATTERN_MM_DD_SLASH = re.compile(r'^(?P<month>\d{1,2})/(?P<day>\d{1,2})$')
_PATTERN_CHINESE_MONTH_DAY = re.compile(r'^(?P<month>\d{1,2})\s*月\s*(?P<day>\d{1,2})\s*日$')


def parse_friend_list_time(
    time_raw: Optional[str],
    screenshot_taken_at: Optional[datetime],
) -> tuple[Optional[str], TimeConfidence]:
    """
    解析 friend_list / chat_list 上的最後訊息時間。

    支援：
    1. 完整月日時分：5月14日 01:36 / 5月14日 下午 3:42（畫面時間獨立解析，不受截圖時間影響）
    2. 時分格式：下午 3:42、14:28（只輸出時分）
    3. 相對日期：今天、昨天
    4. 月日格式：05/04、4月20日

    注意：
    - 只有日期沒有時分時，只輸出日期，不補時間或年份，信心度用 INFERRED
    - 無法解析時回傳 (None, MISSING)
    """

    if not time_raw or not str(time_raw).strip():
        return None, TimeConfidence.MISSING

    raw = str(time_raw).strip()

    # 1. 先復用聊天訊息時間解析（含新增的月日時分格式）
    # 注意：完整月日時分格式（如 "5月14日 01:36"）不需要 screenshot_taken_at 也能解析
    context = ScreenshotContext(
        source_file="friend_list",
        screenshot_taken_at=None,
        app_source=AppSource.UNKNOWN,
        chat_header=None,
        chat_type=ChatType.UNKNOWN,
    )
    parsed, conf = parse_message_time(raw, context)
    if parsed is not None:
        return parsed, conf

    # 以下格式不再使用截圖時間；缺少的日期/時間部分不補值。

    # 2. 今天 / 昨天
    if raw == "今天":
        return "今天", TimeConfidence.INFERRED

    if raw == "昨天":
        return "昨天", TimeConfidence.INFERRED

    # 3. 05/04
    m = _PATTERN_MM_DD_SLASH.match(raw)
    if m:
        month = int(m["month"])
        day = int(m["day"])

        try:
            datetime(2000, month, day)
        except ValueError:
            return None, TimeConfidence.MISSING

        return f"{month:02d}/{day:02d}", TimeConfidence.INFERRED

    # 4. 4月20日
    m = _PATTERN_CHINESE_MONTH_DAY.match(raw)
    if m:
        month = int(m["month"])
        day = int(m["day"])

        try:
            datetime(2000, month, day)
        except ValueError:
            return None, TimeConfidence.MISSING

        return _format_chinese_month_day(month, day), TimeConfidence.INFERRED

    return None, TimeConfidence.MISSING

#==================================================================
# 發話者正規化 (最小版)
#==================================================================

# right-side => 一律標為"發起人"(截圖本人)
# left-side => 有名子就用，沒有就 None

SPEAKER_SELF_LABEL = "發起人"

def normalize_speaker(
        speaker_raw : Optional[str] ,
        side : Side,
        chat_type : ChatType = ChatType.UNKNOWN,
        chat_header : Optional[str] = None , 
) -> Optional[str]:
    """
    right-side：一律為發起人
    left-sdie：
    1.若畫面有 user_name ，優先使用 user_name
    2.若是direct 且 user_name 缺失，使用 chat_header 補對方名稱
    3.其餘回 None
    
    """
    if side == Side.RIGHT:
        return SPEAKER_SELF_LABEL

    if speaker_raw:
        return speaker_raw
    
    if chat_type == ChatType.DIRECT and chat_header:
        return chat_header
    
    return None

def _clean_optional_str(value) -> Optional[str]:
    """
    將空字串、null-like 文字轉成 None。
    friend_list normalizer 專用的小工具。
    """
    if value is None:
        return None

    s = str(value).strip()
    if not s or s.lower() in {"null", "none", "n/a", "未知", "無"}:
        return None

    return s

def normalize_friends(observation: dict) -> list[NormalizedFriend]:
    """
    把 friend_list observation 轉成 NormalizedFriend list。
    """

    if observation.get("screen_type") != "friend_list":
        return []

    source_file = observation.get("source_file", "unknown")
    file_hash = _file_hash_short(source_file)

    from output_parser import extract_timestep
    screenshot_taken_at = extract_timestep(source_file)

    raw_app = (observation.get("app_source") or "unknown").lower()

    if raw_app == "line":
        app_source = AppSource.LINE
    elif raw_app in ("wechat", "weixin", "微信"):
        app_source = AppSource.WECHAT
    else:
        app_source = AppSource.UNKNOWN

    results: list[NormalizedFriend] = []

    for seq, friend in enumerate(observation.get("friends", [])):
        if not isinstance(friend, dict):
            continue

        display_name = _clean_optional_str(friend.get("display_name"))
        if not display_name:
            continue

        # 新欄位：最後訊息
        last_message_raw = (
            _clean_optional_str(friend.get("last_message"))
            or _clean_optional_str(friend.get("last_conversation"))
            or _clean_optional_str(friend.get("last_covation"))
        )

        # 新欄位：最後訊息時間 raw
        last_message_time_raw = (
            _clean_optional_str(friend.get("last_message_time_raw"))
            or _clean_optional_str(friend.get("last_message_time"))
            or _clean_optional_str(friend.get("last_conversation_time"))
            or _clean_optional_str(friend.get("last_conversation_date"))
            or _clean_optional_str(friend.get("last_convation_date"))
            or _clean_optional_str(friend.get("time"))
        )

        # 新欄位：最後訊息時間 parsed
        last_time_parsed, last_time_confidence = parse_friend_list_time(
            last_message_time_raw,
            screenshot_taken_at,
        )

        results.append(
            NormalizedFriend(
                friend_id=f"{file_hash}_{seq}",
                source_file=source_file,
                sequence=seq,
                display_name_raw=display_name,
                display_name_canonical=display_name,
                last_message_raw=last_message_raw,
                last_message_time_raw=last_message_time_raw,
                last_message_time_parsed=last_time_parsed,
                last_message_time_confidence=last_time_confidence,
                app_source=app_source,
                screenshot_taken_at=screenshot_taken_at,
                client_id = observation.get("client_id" , "default")
            )
        )

    return results
#==================================================================
# 主整合函數入口 -- normalize()
#==================================================================
# Phase 2 的主入口，接受 Phase 1 的 observation
# 輸出 list[NormalizedMessage]，供下游 Phase 3 使用

import hashlib
import logging

# logger - 失敗訊息不 print 到輸出，走 logging 
logger = logging.getLogger(__name__)

def _file_hash_short(file_name : str , length : int = 8) -> str:
    """
    從檔案名稱產生短 hash ，作為 message_id 的前綴
    """
    return hashlib.md5(file_name.encode("utf-8")).hexdigest()[:length]

_EXACT_TYPE_LABELS = {
    # ---- 貼圖 ----
    "[貼圖]": MessageType.STICKER,
    "[sticker]": MessageType.STICKER,

    # ---- 圖片 / 照片 ----
    "[圖片]": MessageType.IMAGE,
    "[照片]": MessageType.IMAGE,
    "[image]": MessageType.IMAGE,
    "[photo]": MessageType.IMAGE,

    # ---- 語音 / 音訊 ----
    "[語音]": MessageType.AUDIO,
    "[音訊]": MessageType.AUDIO,
    "[voice]": MessageType.AUDIO,
    "[audio]": MessageType.AUDIO,

    # ---- 通話 ----
    "[通話]": MessageType.CALL,
    "[電話]": MessageType.CALL,
    "[call]": MessageType.CALL,

    # ---- 位置 / 地圖 ----
    "[位置]": MessageType.LOCATION,
    "[位置資訊]": MessageType.LOCATION,
    "[地圖]": MessageType.LOCATION,
    "[location]": MessageType.LOCATION,
    "[map]": MessageType.LOCATION,

    # ---- 聯絡人 ----
    "[聯絡人]": MessageType.CONTACT,
    "[名片]": MessageType.CONTACT,
    "[contact]": MessageType.CONTACT,

    # ---- 檔案 ----
    "[檔案]": MessageType.FILE,
    "[文件]": MessageType.FILE,
    "[file]": MessageType.FILE,
    "[document]": MessageType.FILE,

    # ---- 未知非文字 ----
    "[未知非文字訊息]": MessageType.UNKNOWN,
}

def _detect_content_type(content : str) -> MessageType:
    """
    從訊息內文推斷 content_type
    未來若用 VLM 直接輸出可停用
    """
    if not content or not content.strip():
        return MessageType.UNKNOWN

    raw = content.strip()

    # 英文標籤比對用，例如 [IMAGE]、[File]
    lower = raw.lower()

    # --------------------------------------------------
    # 1. 最安全情況：完全等於固定標籤
    # --------------------------------------------------
    if raw in _EXACT_TYPE_LABELS:
        return _EXACT_TYPE_LABELS[raw]

    if lower in _EXACT_TYPE_LABELS:
        return _EXACT_TYPE_LABELS[lower]

    # --------------------------------------------------
    # 2. 容忍不同括號與空白格式
    #
    # 例如：
    # [ 圖片 ]
    # 【圖片】
    # （圖片）
    # (image)
    # --------------------------------------------------
    m = re.fullmatch(
        r"[\[\【\(（]\s*"
        r"(貼圖|sticker|"
        r"圖片|照片|image|photo|"
        r"語音|音訊|voice|audio|"
        r"通話|電話|call|"
        r"位置|位置資訊|地圖|location|map|"
        r"聯絡人|名片|contact|"
        r"檔案|文件|file|document|"
        r"未知非文字訊息)"
        r"\s*[\]\】\)）]",
        lower,
        flags=re.IGNORECASE,
    )

    if m:
        label = m.group(1).lower()

        if label in ("貼圖", "sticker"):
            return MessageType.STICKER

        if label in ("圖片", "照片", "image", "photo"):
            return MessageType.IMAGE

        if label in ("語音", "音訊", "voice", "audio"):
            return MessageType.AUDIO

        if label in ("通話", "電話", "call"):
            return MessageType.CALL

        if label in ("位置", "位置資訊", "地圖", "location", "map"):
            return MessageType.LOCATION

        if label in ("聯絡人", "名片", "contact"):
            return MessageType.CONTACT

        if label in ("檔案", "文件", "file", "document"):
            return MessageType.FILE

        if label == "未知非文字訊息":
            return MessageType.UNKNOWN

    # --------------------------------------------------
    # 3. 檔案卡片格式：整句看起來就是附件
    #
    # 例如：
    # report.pdf
    # report.pdf 1.2 MB
    # data.xlsx 35 KB
    # --------------------------------------------------
    file_card_pattern = re.fullmatch(
        r".+\."
        r"(pdf|doc|docx|xls|xlsx|ppt|pptx|zip|rar|7z|txt|csv|json|py|jpg|jpeg|png|mp4|mp3|wav|m4a)"
        r"(\s+\d+(\.\d+)?\s*(kb|mb|gb))?",
        lower,
        flags=re.IGNORECASE,
    )

    if file_card_pattern:
        return MessageType.FILE

    # --------------------------------------------------
    # 4. 預設：一般文字訊息
    # --------------------------------------------------
    return MessageType.TEXT

def _build_screenshot_context(observation : dict) -> ScreenshotContext:
    """
    從 observation 抽取整個 normalize() 呼叫的 context
    """
    from output_parser import extract_timestep
    
    source_file = observation.get("source_file" , "unknown")

    # app_source 大小寫容錯:
    # VLM prompt 要求輸出 "LINE" / "WeChat",但實際可能收到各種大小寫變形
    # 統一轉小寫再 match enum(enum 值也都是小寫)
    raw_app = (observation.get("app_source") or "unknown").lower()
    raw_chat_type = (observation.get("chat_type") or "unknown").lower()

    try:
        app_source = AppSource(raw_app)
    except ValueError:
        app_source = AppSource.UNKNOWN

    try:
        chat_type = ChatType(raw_chat_type)
    except ValueError:
        chat_type = ChatType.UNKNOWN

    return  ScreenshotContext(
        source_file = source_file,
        screenshot_taken_at = extract_timestep(source_file),
        app_source = app_source,
        chat_header = observation.get("chat_header"),
        chat_type = chat_type,
        client_id = observation.get("client_id" , "default")
    )

def _normalize_one_message(
        raw_msg : dict,
        sequence  : int,
        context : ScreenshotContext,
        file_hash : str,
) -> Optional[NormalizedMessage]:
    
    try:
        # --- side 解析 ---
        raw_side = (raw_msg.get("side") or "left").lower()
        side = Side.RIGHT if raw_side == "right" else Side.LEFT

        # --- context 檢查 ---
        content = raw_msg.get("message") or ""
        content = content.strip() 
        if not content:
            # 空訊息，跳過(NormalizedMessage 的 min_length = 1 也會擋)
            logger.debug(f"skip empty mssage at seq = {sequence}")
            return None
        
        # --- 時間解析 ---
        time_raw = raw_msg.get("time")
        time_parsed , time_conf = parse_message_time(time_raw , context)

        # --- 發話者正規化 ---
        speaker_raw = raw_msg.get("user_name")
        speaker_canonical = normalize_speaker(
            speaker_raw = speaker_raw, 
            side = side,
            chat_type = context.chat_type,
            chat_header = context.chat_header,
            )
        
        # --- 組裝 ---
        return NormalizedMessage(
            message_id = f"{file_hash}_{sequence}",
            source_file = context.source_file,
            sequence = sequence,
            content = content,
            content_type = _detect_content_type(content),
            speaker_raw = speaker_raw,
            speaker_canonical = speaker_canonical,
            side = side,
            is_self = (side == Side.RIGHT),
            time_raw = time_raw,
            time_parsed = time_parsed,
            time_confidence = time_conf,
            app_source = context.app_source,
            chat_header = context.chat_header,
            chat_type = context.chat_type,
            screenshot_taken_at = context.screenshot_taken_at,
            client_id = context.client_id,
        )
    except Exception as e:
        # # Pydantic ValidationError 或其他例外，記錄但不炸
        logger.warning(
            f"normalize failed at {context.source_file}[{sequence}]: {e}"
        )
        return None

TIME_PROPAGATION_MAX_DISTANCE = 5

def _propagate_times_within_screenshot(
        messages : list[NormalizedMessage],
):
    if not messages:
        return messages
    # current_anchor:目前持有的，最近一個 EXACT 錨點
    # anchor_idx:該錨點在 list 中的位置(用來算距離)
    current_anchor : Optional[NormalizedMessage] = None
    anchor_idx : Optional[int] = None
    propagated : list[NormalizedMessage] = []

    for i , m in enumerate(messages):
        # case 1 : 這則有EXACT時間，變成新錨點
        # use_enum_values = True , 讓 model 內存的是 str 要用 .value 比對
        if(
            m.time_parsed is not None
            and m.time_confidence == TimeConfidence.EXACT.value
        ):
            current_anchor = m
            anchor_idx = i
            propagated.append(m)
            continue
        # case 2 : 已有 INFERRED 時間 (假如有人手動標記，不應該覆蓋)
        if m.time_parsed is not None:
            propagated.append(m)
            continue
        # case 3 : time_parsed 是 None , 需要補時間
        if current_anchor is None or anchor_idx is None:
            propagated.append(m)
            continue
        # 超過距離上限，維持None
        distance = i - anchor_idx
        if distance > TIME_PROPAGATION_MAX_DISTANCE:
            propagated.append(m)
            continue
        # 安全範圍內，套用錨點時間
        propagated.append(m.model_copy(update={
            "time_parsed": current_anchor.time_parsed,
            "time_confidence": TimeConfidence.INFERRED.value,
        }))
    return propagated

def normalize(observation : dict) -> list[NormalizedMessage]:
    """
    把單一截圖的 observation 轉成 NormalizedMessage list，
    observation 為 extract_observations 輸出，必須是 chat_detail 類型
    """
    if observation.get("screen_type") != "chat_detail":
        return []
    
    context = _build_screenshot_context(observation)
    file_hash = _file_hash_short(context.source_file)

    results : list[NormalizedMessage] = []
    for seq , raw_msg in enumerate(observation.get("messages" , [])):
        msg = _normalize_one_message(raw_msg , seq , context , file_hash)
        if msg is not None:
            results.append(msg)

    results = _propagate_times_within_screenshot(results)
    return results
