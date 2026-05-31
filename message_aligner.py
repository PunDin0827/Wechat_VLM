"""
message_aligner.py — 訊息列表的軟對齊器
============================================
解決 eval_pipeline.py 用序號對齊造成的「錯位連鎖錯誤」。

核心演算法:Needleman-Wunsch 全域序列對齊
─────────────────────────────────────
1. 建立 2D 代價矩陣 D[i][j]
2. 動態規劃填表:每格取「對角配對/跳過 gold/跳過 pred」三者最小
3. 從右下角回溯,還原最佳配對方式

對外接口:
- align_messages(pred_msgs, gold_msgs) → AlignmentResult
- align_friends(pred_friends, gold_friends) → AlignmentResult

使用範例:
    from message_aligner import align_messages
    result = align_messages(pred_normalized_msgs, gold_msg_labels)
    for pair in result.pairs:
        if pair.is_match:
            # 比對 pair.pred 和 pair.gold
            ...
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Callable, Any
import unicodedata
import re


# ============================================================
# §1  文字相似度計算
# ============================================================

def _normalize_text(text: Optional[str]) -> str:
    """
    文字正規化:消除全半形、零寬字元、多餘空白等雜訊。
    
    這個函式和 output_parser.normalize_text 同步,確保
    評測時的文字比對和系統去重時用同一套標準。
    """
    if not text:
        return ""
    # NFKC: 全形數字/字母轉半形,相容性合併
    text = unicodedata.normalize("NFKC", str(text))
    # 移除常見的零寬字元(VLM 偶爾會吐出來)
    text = re.sub(r'[\u200b\u200c\u200d\ufeff]', '', text)
    # 多空白合併為單一空白
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _levenshtein_distance(a: str, b: str) -> int:
    """
    計算兩字串的 Levenshtein 距離(編輯距離)。
    
    定義:把 a 變成 b 所需的最少「插入/刪除/替換」次數。
    例:
        levenshtein("abc", "abc")  = 0   完全相同
        levenshtein("abc", "abd")  = 1   一次替換
        levenshtein("abc", "ab")   = 1   一次刪除
        levenshtein("abc", "abcd") = 1   一次插入
    
    複雜度:O(len(a) × len(b)),對訊息泡泡(通常 < 100 字)綽綽有餘。
    """
    if not a:
        return len(b)
    if not b:
        return len(a)
    
    # dp[i][j] = a[:i] 變成 b[:j] 的編輯距離
    # 為節省空間,只保留兩列(prev / curr)
    prev = list(range(len(b) + 1))   # 第 0 列:空字串 → b[:j] 需要 j 次插入
    
    for i in range(1, len(a) + 1):
        curr = [i] + [0] * len(b)    # 每列開頭:a[:i] → 空字串 需要 i 次刪除
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1]          # 字元相同,無需操作
            else:
                # 取「刪除/插入/替換」三者最小,+1 代表本次操作
                curr[j] = 1 + min(
                    prev[j],         # 刪除 a[i-1]
                    curr[j - 1],     # 插入 b[j-1]
                    prev[j - 1],     # 替換 a[i-1] 為 b[j-1]
                )
        prev = curr
    
    return prev[len(b)]


def text_similarity(a: Optional[str], b: Optional[str]) -> float:
    """
    回傳 0.0~1.0 的相似度,越接近 1 越相似。
    
    定義:1 - (編輯距離 / 較長字串的長度)
    
    特殊處理:
    - 兩者都空 → 1.0(都「沒填寫」算同類)
    - 一空一不空 → 0.0(只有一邊填了視為完全不同)
    - 正規化後完全相等 → 1.0(快速路徑,跳過編輯距離計算)
    """
    a_norm = _normalize_text(a)
    b_norm = _normalize_text(b)
    
    if not a_norm and not b_norm:
        return 1.0
    if not a_norm or not b_norm:
        return 0.0
    if a_norm == b_norm:
        return 1.0
    
    max_len = max(len(a_norm), len(b_norm))
    dist = _levenshtein_distance(a_norm, b_norm)
    return max(0.0, 1.0 - dist / max_len)


# ============================================================
# §2  訊息間的相似度函式(cost function)
# ============================================================

# 各欄位的權重設定 — 加總必須為 1.0
# 設計理由:
#   content 是最直接的識別資訊,給最高權重
#   side 不一致幾乎不可能是同一則,給高權重
#   time_raw 可能因 VLM 漏抓而 None,給中等權重(避免過度懲罰)
#   speaker 若 content 對了多半也對,給最低權重
MSG_WEIGHTS = {
    "content": 0.50,
    "side":    0.20,
    "time":    0.20,
    "speaker": 0.10,
}


def _get_field(obj: Any, name: str) -> Any:
    """
    統一取欄位的小工具。
    
    為什麼需要?系統輸出的訊息是 NormalizedMessage(Pydantic 物件,用 .attr),
    人工標籤是 MessageLabel(也是 Pydantic),
    但有時也可能傳進 dict(例如解析後還沒包裝)。
    這個 helper 讓兩種都能用,降低呼叫端的負擔。
    """
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def message_similarity(pred_msg: Any, gold_msg: Any) -> float:
    """
    計算兩則訊息的相似度,回傳 0.0~1.0。
    
    分數 = Σ(各欄位相似度 × 權重)
    
    pred_msg / gold_msg 可以是:
    - NormalizedMessage(系統側,Pydantic)
    - MessageLabel(標籤側,Pydantic)
    - dict(便於測試)
    """
    # ── content:用 Levenshtein 相似度,容忍 OCR 雜訊 ──
    content_sim = text_similarity(
        _get_field(pred_msg, "content"),
        _get_field(gold_msg, "content"),
    )
    
    # ── side:嚴格比對,因為值很短(left/right)沒有模糊空間 ──
    p_side = str(_get_field(pred_msg, "side") or "").lower()
    g_side = str(_get_field(gold_msg, "side") or "").lower()
    side_sim = 1.0 if p_side == g_side else 0.0
    
    # ── time_raw:正規化後嚴格比對 ──
    # 不用 Levenshtein 因為時間字串短,小差異意義大(15:42 vs 15:43 是不同時間)
    p_time = _normalize_text(_get_field(pred_msg, "time_raw"))
    g_time = _normalize_text(_get_field(gold_msg, "time_raw"))
    if not p_time and not g_time:
        time_sim = 1.0      # 都沒時間視為同類
    elif not p_time or not g_time:
        time_sim = 0.5      # 一邊有一邊沒有,不強烈懲罰(VLM 常漏時間)
    else:
        time_sim = 1.0 if p_time == g_time else 0.0
    
    # ── speaker:輕度容錯的相等比對 ──
    p_speaker = _normalize_text(_get_field(pred_msg, "speaker_canonical"))
    g_speaker = _normalize_text(_get_field(gold_msg, "speaker_canonical"))
    if not p_speaker and not g_speaker:
        speaker_sim = 1.0
    elif not p_speaker or not g_speaker:
        speaker_sim = 0.5
    else:
        speaker_sim = 1.0 if p_speaker == g_speaker else 0.0
    
    return (
        content_sim * MSG_WEIGHTS["content"]
        + side_sim    * MSG_WEIGHTS["side"]
        + time_sim    * MSG_WEIGHTS["time"]
        + speaker_sim * MSG_WEIGHTS["speaker"]
    )


def friend_similarity(pred_friend: Any, gold_friend: Any) -> float:
    """
    計算兩筆好友資料的相似度。
    
    friend_list 沒有 side / speaker 概念,改用三欄位的 Levenshtein 加權。
    """
    name_sim = text_similarity(
        _get_field(pred_friend, "display_name"),
        _get_field(gold_friend, "display_name"),
    )
    msg_sim = text_similarity(
        _get_field(pred_friend, "last_message"),
        _get_field(gold_friend, "last_message"),
    )
    time_sim = text_similarity(
        _get_field(pred_friend, "last_message_time_raw"),
        _get_field(gold_friend, "last_message_time_raw"),
    )
    # display_name 最重要(其他兩個可能 None),給最高權重
    return 0.6 * name_sim + 0.25 * msg_sim + 0.15 * time_sim


# ============================================================
# §3  對齊結果的資料結構
# ============================================================

@dataclass
class AlignmentPair:
    """
    一個配對結果。
    
    狀態組合:
    - is_match    True  + pred not None + gold not None  → 配對成功
    - is_extra    True  + pred not None + gold is None   → 系統多抓
    - is_missed   True  + pred is None  + gold not None  → 系統漏抓
    """
    pred:       Optional[Any] = None     # 系統側的訊息物件,None 代表這位置系統沒抓
    gold:       Optional[Any] = None     # 標籤側的訊息物件,None 代表這位置標籤沒寫
    similarity: float = 0.0              # 配對成功時的相似度分數,0~1
    pred_idx:   Optional[int] = None     # 在原 pred_msgs list 中的 index(供 debug)
    gold_idx:   Optional[int] = None     # 在原 gold_msgs list 中的 index
    
    @property
    def is_match(self) -> bool:
        """是否成功配對(雙方都不是 None)。"""
        return self.pred is not None and self.gold is not None
    
    @property
    def is_extra(self) -> bool:
        """系統多抓:有 pred 但無 gold。"""
        return self.pred is not None and self.gold is None
    
    @property
    def is_missed(self) -> bool:
        """系統漏抓:有 gold 但無 pred。"""
        return self.pred is None and self.gold is not None


@dataclass
class AlignmentResult:
    """
    完整對齊結果。
    
    呼叫端可用 pairs 逐個取出配對,或用 matched/extras/missed 取出特定類型。
    """
    pairs:       list[AlignmentPair] = field(default_factory=list)
    n_pred:      int = 0
    n_gold:      int = 0
    total_cost:  float = 0.0    # 對齊的總代價,越低代表對得越好
    
    @property
    def n_matched(self) -> int:
        return sum(1 for p in self.pairs if p.is_match)
    
    @property
    def n_extra(self) -> int:
        return sum(1 for p in self.pairs if p.is_extra)
    
    @property
    def n_missed(self) -> int:
        return sum(1 for p in self.pairs if p.is_missed)
    
    @property
    def matched_pairs(self) -> list[AlignmentPair]:
        """只取成功配對的 pairs,通常下游做欄位比對時只用這些。"""
        return [p for p in self.pairs if p.is_match]


# ============================================================
# §4  核心:Needleman-Wunsch 對齊演算法
# ============================================================

# 跳過(gap)的代價 — 大於最差配對代價,才會優先選配對
# 設 0.6:當配對相似度 < 0.4 時,系統會選擇「跳過」而非「強迫配對」
GAP_COST = 0.6

# 配對門檻:相似度低於此值,即使對齊上了也會在後處理時拆成 extra+missed
# 設 0.3:防止「兩則完全不同的訊息被勉強對齊」
MATCH_THRESHOLD = 0.3


def _align_with_similarity(
    pred_list: list[Any],
    gold_list: list[Any],
    similarity_fn: Callable[[Any, Any], float],
) -> AlignmentResult:
    """
    通用對齊核心:給定相似度函式,執行 Needleman-Wunsch。
    
    參數:
        pred_list:      系統輸出的物件列表
        gold_list:      標籤的物件列表
        similarity_fn:  二元函式,(pred, gold) → 0~1 相似度
    
    回傳:
        AlignmentResult,內含逐個配對結果
    """
    n = len(gold_list)
    m = len(pred_list)
    
    # ── 邊界情況:有一邊是空的 ──
    # 全部當成「漏抓」或「多抓」,不用跑 DP
    if n == 0 and m == 0:
        return AlignmentResult(pairs=[], n_pred=0, n_gold=0, total_cost=0.0)
    if n == 0:
        # gold 是空的,所有 pred 都是 extra
        pairs = [
            AlignmentPair(pred=p, pred_idx=i)
            for i, p in enumerate(pred_list)
        ]
        return AlignmentResult(
            pairs=pairs, n_pred=m, n_gold=0,
            total_cost=m * GAP_COST,
        )
    if m == 0:
        # pred 是空的,所有 gold 都是 missed
        pairs = [
            AlignmentPair(gold=g, gold_idx=j)
            for j, g in enumerate(gold_list)
        ]
        return AlignmentResult(
            pairs=pairs, n_pred=0, n_gold=n,
            total_cost=n * GAP_COST,
        )
    
    # ── Step 1:建立代價矩陣 ──
    # D[i][j] = 對齊 gold[:i] 和 pred[:j] 的最小總代價
    # 大小 (n+1) × (m+1),多一行一列代表「空前綴」
    D = [[0.0] * (m + 1) for _ in range(n + 1)]
    
    # 初始化第 0 列(對 pred[:j]):
    # gold 是空的,所有 pred[:j] 都當成 extra,代價 = j × GAP_COST
    for j in range(m + 1):
        D[0][j] = j * GAP_COST
    
    # 初始化第 0 行(對 gold[:i]):
    # pred 是空的,所有 gold[:i] 都當成 missed,代價 = i × GAP_COST
    for i in range(n + 1):
        D[i][0] = i * GAP_COST
    
    # ── Step 2:填表(動態規劃)──
    # 每格的值來自三個方向:對角(配對)、上方(跳過 gold)、左方(跳過 pred)
    # 同時記錄選擇了哪個方向,供回溯使用
    # backtrack[i][j] ∈ {"diag", "up", "left"}
    backtrack = [["" for _ in range(m + 1)] for _ in range(n + 1)]
    
    # 第 0 列/行的 backtrack
    for i in range(1, n + 1):
        backtrack[i][0] = "up"      # 只能從上方來(全 missed)
    for j in range(1, m + 1):
        backtrack[0][j] = "left"    # 只能從左方來(全 extra)
    
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            # 計算配對代價:1 - similarity,讓「越像越便宜」
            sim = similarity_fn(pred_list[j - 1], gold_list[i - 1])
            match_cost = 1.0 - sim
            
            # 三個來源
            diag = D[i - 1][j - 1] + match_cost    # 對角:配對 gold[i-1] ↔ pred[j-1]
            up   = D[i - 1][j]     + GAP_COST       # 上方:gold[i-1] 是 missed
            left = D[i][j - 1]     + GAP_COST       # 左方:pred[j-1] 是 extra
            
            # 取最小值,並記錄方向
            # 同分時優先順序:diag > up > left(傾向於配對)
            if diag <= up and diag <= left:
                D[i][j] = diag
                backtrack[i][j] = "diag"
            elif up <= left:
                D[i][j] = up
                backtrack[i][j] = "up"
            else:
                D[i][j] = left
                backtrack[i][j] = "left"
    
    # ── Step 3:回溯路徑 ──
    # 從右下角 (n, m) 沿著 backtrack 走到左上角 (0, 0),
    # 沿途記錄每一步的配對方式
    pairs_reversed = []
    i, j = n, m
    while i > 0 or j > 0:
        direction = backtrack[i][j]
        
        if direction == "diag":
            # 配對:gold[i-1] ↔ pred[j-1]
            sim = similarity_fn(pred_list[j - 1], gold_list[i - 1])
            
            # 後處理:如果相似度太低,拆成 extra + missed
            # 這避免「兩則完全不同的訊息被勉強對齊」的偽匹配
            if sim < MATCH_THRESHOLD:
                pairs_reversed.append(AlignmentPair(
                    pred=pred_list[j - 1], pred_idx=j - 1,
                    similarity=sim,
                ))
                pairs_reversed.append(AlignmentPair(
                    gold=gold_list[i - 1], gold_idx=i - 1,
                    similarity=sim,
                ))
            else:
                pairs_reversed.append(AlignmentPair(
                    pred=pred_list[j - 1], pred_idx=j - 1,
                    gold=gold_list[i - 1], gold_idx=i - 1,
                    similarity=sim,
                ))
            i -= 1
            j -= 1
        elif direction == "up":
            # gold[i-1] 是 missed
            pairs_reversed.append(AlignmentPair(
                gold=gold_list[i - 1], gold_idx=i - 1,
            ))
            i -= 1
        else:  # "left"
            # pred[j-1] 是 extra
            pairs_reversed.append(AlignmentPair(
                pred=pred_list[j - 1], pred_idx=j - 1,
            ))
            j -= 1
    
    # 因為是從右下回溯到左上,要反轉才是正向順序
    pairs = list(reversed(pairs_reversed))
    
    return AlignmentResult(
        pairs=pairs,
        n_pred=m,
        n_gold=n,
        total_cost=D[n][m],
    )


# ============================================================
# §5  對外接口:align_messages / align_friends
# ============================================================

def align_messages(
    pred_msgs: list[Any],
    gold_msgs: list[Any],
) -> AlignmentResult:
    """
    對齊兩個訊息列表,回傳最佳配對方式。
    
    用法範例:
        from message_aligner import align_messages
        
        result = align_messages(pred_normalized_msgs, gold_msg_labels)
        
        print(f"配對成功: {result.n_matched}")
        print(f"系統多抓: {result.n_extra}")
        print(f"系統漏抓: {result.n_missed}")
        
        for pair in result.matched_pairs:
            # 對成功配對的進行欄位比對
            if pair.pred.content == pair.gold.content:
                ...
    """
    return _align_with_similarity(pred_msgs, gold_msgs, message_similarity)


def align_friends(
    pred_friends: list[Any],
    gold_friends: list[Any],
) -> AlignmentResult:
    """對齊兩個好友列表,用 friend_similarity 作為相似度函式。"""
    return _align_with_similarity(pred_friends, gold_friends, friend_similarity)


# ============================================================
# §6  自我測試:確保演算法正確
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  message_aligner.py 自我測試")
    print("=" * 60)
    
    # ── 測試 1:完全相同 ──
    pred = [
        {"side": "left",  "content": "你好",   "speaker_canonical": None,    "time_raw": "下午 3:42"},
        {"side": "right", "content": "嗨!",   "speaker_canonical": "發起人", "time_raw": "下午 3:43"},
    ]
    gold = [
        {"side": "left",  "content": "你好",   "speaker_canonical": None,    "time_raw": "下午 3:42"},
        {"side": "right", "content": "嗨!",   "speaker_canonical": "發起人", "time_raw": "下午 3:43"},
    ]
    r = align_messages(pred, gold)
    print(f"\n[測試 1] 完全相同")
    print(f"  matched={r.n_matched}, extra={r.n_extra}, missed={r.n_missed}")
    assert r.n_matched == 2 and r.n_extra == 0 and r.n_missed == 0
    
    # ── 測試 2:系統漏抓第一則(原本會錯位的場景)──
    pred = [
        {"side": "right", "content": "嗨!",   "speaker_canonical": "發起人", "time_raw": "下午 3:43"},
        {"side": "left",  "content": "在嗎",  "speaker_canonical": None,    "time_raw": "下午 3:45"},
    ]
    gold = [
        {"side": "left",  "content": "你好",   "speaker_canonical": None,    "time_raw": "下午 3:42"},
        {"side": "right", "content": "嗨!",   "speaker_canonical": "發起人", "time_raw": "下午 3:43"},
        {"side": "left",  "content": "在嗎",  "speaker_canonical": None,    "time_raw": "下午 3:45"},
    ]
    r = align_messages(pred, gold)
    print(f"\n[測試 2] 漏抓第一則(原本會錯位)")
    print(f"  matched={r.n_matched}, extra={r.n_extra}, missed={r.n_missed}")
    print(f"  總代價={r.total_cost:.2f}")
    # 期待:2 個配對成功、0 個多抓、1 個漏抓(第一則「你好」)
    assert r.n_matched == 2, f"期待 2 個配對,實際 {r.n_matched}"
    assert r.n_missed == 1, f"期待 1 個漏抓,實際 {r.n_missed}"
    for pair in r.pairs:
        if pair.is_missed:
            print(f"  → 漏抓: gold[{pair.gold_idx}] = {pair.gold['content']}")
    
    # ── 測試 3:OCR 細節差異(typo) ──
    pred = [
        {"side": "left", "content": "你好嗎", "speaker_canonical": None, "time_raw": None},
    ]
    gold = [
        {"side": "left", "content": "你好嗎?", "speaker_canonical": None, "time_raw": None},
    ]
    r = align_messages(pred, gold)
    print(f"\n[測試 3] OCR 細節差異")
    print(f"  matched={r.n_matched}, similarity={r.pairs[0].similarity:.3f}")
    assert r.n_matched == 1, "OCR 細節差異應該還是能配對成功"
    
    # ── 測試 4:系統多抓(把日期分隔線當訊息) ──
    pred = [
        {"side": "left", "content": "你好",            "speaker_canonical": None, "time_raw": "下午 3:42"},
        {"side": "left", "content": "2024年10月25日",  "speaker_canonical": None, "time_raw": None},  # 誤抓
        {"side": "right","content": "嗨!",            "speaker_canonical": "發起人", "time_raw": "下午 3:43"},
    ]
    gold = [
        {"side": "left", "content": "你好",            "speaker_canonical": None, "time_raw": "下午 3:42"},
        {"side": "right","content": "嗨!",            "speaker_canonical": "發起人", "time_raw": "下午 3:43"},
    ]
    r = align_messages(pred, gold)
    print(f"\n[測試 4] 系統多抓日期分隔線")
    print(f"  matched={r.n_matched}, extra={r.n_extra}, missed={r.n_missed}")
    for pair in r.pairs:
        if pair.is_extra:
            print(f"  → 多抓: pred[{pair.pred_idx}] = {pair.pred['content']}")
    assert r.n_matched == 2 and r.n_extra == 1 and r.n_missed == 0
    
    # ── 測試 5:全空列表 ──
    r = align_messages([], [])
    print(f"\n[測試 5] 雙空列表")
    print(f"  pairs={len(r.pairs)}")
    assert len(r.pairs) == 0
    
    # ── 測試 6:Levenshtein 相似度 ──
    print(f"\n[測試 6] 文字相似度")
    print(f"  '你好嗎' vs '你好嗎?'  = {text_similarity('你好嗎', '你好嗎?'):.3f}")
    print(f"  '早安'   vs '晚安'    = {text_similarity('早安', '晚安'):.3f}")
    print(f"  '完全不同' vs '毫無關係' = {text_similarity('完全不同', '毫無關係'):.3f}")
    print(f"  ''     vs ''        = {text_similarity('', ''):.3f}")
    print(f"  '你好'   vs None      = {text_similarity('你好', None):.3f}")
    
    print("\n" + "=" * 60)
    print("  ✅ 所有測試通過")
    print("=" * 60)
