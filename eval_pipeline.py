"""
eval_pipeline.py — 混淆矩陣 & 欄位級精準度評測(軟對齊版本,解法 A)
================================================================
與 v1 的差異:
  - 訊息和好友列表的配對,從「序號對齊」改為「軟對齊」
  - 解決漏抓/多抓造成的「錯位連鎖錯誤」問題
  - 新增 extra / missed 細項統計

★ 解法 A 改動:
  - 欄位分母固定 = len(gold),不再隨配對結果浮動
  - 漏抓(missed)的 gold 對所有欄位都算「不 match」
  - 多抓(extra)不進入欄位分母(因為沒有 gold 可以對)
  - 這樣可確保「同一份 golden sample」下,評測分母完全可重現

與 v2 的差異:
  - 多了 pred_time_msgs 參數,可從 normalized_messages.json 覆蓋 time 欄位
  - time_raw 用「只比時分」的寬鬆規則(_time_matches_by_hour_minute)

輸入:
  - 系統輸出(parse_screen_output / normalize 的結果)
  - 人工標籤(GoldenLabel)
  - 可選: normalized_messages.json
    只用來覆蓋 chat_detail 訊息的 time 欄位，其餘 prediction 仍維持 result_sum.json 的解析結果

輸出:
  - Stage 1 混淆矩陣(screen_type / app_source / chat_type)
  - Stage 2 欄位級 Accuracy / Precision / Recall / F1
  - 系統多抓 / 漏抓統計
  - 每張圖的逐項 diff 報告

執行:
  python -m src.evaluation.eval_pipeline --labels ./examples/labels --results ./examples/sample_prediction.json
"""

from __future__ import annotations

import json
import os
import argparse
import sys
import re
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np

# ---------- 本專案模組 ----------
from golden_label import (
    GoldenLabel,
    ScreenType,
    AppSource,
    ChatType,
    load_all_labels,
)
from output_parser import parse_screen_output, normalize_text
from normalizer import normalize, NormalizedMessage

# ★ 新增:軟對齊模組
from message_aligner import (
    align_messages,
    align_friends,
    AlignmentResult,
)


# ============================================================
# §1  混淆矩陣工具(維持與 v1 相同)
# ============================================================

class ConfusionMatrix:
    """多類別混淆矩陣,支援任意 label 集合。"""

    def __init__(self, labels: list[str]):
        self.labels = labels
        self._idx = {l: i for i, l in enumerate(labels)}
        n = len(labels)
        self._mat = np.zeros((n, n), dtype=int)

    def add(self, pred: str, true: str) -> None:
        ti = self._idx.get(true, -1)
        pi = self._idx.get(pred, -1)
        if ti == -1 or pi == -1:
            return
        self._mat[ti][pi] += 1

    def _per_class(self) -> dict[str, dict]:
        result = {}
        for i, label in enumerate(self.labels):
            tp = self._mat[i][i]
            fp = self._mat[:, i].sum() - tp
            fn = self._mat[i, :].sum() - tp
            support = self._mat[i, :].sum()
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0 else 0.0
            )
            result[label] = dict(
                tp=int(tp), fp=int(fp), fn=int(fn),
                support=int(support),
                precision=round(precision, 4),
                recall=round(recall, 4),
                f1=round(f1, 4),
            )
        return result

    def accuracy(self) -> float:
        total = self._mat.sum()
        return float(self._mat.diagonal().sum() / total) if total > 0 else 0.0

    def report(self, title: str = "") -> str:
        lines = []
        if title:
            lines.append(f"\n{'='*55}")
            lines.append(f"  {title}")
            lines.append(f"{'='*55}")

        col_w = max(len(l) for l in self.labels) + 2
        header = " " * col_w + "".join(l.ljust(col_w) for l in self.labels)
        lines.append("  (行=True, 列=Pred)")
        lines.append(header)
        for i, label in enumerate(self.labels):
            row = label.ljust(col_w) + "".join(
                str(self._mat[i][j]).ljust(col_w) for j in range(len(self.labels))
            )
            lines.append(row)

        lines.append("")
        lines.append(f"  {'Label':<18} {'Prec':>6} {'Rec':>6} {'F1':>6} {'Sup':>6}")
        lines.append(f"  {'-'*44}")
        per = self._per_class()
        for label, m in per.items():
            lines.append(
                f"  {label:<18} {m['precision']:>6.3f} {m['recall']:>6.3f}"
                f" {m['f1']:>6.3f} {m['support']:>6}"
            )
        lines.append(f"  {'─'*44}")
        lines.append(f"  Overall Accuracy: {self.accuracy():.4f}  (N={self._mat.sum()})")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "labels": self.labels,
            "matrix": self._mat.tolist(),
            "per_class": self._per_class(),
            "accuracy": self.accuracy(),
        }


# ============================================================
# §2  Stage 2 欄位級比對(★ 軟對齊改造)
# ============================================================

def _norm(text: Optional[str]) -> str:
    if not text:
        return ""
    return normalize_text(str(text))


def _get_field(obj, name):
    """統一從 Pydantic 物件或 dict 取欄位。"""
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _get_time_value(obj) -> Optional[str]:
    """
    從訊息物件取出可用的時間欄位。

    為了相容不同 normalized_messages.json 版本，依序嘗試:
    - time_raw: 既有評測欄位名稱
    - time: 常見的 normalized time 欄位名稱
    - normalized_time / time_normalized: 明確標示已正規化的時間欄位
    """
    for field in ("time_parsed","normalized_time", "time_normalized"):
        value = _get_field(obj, field)
        if value not in (None, ""):
            return str(value)
    return None


def _get_by_index(items: Optional[list], idx: Optional[int]):
    """安全地用 index 取 list item，避免 normalized_messages 數量不一致時中斷評測。"""
    if items is None or idx is None:
        return None
    if idx < 0 or idx >= len(items):
        return None
    return items[idx]


def _extract_hour_minute(value: Optional[str]) -> Optional[tuple[int, int]]:
    """
    從時間字串抽出「小時、分鐘」，忽略年月日。

    支援例子:
    - "下午 3:42"        -> (15, 42)
    - "15:42"           -> (15, 42)
    - "2024/10/25 15:42" -> (15, 42)，日期會被忽略
    - "上午12:05"        -> (0, 5)
    - "下午12:05"        -> (12, 5)
    - "3點42分"          -> (3, 42)

    注意:
    - 純日期如 "10/25" 不會被誤判成 10:25，因為這裡只接受冒號時間或中文「點/時」格式。
    - 如果沒有可辨識的小時/分鐘，回傳 None。
    """
    if value in (None, ""):
        return None

    text = _norm(str(value))
    if not text:
        return None

    # 判斷上午/下午語意。英文 AM/PM 也一起支援。
    has_pm = bool(re.search(r"(下午|晚上|傍晚|PM|pm|p\.m\.)", text))
    has_am = bool(re.search(r"(上午|早上|凌晨|AM|am|a\.m\.)", text))

    hour = minute = None

    # 優先抓冒號格式: 15:42, 15：42, 2024-10-25T15:42:00
    # (?<!\d) 避免抓到更長數字中間；分鐘限制 00-59。
    colon_match = re.search(r"(?<!\d)(\d{1,2})\s*[:：]\s*([0-5]\d)(?!\d)", text)
    if colon_match:
        hour = int(colon_match.group(1))
        minute = int(colon_match.group(2))
    else:
        # 中文格式: 3點42分 / 3時42分 / 3 点 42
        zh_match = re.search(r"(?<!\d)(\d{1,2})\s*[點点時时]\s*([0-5]?\d)?\s*分?", text)
        if zh_match:
            hour = int(zh_match.group(1))
            minute = int(zh_match.group(2) or 0)

    if hour is None or minute is None:
        return None

    # 合法性檢查。12 小時制允許 1~12；24 小時制允許 0~23。
    if has_am or has_pm:
        if hour < 1 or hour > 12:
            return None
        if has_pm and hour != 12:
            hour += 12
        if has_am and hour == 12:
            hour = 0
    else:
        if hour < 0 or hour > 23:
            return None

    return hour, minute


def _time_matches_by_hour_minute(pred_value: Optional[str], gold_value: Optional[str]) -> bool:
    """
    時間欄位寬鬆比對：只比較小時與分鐘，忽略年月日。

    規則:
    1. pred/gold 都沒有時間 -> match
    2. 其中一邊沒有時間 -> mismatch
    3. 兩邊都可抽出 HH:MM -> 只比較 HH:MM
    4. 非空但抽不出 HH:MM -> mismatch，避免把純日期或其他文字誤算為正確
    """
    pred_empty = pred_value in (None, "") or _norm(str(pred_value)) == ""
    gold_empty = gold_value in (None, "") or _norm(str(gold_value)) == ""

    if pred_empty and gold_empty:
        return True
    if pred_empty or gold_empty:
        return False

    pred_hm = _extract_hour_minute(pred_value)
    gold_hm = _extract_hour_minute(gold_value)
    if pred_hm is None or gold_hm is None:
        return False

    return pred_hm == gold_hm


def _extract_normalized_message_list(entry) -> list:
    """
    從 normalized_messages.json 的單筆 entry 中取出訊息列表。

    支援幾種常見格式:
    1. {"file_name": "xxx.png", "messages": [...]}
    2. {"file_name": "xxx.png", "normalized_messages": [...]}
    3. {"file_name": "xxx.png", "stage2": {"messages": [...]}}
    4. 直接就是 list[message]
    """
    if isinstance(entry, list):
        return entry
    if not isinstance(entry, dict):
        return []

    for key in ("normalized_messages", "messages", "chat_messages", "items"):
        value = entry.get(key)
        if isinstance(value, list):
            return value

    stage2 = entry.get("stage2")
    if isinstance(stage2, dict) and isinstance(stage2.get("messages"), list):
        return stage2["messages"]

    return []


def _load_normalized_messages_map(normalized_json: Optional[str]) -> dict[str, list]:
    """
    載入 normalized_messages.json，建立 file_name -> normalized message list 的索引。

    會同時放入原始 file_name 與 basename 兩種 key，避免 label.source_file
    是完整路徑、result_sum.json 只有檔名時對不到。
    """
    if not normalized_json:
        return {}

    path = Path(normalized_json)
    if not path.exists():
        raise FileNotFoundError(f"normalized_messages.json 不存在: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    mapping: dict[str, list] = {}

    def put_mapping(file_name: Optional[str], messages: list) -> None:
        if not file_name:
            return
        mapping[file_name] = messages
        mapping[Path(file_name).name] = messages

    # 格式 A: {"xxx.png": [...], "yyy.png": [...]}
    if isinstance(data, dict):
        container = data.get("results") or data.get("items") or data.get("data")
        if isinstance(container, list):
            data = container
        else:
            for key, value in data.items():
                messages = _extract_normalized_message_list(value)
                if messages:
                    put_mapping(key, messages)
            return mapping

    # 格式 B: [{"file_name": "xxx.png", "messages": [...]}, ...]
    if isinstance(data, list):
        for entry in data:
            if not isinstance(entry, dict):
                continue
            file_name = (
                entry.get("file_name")
                or entry.get("source_file")
                or entry.get("image")
                or entry.get("path")
            )
            messages = _extract_normalized_message_list(entry)
            if messages:
                put_mapping(file_name, messages)
        if not mapping and data and isinstance(data[0], dict) and "content" in data[0]:
            from collections import defaultdict as _defaultdict
            grouped = _defaultdict(list)
            for entry in data:
                if not isinstance(entry, dict):
                    continue
                fn = (
                    entry.get("file_name")
                    or entry.get("source_file")
                )
                if fn:
                    grouped[fn].append(entry)
            for fn, msgs in grouped.items():
                msgs.sort(key=lambda m: m.get("sequence", 0))
                put_mapping(fn, msgs) 

    return mapping


def _match_messages_soft(
    pred_msgs: list,
    gold_msgs: list,
    pred_time_msgs: Optional[list] = None,
) -> dict:
    """
    ★ 解法 A(v3 版):分母固定 = len(gold),保留 time 覆蓋邏輯。
    
    流程:
    1. 呼叫 align_messages 取得 AlignmentResult
       對齊仍用 result_sum.json 解析出的 pred_msgs,避免 normalized time 影響配對。
    2. 建立 gold_idx -> matched_pair 的查找表(O(1) lookup)
    3. 逐個遍歷 gold,不論有沒有配對都讓欄位 total += 1
       - 有配對 → 比對欄位,match 則 +1
       - 沒配對(漏抓) → 不 match,只貢獻 total
    4. time_raw 額外可從 normalized_messages.json 取值,並用「只比時分」的寬鬆規則
    
    這樣分母 = len(gold_msgs),完全不會隨系統輸出變動,
    解決原本「同樣 golden 但每次跑分母不同」的問題。
    """
    # 執行軟對齊(對齊邏輯本身不變)
    align_result = align_messages(pred_msgs, gold_msgs)
    
    # ── 步驟 1:建立 gold_idx → matched_pair 的快速查找表 ──
    gold_to_pair = {
        pair.gold_idx: pair
        for pair in align_result.matched_pairs
    }
    
    # ── 步驟 2:統計容器 ──
    stats = defaultdict(lambda: {"match": 0, "total": 0})
    
    # ── 步驟 3:逐個遍歷 gold(分母來源,固定為 len(gold))──
    for j, g in enumerate(gold_msgs):
        pair = gold_to_pair.get(j)   # 可能是 None(該 gold 被漏抓)
        p = pair.pred if pair else None
        pred_idx = pair.pred_idx if pair else None
        
        # side
        stats["side"]["total"] += 1
        if p is not None and _norm(_get_field(p, "side")) == _norm(_get_field(g, "side")):
            stats["side"]["match"] += 1
        
        # speaker_canonical
        stats["speaker"]["total"] += 1
        if p is not None and _norm(_get_field(p, "speaker_canonical")) == _norm(_get_field(g, "speaker_canonical")):
            stats["speaker"]["match"] += 1
        
        # content(嚴格相等)
        stats["content_exact"]["total"] += 1
        if p is not None and _norm(_get_field(p, "content")) == _norm(_get_field(g, "content")):
            stats["content_exact"]["match"] += 1
        
        # content_type
        stats["content_type"]["total"] += 1
        if p is not None and _norm(_get_field(p, "content_type")) == _norm(_get_field(g, "content_type")):
            stats["content_type"]["match"] += 1
        
        # time_raw(用寬鬆規則 + 可從 normalized_messages 覆蓋)
        stats["time_raw"]["total"] += 1
        if p is not None:
            # 若有提供 normalized_messages.json,時間 prediction 從該檔案取
            time_pred_obj = _get_by_index(pred_time_msgs, pred_idx) or p
            pred_time_value = _get_time_value(time_pred_obj)
            gold_time_value = str(_get_field(g, "time_raw") or "")
            if _time_matches_by_hour_minute(pred_time_value, gold_time_value):
                stats["time_raw"]["match"] += 1
    
    return {
        "n_pred":      align_result.n_pred,
        "n_gold":      align_result.n_gold,
        "n_paired":    align_result.n_matched,
        "extra_pred":  align_result.n_extra,
        "missed_gold": align_result.n_missed,
        "fields":      {k: dict(v) for k, v in stats.items()},
        # ★ 新增:對齊診斷資訊
        "alignment_cost": round(align_result.total_cost, 3),
        "alignment_details": [
            {
                "kind": "matched" if pair.is_match else ("extra" if pair.is_extra else "missed"),
                "similarity": round(pair.similarity, 3) if pair.is_match else None,
                "pred_idx": pair.pred_idx,
                "gold_idx": pair.gold_idx,
                "pred_content": _norm(_get_field(pair.pred, "content")) if pair.pred else None,
                "gold_content": _norm(_get_field(pair.gold, "content")) if pair.gold else None,
                "pred_time_for_eval": _norm(_get_time_value(_get_by_index(pred_time_msgs, pair.pred_idx) or pair.pred)) if pair.pred else None,
                "gold_time": _norm(_get_time_value(pair.gold)) if pair.gold else None,
                "pred_time_hm": _extract_hour_minute(_get_time_value(_get_by_index(pred_time_msgs, pair.pred_idx) or pair.pred)) if pair.pred else None,
                "gold_time_hm": _extract_hour_minute(_get_time_value(pair.gold)) if pair.gold else None,
            }
            for pair in align_result.pairs
        ],
        "time_source": "normalized_messages.json" if pred_time_msgs is not None else "result_sum.json",
        "time_match_rule": "compare hour/minute only; ignore month/day/date",
        # ★ 標明分母策略,方便日後追溯
        "denominator_policy": "gold_total",
    }


def _match_friends_soft(pred_friends: list, gold_friends: list) -> dict:
    """
    ★ 解法 A:friend_list 分母也固定 = len(gold_friends)。
    
    邏輯與 _match_messages_soft 一致,只是欄位不同。
    """
    align_result = align_friends(pred_friends, gold_friends)
    
    # 建立 gold_idx -> matched_pair 查找表
    gold_to_pair = {
        pair.gold_idx: pair
        for pair in align_result.matched_pairs
    }
    
    stats = defaultdict(lambda: {"match": 0, "total": 0})
    
    # 逐個遍歷 gold,分母固定 = len(gold_friends)
    for j, g in enumerate(gold_friends):
        pair = gold_to_pair.get(j)
        p = pair.pred if pair else None
        
        for field in ("display_name", "last_message", "last_message_time_raw"):
            stats[field]["total"] += 1
            if p is not None and _norm(_get_field(p, field)) == _norm(_get_field(g, field)):
                stats[field]["match"] += 1
    
    return {
        "n_pred":      align_result.n_pred,
        "n_gold":      align_result.n_gold,
        "n_paired":    align_result.n_matched,
        "extra_pred":  align_result.n_extra,
        "missed_gold": align_result.n_missed,
        "fields":      {k: dict(v) for k, v in stats.items()},
        "alignment_cost": round(align_result.total_cost, 3),
        "denominator_policy": "gold_total",
    }


# ============================================================
# §3  單張圖的完整比對
# ============================================================

def _get_system_output(result_entry: dict) -> dict:
    return parse_screen_output(result_entry.get("output_text", ""))


def evaluate_one(
    label: GoldenLabel,
    system_parsed: dict,
    system_normalized: list[NormalizedMessage],
    normalized_time_messages: Optional[list] = None,
) -> dict:
    """單張圖的評測結果。"""
    report = {
        "source_file":  label.source_file,
        "annotator":    label.annotator,
        "confidence":   label.annotation_confidence,
        "stage1":       {},
        "stage2":       None,
        "notes":        label.notes,
    }

    # ── Stage 1 比對 ──
    pred_screen = (system_parsed.get("screen_type") or "unknown").lower()
    pred_app    = (system_parsed.get("app_source")  or "unknown")
    pred_chat   = (system_parsed.get("chat_type")   or "unknown").lower()

    gold_screen = label.stage1.screen_type
    gold_app    = label.stage1.app_source
    gold_chat   = label.stage1.chat_type

    report["stage1"] = {
        "screen_type": {
            "pred": pred_screen, "gold": gold_screen,
            "match": pred_screen == gold_screen,
        },
        "app_source": {
            "pred": pred_app, "gold": gold_app,
            "match": _norm(pred_app) == _norm(gold_app),
        },
        "chat_type": {
            "pred": pred_chat, "gold": gold_chat,
            "match": pred_chat == gold_chat,
        },
    }

    # ── Stage 2 比對(★ 改用軟對齊版本)──
    screen_type = label.stage1.screen_type
    if screen_type == "chat_detail":
        report["stage2"] = _match_messages_soft(
            system_normalized,
            label.stage2.messages,
            pred_time_msgs=normalized_time_messages,
        )
    elif screen_type == "friend_list":
        pred_friends = system_parsed.get("friends", [])
        report["stage2"] = _match_friends_soft(
            pred_friends,
            label.stage2.friends,
        )

    return report


# ============================================================
# §4  批次評測 & 彙總
# ============================================================

SCREEN_LABELS = ["chat_detail", "chat_list", "friend_list", "other", "unknown"]
APP_LABELS    = ["LINE", "WeChat", "unknown"]
CHAT_LABELS   = ["direct", "group", "unknown", "n/a"]


class EvalSummary:
    """彙總所有標籤圖片的評測結果。"""

    def __init__(self):
        self.cm_screen = ConfusionMatrix(SCREEN_LABELS)
        self.cm_app    = ConfusionMatrix(APP_LABELS)
        self.cm_chat   = ConfusionMatrix(CHAT_LABELS)
        self.reports: list[dict] = []

        # chat_detail 累計
        self._s2_msg_field_totals = defaultdict(lambda: {"match": 0, "total": 0})
        self._s2_msg_count_errors = []
        # ★ 新增:多抓 / 漏抓累計
        self._s2_msg_extras = 0
        self._s2_msg_missed = 0
        self._s2_msg_paired = 0
        self._s2_msg_total_gold = 0
        self._s2_msg_total_pred = 0

        # friend_list 累計
        self._s2_friend_field_totals = defaultdict(lambda: {"match": 0, "total": 0})
        self._s2_friend_extras = 0
        self._s2_friend_missed = 0
        self._s2_friend_paired = 0

    def add(self, report: dict) -> None:
        self.reports.append(report)

        s1 = report["stage1"]
        self.cm_screen.add(pred=s1["screen_type"]["pred"], true=s1["screen_type"]["gold"])
        self.cm_app.add(   pred=s1["app_source"]["pred"],  true=s1["app_source"]["gold"])
        self.cm_chat.add(  pred=s1["chat_type"]["pred"],   true=s1["chat_type"]["gold"])

        s2 = report.get("stage2")
        if not s2 or "fields" not in s2:
            return
        
        # 判斷是 chat_detail 還是 friend_list(由 fields 的 key 推斷)
        is_chat_detail = "side" in s2["fields"]
        
        if is_chat_detail:
            count_err = abs(s2["n_pred"] - s2["n_gold"])
            self._s2_msg_count_errors.append(count_err)
            self._s2_msg_extras     += s2["extra_pred"]
            self._s2_msg_missed     += s2["missed_gold"]
            self._s2_msg_paired     += s2["n_paired"]
            self._s2_msg_total_gold += s2["n_gold"]
            self._s2_msg_total_pred += s2["n_pred"]
            for field, vals in s2["fields"].items():
                self._s2_msg_field_totals[field]["match"] += vals["match"]
                self._s2_msg_field_totals[field]["total"] += vals["total"]
        else:
            # friend_list
            self._s2_friend_extras += s2["extra_pred"]
            self._s2_friend_missed += s2["missed_gold"]
            self._s2_friend_paired += s2["n_paired"]
            for field, vals in s2["fields"].items():
                self._s2_friend_field_totals[field]["match"] += vals["match"]
                self._s2_friend_field_totals[field]["total"] += vals["total"]

    def _field_accuracy_table(self, totals: dict, title: str) -> str:
        lines = [f"\n  [{title} 欄位精準度]"]
        lines.append(f"  {'Field':<22} {'Acc':>6}  ({'{match}/{total}'})")
        lines.append(f"  {'─'*40}")
        for field, v in sorted(totals.items()):
            t = v["total"]
            m = v["match"]
            acc = m / t if t > 0 else 0.0
            lines.append(f"  {field:<22} {acc:>6.3f}  ({m}/{t})")
        return "\n".join(lines)

    def print_report(self) -> None:
        print(self.cm_screen.report("Stage 1 — screen_type 混淆矩陣"))
        print(self.cm_app.report(   "Stage 1 — app_source 混淆矩陣"))
        print(self.cm_chat.report(  "Stage 1 — chat_type 混淆矩陣 (僅 chat_detail)"))

        # ── chat_detail 報告 ──
        if self._s2_msg_field_totals:
            print("\n" + "="*55)
            print("  Stage 2 — chat_detail 訊息提取精準度(軟對齊版)")
            print("="*55)
            errs = self._s2_msg_count_errors
            avg_err = sum(errs) / len(errs) if errs else 0
            print(f"  訊息數量誤差 |pred-gold| 平均: {avg_err:.2f}  (N={len(errs)})")
            print(f"  ┌─ 配對統計(訊息總計) ─────────────────────")
            print(f"  │  Gold 訊息總數:  {self._s2_msg_total_gold}")
            print(f"  │  Pred 訊息總數:  {self._s2_msg_total_pred}")
            print(f"  │  配對成功:       {self._s2_msg_paired}")
            print(f"  │  系統多抓(extra): {self._s2_msg_extras}")
            print(f"  │  系統漏抓(missed): {self._s2_msg_missed}")
            
            # ★ 新增:Precision / Recall / F1(訊息層級)
            tp = self._s2_msg_paired
            fp = self._s2_msg_extras
            fn = self._s2_msg_missed
            msg_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            msg_recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            msg_f1 = (
                2 * msg_precision * msg_recall / (msg_precision + msg_recall)
                if (msg_precision + msg_recall) > 0 else 0.0
            )
            print(f"  │  ─────────────────────────────────────")
            print(f"  │  訊息層級 Precision: {msg_precision:.3f}")
            print(f"  │  訊息層級 Recall:    {msg_recall:.3f}")
            print(f"  │  訊息層級 F1:        {msg_f1:.3f}")
            print(f"  └──────────────────────────────────────────")
            
            print(self._field_accuracy_table(
                self._s2_msg_field_totals, "chat_detail(僅配對成功的訊息)"
            ))

        # ── friend_list 報告 ──
        if self._s2_friend_field_totals:
            print("\n" + "="*55)
            print("  Stage 2 — friend_list 提取精準度(軟對齊版)")
            print("="*55)
            print(f"  配對成功: {self._s2_friend_paired}")
            print(f"  系統多抓: {self._s2_friend_extras}")
            print(f"  系統漏抓: {self._s2_friend_missed}")
            print(self._field_accuracy_table(
                self._s2_friend_field_totals, "friend_list"
            ))

    def to_dict(self) -> dict:
        # 計算訊息層級 P/R/F1
        tp = self._s2_msg_paired
        fp = self._s2_msg_extras
        fn = self._s2_msg_missed
        msg_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        msg_recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        msg_f1 = (
            2 * msg_precision * msg_recall / (msg_precision + msg_recall)
            if (msg_precision + msg_recall) > 0 else 0.0
        )
        
        return {
            "stage1": {
                "screen_type": self.cm_screen.to_dict(),
                "app_source":  self.cm_app.to_dict(),
                "chat_type":   self.cm_chat.to_dict(),
            },
            "stage2_chat_detail": {
                "msg_count_errors": self._s2_msg_count_errors,
                "totals": {
                    "n_pred":  self._s2_msg_total_pred,
                    "n_gold":  self._s2_msg_total_gold,
                    "paired":  self._s2_msg_paired,
                    "extras":  self._s2_msg_extras,
                    "missed":  self._s2_msg_missed,
                },
                "message_level_metrics": {
                    "precision": round(msg_precision, 4),
                    "recall":    round(msg_recall, 4),
                    "f1":        round(msg_f1, 4),
                },
                "field_accuracy": {
                    k: {"match": v["match"], "total": v["total"],
                        "accuracy": v["match"]/v["total"] if v["total"] else 0}
                    for k, v in self._s2_msg_field_totals.items()
                },
            },
            "stage2_friend_list": {
                "totals": {
                    "paired":  self._s2_friend_paired,
                    "extras":  self._s2_friend_extras,
                    "missed":  self._s2_friend_missed,
                },
                "field_accuracy": {
                    k: {"match": v["match"], "total": v["total"],
                        "accuracy": v["match"]/v["total"] if v["total"] else 0}
                    for k, v in self._s2_friend_field_totals.items()
                },
            },
            "per_image_reports": self.reports,
        }


# ============================================================
# §5  CLI 入口
# ============================================================

def run_eval(
    label_dir: str,
    result_json: str,
    output_json: Optional[str] = None,
    normalized_messages_json: Optional[str] = None,
):
    """主流程:載入標籤 + 結果,執行評測並列印報告。"""

    label_path = Path(label_dir)
    if label_path.is_file():
        labels = [GoldenLabel.from_file(label_path)]
    else:
        labels = load_all_labels(label_path)
    if not labels:
        print(f"[ERROR] 找不到標籤檔: {label_dir}")
        sys.exit(1)
    label_map: dict[str, GoldenLabel] = {l.source_file: l for l in labels}
    print(f"[INFO] 載入 {len(labels)} 份人工標籤")

    results: list[dict] = json.loads(Path(result_json).read_text(encoding="utf-8"))
    result_map: dict[str, dict] = {r["file_name"]: r for r in results}
    print(f"[INFO] 載入 {len(results)} 筆系統輸出")

    normalized_msg_map = _load_normalized_messages_map(normalized_messages_json)
    if normalized_messages_json:
        print(f"[INFO] 載入 normalized_messages 對應檔: {len(normalized_msg_map)} 個 key")

    summary = EvalSummary()
    skipped = 0

    for file_name, label in label_map.items():
        if file_name not in result_map:
            print(f"  [SKIP] 系統輸出缺少: {file_name}")
            skipped += 1
            continue

        result_entry = result_map[file_name]
        parsed       = _get_system_output(result_entry)

        from output_parser import extract_observations
        obs  = extract_observations(result_entry.get("output_text", ""), file_name)
        norm = normalize(obs)

        normalized_time_messages = (
            normalized_msg_map.get(file_name)
            or normalized_msg_map.get(Path(file_name).name)
            or normalized_msg_map.get(label.source_file)
            or normalized_msg_map.get(Path(label.source_file).name)
        )

        report = evaluate_one(
            label,
            parsed,
            norm,
            normalized_time_messages=normalized_time_messages,
        )
        summary.add(report)

    print(f"\n[INFO] 評測完成:{len(summary.reports)} 張,跳過 {skipped} 張\n")
    summary.print_report()

    if output_json:
        import os

        out_path = os.path.abspath(output_json)
        json_content = json.dumps(summary.to_dict(), ensure_ascii=False, indent=2)

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(json_content)

        # 不用 exists()，直接嘗試讀回驗證
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                read_back = f.read(100)

            print(f"\n[INFO] 詳細結果已存至:")
            print(f"       路徑: {out_path}")
            print(f"       驗證: 可成功讀回")
        except Exception as e:
            print(f"\n[ERROR] 寫入後無法讀回:")
            print(f"        路徑: {out_path}")
            print(f"        錯誤: {type(e).__name__}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="eval_pipeline — 軟對齊評測（公開精簡版）"
    )
    parser.add_argument("--labels",  required=True, help="人工標籤檔或標籤目錄(.json)")
    parser.add_argument("--results", required=True, help="系統輸出 result_sum.json")
    parser.add_argument("--output",  default=None,  help="匯出評測結果的 JSON 路徑(可選)")
    parser.add_argument(
        "--normalized-messages",
        default=None,
        help="normalized_messages.json 路徑；若提供，只用其中的 time 欄位做 time_raw 評測",
    )
    args = parser.parse_args()

    run_eval(
        args.labels,
        args.results,
        args.output,
        normalized_messages_json=args.normalized_messages,
    )