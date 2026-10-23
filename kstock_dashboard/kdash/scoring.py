"""종목 단기 방향 점수 모델.

세부 지표 점수(-1 ~ +1) → 영역 점수(가중평균) → 최종 점수(-100 ~ +100).
가중치와 기준값은 config/model.yaml 에서 읽는다.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "model.yaml"


def load_config(path: Path | str = CONFIG_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def sector_profiles(cfg: dict) -> dict:
    return cfg["sector_profiles"]


def profile_config(cfg: dict, sector: str) -> dict:
    """업종 프로필로 A 영역(글로벌 연관 지표)과 E 영역 업황 지표 이름을 채운 설정."""
    prof = cfg["sector_profiles"].get(sector) or cfg["sector_profiles"]["default"]
    out = copy.deepcopy(cfg)
    out["areas"]["global"]["indicators"] = [
        {"key": f"peer_{p['ticker']}", "label": f"{p['label']} 전일 수익률", "weight": p["weight"],
         "rule": "linear", "inputs": [f"peer:{p['ticker']}"], "scale": p["scale"]}
        for p in prof["peers"]
    ]
    for ind in out["areas"]["industry"]["indicators"]:
        if ind["key"] == "industry":
            ind["label"] = prof.get("industry_label", ind["label"])
    return out


def peer_tickers(cfg: dict, sectors) -> list[str]:
    seen = []
    for sec in sectors:
        prof = cfg["sector_profiles"].get(sec) or cfg["sector_profiles"]["default"]
        for p in prof["peers"]:
            if p["ticker"] not in seen:
                seen.append(p["ticker"])
    return seen


def clip(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _missing(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


# ---------------------------------------------------------------- 규칙 함수
def rule_linear(x, scale, sign=1):
    return clip(sign * x / scale)


def rule_discrete(x):
    return clip(float(x))


def rule_trend(close, ma5, ma20):
    if close > ma5 > ma20:
        return 1.0
    if close < ma5 < ma20:
        return -1.0
    return 0.0


def rule_rsi(rsi):
    if rsi > 75:
        return -0.5   # 과열
    if rsi < 25:
        return 0.5    # 과매도 반등
    if 50 <= rsi <= 70:
        return 0.3
    if 30 <= rsi < 50:
        return -0.3
    return 0.0


def rule_volume(vol_ratio, day_ret):
    if vol_ratio >= 1.5 and day_ret > 0:
        return 1.0
    if vol_ratio >= 1.5 and day_ret < 0:
        return -1.0
    return 0.0


def rule_reversal(ret_3d):
    if ret_3d >= 0.08:
        return -0.5
    if ret_3d <= -0.08:
        return 0.5
    return 0.0


def rule_streak(days):
    return clip(days / 5)


RULES = {
    "trend": rule_trend,
    "rsi": rule_rsi,
    "volume": rule_volume,
    "reversal": rule_reversal,
    "streak": rule_streak,
    "discrete": rule_discrete,
}


def score_indicator(spec: dict, snapshot: dict) -> float | None:
    values = [snapshot.get(k) for k in spec["inputs"]]
    if any(_missing(v) for v in values):
        return None
    if spec["rule"] == "linear":
        return rule_linear(values[0], spec["scale"], spec.get("sign", 1))
    return RULES[spec["rule"]](*values)


# ---------------------------------------------------------------- 결과 구조
@dataclass
class AreaResult:
    key: str
    label: str
    weight: float
    score: float | None
    contribution: float
    indicators: list[dict]


@dataclass
class ScoreResult:
    horizon: str
    raw_score: float | None
    score: float | None
    judgement: str
    confidence: str
    multiplier: float
    missing_weight: float
    areas: list[AreaResult]
    notes: list[str] = field(default_factory=list)


def judge(score: float, judgement_cfg: list[dict], widen: float = 1.0) -> str:
    for band in judgement_cfg:
        bound = band["min"] if band["min"] <= -100 else band["min"] * widen
        if score >= bound:
            return band["label"]
    return judgement_cfg[-1]["label"]


def compute_score(
    snapshot: dict,
    horizon: str = "1d",
    cfg: dict | None = None,
    event_multiplier: float = 1.0,
    event_notes: list[str] | None = None,
    multi_tier1: bool = False,
) -> ScoreResult:
    """snapshot(원시 지표 값 dict)으로 점수를 계산한다.

    event_multiplier / event_notes / multi_tier1 은 calendar.risk_adjustment 의 결과를 넘긴다.
    """
    cfg = cfg or load_config()
    filters = cfg["filters"]
    area_weights = cfg["horizons"][horizon]["areas"]
    notes = list(event_notes or [])

    areas: list[AreaResult] = []
    missing_weight = 0.0
    for area_key, area_w in area_weights.items():
        area_cfg = cfg["areas"][area_key]
        rows, total_w, acc = [], 0.0, 0.0
        for spec in area_cfg["indicators"]:
            s = score_indicator(spec, snapshot)
            rows.append({
                "지표": spec["label"],
                "입력값": [snapshot.get(k) for k in spec["inputs"]],
                "점수": s,
                "비중": spec["weight"],
            })
            if s is None:
                missing_weight += area_w * spec["weight"]
            else:
                total_w += spec["weight"]
                acc += spec["weight"] * s
        area_score = acc / total_w if total_w > 0 else None
        areas.append(AreaResult(area_key, area_cfg["label"], area_w, area_score, 0.0, rows))

    available = [a for a in areas if a.score is not None]
    avail_w = sum(a.weight for a in available)
    if not available:
        return ScoreResult(horizon, None, None, "판단 보류", "낮음", event_multiplier,
                           missing_weight, areas, notes + ["사용 가능한 데이터가 없습니다."])

    raw = 0.0
    for a in available:
        a.contribution = 100 * a.weight / avail_w * a.score
        raw += a.contribution
    adjusted = raw * event_multiplier

    low_conf = False
    if event_multiplier < 1.0:
        notes.append(f"이벤트 위험으로 점수에 ×{event_multiplier:g} 적용")
        if horizon == "1d" and event_multiplier <= filters["event_multiplier"]["1d"][1]:
            low_conf = True
    if multi_tier1:
        low_conf = True
        notes.append("예측 기간 안에 1등급 이벤트가 2건 이상 → 관망 권고")

    widen = 1.0
    vol = max((snapshot.get(k) or 0) for k in ("vix", "vkospi"))
    if vol > filters["high_vol_level"]:
        widen = filters["high_vol_threshold_mult"]
        low_conf = True
        notes.append(f"고변동성(VIX/VKOSPI {vol:.1f}) → 판단 구간 ×{widen:g}")

    lvl = filters["conflict_level"]
    if any(a.score >= lvl for a in available) and any(a.score <= -lvl for a in available):
        low_conf = True
        notes.append("영역 간 신호 충돌")

    if missing_weight > filters["max_missing_weight"]:
        notes.append(f"결측 가중치 {missing_weight:.0%} > {filters['max_missing_weight']:.0%} → 판단 보류")
        return ScoreResult(horizon, raw, adjusted, "판단 보류", "낮음", event_multiplier,
                           missing_weight, areas, notes)
    if missing_weight > 0:
        notes.append(f"결측 지표 가중치 {missing_weight:.0%}는 나머지로 재배분")

    return ScoreResult(
        horizon=horizon,
        raw_score=raw,
        score=adjusted,
        judgement=judge(adjusted, cfg["judgement"], widen),
        confidence="낮음" if low_conf else "보통",
        multiplier=event_multiplier,
        missing_weight=missing_weight,
        areas=areas,
        notes=notes,
    )
