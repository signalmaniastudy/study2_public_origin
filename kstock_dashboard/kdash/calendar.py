"""이벤트 달력: 일정 로드, 한국시간 변환, 한국장 반영일 계산, 위험 필터."""
from __future__ import annotations

from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import yaml

KST = ZoneInfo("Asia/Seoul")
TZ = {"KST": KST, "ET": ZoneInfo("America/New_York")}
KRX_CLOSE = time(15, 30)
EVENTS_PATH = Path(__file__).resolve().parent.parent / "data" / "events.yaml"
TIER_ICON = {1: "🔴", 2: "🟠", 3: "⚪"}


def load_calendar_file(path: Path | str = EVENTS_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def krx_holidays(raw: dict) -> set[date]:
    return {pd.Timestamp(d).date() for d in raw.get("krx_holidays", [])}


def is_trading_day(d: date, holidays: set[date]) -> bool:
    return d.weekday() < 5 and d not in holidays


def next_trading_day(d: date, holidays: set[date]) -> date:
    d += timedelta(days=1)
    while not is_trading_day(d, holidays):
        d += timedelta(days=1)
    return d


def trading_days(start: date, n: int, holidays: set[date]) -> list[date]:
    """start(거래일이 아니면 다음 거래일)부터 n 거래일."""
    d = start if is_trading_day(start, holidays) else next_trading_day(start, holidays)
    out = [d]
    while len(out) < n:
        d = next_trading_day(d, holidays)
        out.append(d)
    return out


def to_kst(d: date, t: str | None, tz: str) -> datetime | None:
    if not t:
        return None
    hh, mm = map(int, t.split(":"))
    return datetime.combine(d, time(hh, mm), tzinfo=TZ[tz]).astimezone(KST)


def impact_date(event_date: date, kst_dt: datetime | None, holidays: set[date]) -> date:
    """이벤트가 처음 반영되는 한국장 거래일.

    한국장 마감(15:30) 전에 나오면 그날, 이후면 다음 거래일. 시각이 없으면 발표일 당일로 본다.
    """
    d = kst_dt.date() if kst_dt else event_date
    after_close = kst_dt is not None and kst_dt.time() >= KRX_CLOSE
    if is_trading_day(d, holidays) and not after_close:
        return d
    return next_trading_day(d, holidays)


def second_thursday(year: int, month: int) -> date:
    d = date(year, month, 1)
    d += timedelta(days=(3 - d.weekday()) % 7)
    return d + timedelta(days=7)


def generate_expiries(years: list[int], holidays: set[date]) -> list[dict]:
    """KRX 옵션 만기: 매월 둘째 목요일(휴장이면 앞당김). 3·6·9·12월은 선물·옵션 동시만기."""
    out = []
    for y in years:
        for m in range(1, 13):
            d = second_thursday(y, m)
            while not is_trading_day(d, holidays):
                d -= timedelta(days=1)
            quad = m in (3, 6, 9, 12)
            out.append({
                "date": d, "time": "15:20", "tz": "KST",
                "title": "선물·옵션 동시만기" if quad else "옵션 만기",
                "region": "KR", "category": "시장 구조", "tier": 1 if quad else 2,
                "status": "rule", "note": "둘째 목요일 규칙으로 생성",
            })
    return out


def generate_jobless_claims(start: date, end: date) -> list[dict]:
    out, d = [], start + timedelta(days=(3 - start.weekday()) % 7)
    while d <= end:
        out.append({
            "date": d, "time": "08:30", "tz": "ET", "title": "미국 신규 실업수당 청구",
            "region": "US", "category": "미국 매크로", "tier": 3, "status": "rule",
            "note": "매주 목요일 규칙으로 생성 (휴일 주는 변경될 수 있음)",
        })
        d += timedelta(days=7)
    return out


def build_events(raw: dict | None = None, include_rules: bool = True) -> pd.DataFrame:
    raw = raw or load_calendar_file()
    holidays = krx_holidays(raw)
    rows = [dict(e) for e in raw.get("events", [])]
    for r in rows:
        r["date"] = pd.Timestamp(r["date"]).date()
    if include_rules and rows:
        years = sorted({r["date"].year for r in rows})
        rows += generate_expiries(years, holidays)
        rows += generate_jobless_claims(date(years[0], 1, 1), date(years[-1], 12, 31))

    for r in rows:
        r["ticker"] = str(r["ticker"]) if r.get("ticker") else ""
        r["sectors"] = list(r.get("sectors") or [])
        r.setdefault("time", None)
        r.setdefault("tz", "KST")
        r.setdefault("status", "estimated")
        r.setdefault("note", "")
        kst = to_kst(r["date"], r["time"], r["tz"])
        r["kst_datetime"] = kst
        r["kst_label"] = kst.strftime("%m/%d %H:%M") if kst else "시각 미정"
        r["impact_date"] = impact_date(r["date"], kst, holidays)
        r["icon"] = TIER_ICON.get(r["tier"], "⚪")
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values(["impact_date", "tier"]).reset_index(drop=True)


def applies_to(events: pd.DataFrame, code: str, sector: str) -> pd.Series:
    """종목에 적용되는 이벤트: 시장 전체 이벤트 + 그 종목 이벤트 + 그 업종 이벤트."""
    if events.empty:
        return pd.Series(dtype=bool)
    market_wide = (events["ticker"] == "") & (events["sectors"].map(len) == 0)
    own = events["ticker"] == code
    sec = events["sectors"].map(lambda s: sector in s)
    return market_wide | own | sec


def risk_adjustment(events: pd.DataFrame, target: date, horizon_days: int,
                    holidays: set[date], cfg_filters: dict, horizon: str):
    """예측 기간(target부터 horizon_days 거래일)에 반영되는 이벤트로 점수 배수를 정한다.

    반환: (배수, 설명 목록, 1등급 이벤트 2건 이상 여부, 해당 이벤트 DataFrame)
    """
    window = trading_days(target, horizon_days, holidays)
    hit = events[events["impact_date"].isin(window)] if not events.empty else events
    mult_by_tier = {int(k): v for k, v in cfg_filters["event_multiplier"][horizon].items()}
    mult, notes = 1.0, []
    for _, e in hit.iterrows():
        m = mult_by_tier.get(int(e["tier"]))
        if m is None:
            continue
        mult = min(mult, m)
        notes.append(f"{e['icon']} {e['impact_date']:%m/%d} 반영: {e['title']}")
    multi_tier1 = int((hit["tier"] == 1).sum()) >= 2 if not hit.empty else False
    return mult, notes, multi_tier1, hit


def next_major_event(events: pd.DataFrame, today: date, tier: int = 1):
    up = events[(events["tier"] <= tier) & (events["impact_date"] >= today)]
    return None if up.empty else up.iloc[0]
