"""종목 목록(검색용)과 관심종목 저장.

- 종목 목록: pykrx로 KOSPI·KOSDAQ 전체를 받아 하루 동안 파일로 캐시한다.
  실패하면 data/stocks.csv(주요 종목, 업종 프로필 포함)만 사용한다.
- 관심종목: watchlist.json 에 저장해 다음 실행 때도 유지한다.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
BUNDLED_CSV = ROOT / "data" / "stocks.csv"
CACHE_CSV = ROOT / "data" / "stock_list_cache.csv"
WATCHLIST_PATH = ROOT / "watchlist.json"
DEFAULT_WATCHLIST = {"codes": ["000660"], "sectors": {}}
COLUMNS = ["code", "name", "market", "sector"]


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str).fillna("")[COLUMNS]


def bundled_stocks() -> pd.DataFrame:
    return _read(BUNDLED_CSV)


def fetch_krx_list() -> pd.DataFrame:
    from pykrx import stock

    rows = []
    for market in ("KOSPI", "KOSDAQ"):
        for code in stock.get_market_ticker_list(market=market):
            rows.append({"code": code, "name": stock.get_market_ticker_name(code), "market": market, "sector": ""})
    if not rows:
        raise ValueError("빈 종목 목록")
    return pd.DataFrame(rows, columns=COLUMNS)


def load_stock_list(try_live: bool = True) -> tuple[pd.DataFrame, str]:
    """반환: (종목 목록, 출처 설명). 업종은 내장 목록에 있는 종목만 채워진다."""
    bundled = bundled_stocks()
    full, source = None, "내장 주요 종목"
    if CACHE_CSV.exists() and date.fromtimestamp(CACHE_CSV.stat().st_mtime) == date.today():
        full, source = _read(CACHE_CSV), "KRX 전체 (오늘 캐시)"
    elif try_live:
        try:
            full = fetch_krx_list()
            full.to_csv(CACHE_CSV, index=False)
            source = "KRX 전체"
        except Exception:
            if CACHE_CSV.exists():
                full, source = _read(CACHE_CSV), "KRX 전체 (이전 캐시)"
    if full is None:
        return bundled, source
    merged = full.drop(columns="sector").merge(bundled[["code", "sector"]], on="code", how="left").fillna("")
    extra = bundled[~bundled["code"].isin(merged["code"])]
    return pd.concat([merged, extra], ignore_index=True)[COLUMNS], source


def search(stocks: pd.DataFrame, query: str, limit: int = 30) -> pd.DataFrame:
    q = query.strip().lower()
    if not q:
        return stocks.iloc[:0]
    hit = stocks[stocks["name"].str.lower().str.contains(q, regex=False) | stocks["code"].str.startswith(q)]
    # 이름이 검색어로 시작하는 종목을 먼저
    return hit.assign(_p=~hit["name"].str.lower().str.startswith(q)).sort_values(["_p", "name"]).drop(columns="_p").head(limit)


def load_watchlist() -> dict:
    try:
        data = json.loads(WATCHLIST_PATH.read_text(encoding="utf-8"))
        return {"codes": [str(c) for c in data.get("codes", [])], "sectors": dict(data.get("sectors", {}))}
    except Exception:
        return {"codes": list(DEFAULT_WATCHLIST["codes"]), "sectors": {}}


def save_watchlist(watch: dict) -> None:
    try:
        WATCHLIST_PATH.write_text(json.dumps(watch, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        pass  # 읽기 전용 환경에서는 세션 동안만 유지


def stock_info(stocks: pd.DataFrame, code: str, watch: dict) -> dict:
    row = stocks[stocks["code"] == code]
    info = row.iloc[0].to_dict() if len(row) else {"code": code, "name": code, "market": "", "sector": ""}
    info["sector"] = watch.get("sectors", {}).get(code) or info.get("sector") or "default"
    return info
