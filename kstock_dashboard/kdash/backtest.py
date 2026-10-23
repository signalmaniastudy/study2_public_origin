"""점수 모델 백테스트.

거래일 T마다 T 이전 데이터로 점수를 계산하고, 실제로 매매할 수 있는
'T일 시가 → T+h-1일 종가' 수익률과 비교한다. (전일 종가 기준으로 비교하면
밤사이 미국장 신호가 이미 시가 갭에 반영된 부분까지 성과로 잡혀 과대평가된다.)
E 영역(수동 입력)과 이벤트 필터는 과거 값이 없어 백테스트에서 제외된다.
"""
from __future__ import annotations

import pandas as pd

from .data import MarketData, StockData, build_snapshot
from .scoring import compute_score, load_config

BINS = [-100, -50, -20, 20, 50, 100]
BIN_LABELS = ["≤-50", "-50~-20", "-20~20", "20~50", "≥50"]


def run_backtest(md: MarketData, sd: StockData, peers: dict, horizon: str = "1d",
                 warmup: int = 30, cfg: dict | None = None) -> pd.DataFrame:
    """cfg 는 종목 업종 프로필이 적용된 설정(scoring.profile_config)을 넘긴다."""
    cfg = cfg or load_config()
    h = cfg["horizons"][horizon]["days"]
    hx = sd.ohlcv
    if hx is None:
        return pd.DataFrame()
    dates = hx.index
    rows = []
    for i in range(warmup, len(dates) - h + 1):
        t = dates[i]
        res = compute_score(build_snapshot(md, sd, peers, t), horizon, cfg)
        if res.score is None:
            continue
        fwd = hx["Close"].iloc[i + h - 1] / hx["Open"].iloc[i] - 1
        rows.append({"date": t, "score": res.score, "judgement": res.judgement, "fwd_ret": fwd})
    return pd.DataFrame(rows)


def summarize(bt: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    if bt.empty:
        return pd.DataFrame(), {}
    b = bt.assign(bin=pd.cut(bt["score"], BINS, labels=BIN_LABELS, include_lowest=True))
    table = b.groupby("bin", observed=False).agg(
        표본수=("fwd_ret", "size"),
        평균수익률=("fwd_ret", "mean"),
        상승확률=("fwd_ret", lambda r: (r > 0).mean() if len(r) else float("nan")),
    )
    ic = bt["score"].rank().corr(bt["fwd_ret"].rank())  # Spearman 순위 상관
    long = bt[bt["score"] >= 30]
    stats = {
        "IC": ic,
        "표본수": len(bt),
        "전체 상승확률": (bt["fwd_ret"] > 0).mean(),
        "+30 이상 진입 횟수": len(long),
        "+30 이상 상승확률": (long["fwd_ret"] > 0).mean() if len(long) else float("nan"),
        "+30 이상 평균수익률": long["fwd_ret"].mean() if len(long) else float("nan"),
    }
    return table, stats
