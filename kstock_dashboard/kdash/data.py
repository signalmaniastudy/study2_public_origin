"""시세·수급 데이터 수집과 점수 모델 입력값(snapshot) 계산.

- 실데이터: yfinance(해외 지수·환율·금리, SK하이닉스 시세), pykrx(투자자별 수급, 공매도)
- 샘플 데이터: 네트워크가 막혀 있거나 시연할 때 쓰는 합성 데이터 (투자 판단에 사용 금지)

시점 정렬 원칙: 한국장 거래일 T를 예측할 때는 날짜가 T보다 앞선 데이터만 쓴다.
(미국 T-1일 종가는 한국 T일 개장 전에 확정되므로 사용 가능)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd

TICKER = "000660"  # SK하이닉스

YF_TICKERS = {
    "hynix": "000660.KS",
    "kospi": "^KS11",
    "kosdaq": "^KQ11",
    "sox": "^SOX",
    "mu": "MU",
    "nvda": "NVDA",
    "ewy": "EWY",
    "nasdaq": "^IXIC",
    "usdkrw": "KRW=X",
    "us10y": "^TNX",
    "vix": "^VIX",
    "wti": "CL=F",
}

LABELS = {
    "hynix": "SK하이닉스", "kospi": "KOSPI", "kosdaq": "KOSDAQ", "sox": "필라델피아 반도체",
    "mu": "마이크론", "nvda": "엔비디아", "ewy": "EWY", "nasdaq": "나스닥", "usdkrw": "원/달러",
    "us10y": "미 국채 10년(%)", "vix": "VIX", "wti": "WTI",
}


@dataclass
class MarketData:
    prices: dict[str, pd.DataFrame]          # name -> OHLCV (index: date)
    flows: pd.DataFrame | None = None        # SK하이닉스 투자자별 순매수 금액 (외국인, 기관)
    kospi_flows: pd.DataFrame | None = None  # KOSPI 투자자별 순매수 금액
    short: pd.DataFrame | None = None        # 공매도 잔고 비중
    source: str = "live"
    errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------- 기술적 지표
def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    diff = close.diff()
    up = diff.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    down = (-diff.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / down)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    prev = df["Close"].shift()
    tr = pd.concat([df["High"] - df["Low"], (df["High"] - prev).abs(), (df["Low"] - prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


# ---------------------------------------------------------------- 실데이터
def load_live(days: int = 500) -> MarketData:
    import yfinance as yf

    start = date.today() - timedelta(days=days)
    prices, errors = {}, []
    for name, tk in YF_TICKERS.items():
        try:
            df = yf.Ticker(tk).history(start=start.isoformat(), auto_adjust=False)
            if df.empty:
                raise ValueError("빈 데이터")
            df.index = pd.to_datetime(df.index.date)
            prices[name] = df[["Open", "High", "Low", "Close", "Volume"]]
        except Exception as e:  # 네트워크·상장폐지 등
            errors.append(f"yfinance {tk}: {e}")

    md = MarketData(prices=prices, source="live", errors=errors)
    try:
        from pykrx import stock

        s, e = start.strftime("%Y%m%d"), date.today().strftime("%Y%m%d")
        f = stock.get_market_trading_value_by_date(s, e, TICKER)
        md.flows = f.rename(columns={"외국인합계": "foreign", "기관합계": "inst"})[["foreign", "inst"]]
        k = stock.get_market_trading_value_by_date(s, e, "KOSPI")
        md.kospi_flows = k.rename(columns={"외국인합계": "foreign"})[["foreign"]]
        sh = stock.get_shorting_balance_by_date(s, e, TICKER)
        md.short = sh.rename(columns={"비중": "ratio"})[["ratio"]]
    except Exception as ex:
        errors.append(f"pykrx: {ex}")
    return md


# ---------------------------------------------------------------- 샘플 데이터
def load_sample(days: int = 500, seed: int = 7) -> MarketData:
    """합성 데이터. 미국 T-1일 공통 요인이 한국 T일 시가 갭에 반영되는 구조만 흉내 낸다."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=days)
    n = len(idx)
    f = rng.normal(0.0004, 0.015, n)  # 글로벌 반도체 공통 요인

    def ohlcv(rets, start, vol=0.01, base_volume=1e6, gap=None):
        close = start * np.exp(np.cumsum(rets))
        open_ = np.r_[close[0], close[:-1]] * (1 + (gap if gap is not None else rng.normal(0, vol / 3, n)))
        high = np.maximum(open_, close) * (1 + abs(rng.normal(0, vol / 2, n)))
        low = np.minimum(open_, close) * (1 - abs(rng.normal(0, vol / 2, n)))
        volume = base_volume * np.exp(rng.normal(0, 0.35, n))
        return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume}, index=idx)

    lag_f = np.r_[0, f[:-1]]
    gap = 0.8 * lag_f + rng.normal(0, 0.005, n)
    hynix_ret = gap + rng.normal(0, 0.018, n)
    prices = {
        "sox": ohlcv(1.2 * f + rng.normal(0, 0.006, n), 5000),
        "mu": ohlcv(1.5 * f + rng.normal(0, 0.012, n), 110),
        "nvda": ohlcv(1.3 * f + rng.normal(0, 0.014, n), 140),
        "nasdaq": ohlcv(0.6 * f + rng.normal(0, 0.006, n), 18000),
        "ewy": ohlcv(0.5 * f + rng.normal(0, 0.008, n), 65),
        "hynix": ohlcv(hynix_ret, 180000, 0.02, 3e6, gap=gap),
        "kospi": ohlcv(0.5 * lag_f + rng.normal(0, 0.008, n), 2600),
        "kosdaq": ohlcv(0.4 * lag_f + rng.normal(0, 0.010, n), 800),
        "usdkrw": ohlcv(-0.1 * f + rng.normal(0, 0.003, n), 1380, 0.003),
        "wti": ohlcv(rng.normal(0, 0.018, n), 72),
    }
    prices["us10y"] = pd.DataFrame({c: 4.2 + np.cumsum(rng.normal(0, 0.04, n)) for c in ["Open", "High", "Low", "Close"]}, index=idx)
    prices["us10y"]["Volume"] = 0
    vix = np.empty(n)
    vix[0] = 16.0
    for i in range(1, n):  # 평균 회귀하는 VIX
        vix[i] = vix[i - 1] + 0.1 * (16 - vix[i - 1]) - 60 * f[i] + rng.normal(0, 0.8)
    vix = np.clip(vix, 10, 45)
    prices["vix"] = pd.DataFrame({"Open": vix, "High": vix * 1.03, "Low": vix * 0.97, "Close": vix, "Volume": 0}, index=idx)

    turnover = prices["hynix"]["Close"] * prices["hynix"]["Volume"]
    foreign = turnover * (0.08 * np.sign(hynix_ret) + rng.normal(0, 0.06, n))
    flows = pd.DataFrame({"foreign": foreign, "inst": turnover * rng.normal(0, 0.05, n)}, index=idx)
    kospi_flows = pd.DataFrame({"foreign": rng.normal(0, 3e11, n)}, index=idx)
    short = pd.DataFrame({"ratio": 1.0 + np.cumsum(rng.normal(0, 0.02, n)).clip(-0.8, 3)}, index=idx)
    return MarketData(prices, flows, kospi_flows, short, source="sample")


def load_market(mode: str = "auto") -> MarketData:
    """mode: auto(실데이터 실패 시 샘플) / live / sample"""
    if mode == "sample":
        return load_sample()
    md = load_live()
    if mode == "auto" and "hynix" not in md.prices:
        sample = load_sample()
        sample.errors = md.errors
        return sample
    return md


# ---------------------------------------------------------------- snapshot
def _before(df: pd.DataFrame | None, target: pd.Timestamp) -> pd.DataFrame | None:
    if df is None or df.empty:
        return None
    out = df[df.index < target]
    return out if len(out) else None


def _ret(df, target, col="Close"):
    d = _before(df, target)
    if d is None or len(d) < 2:
        return None
    return float(d[col].iloc[-1] / d[col].iloc[-2] - 1)


def _streak(s: pd.Series) -> int:
    sign = np.sign(s.iloc[-1])
    if sign == 0:
        return 0
    n = 0
    for v in reversed(s.to_numpy()):
        if np.sign(v) != sign:
            break
        n += 1
    return int(sign * n)


def build_snapshot(md: MarketData, target: date | pd.Timestamp, manual: dict | None = None) -> dict:
    """한국장 거래일 target을 예측하기 위한 원시 지표 값 (target 이전 데이터만 사용)."""
    t = pd.Timestamp(target)
    p = md.prices
    snap: dict = {}

    snap["sox_ret"] = _ret(p.get("sox"), t)
    snap["mu_ret"] = _ret(p.get("mu"), t)
    snap["nvda_ret"] = _ret(p.get("nvda"), t)
    snap["ewy_ret"] = _ret(p.get("ewy"), t)

    hx = _before(p.get("hynix"), t)
    if hx is not None and len(hx) >= 25:
        c = hx["Close"]
        snap.update({
            "close": float(c.iloc[-1]),
            "ma5": float(c.rolling(5).mean().iloc[-1]),
            "ma20": float(c.rolling(20).mean().iloc[-1]),
            "rsi14": float(rsi(c).iloc[-1]),
            "vol_ratio": float(hx["Volume"].iloc[-1] / hx["Volume"].iloc[-21:-1].mean()),
            "day_ret": float(c.iloc[-1] / c.iloc[-2] - 1),
            "ret_3d": float(c.iloc[-1] / c.iloc[-4] - 1),
            "atr14": float(atr(hx).iloc[-1]),
        })
        turnover20 = float((hx["Close"] * hx["Volume"]).iloc[-20:].mean())
    else:
        turnover20 = None

    fl = _before(md.flows, t)
    if fl is not None and len(fl) >= 5 and turnover20:
        snap["foreign_5d_ratio"] = float(fl["foreign"].iloc[-5:].sum() / turnover20)
        snap["inst_5d_ratio"] = float(fl["inst"].iloc[-5:].sum() / turnover20)
        snap["foreign_streak"] = _streak(fl["foreign"])

    # 공매도 잔고는 T+2일에 공개되므로 2거래일 늦춰서 사용
    sh = _before(md.short, t)
    if sh is not None and len(sh) >= 8:
        r = sh["ratio"].iloc[:-2]
        snap["short_change"] = float(r.iloc[-1] / r.iloc[-6] - 1)

    snap["usdkrw_chg"] = _ret(p.get("usdkrw"), t)
    us10y = _before(p.get("us10y"), t)
    if us10y is not None and len(us10y) >= 2:
        snap["us10y_chg_bp"] = float((us10y["Close"].iloc[-1] - us10y["Close"].iloc[-2]) * 100)
    snap["vix_chg"] = _ret(p.get("vix"), t)
    vix = _before(p.get("vix"), t)
    if vix is not None:
        snap["vix"] = float(vix["Close"].iloc[-1])
    kf = _before(md.kospi_flows, t)
    if kf is not None:
        snap["kospi_foreign_net"] = float(kf["foreign"].iloc[-1])

    for k, v in (manual or {}).items():
        if v is not None:
            snap[k] = v
    return snap


def latest_quotes(md: MarketData) -> pd.DataFrame:
    rows = []
    for name, df in md.prices.items():
        if df is None or len(df) < 2:
            continue
        last, prev = df["Close"].iloc[-1], df["Close"].iloc[-2]
        chg = (last - prev) * 100 if name == "us10y" else last / prev - 1
        rows.append({"key": name, "지표": LABELS.get(name, name), "기준일": df.index[-1].date(),
                     "현재값": float(last), "변화": float(chg)})
    return pd.DataFrame(rows)
