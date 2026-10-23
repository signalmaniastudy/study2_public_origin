"""시세·수급 데이터 수집과 점수 모델 입력값(snapshot) 계산.

- 공통 시장 데이터: 국내외 지수, 환율, 금리, VIX, KOSPI 외국인 수급
- 종목 데이터: 종목 시세(yfinance), 투자자별 수급·공매도(pykrx)
- 해외 연관 지표(peers): 업종 프로필별 해외 지수·종목 (config/model.yaml)
- 샘플 데이터: 네트워크가 막혀 있거나 시연할 때 쓰는 합성 데이터 (투자 판단에 사용 금지)

시점 정렬 원칙: 한국장 거래일 T를 예측할 때는 날짜가 T보다 앞선 데이터만 쓴다.
(미국 T-1일 종가는 한국 T일 개장 전에 확정되므로 사용 가능)
"""
from __future__ import annotations

import zlib
from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd

HISTORY_DAYS = 500

COMMON_TICKERS = {
    "kospi": "^KS11",
    "kosdaq": "^KQ11",
    "sp500": "^GSPC",
    "nasdaq": "^IXIC",
    "sox": "^SOX",
    "ewy": "EWY",
    "usdkrw": "KRW=X",
    "us10y": "^TNX",
    "vix": "^VIX",
    "wti": "CL=F",
}

LABELS = {
    "kospi": "KOSPI", "kosdaq": "KOSDAQ", "sp500": "S&P 500", "nasdaq": "나스닥", "sox": "필라델피아 반도체",
    "ewy": "EWY", "usdkrw": "원/달러", "us10y": "미 국채 10년(%)", "vix": "VIX", "wti": "WTI",
}

YF_SUFFIX = {"KOSPI": ".KS", "KOSDAQ": ".KQ"}


@dataclass
class MarketData:
    """모든 종목이 함께 쓰는 시장 데이터."""
    prices: dict[str, pd.DataFrame]          # name -> OHLCV (index: date)
    kospi_flows: pd.DataFrame | None = None  # KOSPI 투자자별 순매수 금액
    source: str = "live"
    errors: list[str] = field(default_factory=list)


@dataclass
class StockData:
    code: str
    ohlcv: pd.DataFrame | None = None
    flows: pd.DataFrame | None = None        # 투자자별 순매수 금액 (foreign, inst)
    short: pd.DataFrame | None = None        # 공매도 잔고 비중 (ratio)
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
def _start(days: int) -> date:
    return date.today() - timedelta(days=days)


def _yf_history(ticker: str, days: int = HISTORY_DAYS) -> pd.DataFrame:
    import yfinance as yf

    df = yf.Ticker(ticker).history(start=_start(days).isoformat(), auto_adjust=False)
    if df.empty:
        raise ValueError("빈 데이터")
    df.index = pd.to_datetime(df.index.date)
    return df[["Open", "High", "Low", "Close", "Volume"]]


def load_common_live(days: int = HISTORY_DAYS) -> MarketData:
    prices, errors = {}, []
    for name, tk in COMMON_TICKERS.items():
        try:
            prices[name] = _yf_history(tk, days)
        except Exception as e:  # 네트워크·상장폐지 등
            errors.append(f"yfinance {tk}: {e}")
    md = MarketData(prices=prices, source="live", errors=errors)
    try:
        from pykrx import stock

        s, e = _start(days).strftime("%Y%m%d"), date.today().strftime("%Y%m%d")
        k = stock.get_market_trading_value_by_date(s, e, "KOSPI")
        md.kospi_flows = k.rename(columns={"외국인합계": "foreign"})[["foreign"]]
    except Exception as ex:
        errors.append(f"pykrx KOSPI 수급: {ex}")
    return md


def load_stock_live(code: str, market: str = "", days: int = HISTORY_DAYS) -> StockData:
    sd = StockData(code=code, source="live")
    suffixes = [YF_SUFFIX[market]] if market in YF_SUFFIX else [".KS", ".KQ"]
    for suf in suffixes:
        try:
            sd.ohlcv = _yf_history(code + suf, days)
            break
        except Exception as e:
            sd.errors.append(f"yfinance {code}{suf}: {e}")
    if sd.ohlcv is not None:
        sd.errors = []
    try:
        from pykrx import stock

        s, e = _start(days).strftime("%Y%m%d"), date.today().strftime("%Y%m%d")
        f = stock.get_market_trading_value_by_date(s, e, code)
        sd.flows = f.rename(columns={"외국인합계": "foreign", "기관합계": "inst"})[["foreign", "inst"]]
        sh = stock.get_shorting_balance_by_date(s, e, code)
        sd.short = sh.rename(columns={"비중": "ratio"})[["ratio"]]
    except Exception as ex:
        sd.errors.append(f"pykrx {code}: {ex}")
    return sd


def load_peers_live(tickers: list[str], days: int = HISTORY_DAYS) -> tuple[dict[str, pd.DataFrame], list[str]]:
    out, errors = {}, []
    for tk in tickers:
        try:
            out[tk] = _yf_history(tk, days)
        except Exception as e:
            errors.append(f"yfinance {tk}: {e}")
    return out, errors


# ---------------------------------------------------------------- 샘플 데이터
def _seed(key: str) -> int:
    return zlib.crc32(key.encode())


def _sample_index(days: int) -> pd.DatetimeIndex:
    return pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=days)


def _sample_ohlcv(key: str, start: float, vol: float, days: int, base_volume: float = 1e6) -> pd.DataFrame:
    """합성 OHLCV. 시가 갭과 장중 변동을 분리해 '시가→종가' 백테스트가 가능하게 만든다."""
    rng = np.random.default_rng(_seed(key))
    idx = _sample_index(days)
    n = len(idx)
    gap = rng.normal(0, vol * 0.4, n)
    rets = gap + rng.normal(0.0003, vol, n)
    close = start * np.exp(np.cumsum(rets))
    open_ = np.r_[close[0], close[:-1]] * np.exp(gap)
    high = np.maximum(open_, close) * (1 + abs(rng.normal(0, vol / 2, n)))
    low = np.minimum(open_, close) * (1 - abs(rng.normal(0, vol / 2, n)))
    volume = base_volume * np.exp(rng.normal(0, 0.35, n))
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume}, index=idx)


def load_common_sample(days: int = HISTORY_DAYS) -> MarketData:
    starts = {"kospi": (2600, 0.008), "kosdaq": (800, 0.01), "sp500": (5800, 0.008), "nasdaq": (18000, 0.01),
              "sox": (5000, 0.018), "ewy": (65, 0.01), "usdkrw": (1380, 0.003), "wti": (72, 0.018)}
    prices = {k: _sample_ohlcv(f"common:{k}", s, v, days) for k, (s, v) in starts.items()}
    idx = _sample_index(days)
    rng = np.random.default_rng(_seed("common:macro"))
    n = len(idx)
    us10y = 4.2 + np.cumsum(rng.normal(0, 0.04, n))
    vix = np.empty(n)
    vix[0] = 16.0
    for i in range(1, n):  # 평균 회귀하는 VIX
        vix[i] = vix[i - 1] + 0.1 * (16 - vix[i - 1]) + rng.normal(0, 1.0)
    vix = np.clip(vix, 10, 45)
    prices["us10y"] = pd.DataFrame({"Open": us10y, "High": us10y, "Low": us10y, "Close": us10y, "Volume": 0}, index=idx)
    prices["vix"] = pd.DataFrame({"Open": vix, "High": vix * 1.03, "Low": vix * 0.97, "Close": vix, "Volume": 0}, index=idx)
    kospi_flows = pd.DataFrame({"foreign": rng.normal(0, 3e11, n)}, index=idx)
    return MarketData(prices, kospi_flows, source="sample")


def load_stock_sample(code: str, days: int = HISTORY_DAYS) -> StockData:
    rng = np.random.default_rng(_seed(f"stock:{code}"))
    start = float(rng.choice([8000, 25000, 60000, 150000, 400000]))
    ohlcv = _sample_ohlcv(f"stock:{code}", start, 0.02, days, base_volume=2e6)
    n = len(ohlcv)
    turnover = ohlcv["Close"] * ohlcv["Volume"]
    day_ret = ohlcv["Close"].pct_change().fillna(0).to_numpy()
    flows = pd.DataFrame({"foreign": turnover * (2 * day_ret + rng.normal(0, 0.06, n)),
                          "inst": turnover * rng.normal(0, 0.05, n)}, index=ohlcv.index)
    short = pd.DataFrame({"ratio": 1.0 + np.cumsum(rng.normal(0, 0.02, n)).clip(-0.8, 3)}, index=ohlcv.index)
    return StockData(code, ohlcv, flows, short, source="sample")


def load_peers_sample(tickers: list[str], days: int = HISTORY_DAYS) -> dict[str, pd.DataFrame]:
    return {tk: _sample_ohlcv(f"peer:{tk}", 100, 0.02, days) for tk in tickers}


# ---------------------------------------------------------------- 모드별 로더
# mode: auto(실데이터 실패 시 샘플) / live / sample
def load_common(mode: str = "auto") -> MarketData:
    if mode == "sample":
        return load_common_sample()
    md = load_common_live()
    if mode == "auto" and not md.prices:
        sample = load_common_sample()
        sample.errors = md.errors
        return sample
    return md


def load_stock(code: str, market: str = "", mode: str = "auto") -> StockData:
    if mode == "sample":
        return load_stock_sample(code)
    sd = load_stock_live(code, market)
    if mode == "auto" and sd.ohlcv is None:
        sample = load_stock_sample(code)
        sample.errors = sd.errors
        return sample
    return sd


def load_peers(tickers: list[str], mode: str = "auto") -> tuple[dict[str, pd.DataFrame], list[str], bool]:
    """반환: (ticker -> OHLCV, 오류 목록, 샘플 사용 여부)"""
    if mode == "sample":
        return load_peers_sample(tickers), [], True
    out, errors = load_peers_live(tickers)
    if mode == "auto" and tickers and not out:
        return load_peers_sample(tickers), errors, True
    return out, errors, False


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


def peer_key(ticker: str) -> str:
    return f"peer:{ticker}"


def build_snapshot(md: MarketData, sd: StockData, peers: dict[str, pd.DataFrame],
                   target: date | pd.Timestamp, manual: dict | None = None) -> dict:
    """한국장 거래일 target을 예측하기 위한 원시 지표 값 (target 이전 데이터만 사용)."""
    t = pd.Timestamp(target)
    p = md.prices
    snap: dict = {}

    for tk, df in peers.items():
        snap[peer_key(tk)] = _ret(df, t)

    hx = _before(sd.ohlcv, t)
    turnover20 = None
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

    fl = _before(sd.flows, t)
    if fl is not None and len(fl) >= 5 and turnover20:
        snap["foreign_5d_ratio"] = float(fl["foreign"].iloc[-5:].sum() / turnover20)
        snap["inst_5d_ratio"] = float(fl["inst"].iloc[-5:].sum() / turnover20)
        snap["foreign_streak"] = _streak(fl["foreign"])

    # 공매도 잔고는 T+2일에 공개되므로 2거래일 늦춰서 사용
    sh = _before(sd.short, t)
    if sh is not None and len(sh) >= 8:
        r = sh["ratio"].iloc[:-2]
        if r.iloc[-6] > 0:
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


def last_change(df: pd.DataFrame | None) -> tuple[float, float] | None:
    if df is None or len(df) < 2:
        return None
    last, prev = df["Close"].iloc[-1], df["Close"].iloc[-2]
    return float(last), float(last / prev - 1)


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
