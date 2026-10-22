from datetime import date, datetime

import pytest

from kdash import calendar as cal
from kdash.data import build_snapshot, load_sample
from kdash.scoring import compute_score, judge, load_config, rule_rsi, rule_trend

CFG = load_config()


def full_snapshot(**over):
    snap = {
        "sox_ret": 0.024, "mu_ret": 0.031, "nvda_ret": 0.018, "ewy_ret": 0.01,
        "foreign_5d_ratio": 0.3, "inst_5d_ratio": 0.1, "foreign_streak": 3, "short_change": -0.02,
        "close": 100, "ma5": 98, "ma20": 95, "rsi14": 76, "vol_ratio": 1.1, "day_ret": 0.01, "ret_3d": 0.03,
        "usdkrw_chg": 0.004, "us10y_chg_bp": 6, "vix_chg": 0.0, "kospi_foreign_net": 0,
        "dram_chg": 0.0, "eps_revision": 0, "export_surprise_pp": 0, "news": 0, "vix": 18,
    }
    snap.update(over)
    return snap


def test_area_weights_sum_to_one():
    for h in CFG["horizons"].values():
        assert sum(h["areas"].values()) == pytest.approx(1.0)
    for a in CFG["areas"].values():
        assert sum(i["weight"] for i in a["indicators"]) == pytest.approx(1.0)


def test_rules():
    assert rule_trend(100, 98, 95) == 1
    assert rule_trend(90, 95, 98) == -1
    assert rule_rsi(80) == -0.5 and rule_rsi(20) == 0.5 and rule_rsi(60) == 0.3


def test_score_bounds_and_direction():
    up = compute_score(full_snapshot(), "1d", CFG)
    assert up.score > 0 and -100 <= up.score <= 100
    down = compute_score(full_snapshot(sox_ret=-0.05, mu_ret=-0.06, nvda_ret=-0.06, ewy_ret=-0.03,
                                       foreign_5d_ratio=-0.6, foreign_streak=-5), "1d", CFG)
    assert down.score < 0


def test_maximum_signal_is_100():
    snap = full_snapshot(sox_ret=1, mu_ret=1, nvda_ret=1, ewy_ret=1, foreign_5d_ratio=9, inst_5d_ratio=9,
                         foreign_streak=9, short_change=-9, rsi14=21, vol_ratio=2, day_ret=0.01, ret_3d=-0.1,
                         usdkrw_chg=-1, us10y_chg_bp=-99, vix_chg=-1, kospi_foreign_net=1e13,
                         dram_chg=1, eps_revision=1, export_surprise_pp=99, news=1)
    r = compute_score(snap, "5d", CFG)
    # 규칙 지표(RSI 0.5, 반전 0.5)는 최대가 1이 아니므로 100보다 작다
    assert 80 < r.score <= 100


def test_missing_data_holds_judgement():
    snap = {k: v for k, v in full_snapshot().items() if not k.startswith(("foreign", "inst", "short"))}
    for k in ("dram_chg", "eps_revision", "export_surprise_pp", "news"):
        snap.pop(k)
    r = compute_score(snap, "5d", CFG)
    assert r.judgement == "판단 보류"


def test_high_vol_widens_bands():
    assert judge(25, CFG["judgement"]) == "상승 우위"
    assert judge(25, CFG["judgement"], widen=1.5) == "중립"
    r = compute_score(full_snapshot(vix=35), "1d", CFG)
    assert r.confidence == "낮음"


def test_event_multiplier_applied():
    r = compute_score(full_snapshot(), "1d", CFG, event_multiplier=0.5)
    assert r.score == pytest.approx(r.raw_score * 0.5)
    assert r.confidence == "낮음"


# ---------------------------------------------------------------- 달력
HOL = {date(2026, 10, 5), date(2026, 10, 9)}


def test_us_evening_release_impacts_next_session():
    kst = cal.to_kst(date(2026, 11, 12), "08:30", "ET")      # 미국 CPI, 서머타임 종료 후
    assert kst == datetime(2026, 11, 12, 22, 30, tzinfo=cal.KST)
    assert cal.impact_date(date(2026, 11, 12), kst, HOL) == date(2026, 11, 13)


def test_dst_conversion():
    assert cal.to_kst(date(2026, 10, 28), "14:00", "ET").hour == 3   # 서머타임 중 FOMC → 03:00
    assert cal.to_kst(date(2026, 12, 9), "14:00", "ET").hour == 4    # 서머타임 종료 → 04:00


def test_fomc_early_morning_same_day_session():
    kst = cal.to_kst(date(2026, 10, 28), "14:00", "ET")
    assert cal.impact_date(date(2026, 10, 28), kst, HOL) == date(2026, 10, 29)


def test_holiday_rolls_forward():
    kst = cal.to_kst(date(2026, 10, 8), "08:30", "ET")       # 목요일 밤 발표, 금요일(10/9) 휴장
    assert cal.impact_date(date(2026, 10, 8), kst, HOL) == date(2026, 10, 12)


def test_option_expiry_second_thursday():
    exp = {e["date"] for e in cal.generate_expiries([2026], set())}
    assert date(2026, 10, 8) in exp and date(2026, 12, 10) in exp


def test_risk_adjustment_uses_window():
    raw = cal.load_calendar_file()
    events = cal.build_events(raw)
    hol = cal.krx_holidays(raw)
    f = CFG["filters"]
    m1, _, _, hit = cal.risk_adjustment(events, date(2026, 10, 29), 1, hol, f, "1d")
    assert m1 == 0.5 and "FOMC 금리 결정" in set(hit["title"])
    m5, _, multi, _ = cal.risk_adjustment(events, date(2026, 10, 26), 5, hol, f, "5d")
    assert m5 == 0.7 and multi


# ---------------------------------------------------------------- 데이터
def test_snapshot_has_no_lookahead():
    md = load_sample(days=120)
    t = md.prices["hynix"].index[-10]
    snap = build_snapshot(md, t)
    prev_close = md.prices["hynix"]["Close"][md.prices["hynix"].index < t].iloc[-1]
    assert snap["close"] == pytest.approx(prev_close)
    r = compute_score(snap, "1d", CFG)
    assert r.score is not None
