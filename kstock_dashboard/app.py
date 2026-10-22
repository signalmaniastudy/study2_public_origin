# 실행: streamlit run app.py   (kstock_dashboard 폴더에서)
from __future__ import annotations

from datetime import date, datetime, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from kdash import calendar as cal
from kdash.backtest import run_backtest, summarize
from kdash.data import load_market, build_snapshot, latest_quotes
from kdash.scoring import compute_score, load_config

st.set_page_config(page_title="한국 증시 대시보드", page_icon="📈", layout="wide")
UP, DOWN = "#d62728", "#1f77b4"  # 한국식: 상승 빨강, 하락 파랑


@st.cache_data(ttl=600, show_spinner="데이터 불러오는 중…")
def get_market(mode: str):
    return load_market(mode), datetime.now(cal.KST)


@st.cache_data(ttl=3600)
def get_events():
    raw = cal.load_calendar_file()
    return raw, cal.build_events(raw)


@st.cache_data(ttl=3600, show_spinner="백테스트 계산 중…")
def get_backtest(mode: str, horizon: str):
    return run_backtest(get_market(mode)[0], horizon)


def target_session(holidays) -> date:
    """예측 대상 한국장 거래일: 장 마감 전이면 오늘, 이후면 다음 거래일."""
    now = datetime.now(cal.KST)
    today = now.date()
    if cal.is_trading_day(today, holidays) and now.time() < cal.KRX_CLOSE:
        return today
    return cal.next_trading_day(today, holidays)


# ---------------------------------------------------------------- 사이드바
cfg = load_config()
raw_cal, events = get_events()
holidays = cal.krx_holidays(raw_cal)

with st.sidebar:
    st.header("설정")
    mode = st.radio("데이터", ["auto", "live", "sample"], index=0,
                    format_func={"auto": "자동 (실패 시 샘플)", "live": "실데이터만", "sample": "샘플"}.get)
    st.subheader("데이터 새로고침")
    refresh_now = st.button("🔄 지금 새로고침", use_container_width=True)
    refresh_min = st.selectbox("자동 새로고침", [0, 1, 5, 10, 30], index=0,
                               format_func=lambda m: "끄기" if m == 0 else f"{m}분마다")
    st.subheader("E. 업황·이벤트 (수동 입력)")
    st.caption("무료 자동 수집이 어려운 지표입니다. 입력하지 않으면 결측으로 처리됩니다.")
    manual = {}
    if st.checkbox("DRAM 현물가 입력"):
        manual["dram_chg"] = st.number_input("DRAM 현물가 2주 변화율(%)", -30.0, 30.0, 0.0, 0.5) / 100
    if st.checkbox("실적 전망 변화 입력"):
        manual["eps_revision"] = st.select_slider("EPS·목표가 1~2주 변화", [-1, 0, 1], 0,
                                                  format_func={-1: "하향", 0: "변화 없음", 1: "상향"}.get)
    if st.checkbox("반도체 수출 서프라이즈 입력"):
        manual["export_surprise_pp"] = st.number_input("수출 증가율 − 직전 추세 (%p)", -50.0, 50.0, 0.0, 1.0)
    if st.checkbox("뉴스 감성 입력"):
        manual["news"] = st.select_slider("뉴스 감성", [-1, 0, 1], 0,
                                          format_func={-1: "부정", 0: "중립", 1: "긍정"}.get)
    if st.checkbox("VKOSPI 입력"):
        manual["vkospi"] = st.number_input("VKOSPI", 5.0, 80.0, 20.0, 0.5)
    st.divider()
    st.caption("분석 보조 도구이며 투자 권유가 아닙니다. 가중치는 백테스트 전 출발값입니다.")

if refresh_now:
    get_market.clear()
    get_backtest.clear()


@st.fragment(run_every=refresh_min * 60 if refresh_min else None)
def auto_refresh():
    """전체 실행 때는 표시만 하고, 타이머로 다시 실행될 때 캐시를 비우고 화면 전체를 새로 그린다."""
    if st.session_state.pop("_full_run", False):
        return
    get_market.clear()
    get_backtest.clear()
    st.rerun(scope="app")


st.session_state["_full_run"] = True
auto_refresh()

md, fetched_at = get_market(mode)
today = datetime.now(cal.KST).date()
target = target_session(holidays)

st.title("📈 한국 증시 단기 대시보드")
st.caption(f"마지막 업데이트: {fetched_at:%Y-%m-%d %H:%M:%S} KST"
           + (f" · {refresh_min}분마다 자동 새로고침" if refresh_min else " · 자동 새로고침 꺼짐")
           + " · 국내 시세는 yfinance 기준 약 15~20분 지연될 수 있음")
if md.source == "sample":
    st.warning("⚠️ **샘플(합성) 데이터**로 표시 중입니다. 화면 확인용이며 실제 시장과 무관합니다."
               + (f" (실데이터 수집 실패 {len(md.errors)}건)" if md.errors else ""))
if md.errors:
    with st.expander(f"데이터 수집 오류 {len(md.errors)}건"):
        for e in md.errors:
            st.caption(e[:200])

# ---------------------------------------------------------------- KPI
quotes = latest_quotes(md).set_index("key")
kpi_keys = ["kospi", "kosdaq", "hynix", "usdkrw", "sox", "vix"]
cols = st.columns(len(kpi_keys))
for c, k in zip(cols, kpi_keys):
    if k in quotes.index:
        q = quotes.loc[k]
        c.metric(q["지표"], f"{q['현재값']:,.2f}", f"{q['변화']:+.2%}", delta_color="inverse" if k in ("usdkrw", "vix") else "normal")

nxt = cal.next_major_event(events, today)
if nxt is not None:
    dday = (nxt["impact_date"] - today).days
    st.markdown(f"**다음 1등급 이벤트:** {nxt['icon']} {nxt['title']} · 한국장 반영 {nxt['impact_date']:%m/%d} "
                f"(**D-{dday}**) · 발표 {nxt['kst_label']} KST")

tab_score, tab_cal, tab_market, tab_bt = st.tabs(["🎯 SK하이닉스 점수", "📅 이벤트 달력", "🌐 시장 지표", "🧪 백테스트"])

# ---------------------------------------------------------------- 점수
with tab_score:
    st.caption(f"예측 대상 거래일: **{target:%Y-%m-%d}** · {target:%Y-%m-%d} 이전에 확정된 데이터만 사용")
    snap = build_snapshot(md, target, manual)
    results = {}
    for h in ("1d", "5d"):
        mult, notes, multi, _ = cal.risk_adjustment(events, target, cfg["horizons"][h]["days"], holidays, cfg["filters"], h)
        results[h] = compute_score(snap, h, cfg, mult, notes, multi)

    c1, c2 = st.columns(2)
    for c, h in zip((c1, c2), ("1d", "5d")):
        r = results[h]
        with c:
            st.subheader(cfg["horizons"][h]["label"])
            if r.score is None:
                st.error("판단 보류")
            else:
                color = UP if r.score > 0 else DOWN
                st.markdown(f"<div style='font-size:3rem;font-weight:700;color:{color}'>{r.score:+.1f}</div>",
                            unsafe_allow_html=True)
                st.markdown(f"**{r.judgement}** · 신뢰도 **{r.confidence}**"
                            + (f" · 원점수 {r.raw_score:+.1f} × {r.multiplier:g}" if r.multiplier < 1 else ""))
            for n in r.notes:
                st.caption("• " + n)
            contrib = pd.DataFrame([{"영역": a.label, "기여도": a.contribution} for a in r.areas if a.score is not None])
            fig = go.Figure(go.Bar(x=contrib["기여도"], y=contrib["영역"], orientation="h",
                                   marker_color=[UP if v > 0 else DOWN for v in contrib["기여도"]]))
            fig.update_layout(height=230, margin=dict(l=0, r=0, t=10, b=0), xaxis_title="점수 기여도",
                              yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, use_container_width=True, key=f"contrib_{h}")

    if "atr14" in snap:
        st.caption(f"매매 규칙 예시 (검증용): 점수 +30 이상 진입 · 손절 1.5×ATR = "
                   f"{1.5 * snap['atr14']:,.0f}원 · 5거래일 경과 또는 점수 0 미만 시 청산")

    with st.expander("세부 지표 보기 (1일 예측 기준)"):
        rows = []
        for a in results["1d"].areas:
            for ind in a.indicators:
                vals = ", ".join("-" if v is None else f"{v:,.4g}" for v in ind["입력값"])
                rows.append({"영역": a.label, "지표": ind["지표"], "입력값": vals, "비중": ind["비중"],
                             "점수": None if ind["점수"] is None else round(ind["점수"], 2)})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

# ---------------------------------------------------------------- 달력
with tab_cal:
    f1, f2, f3 = st.columns([2, 2, 1])
    view = f1.radio("보기", ["2주", "월간 목록"], horizontal=True)
    cats = sorted(events["category"].unique())
    sel_cats = f2.multiselect("분류", cats, default=cats)
    show_t3 = f3.checkbox("3등급 표시", value=False)
    ev = events[events["category"].isin(sel_cats) & ((events["tier"] <= 2) | show_t3)]

    if view == "2주":
        monday = today - timedelta(days=today.weekday())
        for w in range(2):
            cols = st.columns(5)
            for i, c in enumerate(cols):
                d = monday + timedelta(days=7 * w + i)
                closed = d in holidays
                head = f"**{'월화수목금'[i]} {d:%m/%d}**" + (" · 휴장" if closed else "") + (" ◀ 오늘" if d == today else "")
                with c.container(border=True):
                    st.markdown(head)
                    day = ev[ev["impact_date"] == d]
                    if day.empty:
                        st.caption("—")
                    for _, e in day.iterrows():
                        st.markdown(f"{e['icon']} {e['title']}  \n<small>발표 {e['kst_label']}"
                                    f"{' · 예상일' if e['status'] == 'estimated' else ''}</small>",
                                    unsafe_allow_html=True)
        st.caption("날짜 칸은 **한국장에 반영되는 거래일** 기준입니다. 예: 미국 CPI(21:30/22:30 발표)는 다음 거래일 칸에 표시")
        win1 = cal.risk_adjustment(events, target, 1, holidays, cfg["filters"], "1d")
        win5 = cal.risk_adjustment(events, target, 5, holidays, cfg["filters"], "5d")
        st.info(f"현재 점수 배수 — 1일 예측 ×{win1[0]:g}, 5일 예측 ×{win5[0]:g}"
                + (" · 1등급 2건 이상: 관망 권고" if win5[2] else ""))
    else:
        month_start = today.replace(day=1)
        month = st.date_input("기준 월", month_start)
        m0 = month.replace(day=1)
        m1 = (m0 + timedelta(days=32)).replace(day=1)
        lst = ev[(ev["impact_date"] >= m0) & (ev["impact_date"] < m1)]
        st.dataframe(
            lst[["impact_date", "icon", "title", "kst_label", "category", "tier", "status", "note"]].rename(columns={
                "impact_date": "한국장 반영일", "icon": "", "title": "이벤트", "kst_label": "발표(KST)",
                "category": "분류", "tier": "등급", "status": "상태", "note": "비고"}),
            hide_index=True, use_container_width=True)
    st.caption("일정은 data/events.yaml 에서 관리합니다. 예상일(estimated)은 공식 캘린더로 확인하세요.")

# ---------------------------------------------------------------- 시장 지표
with tab_market:
    hx = md.prices.get("hynix")
    if hx is not None:
        h = hx.iloc[-120:]
        fig = go.Figure(go.Candlestick(x=h.index, open=h["Open"], high=h["High"], low=h["Low"], close=h["Close"],
                                       increasing_line_color=UP, decreasing_line_color=DOWN, name="SK하이닉스"))
        for n, color in ((5, "#ff7f0e"), (20, "#2ca02c")):
            fig.add_trace(go.Scatter(x=h.index, y=hx["Close"].rolling(n).mean().iloc[-120:], name=f"{n}일선",
                                     line=dict(width=1.5, color=color)))
        fig.update_layout(height=380, margin=dict(l=0, r=0, t=30, b=0), title="SK하이닉스 (최근 120거래일)",
                          xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    if md.flows is not None:
        fl = md.flows.iloc[-40:] / 1e8
        fig = go.Figure([go.Bar(x=fl.index, y=fl["foreign"], name="외국인"), go.Bar(x=fl.index, y=fl["inst"], name="기관")])
        fig.update_layout(height=320, barmode="group", title="SK하이닉스 순매수 (억 원)", margin=dict(l=0, r=0, t=30, b=0))
        c1.plotly_chart(fig, use_container_width=True)
    else:
        c1.info("수급 데이터 없음 (pykrx 수집 실패)")
    tbl = quotes.reset_index(drop=True)
    tbl["변화"] = [f"{v:+.1f}bp" if k == "us10y" else f"{v:+.2%}" for k, v in zip(quotes.index, quotes["변화"])]
    tbl["현재값"] = tbl["현재값"].map(lambda v: f"{v:,.2f}")
    c2.dataframe(tbl, hide_index=True, use_container_width=True, height=320)

# ---------------------------------------------------------------- 백테스트
with tab_bt:
    st.caption("A~D 영역만 사용 (E 영역과 이벤트 필터는 과거 값이 없어 제외). "
               "수익률은 실제 매매가 가능한 **T일 시가 → T+h-1일 종가** 기준이며 거래비용은 반영하지 않았습니다.")
    if md.source == "sample":
        st.warning("샘플 데이터의 백테스트 결과는 의미가 없습니다. 실데이터로 실행하세요.")
    hz = st.radio("예측 기간", ["1d", "5d"], horizontal=True, format_func=lambda h: cfg["horizons"][h]["label"])
    if st.button("백테스트 실행"):
        bt = get_backtest(mode, hz)
        table, stats = summarize(bt)
        m = st.columns(4)
        m[0].metric("표본 수", stats.get("표본수", 0))
        m[1].metric("IC (Spearman)", f"{stats.get('IC', float('nan')):.3f}")
        m[2].metric("전체 상승확률", f"{stats.get('전체 상승확률', float('nan')):.1%}")
        m[3].metric("+30 이상 상승확률", f"{stats.get('+30 이상 상승확률', float('nan')):.1%}",
                    help=f"진입 {stats.get('+30 이상 진입 횟수', 0)}회")
        st.dataframe(table.style.format({"평균수익률": "{:+.2%}", "상승확률": "{:.1%}"}), use_container_width=True)
        st.caption("판단 기준: IC 0.05 이상이면 의미 있는 수준. 점수 구간이 높을수록 평균 수익률이 높아지는지 확인하세요.")
