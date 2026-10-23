# 실행: streamlit run app.py   (kstock_dashboard 폴더에서)
from __future__ import annotations

from datetime import date, datetime, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from kdash import calendar as cal
from kdash.backtest import run_backtest, summarize
from kdash.data import build_snapshot, last_change, latest_quotes, load_common, load_peers, load_stock
from kdash.scoring import compute_score, load_config, peer_tickers, profile_config
from kdash.stocks import load_stock_list, load_watchlist, save_watchlist, stock_info

st.set_page_config(page_title="한국 증시 대시보드", page_icon="📈", layout="wide")
UP, DOWN = "#d62728", "#1f77b4"  # 한국식: 상승 빨강, 하락 파랑
MAX_WATCH = 20


# ---------------------------------------------------------------- 캐시된 로더
@st.cache_data(ttl=600, show_spinner="시장 데이터 불러오는 중…")
def get_common(mode: str):
    return load_common(mode), datetime.now(cal.KST)


@st.cache_data(ttl=600, show_spinner="종목 데이터 불러오는 중…")
def get_stock(code: str, market: str, mode: str):
    return load_stock(code, market, mode)


@st.cache_data(ttl=600, show_spinner="해외 연관 지표 불러오는 중…")
def get_peers(tickers: tuple, mode: str):
    return load_peers(list(tickers), mode)


@st.cache_data(ttl=86400, show_spinner="종목 목록 불러오는 중…")
def get_stock_list():
    return load_stock_list()


@st.cache_data(ttl=3600)
def get_events():
    raw = cal.load_calendar_file()
    return raw, cal.build_events(raw)


@st.cache_data(ttl=3600, show_spinner="백테스트 계산 중…")
def get_backtest(code: str, market: str, sector: str, mode: str, horizon: str):
    scfg = profile_config(load_config(), sector)
    peers = get_peers(tuple(peer_tickers(scfg, [sector])), mode)[0]
    return run_backtest(get_common(mode)[0], get_stock(code, market, mode), peers, horizon, cfg=scfg)


def clear_data_cache():
    for f in (get_common, get_stock, get_peers, get_backtest):
        f.clear()


def target_session(holidays) -> date:
    """예측 대상 한국장 거래일: 장 마감 전이면 오늘, 이후면 다음 거래일."""
    now = datetime.now(cal.KST)
    today = now.date()
    if cal.is_trading_day(today, holidays) and now.time() < cal.KRX_CLOSE:
        return today
    return cal.next_trading_day(today, holidays)


# ---------------------------------------------------------------- 상태
cfg = load_config()
profiles = cfg["sector_profiles"]
raw_cal, events = get_events()
holidays = cal.krx_holidays(raw_cal)
stocks_df, list_source = get_stock_list()
names = dict(zip(stocks_df["code"], stocks_df["name"]))

ss = st.session_state
if "watch" not in ss:
    ss.watch = load_watchlist()
    ss.watch_sel = list(ss.watch["codes"])
if "manual" not in ss:
    ss.manual = {}  # code -> E 영역 수동 입력값


def label(code: str) -> str:
    return f"{names.get(code, code)} ({code})"


def on_watch_change():
    ss.watch["codes"] = list(ss.watch_sel)
    save_watchlist(ss.watch)


def add_code():
    code = ss.add_code_input.strip().upper()
    ss.add_code_input = ""
    if len(code) == 6 and code.isalnum() and code not in ss.watch_sel and len(ss.watch_sel) < MAX_WATCH:
        ss.watch_sel = ss.watch_sel + [code]
        on_watch_change()


def set_sector(code: str):
    ss.watch.setdefault("sectors", {})[code] = ss[f"sector_{code}"]
    save_watchlist(ss.watch)


def set_manual(code: str, field: str, key: str, scale: float = 1.0):
    v = ss[key]
    ss.manual.setdefault(code, {})[field] = None if v is None or v == "미입력" else v * scale


def on_table_select():
    rows = ss.overview_table.selection.rows
    if rows and rows[0] < len(ss.watch["codes"]):
        ss.detail_code = ss.watch["codes"][rows[0]]


# ---------------------------------------------------------------- 사이드바
with st.sidebar:
    st.header("설정")
    mode = st.radio("데이터", ["auto", "live", "sample"], index=0,
                    format_func={"auto": "자동 (실패 시 샘플)", "live": "실데이터만", "sample": "샘플"}.get)

    st.subheader("⭐ 관심종목")
    options = list(stocks_df["code"]) + [c for c in ss.watch_sel if c not in names]
    st.multiselect("종목 검색 (이름 또는 코드 입력)", options, key="watch_sel", format_func=label,
                   max_selections=MAX_WATCH, on_change=on_watch_change, placeholder="예: 삼성전자, 005930")
    st.caption(f"검색 목록: {list_source} {len(stocks_df):,}개 · 관심종목은 watchlist.json 에 저장됩니다.")
    with st.expander("목록에 없는 종목 코드 직접 추가"):
        st.text_input("종목 코드 6자리", key="add_code_input", placeholder="예: 000660")
        st.button("추가", on_click=add_code)

    st.subheader("데이터 새로고침")
    refresh_now = st.button("🔄 지금 새로고침", use_container_width=True)
    refresh_min = st.selectbox("자동 새로고침", [0, 1, 5, 10, 30], index=0,
                               format_func=lambda m: "끄기" if m == 0 else f"{m}분마다")
    vkospi = st.number_input("VKOSPI (선택 입력)", 5.0, 80.0, value=None, step=0.5,
                             help="입력하면 30 초과 시 고변동성 필터가 적용됩니다.")
    st.divider()
    st.caption("분석 보조 도구이며 투자 권유가 아닙니다. 가중치는 백테스트 전 출발값입니다.")

if refresh_now:
    clear_data_cache()


@st.fragment(run_every=refresh_min * 60 if refresh_min else None)
def auto_refresh():
    """전체 실행 때는 표시만 하고, 타이머로 다시 실행될 때 캐시를 비우고 화면 전체를 새로 그린다."""
    if ss.pop("_full_run", False):
        return
    clear_data_cache()
    st.rerun(scope="app")


ss["_full_run"] = True
auto_refresh()

# ---------------------------------------------------------------- 데이터
md, fetched_at = get_common(mode)
today = datetime.now(cal.KST).date()
target = target_session(holidays)
watch_codes = list(ss.watch["codes"])
infos = {c: stock_info(stocks_df, c, ss.watch) for c in watch_codes}
peers, peer_errors, peers_sample = get_peers(tuple(peer_tickers(cfg, [i["sector"] for i in infos.values()])), mode)

sample_parts, errors = [], list(md.errors) + peer_errors
if md.source == "sample":
    sample_parts.append("시장 지표")
if peers_sample:
    sample_parts.append("해외 연관 지표")


def analyze(code: str) -> dict:
    info = infos[code]
    sd = get_stock(code, info["market"], mode)
    scfg = profile_config(cfg, info["sector"])
    manual = dict(ss.manual.get(code, {}))
    if vkospi is not None:
        manual["vkospi"] = vkospi
    snap = build_snapshot(md, sd, peers, target, manual)
    stock_events = events[cal.applies_to(events, code, info["sector"])] if not events.empty else events
    results = {}
    for h in ("1d", "5d"):
        mult, notes, multi, _ = cal.risk_adjustment(stock_events, target, cfg["horizons"][h]["days"],
                                                    holidays, cfg["filters"], h)
        results[h] = compute_score(snap, h, scfg, mult, notes, multi)
    return {"info": info, "sd": sd, "cfg": scfg, "snap": snap, "results": results, "events": stock_events}


analyses = {c: analyze(c) for c in watch_codes}
for c, a in analyses.items():
    if a["sd"].source == "sample":
        sample_parts.append(names.get(c, c))
    errors += a["sd"].errors

# ---------------------------------------------------------------- 상단
st.title("📈 한국 증시 단기 대시보드")
st.caption(f"마지막 업데이트: {fetched_at:%Y-%m-%d %H:%M:%S} KST"
           + (f" · {refresh_min}분마다 자동 새로고침" if refresh_min else " · 자동 새로고침 꺼짐")
           + " · 국내 시세는 yfinance 기준 약 15~20분 지연될 수 있음")
if sample_parts:
    st.warning("⚠️ 다음 항목은 **샘플(합성) 데이터**로 표시 중이며 실제 시장과 무관합니다: " + ", ".join(sample_parts))
if errors:
    with st.expander(f"데이터 수집 오류 {len(errors)}건"):
        for e in errors:
            st.caption(e[:200])

quotes = latest_quotes(md).set_index("key")
kpi_keys = ["kospi", "kosdaq", "usdkrw", "nasdaq", "sox", "vix"]
cols = st.columns(len(kpi_keys))
for c, k in zip(cols, kpi_keys):
    if k in quotes.index:
        q = quotes.loc[k]
        c.metric(q["지표"], f"{q['현재값']:,.2f}", f"{q['변화']:+.2%}",
                 delta_color="inverse" if k in ("usdkrw", "vix") else "normal")

market_wide = events[(events["ticker"] == "") & (events["sectors"].map(len) == 0)] if not events.empty else events
nxt = cal.next_major_event(market_wide, today)
if nxt is not None:
    st.markdown(f"**다음 1등급 시장 이벤트:** {nxt['icon']} {nxt['title']} · 한국장 반영 {nxt['impact_date']:%m/%d} "
                f"(**D-{(nxt['impact_date'] - today).days}**) · 발표 {nxt['kst_label']} KST")

if not watch_codes:
    st.info("왼쪽 사이드바의 **⭐ 관심종목**에서 종목을 검색해 추가하세요. (사이드바가 안 보이면 왼쪽 위 `>` 버튼)")
    st.stop()

if ss.get("detail_code") not in watch_codes:
    ss.detail_code = watch_codes[0]

tab_watch, tab_detail, tab_cal, tab_market, tab_bt = st.tabs(
    ["⭐ 관심종목", "🎯 종목 상세", "📅 이벤트 달력", "🌐 시장 지표", "🧪 백테스트"])


def fmt_score(v):
    return None if v is None else round(v, 1)


def next_stock_event(a: dict):
    ev = a["events"]
    own = ev[((ev["ticker"] != "") | (ev["sectors"].map(len) > 0)) & (ev["impact_date"] >= today) & (ev["tier"] <= 2)]
    if own.empty:
        return ""
    e = own.iloc[0]
    return f"{e['icon']} {e['impact_date']:%m/%d} {e['title']}"


# ---------------------------------------------------------------- 관심종목
with tab_watch:
    st.caption(f"예측 대상 거래일: **{target:%Y-%m-%d}** · 행을 클릭하면 **🎯 종목 상세** 탭에서 그 종목이 선택됩니다.")
    rows = []
    for code in watch_codes:
        a = analyses[code]
        lc = last_change(a["sd"].ohlcv)
        r1, r5 = a["results"]["1d"], a["results"]["5d"]
        rows.append({
            "종목": label(code),
            "업종": profiles.get(a["info"]["sector"], profiles["default"])["label"],
            "현재가": lc[0] if lc else None,
            "등락률": lc[1] if lc else None,
            "1일 점수": fmt_score(r1.score), "1일 판단": r1.judgement,
            "5일 점수": fmt_score(r5.score), "5일 판단": r5.judgement,
            "신뢰도(5일)": r5.confidence,
            "관련 이벤트": next_stock_event(a),
            "데이터": "샘플" if a["sd"].source == "sample" else "실데이터",
        })
    table = pd.DataFrame(rows)

    def color(v):
        if v is None or pd.isna(v) or v == 0:
            return ""
        return f"color: {UP if v > 0 else DOWN}; font-weight: 600"

    styled = (table.style
              .map(color, subset=["등락률", "1일 점수", "5일 점수"])
              .format({"현재가": "{:,.0f}", "등락률": "{:+.2%}", "1일 점수": "{:+.1f}", "5일 점수": "{:+.1f}"},
                      na_rep="-"))
    st.dataframe(styled, hide_index=True, use_container_width=True, key="overview_table",
                 on_select=on_table_select, selection_mode="single-row")

    chart = table.dropna(subset=["5일 점수"])
    if len(chart):
        fig = go.Figure()
        fig.add_trace(go.Bar(name="1일 점수", x=chart["종목"], y=chart["1일 점수"], marker_color="#9ecae1"))
        fig.add_trace(go.Bar(name="5일 점수", x=chart["종목"], y=chart["5일 점수"],
                             marker_color=[UP if v > 0 else DOWN for v in chart["5일 점수"]]))
        for y in (20, -20):
            fig.add_hline(y=y, line_dash="dot", line_color="gray")
        fig.update_layout(height=320, barmode="group", yaxis_range=[-100, 100], margin=dict(l=0, r=0, t=30, b=0),
                          title="관심종목 점수 비교 (점선: ±20 중립 구간)")
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------- 종목 상세
with tab_detail:
    code = st.selectbox("종목 선택", watch_codes, key="detail_code", format_func=label)
    a = analyses[code]
    info, sd, scfg, snap, results = a["info"], a["sd"], a["cfg"], a["snap"], a["results"]
    st.caption(f"예측 대상 거래일: **{target:%Y-%m-%d}** · {target:%Y-%m-%d} 이전에 확정된 데이터만 사용")

    s1, s2 = st.columns([1, 2])
    sector_keys = list(profiles)
    s1.selectbox("업종 프로필", sector_keys, index=sector_keys.index(info["sector"]) if info["sector"] in sector_keys else 0,
                 key=f"sector_{code}", format_func=lambda k: profiles[k]["label"], on_change=set_sector, args=(code,),
                 help="A 영역(글로벌 연관 지표)에 쓸 해외 지표를 정합니다. 바꾸면 저장됩니다.")
    s2.caption("A 영역 연관 지표: " + ", ".join(p["label"] for p in profiles[info["sector"]]["peers"]))

    c1, c2 = st.columns(2)
    for c, h in zip((c1, c2), ("1d", "5d")):
        r = results[h]
        with c:
            st.subheader(cfg["horizons"][h]["label"])
            if r.score is None or r.judgement == "판단 보류":
                st.error("판단 보류")
            if r.score is not None:
                color_ = UP if r.score > 0 else DOWN
                st.markdown(f"<div style='font-size:3rem;font-weight:700;color:{color_}'>{r.score:+.1f}</div>",
                            unsafe_allow_html=True)
                st.markdown(f"**{r.judgement}** · 신뢰도 **{r.confidence}**"
                            + (f" · 원점수 {r.raw_score:+.1f} × {r.multiplier:g}" if r.multiplier < 1 else ""))
            for n in r.notes:
                st.caption("• " + n)
            contrib = pd.DataFrame([{"영역": ar.label, "기여도": ar.contribution} for ar in r.areas if ar.score is not None])
            if len(contrib):
                fig = go.Figure(go.Bar(x=contrib["기여도"], y=contrib["영역"], orientation="h",
                                       marker_color=[UP if v > 0 else DOWN for v in contrib["기여도"]]))
                fig.update_layout(height=230, margin=dict(l=0, r=0, t=10, b=0), xaxis_title="점수 기여도",
                                  yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig, use_container_width=True, key=f"contrib_{code}_{h}")

    if "atr14" in snap:
        st.caption(f"매매 규칙 예시 (검증용): 점수 +30 이상 진입 · 손절 1.5×ATR = "
                   f"{1.5 * snap['atr14']:,.0f}원 · 5거래일 경과 또는 점수 0 미만 시 청산")

    with st.expander("✏️ E 영역 수동 입력 (이 종목)"):
        st.caption("무료 자동 수집이 어려운 지표입니다. 비워 두면 결측으로 처리되고 가중치는 나머지 영역에 재배분됩니다.")
        cur = ss.manual.get(code, {})
        m1, m2, m3, m4 = st.columns(4)
        k = f"m_{code}_industry"
        m1.number_input(profiles[info["sector"]].get("industry_label", "업황 지표 2주 변화율") + " (%)", -50.0, 50.0,
                        value=None if cur.get("industry_chg") is None else cur["industry_chg"] * 100, step=0.5,
                        key=k, on_change=set_manual, args=(code, "industry_chg", k, 0.01))
        k = f"m_{code}_eps"
        m2.selectbox("실적 전망·목표가 1~2주 변화", ["미입력", -1, 0, 1], key=k,
                     index=["미입력", -1, 0, 1].index(cur.get("eps_revision", "미입력") if cur.get("eps_revision") is not None else "미입력"),
                     format_func={"미입력": "미입력", -1: "하향", 0: "변화 없음", 1: "상향"}.get,
                     on_change=set_manual, args=(code, "eps_revision", k))
        k = f"m_{code}_export"
        m3.number_input("수출 서프라이즈 (%p)", -50.0, 50.0, value=cur.get("export_surprise_pp"), step=1.0,
                        key=k, on_change=set_manual, args=(code, "export_surprise_pp", k))
        k = f"m_{code}_news"
        m4.selectbox("뉴스 감성", ["미입력", -1, 0, 1], key=k,
                     index=["미입력", -1, 0, 1].index(cur.get("news") if cur.get("news") is not None else "미입력"),
                     format_func={"미입력": "미입력", -1: "부정", 0: "중립", 1: "긍정"}.get,
                     on_change=set_manual, args=(code, "news", k))

    with st.expander("세부 지표 보기 (1일 예측 기준)"):
        drows = []
        for ar in results["1d"].areas:
            for ind in ar.indicators:
                vals = ", ".join("-" if v is None else f"{v:,.4g}" for v in ind["입력값"])
                drows.append({"영역": ar.label, "지표": ind["지표"], "입력값": vals, "비중": ind["비중"],
                              "점수": None if ind["점수"] is None else round(ind["점수"], 2)})
        st.dataframe(pd.DataFrame(drows), hide_index=True, use_container_width=True)

    if sd.ohlcv is not None:
        hx = sd.ohlcv
        h = hx.iloc[-120:]
        fig = go.Figure(go.Candlestick(x=h.index, open=h["Open"], high=h["High"], low=h["Low"], close=h["Close"],
                                       increasing_line_color=UP, decreasing_line_color=DOWN, name=names.get(code, code)))
        for n, color_ in ((5, "#ff7f0e"), (20, "#2ca02c")):
            fig.add_trace(go.Scatter(x=h.index, y=hx["Close"].rolling(n).mean().iloc[-120:], name=f"{n}일선",
                                     line=dict(width=1.5, color=color_)))
        fig.update_layout(height=380, margin=dict(l=0, r=0, t=30, b=0), title=f"{label(code)} (최근 120거래일)",
                          xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True, key=f"candle_{code}")
    if sd.flows is not None:
        fl = sd.flows.iloc[-40:] / 1e8
        fig = go.Figure([go.Bar(x=fl.index, y=fl["foreign"], name="외국인"), go.Bar(x=fl.index, y=fl["inst"], name="기관")])
        fig.update_layout(height=300, barmode="group", title="순매수 (억 원)", margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig, use_container_width=True, key=f"flows_{code}")
    else:
        st.info("수급 데이터 없음 (pykrx 수집 실패)")

# ---------------------------------------------------------------- 달력
with tab_cal:
    f1, f2, f3 = st.columns([2, 2, 1])
    view = f1.radio("보기", ["2주", "월간 목록"], horizontal=True)
    cats = sorted(events["category"].unique())
    sel_cats = f2.multiselect("분류", cats, default=cats)
    show_t3 = f3.checkbox("3등급 표시", value=False)
    relevant = pd.Series(False, index=events.index)
    for code in watch_codes:
        relevant |= cal.applies_to(events, code, infos[code]["sector"])
    ev = events[relevant & events["category"].isin(sel_cats) & ((events["tier"] <= 2) | show_t3)].copy()
    ev["표시"] = ev["title"] + ev["sectors"].map(
        lambda s: f" [{', '.join(profiles[x]['label'] for x in s if x in profiles)}]" if s else "")

    if view == "2주":
        monday = today - timedelta(days=today.weekday())
        for w in range(2):
            cols = st.columns(5)
            for i, c in enumerate(cols):
                d = monday + timedelta(days=7 * w + i)
                head = f"**{'월화수목금'[i]} {d:%m/%d}**" + (" · 휴장" if d in holidays else "") + (" ◀ 오늘" if d == today else "")
                with c.container(border=True):
                    st.markdown(head)
                    day = ev[ev["impact_date"] == d]
                    if day.empty:
                        st.caption("—")
                    for _, e in day.iterrows():
                        st.markdown(f"{e['icon']} {e['표시']}  \n<small>발표 {e['kst_label']}"
                                    f"{' · 예상일' if e['status'] == 'estimated' else ''}</small>",
                                    unsafe_allow_html=True)
        st.caption("관심종목에 해당하는 이벤트만 표시합니다(시장 전체 + 종목·업종 이벤트). "
                   "날짜 칸은 **한국장에 반영되는 거래일** 기준입니다.")
        mrows = [{"종목": label(c), "1일 예측 배수": analyses[c]["results"]["1d"].multiplier,
                  "5일 예측 배수": analyses[c]["results"]["5d"].multiplier} for c in watch_codes]
        st.dataframe(pd.DataFrame(mrows), hide_index=True, use_container_width=True)
    else:
        month = st.date_input("기준 월", today.replace(day=1))
        m0 = month.replace(day=1)
        m1 = (m0 + timedelta(days=32)).replace(day=1)
        lst = ev[(ev["impact_date"] >= m0) & (ev["impact_date"] < m1)]
        st.dataframe(
            lst[["impact_date", "icon", "표시", "kst_label", "category", "tier", "status", "note"]].rename(columns={
                "impact_date": "한국장 반영일", "icon": "", "표시": "이벤트", "kst_label": "발표(KST)",
                "category": "분류", "tier": "등급", "status": "상태", "note": "비고"}),
            hide_index=True, use_container_width=True)
    st.caption("일정은 data/events.yaml 에서 관리합니다. 예상일(estimated)은 공식 캘린더로 확인하세요.")

# ---------------------------------------------------------------- 시장 지표
with tab_market:
    kospi = md.prices.get("kospi")
    c1, c2 = st.columns([3, 2])
    if kospi is not None:
        h = kospi.iloc[-120:]
        fig = go.Figure(go.Candlestick(x=h.index, open=h["Open"], high=h["High"], low=h["Low"], close=h["Close"],
                                       increasing_line_color=UP, decreasing_line_color=DOWN, name="KOSPI"))
        fig.update_layout(height=400, margin=dict(l=0, r=0, t=30, b=0), title="KOSPI (최근 120거래일)",
                          xaxis_rangeslider_visible=False)
        c1.plotly_chart(fig, use_container_width=True)
    tbl = quotes.reset_index(drop=True)
    tbl["변화"] = [f"{v:+.1f}bp" if k == "us10y" else f"{v:+.2%}" for k, v in zip(quotes.index, quotes["변화"])]
    tbl["현재값"] = tbl["현재값"].map(lambda v: f"{v:,.2f}")
    c2.dataframe(tbl, hide_index=True, use_container_width=True, height=400)

# ---------------------------------------------------------------- 백테스트
with tab_bt:
    st.caption("A~D 영역만 사용 (E 영역과 이벤트 필터는 과거 값이 없어 제외). "
               "수익률은 실제 매매가 가능한 **T일 시가 → T+h-1일 종가** 기준이며 거래비용은 반영하지 않았습니다.")
    b1, b2 = st.columns(2)
    bcode = b1.selectbox("종목", watch_codes, index=watch_codes.index(ss.detail_code), format_func=label, key="bt_code")
    hz = b2.radio("예측 기간", ["1d", "5d"], horizontal=True, format_func=lambda h: cfg["horizons"][h]["label"])
    if analyses[bcode]["sd"].source == "sample" or md.source == "sample":
        st.warning("샘플 데이터의 백테스트 결과는 의미가 없습니다. 실데이터로 실행하세요.")
    if st.button("백테스트 실행"):
        binfo = infos[bcode]
        bt = get_backtest(bcode, binfo["market"], binfo["sector"], mode, hz)
        table, stats = summarize(bt)
        if not stats:
            st.error("백테스트할 데이터가 없습니다.")
        else:
            m = st.columns(4)
            m[0].metric("표본 수", stats["표본수"])
            m[1].metric("IC (Spearman)", f"{stats['IC']:.3f}")
            m[2].metric("전체 상승확률", f"{stats['전체 상승확률']:.1%}")
            m[3].metric("+30 이상 상승확률", f"{stats['+30 이상 상승확률']:.1%}",
                        help=f"진입 {stats['+30 이상 진입 횟수']}회")
            st.dataframe(table.style.format({"평균수익률": "{:+.2%}", "상승확률": "{:.1%}"}), use_container_width=True)
            st.caption("판단 기준: IC 0.05 이상이면 의미 있는 수준. 점수 구간이 높을수록 평균 수익률이 높아지는지 확인하세요.")
