# dashboard_app.py
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import yfinance as yf

from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, roc_auc_score

import matplotlib.pyplot as plt
import streamlit as st
from datetime import date

# =========================
# 유틸: 다운로드/정리
# =========================
def _flatten_cols(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [str(t[0]) for t in df.columns]
    else:
        df.columns = [str(c) for c in df.columns]
    return df

def _download_ohlcv(ticker: str, start="2015-01-01") -> pd.DataFrame:
    df = yf.download(ticker, start=start, interval="1d",
                     auto_adjust=True, progress=False, threads=False)
    if df is None or df.empty:
        raise ValueError(f"empty data for {ticker}")
    df = _flatten_cols(df).copy()
    need = ["Open","High","Low","Close","Volume"]
    for c in need:
        if c not in df.columns:
            if c == "Volume": df[c] = 0.0
            else: df[c] = df["Close"]
    df = df[need].sort_index().dropna()
    # 중복 인덱스 제거(마지막 값 유지)
    df = df[~df.index.duplicated(keep="last")]
    return df

def _download_close(ticker: str, start="2015-01-01") -> pd.Series | None:
    try:
        df = _download_ohlcv(ticker, start)
        return pd.to_numeric(df["Close"], errors="coerce").dropna()
    except Exception:
        return None

# =========================
# 내부(가격/거래량) 피처
# =========================
def _roll_pct_rank(s: pd.Series, win=252):
    return s.rolling(win, min_periods=int(win*0.5)).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) else np.nan, raw=False
    )

def add_basic_indicators(feat: pd.DataFrame, close: pd.Series) -> pd.DataFrame:
    d = close.diff()
    gain = d.where(d>0, 0.0); loss = (-d).where(d<0, 0.0)
    ag = gain.rolling(14, min_periods=14).mean()
    al = loss.rolling(14, min_periods=14).mean()
    rs = ag / (al.replace(0, np.nan))
    feat["rsi14"] = 100 - (100/(1+rs))

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    feat["macd_line"] = macd
    feat["macd_hist"] = macd - signal
    return feat

def make_internal_features(df: pd.DataFrame, pack_level:int=1) -> pd.DataFrame:
    """
    pack_level:
      0: 최소(원래 4개 중심)
      1: 내부 피처(모멘텀/변동성/밴드/리스크) 확장
      2: level1 + 간단 레짐/캘린더 상호작용 + 거시블록(아래서 추가)
    """
    # ✅ 여기서 squeeze()로 항상 Series 보장 (에러 원인 해결 포인트)
    close = pd.to_numeric(df["Close"].squeeze(),  errors="coerce")
    high  = pd.to_numeric(df["High"].squeeze(),   errors="coerce")
    low   = pd.to_numeric(df["Low"].squeeze(),    errors="coerce")
    vol   = pd.to_numeric(df["Volume"].squeeze(), errors="coerce")
    logp  = np.log(close)
    ret1  = close.pct_change()

    feat = pd.DataFrame(index=df.index)

    # level 0
    ma20 = close.rolling(20).mean()
    feat["ret1"]       = ret1
    feat["mom20"]      = (close - ma20) / (ma20.replace(0, np.nan))
    feat["volatility"] = ret1.rolling(14).std()
    feat["vol_ratio"]  = vol / (vol.rolling(20).mean() + 1e-12)

    if pack_level >= 1:
        ma60 = close.rolling(60).mean()
        feat["ret5"]   = close.pct_change(5)
        feat["ret10"]  = close.pct_change(10)
        feat["mom60"]  = (close - ma60) / (ma60.replace(0,np.nan))

        z20_mu = close.rolling(20).mean()
        z20_sd = close.rolling(20).std()
        feat["z20"] = (close - z20_mu) / (z20_sd.replace(0,np.nan))
        z60_mu = close.rolling(60).mean()
        z60_sd = close.rolling(60).std()
        feat["z60"] = (close - z60_mu) / (z60_sd.replace(0,np.nan))

        lr = logp.diff()
        feat["vol10"] = lr.rolling(10).std()
        feat["vol20"] = lr.rolling(20).std()
        feat["vol60"] = lr.rolling(60).std()

        pos_ret = lr.where(lr>0, np.nan)
        neg_ret = lr.where(lr<0, np.nan)
        feat["upvol20"]   = pos_ret.rolling(20, min_periods=10).std()
        feat["downvol20"] = neg_ret.rolling(20, min_periods=10).std()
        feat["skew20"]    = lr.rolling(20, min_periods=10).skew()
        feat["kurt20"]    = lr.rolling(20, min_periods=10).kurt()

        sd20 = close.rolling(20).std()
        up = z20_mu + 2*sd20; lo = z20_mu - 2*sd20
        feat["bb_width"] = (up-lo) / (z20_mu.replace(0,np.nan))
        feat["bb_pos"]   = (close - lo) / ((up-lo).replace(0,np.nan))

        tr = pd.concat([(high-low).abs(),
                        (high-close.shift()).abs(),
                        (low -close.shift()).abs()], axis=1).max(axis=1)
        atr14 = tr.rolling(14).mean()
        feat["atrp14"] = atr14 / (close.replace(0,np.nan))

        cummax = close.cummax()
        dd = close/cummax - 1.0
        feat["dd_min63"] = dd.rolling(63, min_periods=10).min()
        hh = close.rolling(252, min_periods=126).max()
        ll = close.rolling(252, min_periods=126).min()
        feat["pos_52w"] = (close - ll) / ((hh - ll).replace(0,np.nan))

        vol20 = lr.rolling(20).std()
        feat["volpct_1y"] = _roll_pct_rank(vol20, win=252)

    if pack_level >= 2:
        dow = df.index.weekday
        dom = df.index.day
        feat["dow_sin"] = np.sin(2*np.pi*(dow/7.0))
        feat["dow_cos"] = np.cos(2*np.pi*(dow/7.0))
        feat["dom_sin"] = np.sin(2*np.pi*(dom/31.0))
        feat["dom_cos"] = np.cos(2*np.pi*(dom/31.0))

        ema20 = close.ewm(span=20, adjust=False).mean()
        ema60 = close.ewm(span=60, adjust=False).mean()
        trend_on = (ema20 > ema60).astype(float)
        for k in ["mom20","vol20","bb_pos"]:
            if k in feat.columns:
                feat[f"{k}_x_on"]  = feat[k] * trend_on
                feat[f"{k}_x_off"] = feat[k] * (1.0 - trend_on)

    # 룩어헤드 방지
    feat = feat.replace([np.inf, -np.inf], np.nan).shift(1)
    return feat

# =========================
# 거시/교차자산 Safe Pack
# =========================
MACRO_TICKS = {
    "spx":  "^GSPC",      # S&P 500
    "nas":  "^IXIC",      # Nasdaq
    "vix":  "^VIX",       # VIX
    "dxy":  "DX-Y.NYB",   # Dollar index (대안: "DXY")
    "us10": "^TNX",       # US 10y (index)
    "gold": "GC=F",       # Gold
    "oil":  "CL=F",       # WTI
}

def add_macro_block(base_close: pd.Series, start="2015-01-01") -> pd.DataFrame:
    xr = np.log(base_close).diff()
    out = pd.DataFrame(index=base_close.index)
    for name, tick in MACRO_TICKS.items():
        s = _download_close(tick, start)
        if s is None or s.empty:
            continue
        s = s.reindex(out.index).ffill()
        out[f"{name}_ret5"]   = s.pct_change(5)
        out[f"{name}_ret20"]  = s.pct_change(20)
        yr = np.log(s).diff()
        out[f"{name}_corr63"] = xr.rolling(63).corr(yr)
        cov = xr.rolling(63).cov(yr)
        var = yr.rolling(63).var()
        out[f"{name}_beta63"] = cov / (var.replace(0,np.nan))
    return out.shift(1)   # 룩어헤드 방지

# =========================
# 데이터셋 빌드
# =========================
@st.cache_data(show_spinner=False)
def build_dataset(symbol: str, start="2015-01-01", pack_level:int=2):
    ohlcv = _download_ohlcv(symbol, start)
    close = pd.to_numeric(ohlcv["Close"].squeeze(), errors="coerce").astype(float)

    feat = make_internal_features(ohlcv, pack_level=pack_level)
    feat = add_basic_indicators(feat, close)

    if pack_level >= 2:
        macro = add_macro_block(close, start=start)
        feat = feat.join(macro, how="left")

    # 라벨: 다음날부터 7영업일 평균 수익률 > 0 ?
    ret1 = close.pct_change()
    y = (ret1.shift(-1).rolling(7).mean() > 0).astype(int).rename("label")

    df = feat.join(y, how="left")

    # 초기 burn-in 컷(롤링 NaN 구간 제거)
    if len(df) > 320:
        df = df.iloc[260:].copy()

    # 컬럼 단위 결측 보수 + 남은 NaN 제거
    for c in feat.columns:
        med = df[c].rolling(20, min_periods=1).median()
        df[c] = df[c].fillna(med)
    df = df.dropna(subset=["label"])

    # 상수열/전부 NaN 컬럼 제거
    nun = df.nunique(dropna=True)
    keep_cols = [c for c in feat.columns if nun.get(c, 2) > 1]
    X = df[keep_cols].copy()
    y = df["label"].astype(int)
    return df, X, y, ohlcv

# =========================
# Walk-forward + Bagging + Calibration
# =========================
@st.cache_resource(show_spinner=False)
def walk_forward_train(X, y, n_splits=6, seeds=(17,42,137), calibrate=True):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    oof_proba = pd.Series(np.nan, index=y.index)

    for tr, te in tscv.split(X):
        preds = []
        for sd in seeds:
            base = RandomForestClassifier(
                n_estimators=400, max_depth=6, min_samples_leaf=5,
                max_features="sqrt", random_state=sd, n_jobs=-1
            )
            if calibrate:
                cal = CalibratedClassifierCV(base, method="sigmoid", cv=3)
                cal.fit(X.iloc[tr], y.iloc[tr])
                p = cal.predict_proba(X.iloc[te])[:,1]
                preds.append(p)
            else:
                base.fit(X.iloc[tr], y.iloc[tr])
                p = base.predict_proba(X.iloc[te])[:,1]
                preds.append(p)
        oof_proba.iloc[te] = np.mean(preds, axis=0)

    mask = oof_proba.notna()
    acc = accuracy_score(y[mask], (oof_proba[mask]>=0.5).astype(int))
    auc = roc_auc_score(y[mask], oof_proba[mask])

    # 최종 배포용: 전체 데이터로 재학습 (bagging)
    fitted_models = []
    for sd in seeds:
        base = RandomForestClassifier(
            n_estimators=500, max_depth=6, min_samples_leaf=5,
            max_features="sqrt", random_state=sd, n_jobs=-1
        )
        if calibrate:
            cal = CalibratedClassifierCV(base, method="sigmoid", cv=3)
            cal.fit(X, y)
            fitted_models.append(cal)
        else:
            base.fit(X, y)
            fitted_models.append(base)

    return acc, auc, fitted_models, oof_proba

def bagged_predict(models, X) -> np.ndarray:
    ps = [m.predict_proba(X)[:,1] for m in models]
    return np.mean(ps, axis=0)

# =========================
# 7일 보유 전략 시뮬레이션/최적화
# =========================
def simulate_weekly_strategy(prob: pd.Series,
                             price: pd.Series,
                             buy_thr: float,
                             sell_thr: float | None = None,
                             long_short: bool = False,
                             hold_days: int = 7,
                             tx_bps: float = 10.0) -> tuple[pd.DataFrame, dict]:
    """신호 발생일에 7일간 포지션 유지, 중첩 방지(쿨다운)."""
    prob = prob.dropna()
    px = price.reindex(prob.index).ffill()
    rets = px.pct_change().fillna(0.0)

    pos = pd.Series(0, index=prob.index, dtype=float)
    i = 0
    idx = prob.index.to_list()
    n = len(idx)

    while i < n:
        p = prob.iloc[i]
        if p >= buy_thr:
            end_i = min(i + hold_days, n)
            pos.iloc[i:end_i] = 1.0
            i = end_i
            continue
        if long_short and sell_thr is not None and p <= sell_thr:
            end_i = min(i + hold_days, n)
            pos.iloc[i:end_i] = -1.0
            i = end_i
            continue
        i += 1

    strat_gross = pos.shift(1).fillna(0) * rets
    trades = pos.diff().abs().fillna(0)
    cost = trades * (tx_bps / 10000.0)
    strat_net = strat_gross - cost

    eq = (1 + strat_net).cumprod()
    mdd = ((eq.cummax() - eq) / eq.cummax()).max()

    days = (pos.index[-1] - pos.index[0]).days
    years = max(days / 365.0, 1e-9)
    final = float(eq.iloc[-1])
    cagr = final ** (1 / years) - 1 if final > 0 else -1.0

    vol = strat_net.std()
    sharpe = (strat_net.mean() / (vol + 1e-12)) * np.sqrt(365.0)

    out = pd.DataFrame({
        "price": px,
        "ret": rets,
        "pos": pos,
        "strat_gross": strat_gross,
        "cost": cost,
        "strat_net": strat_net,
        "equity": eq
    })
    stats = {
        "final_equity": final,
        "CAGR": cagr,
        "Sharpe": sharpe,
        "MDD": float(mdd),
        "Trades": int(trades.sum()/2)  # 진입/청산 합쳐서 1회로 간주
    }
    return out, stats

def optimize_thresholds(prob: pd.Series,
                        price: pd.Series,
                        long_short: bool = False,
                        hold_days: int = 7,
                        tx_bps: float = 10.0,
                        buy_grid = None,
                        sell_grid = None) -> dict:
    """OOF 확률로 Sharpe 최대화 임계값 탐색."""
    if buy_grid is None:
        buy_grid = np.arange(0.55, 0.90, 0.01)   # 보수적 진입
    if long_short:
        if sell_grid is None:
            sell_grid = np.arange(0.10, 0.45, 0.01)
    else:
        sell_grid = [None]

    best = None
    best_stats = None
    best_bt = None
    best_score = -1e9

    for b in buy_grid:
        for s in sell_grid:
            bt, st = simulate_weekly_strategy(prob, price, b, s, long_short, hold_days, tx_bps)
            score = st["Sharpe"]
            if np.isfinite(score) and score > best_score:
                best_score = score
                best = (float(b), float(s) if s is not None else None)
                best_stats = st
                best_bt = bt

    return {
        "best_buy": best[0],
        "best_sell": best[1],
        "stats": best_stats,
        "bt": best_bt
    }

def pretty_signal(p: float, buy_thr=0.6, sell_thr=None, long_short=False) -> str:
    if p >= buy_thr: return "📈 매수"
    if long_short and (sell_thr is not None) and p <= sell_thr: return "📉 매도"
    return "⏸ 관망"

# =========================
# 실행/리포팅 래퍼 (출력은 반환용)
# =========================
def nearest_prev(idx: pd.DatetimeIndex, d):
    d = pd.to_datetime(d).normalize()
    cand = idx[idx<=d]
    return cand.max() if len(cand) else None

def run_asset(symbol: str,
              asof_date,
              pack_level=2,
              long_short=False,
              tx_bps=10.0,
              hold_days=7):
    # 데이터 & 피처
    df, X, y, ohlcv = build_dataset(symbol, pack_level=pack_level)
    close = pd.to_numeric(ohlcv["Close"].squeeze(), errors="coerce").reindex(df.index).ffill()

    # 학습 + OOF 확률
    acc, auc, models, oof_prob = walk_forward_train(X, y, n_splits=6, seeds=(17,42,137), calibrate=True)

    # 임계값 최적화(OoF)
    tuned = optimize_thresholds(oof_prob.dropna(), close, long_short=long_short,
                                hold_days=hold_days, tx_bps=tx_bps)
    bthr = tuned["best_buy"]
    sthr = tuned["best_sell"]
    st = tuned["stats"]
    bt = tuned["bt"]

    # 전체 구간 확률(보고/예측용)
    full_prob = pd.Series(bagged_predict(models, X), index=X.index)

    # 기준일 1개만 처리
    use_d = nearest_prev(df.index, asof_date)
    if use_d is not None:
        p = float(full_prob.loc[use_d])
        sig_row = {"기준일": use_d.date(),
                   "신호": pretty_signal(p, bthr, sthr, long_short),
                   "상승 확률": round(p, 3)}
    else:
        sig_row = {"기준일": pd.to_datetime(asof_date).date(),
                   "신호": "데이터 없음",
                   "상승 확률": np.nan}

    return {
        "df": df,
        "features": X.columns.tolist(),
        "models": models,
        "walkforward": {"acc": acc, "auc": auc},
        "thresholds": {"buy_thr": bthr, "sell_thr": sthr},
        "bt_stats": st,
        "bt_df": bt,
        "signal": sig_row,
        "oof_prob": oof_prob,
        "full_prob": full_prob,
        "close": close
    }

# =========================
# Streamlit 대시보드
# =========================
st.set_page_config(page_title="Crypto ML Trading Dashboard", layout="wide")

def main():
    st.title("📈 Blockchain Quant Lab")
    st.caption("랜덤포레스트 + 캘리브레이션 / 5일 보유 전략 + 임계값 최적화")

    # ---- 사이드바 설정 (기준일은 1개만) ----
    with st.sidebar:
        asset = st.selectbox("자산 선택", ["BTC-USD", "ETH-USD"], index=0)
        pack_level = st.selectbox("Safe Pack 단계", [0,1,2], index=2,
                                  help="0: 최소 / 1: 내부 확장 / 2: 거시+상호작용 포함")
        long_short = st.checkbox("롱·숏 허용(낮은 확률은 7일 숏)", value=False)
        tx_bps = st.slider("거래비용 (bps)", 0, 50, 10, 1)
        hold_days = st.slider("보유일수(일)", 3, 14, 7, 1)
        asof = st.date_input("기준일", value=date(2025, 8, 6))
        run_btn = st.button("실행", type="primary", use_container_width=True)

    if not run_btn:
        st.info("왼쪽 사이드바에서 설정 후 **실행**을 눌러주세요.")
        return

    with st.spinner("모델 학습 및 최적화 중..."):
        res = run_asset(
            symbol=asset,
            asof_date=pd.to_datetime(asof),
            pack_level=pack_level,
            long_short=long_short,
            tx_bps=float(tx_bps),
            hold_days=int(hold_days),
        )

    # ---- 상단 메트릭 ----
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("자산", asset.replace("-USD",""))
    c2.metric("WF Accuracy", f"{res['walkforward']['acc']:.3f}")
    c3.metric("WF AUC", f"{res['walkforward']['auc']:.3f}")
    thr_text = f"buy {res['thresholds']['buy_thr']:.3f}"
    if res['thresholds']['sell_thr'] is not None:
        thr_text += f" / sell {res['thresholds']['sell_thr']:.3f}"
    c4.metric("최적 Threshold", thr_text)

    # ---- 기준일 신호 ----
    st.subheader("📌 기준일 신호")
    st.table(pd.DataFrame([res["signal"]]))

    # ---- 백테스트 요약 ----
    st.subheader("📉 OOF 기반 백테스트 요약")
    s = res["bt_stats"]
    cc1, cc2, cc3, cc4 = st.columns(4)
    cc1.metric("Sharpe", f"{s['Sharpe']:.2f}")
    cc2.metric("CAGR", f"{s['CAGR']*100:.2f}%")
    cc3.metric("MDD", f"{s['MDD']*100:.2f}%")
    cc4.metric("Trades", f"{s['Trades']}")

    # 마지막 10행 표
    st.dataframe(res["bt_df"][["price","pos","strat_net","equity"]].tail(10))

    # ---- 그래프 (Equity & Price) ----
    st.subheader("📈 Equity / Price")
    fig, ax = plt.subplots(figsize=(10,4))
    res["bt_df"]["equity"].plot(ax=ax, label="Strategy Equity")
    ax2 = ax.twinx()
    res["bt_df"]["price"].plot(ax=ax2, alpha=0.5, label="Price", color="gray")
    ax.set_ylabel("Equity")
    ax2.set_ylabel("Price")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")
    st.pyplot(fig)

    # ---- 설명 박스 ----
    with st.expander("해석 가이드"):
        st.markdown(
            """
- **WF Accuracy / AUC**: 시계열 분할로 얻은 검증 성능입니다. (데이터 누수 방지)
- **최적 Threshold**: OOF 확률로 **Sharpe**가 가장 높도록 찾은 매수/매도 기준치입니다.
- **신호**: 기준일 시점의 상승확률이 임계값을 넘으면 `📈 매수`, (롱·숏 허용 시) 충분히 낮으면 `📉 매도`, 그 외 `⏸ 관망`.
- **7일 보유 전략**: 신호 발생일에 진입하여 **7일간 보유**(중첩 방지). 거래마다 **비용(bps)** 차감.
- **CAGR / MDD / Sharpe**: 전략의 장기 수익률, 최대손실폭, 수익대비 변동성 지표입니다.
            """
        )

if __name__ == "__main__":
    main()
