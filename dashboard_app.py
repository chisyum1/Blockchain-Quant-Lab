import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import yfinance as yf

import matplotlib.pyplot as plt
import streamlit as st
from datetime import date

# =========================
# (기존 코드의 보조지표/거시 블록 함수들 유지)
# =========================
# ... (여기에는 _download_ohlcv, make_internal_features, add_macro_block 등
# 기존 dashboard_app.py의 feature 관련 함수들이 그대로 들어갑니다)

# =========================
# 데이터셋 빌드 (기존 그대로)
# =========================
@st.cache_data(show_spinner=False)
def build_dataset(symbol: str, start="2015-01-01", pack_level:int=2):
    # 기존 코드와 동일 (내부 피처 + 거시 Safe Pack 포함)
    ...
    return df, X, y, ohlcv

# =========================
# 딥러닝: LSTM / CNN
# =========================
import tensorflow as tf
from tensorflow.keras import layers

def make_sequences(X: pd.DataFrame, y: pd.Series, window=30):
    Xs, ys = [], []
    for i in range(len(X) - window):
        Xs.append(X.iloc[i:i+window].values)
        ys.append(y.iloc[i+window])
    return np.array(Xs), np.array(ys)

def build_lstm_model(input_shape):
    model = tf.keras.Sequential([
        layers.LSTM(64, return_sequences=True, input_shape=input_shape),
        layers.LSTM(32),
        layers.Dense(16, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy",
                  metrics=["accuracy", tf.keras.metrics.AUC(name="auc")])
    return model

def build_cnn_model(input_shape):
    model = tf.keras.Sequential([
        layers.Conv1D(64, kernel_size=3, activation="relu", input_shape=input_shape),
        layers.MaxPooling1D(pool_size=2),
        layers.Conv1D(128, kernel_size=3, activation="relu"),
        layers.GlobalMaxPooling1D(),
        layers.Dense(32, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy",
                  metrics=["accuracy", tf.keras.metrics.AUC(name="auc")])
    return model

# =========================
# 실행 래퍼
# =========================
def run_asset(symbol: str,
              asof_date,
              pack_level=2,
              model_type="LSTM"):

    df, X, y, ohlcv = build_dataset(symbol, pack_level=pack_level)
    close = pd.to_numeric(ohlcv["Close"].squeeze(), errors="coerce").reindex(df.index).ffill()

    # --- 시퀀스 데이터 준비 ---
    window = 30
    X_seq, y_seq = make_sequences(X, y, window=window)
    split = int(len(X_seq)*0.8)
    X_train, X_val = X_seq[:split], X_seq[split:]
    y_train, y_val = y_seq[:split], y_seq[split:]

    # --- 모델 선택 ---
    if model_type == "LSTM":
        model = build_lstm_model(input_shape=(window, X_seq.shape[2]))
    else:
        model = build_cnn_model(input_shape=(window, X_seq.shape[2]))

    # --- 학습 ---
    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=10, batch_size=32, verbose=0)

    # --- 예측 ---
    prob = model.predict(X_seq, verbose=0).flatten()
    prob_series = pd.Series(prob, index=df.index[window:])

    # 성능 지표
    from sklearn.metrics import accuracy_score, roc_auc_score
    acc = accuracy_score(y.iloc[window:], (prob_series >= 0.5).astype(int))
    auc = roc_auc_score(y.iloc[window:], prob_series)

    # 기준일 신호
    use_d = df.index[df.index <= asof_date][-1]
    if use_d in prob_series.index:
        p = float(prob_series.loc[use_d])
        sig = "📈 매수" if p >= 0.5 else "⏸ 관망"
        sig_row = {"기준일": use_d.date(), "신호": sig, "상승 확률": round(p,3)}
    else:
        sig_row = {"기준일": asof_date.date(), "신호": "데이터 없음", "상승 확률": np.nan}

    return {
        "acc": acc, "auc": auc,
        "signal": sig_row,
        "prob": prob_series,
        "close": close
    }

# =========================
# Streamlit 대시보드
# =========================
st.set_page_config(page_title="Crypto DL Trading Dashboard", layout="wide")

def main():
    st.title("📈 Crypto Deep Learning Trading Lab")
    st.caption("LSTM / 1D CNN 기반 + 기존 보조지표 + 거시 Safe Pack 유지")

    with st.sidebar:
        asset = st.selectbox("자산 선택", ["BTC-USD", "ETH-USD"], index=0)
        model_type = st.selectbox("모델 선택", ["LSTM", "1D CNN"], index=0)
        pack_level = st.selectbox("Safe Pack 단계", [0,1,2], index=2)
        asof = st.date_input("기준일", value=date(2025, 8, 6))
        run_btn = st.button("실행", type="primary", use_container_width=True)

    if not run_btn:
        st.info("왼쪽 사이드바에서 실행을 눌러주세요.")
        return

    with st.spinner("모델 학습 중..."):
        res = run_asset(
            symbol=asset,
            asof_date=pd.to_datetime(asof),
            pack_level=pack_level,
            model_type=model_type
        )

    # 메트릭
    c1, c2 = st.columns(2)
    c1.metric("Accuracy", f"{res['acc']:.3f}")
    c2.metric("AUC", f"{res['auc']:.3f}")

    # 기준일 신호
    st.subheader("📌 기준일 신호")
    st.table(pd.DataFrame([res["signal"]]))

    # 그래프
    st.subheader("📈 상승 확률 & 가격")
    fig, ax = plt.subplots(figsize=(10,4))
    res["prob"].plot(ax=ax, label="상승 확률", color="blue")
    ax2 = ax.twinx()
    res["close"].plot(ax=ax2, label="Price", color="gray", alpha=0.5)
    ax.set_ylabel("Probability")
    ax2.set_ylabel("Price")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")
    st.pyplot(fig)

if __name__ == "__main__":
    main()
