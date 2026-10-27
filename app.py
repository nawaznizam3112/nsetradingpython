
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

st.set_page_config(
    page_title="NSE Intraday Intelligence",
    page_icon="📈",
    layout="wide"
)

st.title("📈 NSE Intraday Intelligence")
st.caption("Historical market data • Scanner • Paper trading")

st.warning(
    "Research and paper-trading tool only. Yahoo Finance is an "
    "unofficial data source. No automatic broker orders are enabled."
)

with st.sidebar:
    st.header("Market data")

    source = st.radio(
        "Choose data source",
        ["Yahoo Finance", "Upload CSV", "Synthetic demo"]
    )

    symbols_text = st.text_area(
        "NSE symbols (Yahoo Finance format)",
        "RELIANCE.NS, HDFCBANK.NS, INFY.NS, "
        "ICICIBANK.NS, TCS.NS"
    )

    interval = st.selectbox(
        "Candle interval",
        ["5m", "15m", "30m", "1d"]
    )

    period_options = {
        "5m": ["5d", "1mo", "2mo"],
        "15m": ["5d", "1mo", "2mo"],
        "30m": ["5d", "1mo", "2mo"],
        "1d": ["1mo", "3mo", "6mo", "1y"]
    }

    period = st.selectbox(
        "Historical period",
        period_options[interval]
    )

    min_rvol = st.slider(
        "Minimum relative volume", 0.5, 3.0, 1.2, 0.1
    )

    min_score = st.slider(
        "Minimum absolute scanner score", 1, 4, 2
    )


REQUIRED = {
    "timestamp", "symbol", "open",
    "high", "low", "close", "volume"
}


@st.cache_data(ttl=300, show_spinner=False)
def fetch_yahoo(symbol, period, interval):
    """Fetch historical candles; cache successful results briefly."""
    ticker = yf.Ticker(symbol)

    df = ticker.history(
        period=period,
        interval=interval,
        auto_adjust=False,
        actions=False
    )

    if df is None or df.empty:
        return pd.DataFrame()

    df = df.reset_index()

    # Yahoo may label the timestamp column differently.
    time_col = next(
        (c for c in df.columns
         if str(c).lower() in ["datetime", "date"]),
        None
    )

    if time_col is None:
        return pd.DataFrame()

    df = df.rename(columns={
        time_col: "timestamp",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume"
    })

    # Handle multi-level columns returned by some yfinance versions.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            c[0] if isinstance(c, tuple) else c
            for c in df.columns
        ]

    needed = [
        "timestamp", "open", "high",
        "low", "close", "volume"
    ]

    if not all(c in df.columns for c in needed):
        return pd.DataFrame()

    df = df[needed].copy()
    df["symbol"] = symbol

    ts = pd.to_datetime(df["timestamp"], errors="coerce")

    if ts.dt.tz is not None:
        ts = ts.dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)

    df["timestamp"] = ts

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=needed)
    df = df[df["close"] > 0]

    return df.sort_values("timestamp")


def load_csv(upload):
    df = pd.read_csv(upload)
    df.columns = [str(c).strip().lower() for c in df.columns]

    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(
            "Missing columns: " + ", ".join(sorted(missing))
        )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"], errors="coerce"
    )
    df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=list(REQUIRED))
    df = df[df["symbol"] != ""]

    if df.empty:
        raise ValueError("No valid candle rows found.")

    if (df[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("Prices must be greater than zero.")

    if (df["volume"] < 0).any():
        raise ValueError("Volume cannot be negative.")

    return df.sort_values(["symbol", "timestamp"])


def add_indicators(g):
    g = g.sort_values("timestamp").copy()

    g["ema9"] = g["close"].ewm(span=9, adjust=False).mean()
    g["ema20"] = g["close"].ewm(span=20, adjust=False).mean()

    delta = g["close"].diff()

    gain = delta.clip(lower=0).ewm(
        alpha=1/14, min_periods=14, adjust=False
    ).mean()

    loss = (-delta.clip(upper=0)).ewm(
        alpha=1/14, min_periods=14, adjust=False
    ).mean()

    rs = gain / loss.replace(0, np.nan)
    g["rsi14"] = 100 - 100 / (1 + rs)

    typical = (g["high"] + g["low"] + g["close"]) / 3
    session = g["timestamp"].dt.date

    cumulative_volume = g["volume"].groupby(session).cumsum()
    cumulative_value = (
        typical * g["volume"]
    ).groupby(session).cumsum()

    g["vwap"] = cumulative_value / cumulative_volume.replace(0, np.nan)

    g["volume_avg20"] = g["volume"].rolling(
        20, min_periods=5
    ).mean()

    g["rvol"] = g["volume"] / g["volume_avg20"].replace(0, np.nan)
    g["change_pct"] = g["close"].pct_change() * 100

    return g


def score_stock(g):
    r = g.iloc[-1]
    score = 0
    reasons = []

    if pd.notna(r["ema9"]) and pd.notna(r["ema20"]):
        if r["ema9"] > r["ema20"]:
            score += 1
            reasons.append("EMA9 above EMA20")
        elif r["ema9"] < r["ema20"]:
            score -= 1
            reasons.append("EMA9 below EMA20")

    if pd.notna(r["vwap"]):
        if r["close"] > r["vwap"]:
            score += 1
            reasons.append("Price above VWAP")
        elif r["close"] < r["vwap"]:
            score -= 1
            reasons.append("Price below VWAP")

    if pd.notna(r["rvol"]) and r["rvol"] >= min_rvol:
        score += 1 if r["change_pct"] >= 0 else -1
        reasons.append(f"Relative volume {r['rvol']:.2f}x")

    if pd.notna(r["rsi14"]):
        if 50 <= r["rsi14"] <= 68:
            score += 1
            reasons.append("Bullish-momentum RSI range")
        elif 32 <= r["rsi14"] < 50:
            score -= 1
            reasons.append("Bearish-momentum RSI range")

    label = (
        "Bullish candidate" if score >= 2
        else "Bearish candidate" if score <= -2
        else "No clear setup"
    )

    return score, label, "; ".join(reasons)


# ------------------ LOAD DATA ------------------

if source == "Yahoo Finance":
    symbols = list(dict.fromkeys(
        s.strip().upper()
        for s in symbols_text.split(",")
        if s.strip()
    ))

    if not symbols:
        st.error("Enter at least one symbol.")
        st.stop()

    if interval != "1d" and period == "2mo":
        st.info(
            "Intraday history is limited by Yahoo Finance. "
            "Availability varies by interval and symbol."
        )

    frames = []
    with st.spinner("Fetching historical candles..."):
        for symbol in symbols:
            try:
                frame = fetch_yahoo(symbol, period, interval)
                if not frame.empty:
                    frames.append(frame)
            except Exception as exc:
                st.warning(f"{symbol}: download failed ({exc})")

    if not frames:
        st.error(
            "No data returned. Check your internet connection, "
            "symbol format, interval, and data availability."
        )
        st.stop()

    raw = pd.concat(frames, ignore_index=True)

elif source == "Upload CSV":
    upload = st.file_uploader("Upload intraday candle CSV", type=["csv"])
    if upload is None:
        st.info("Upload a CSV to continue.")
        st.stop()

    try:
        raw = load_csv(upload)
    except Exception as exc:
        st.error(f"CSV error: {exc}")
        st.stop()

else:
    st.info(
        "Synthetic mode is for testing the interface only. "
        "Its prices are not real."
    )

    rng = np.random.default_rng(12)
    times = pd.date_range(
        "2026-10-26 09:15", periods=75, freq="5min"
    )
    frames = []

    for i, symbol in enumerate(
        ["RELIANCE", "HDFCBANK", "INFY", "TCS"]
    ):
        close = (500 + i * 100) * np.cumprod(
            1 + rng.normal(0, 0.002, len(times))
        )
        open_ = np.r_[close[0], close[:-1]]

        frames.append(pd.DataFrame({
            "timestamp": times,
            "symbol": symbol,
            "open": open_,
            "high": np.maximum(open_, close) * 1.001,
            "low": np.minimum(open_, close) * 0.999,
            "close": close,
            "volume": rng.integers(1000, 9000, len(times))
        }))

    raw = pd.concat(frames, ignore_index=True)


# ------------------ ANALYSE DATA ------------------

data = pd.concat(
    [add_indicators(g) for _, g in raw.groupby("symbol")],
    ignore_index=True
).sort_values(["symbol", "timestamp"])

latest_time = data["timestamp"].max()

st.subheader("Data status")

c1, c2, c3 = st.columns(3)
c1.metric("Stocks loaded", data["symbol"].nunique())
c2.metric("Candles loaded", f"{len(data):,}")
c3.metric("Latest candle", latest_time.strftime("%d %b %H:%M"))

if source == "Yahoo Finance":
    st.caption(
        "Historical-data retrieval is not a live-trading feed. "
        "Verify the timestamp and data against another reliable source."
    )

    if latest_time.date() < pd.Timestamp.now().date():
        st.warning(
            "The newest candle is from an earlier date. "
            "Do not treat this as today's market assessment."
        )

results = []

for symbol, g in data.groupby("symbol"):
    g = g.sort_values("timestamp")
    score, label, evidence = score_stock(g)
    r = g.iloc[-1]

    results.append({
        "Symbol": symbol,
        "Close": r["close"],
        "Change %": r["change_pct"],
        "RSI": r["rsi14"],
        "VWAP": r["vwap"],
        "RVOL": r["rvol"],
        "Score": score,
        "Assessment": label,
        "Evidence": evidence,
        "Timestamp": r["timestamp"]
    })

scan = pd.DataFrame(results).sort_values("Score", ascending=False)

st.subheader("Stock scanner")
st.caption(
    "These are rule-based research classifications, not predictions "
    "or buy/sell recommendations."
)

filtered = scan[scan["Score"].abs() >= min_score]

st.dataframe(
    filtered,
    use_container_width=True,
    hide_index=True
)

st.download_button(
    "Download scanner CSV",
    scan.to_csv(index=False).encode("utf-8"),
    "scanner_results.csv",
    "text/csv"
)

with st.expander("Inspect stock chart"):
    chosen = st.selectbox(
        "Choose stock", sorted(data["symbol"].unique())
    )
    g = data[data["symbol"] == chosen].sort_values("timestamp")

    st.line_chart(
        g.set_index("timestamp")[["close", "ema9", "ema20", "vwap"]]
    )

st.subheader("Paper-trade risk calculator")

with st.form("paper_trade"):
    side = st.selectbox("Direction", ["Long", "Short"])
    entry = st.number_input("Hypothetical entry price", min_value=0.01, value=100.0)
    stop = st.number_input("Stop-loss price", min_value=0.01, value=98.0)
    target = st.number_input("Target price", min_value=0.01, value=104.0)
    risk_budget = st.number_input("Maximum planned price risk (₹)", min_value=1.0, value=500.0)

    submitted = st.form_submit_button("Calculate")

if submitted:
    valid = (
        side == "Long" and stop < entry < target
    ) or (
        side == "Short" and target < entry < stop
    )

    if not valid:
        st.error("Check the order of entry, stop and target prices.")
    else:
        per_share_risk = abs(entry - stop)
        quantity = int(risk_budget // per_share_risk)
        reward_risk = abs(target - entry) / per_share_risk

        a, b, c = st.columns(3)
        a.metric("Illustrative shares", f"{quantity:,}")
        b.metric("Gross planned risk", f"₹{quantity * per_share_risk:,.2f}")
        c.metric("Gross reward/risk", f"{reward_risk:.2f}:1")

        st.caption(
            "Excludes brokerage, taxes, slippage, liquidity and gap risk. "
            "No order has been placed."
        )

st.subheader("AI news research")
st.info(
    "Not connected yet. After we validate real historical candles, "
    "we will add source-linked news research and a backtesting engine."
)

st.caption(
    "NSE Intraday Intelligence v0.3 • Historical data only • "
    "No live orders • No guarantee of profitability"
)

