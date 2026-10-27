import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path

st.set_page_config(page_title="NSE Intraday Intelligence", page_icon="📈", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
[data-testid="stMetric"] {background: rgba(128,128,128,.08); padding: 14px; border-radius: 12px;}
.small-note {font-size: .88rem; opacity: .8;}
</style>
""", unsafe_allow_html=True)

st.title("📈 NSE Intraday Intelligence")
st.caption("Research • Scanner • Trading-day assessment • Paper trading")
st.warning("EDUCATIONAL / PAPER-TRADING ONLY. This starter uses uploaded or synthetic sample data. It does not provide live NSE quotes, personalized financial advice, or live order placement.")

with st.sidebar:
    st.header("Control panel")
    mode = st.radio("Data source", ["Demo data", "Upload intraday CSV"], help="Demo data is synthetic and must never be treated as a real market signal.")
    st.divider()
    st.subheader("Scanner rules")
    min_rvol = st.slider("Minimum relative volume", 0.5, 3.0, 1.2, 0.1)
    min_score = st.slider("Minimum setup score", 1, 5, 3)
    st.caption("Rules are transparent heuristics, not a validated profitable strategy.")

REQUIRED = {"timestamp", "symbol", "open", "high", "low", "close", "volume"}

def make_demo():
    """Create clearly synthetic intraday candles so the UI works before data integration."""
    rng = np.random.default_rng(42)
    frames = []
    times = pd.date_range("2026-10-26 09:15", periods=75, freq="5min")
    symbols = ["RELIANCE", "HDFCBANK", "ICICIBANK", "INFY", "TCS", "SBIN", "ITC", "LT"]
    for i, sym in enumerate(symbols):
        drift = (i - 3.5) * 0.00008
        returns = rng.normal(drift, 0.0025, len(times))
        start = 250 + i * 180
        close = start * np.cumprod(1 + returns)
        prev = np.r_[start, close[:-1]]
        op = prev
        high = np.maximum(op, close) * (1 + rng.uniform(0.0001, 0.002, len(times)))
        low = np.minimum(op, close) * (1 - rng.uniform(0.0001, 0.002, len(times)))
        vol = rng.integers(1000, 9000, len(times))
        frames.append(pd.DataFrame({"timestamp": times, "symbol": sym, "open": op, "high": high, "low": low, "close": close, "volume": vol}))
    return pd.concat(frames, ignore_index=True)

def parse_upload(upload):
    df = pd.read_csv(upload)
    df.columns = [str(c).strip().lower() for c in df.columns]
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError("Missing columns: " + ", ".join(sorted(missing)))
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()
    df = df.dropna(subset=list(REQUIRED)).sort_values(["symbol", "timestamp"])
    if df.empty:
        raise ValueError("No valid rows found. Check the CSV values.")
    return df

def add_indicators(g):
    g = g.sort_values("timestamp").copy()
    g["ema9"] = g["close"].ewm(span=9, adjust=False).mean()
    g["ema20"] = g["close"].ewm(span=20, adjust=False).mean()
    delta = g["close"].diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    g["rsi14"] = 100 - (100 / (1 + rs))
    typical = (g["high"] + g["low"] + g["close"]) / 3
    day = g["timestamp"].dt.date
    g["vwap"] = (typical * g["volume"]).groupby(day).cumsum() / g["volume"].groupby(day).cumsum().replace(0, np.nan)
    g["volume_avg20"] = g["volume"].rolling(20, min_periods=5).mean()
    g["rvol"] = g["volume"] / g["volume_avg20"].replace(0, np.nan)
    g["change_pct"] = g["close"].pct_change() * 100
    return g

def score_latest(g, min_rvol):
    row = g.iloc[-1]
    score = 0
    reasons = []
    if pd.notna(row.get("ema9")) and pd.notna(row.get("ema20")):
        if row["ema9"] > row["ema20"]:
            score += 1; reasons.append("EMA9 above EMA20")
        elif row["ema9"] < row["ema20"]:
            score -= 1; reasons.append("EMA9 below EMA20")
    if pd.notna(row.get("vwap")):
        if row["close"] > row["vwap"]:
            score += 1; reasons.append("Price above VWAP")
        elif row["close"] < row["vwap"]:
            score -= 1; reasons.append("Price below VWAP")
    if pd.notna(row.get("rvol")) and row["rvol"] >= min_rvol:
        score += 1 if row["change_pct"] >= 0 else -1
        reasons.append(f"Relative volume elevated ({row['rvol']:.2f}x)")
    if pd.notna(row.get("rsi14")):
        if 50 <= row["rsi14"] <= 68:
            score += 1; reasons.append("RSI supports bullish momentum")
        elif 32 <= row["rsi14"] < 50:
            score -= 1; reasons.append("RSI supports bearish momentum")
    if score >= 2:
        label = "Bullish candidate"
    elif score <= -2:
        label = "Bearish candidate"
    else:
        label = "No clear setup"
    return score, label, "; ".join(reasons) or "Insufficient indicator history"

try:
    if mode == "Demo data":
        raw = make_demo()
        st.info("Demo mode: all prices and volumes are synthetic. No real stock recommendation can be inferred.")
    else:
        upload = st.file_uploader("Upload intraday candles CSV", type=["csv"])
        st.markdown("Required columns: `timestamp, symbol, open, high, low, close, volume`. Use consistent 1-minute or 5-minute candles and timestamps in India time.")
        if upload is None:
            st.stop()
        raw = parse_upload(upload)

    data = pd.concat([add_indicators(g) for _, g in raw.groupby("symbol")], ignore_index=True)
    latest = data.sort_values("timestamp").groupby("symbol", as_index=False).tail(1).copy()
    scored = []
    for sym, g in data.groupby("symbol"):
        score, label, reason = score_latest(g, min_rvol)
        last = g.sort_values("timestamp").iloc[-1]
        scored.append({"Symbol": sym, "Last close": last["close"], "Change %": last["change_pct"], "RSI(14)": last["rsi14"], "RVOL": last["rvol"], "Score": score, "Assessment": label, "Evidence": reason, "Last candle": last["timestamp"]})
    scan = pd.DataFrame(scored).sort_values("Score", ascending=False)

    n = len(scan)
    bullish = int((scan["Assessment"] == "Bullish candidate").sum())
    bearish = int((scan["Assessment"] == "Bearish candidate").sum())
    st.subheader("Today's market assessment")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Symbols analysed", n)
    c2.metric("Bullish candidates", bullish)
    c3.metric("Bearish candidates", bearish)
    c4.metric("Latest candle", str(pd.to_datetime(data["timestamp"]).max().strftime("%d %b %H:%M")))
    if mode == "Demo data":
        st.warning("Market-day status: UNAVAILABLE — synthetic demo data cannot classify today's real market.")
    else:
        st.info("Market-day assessment is provisional. A reliable market-regime assessment also needs a valid NIFTY 50/index dataset and a freshness check.")

    st.subheader("Stock scanner")
    st.caption("Scores are rule-based screening signals, not probabilities of profit. A score threshold is not a buy/sell instruction.")
    view = scan[scan["Score"].abs() >= min_score]
    if view.empty:
        st.write("No symbols meet the selected score threshold.")
    else:
        st.dataframe(view, use_container_width=True, hide_index=True,
                     column_config={"Last close": st.column_config.NumberColumn(format="₹%.2f"),
                                    "Change %": st.column_config.NumberColumn(format="%.2f%%"),
                                    "RSI(14)": st.column_config.NumberColumn(format="%.1f"),
                                    "RVOL": st.column_config.NumberColumn(format="%.2f x")})
    with st.expander("Inspect a stock's candles and indicators"):
        sym = st.selectbox("Choose symbol", sorted(data["symbol"].unique()))
        g = data[data["symbol"] == sym].sort_values("timestamp")
        st.line_chart(g.set_index("timestamp")[["close", "ema9", "ema20", "vwap"]])
        st.dataframe(g.tail(15)[["timestamp","open","high","low","close","volume","rsi14","rvol","vwap"]], use_container_width=True, hide_index=True)

    st.subheader("AI news research — next integration")
    st.write("This starter does not yet retrieve or interpret live news. In the next stage, we will add a news/search provider and a separate source panel with headline, publisher, timestamp, link, and an evidence-based summary.")
    query = st.text_input("Research question (demo only)", placeholder="e.g., What recent announcements could affect RELIANCE?")
    if st.button("Prepare research request"):
        if query.strip():
            st.info("Research request saved in this session's display only. Web search is not connected in this starter; no news has been retrieved.")
        else:
            st.warning("Enter a question first.")

    st.subheader("Paper-trading planner")
    st.caption("Planning only. No order is sent to any broker.")
    with st.form("paper_plan"):
        col1, col2, col3 = st.columns(3)
        symbol = col1.selectbox("Symbol", sorted(data["symbol"].unique()), key="paper_symbol")
        side = col2.selectbox("Direction", ["Long", "Short"])
        entry = col3.number_input("Hypothetical entry price (₹)", min_value=0.01, value=100.0, step=0.5)
        col4, col5, col6 = st.columns(3)
        stop = col4.number_input("Stop-loss price (₹)", min_value=0.01, value=98.0, step=0.5)
        target = col5.number_input("Target price (₹)", min_value=0.01, value=104.0, step=0.5)
        risk_budget = col6.number_input("Maximum planned risk (₹)", min_value=1.0, value=500.0, step=100.0)
        submitted = st.form_submit_button("Calculate paper plan")
    if submitted:
        valid = (side == "Long" and stop < entry < target) or (side == "Short" and target < entry < stop)
        if not valid:
            st.error("Check price order: Long requires stop < entry < target; Short requires target < entry < stop.")
        else:
            risk_per_share = abs(entry - stop)
            qty = int(risk_budget // risk_per_share)
            reward_per_share = abs(target - entry)
            rr = reward_per_share / risk_per_share
            p1, p2, p3 = st.columns(3)
            p1.metric("Illustrative quantity", f"{qty:,} shares")
            p2.metric("Planned price risk", f"₹{qty*risk_per_share:,.2f}")
            p3.metric("Gross reward/risk", f"{rr:.2f} : 1")
            st.caption("Quantity excludes liquidity, lot/market constraints, slippage, brokerage, taxes and gap risk. This is not a recommendation to enter.")
    st.download_button("Download scanner results (CSV)", scan.to_csv(index=False).encode("utf-8"), "scanner_results.csv", "text/csv")

    st.divider()
    st.caption("Build status: Version 0.1 • Demo/CSV scanner • no live data • no connected AI search • no broker execution.")
except Exception as e:
    st.error(f"Could not process the data: {e}")
    st.markdown("Check that the CSV includes all required columns and valid numeric candle data.")
