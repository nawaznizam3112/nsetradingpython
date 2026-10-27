
import math
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf


# ============================================================
# NSE INTRADAY INTELLIGENCE + HISTORICAL BACKTESTER
# Research and paper trading only. No broker order execution.
# ============================================================

st.set_page_config(
    page_title="NSE Intraday Intelligence",
    page_icon="📈",
    layout="wide",
)

st.title("📈 NSE Intraday Intelligence")
st.caption(
    "Historical market scanner, transparent strategy backtester, "
    "and paper-trading research tool. Not investment advice."
)

DEFAULT_SYMBOLS = (
    "RELIANCE.NS, HDFCBANK.NS, INFY.NS, "
    "ICICIBANK.NS, TCS.NS"
)


# ----------------------------
# Market data
# ----------------------------

@st.cache_data(ttl=300, show_spinner=False)
def fetch_yahoo(symbol: str, period: str, interval: str) -> pd.DataFrame:
    """Fetch and normalize Yahoo Finance OHLCV candles."""
    df = yf.Ticker(symbol).history(
        period=period,
        interval=interval,
        auto_adjust=False,
        actions=False,
        prepost=False,
    )

    if df is None or df.empty:
        return pd.DataFrame()

    # Flatten MultiIndex columns if Yahoo returns them.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            c[0] if isinstance(c, tuple) else c
            for c in df.columns
        ]

    df = df.reset_index()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            c[0] if isinstance(c, tuple) else c
            for c in df.columns
        ]

    # Normalize column names regardless of Date/Datetime capitalization.
    rename = {}
    for col in df.columns:
        key = str(col).strip().lower()
        if key in ("datetime", "date", "timestamp", "index"):
            rename[col] = "timestamp"
        elif key == "open":
            rename[col] = "open"
        elif key == "high":
            rename[col] = "high"
        elif key == "low":
            rename[col] = "low"
        elif key == "close":
            rename[col] = "close"
        elif key == "volume":
            rename[col] = "volume"

    df = df.rename(columns=rename)

    required = [
        "timestamp", "open", "high", "low", "close", "volume"
    ]
    if not all(col in df.columns for col in required):
        return pd.DataFrame()

    df = df[required].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # Use Indian market time for display and daily grouping.
    try:
        if df["timestamp"].dt.tz is not None:
            df["timestamp"] = (
                df["timestamp"]
                .dt.tz_convert("Asia/Kolkata")
                .dt.tz_localize(None)
            )
    except (AttributeError, TypeError):
        pass

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["timestamp", "open", "high", "low", "close"])
    df = df.sort_values("timestamp").drop_duplicates("timestamp")
    return df.reset_index(drop=True)


def load_uploaded_csv(uploaded_file) -> pd.DataFrame:
    """Read a CSV with timestamp/date and OHLCV columns."""
    if uploaded_file is None:
        return pd.DataFrame()

    df = pd.read_csv(uploaded_file)
    df.columns = [str(c).strip().lower() for c in df.columns]

    aliases = {
        "date": "timestamp",
        "datetime": "timestamp",
        "time": "timestamp",
        "adj close": "close",
    }
    df = df.rename(columns=aliases)

    required = ["timestamp", "open", "high", "low", "close", "volume"]
    if not all(c in df.columns for c in required):
        raise ValueError(
            "CSV must contain timestamp/date, open, high, low, close, volume."
        )

    df = df[required].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["timestamp", "open", "high", "low", "close"])
    return df.sort_values("timestamp").reset_index(drop=True)


# ----------------------------
# Indicators and signal rules
# ----------------------------

def add_indicators(
    source: pd.DataFrame,
    min_rsi: float,
    max_rsi: float,
    min_rvol: float,
    require_rvol: bool,
) -> pd.DataFrame:
    df = source.copy().sort_values("timestamp").reset_index(drop=True)

    df["ema9"] = df["close"].ewm(span=9, adjust=False).mean()
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()

    # Wilder-style smoothed RSI approximation.
    change = df["close"].diff()
    gain = change.clip(lower=0)
    loss = -change.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))
    df.loc[(avg_loss == 0) & (avg_gain > 0), "rsi"] = 100
    df.loc[(avg_gain == 0) & (avg_loss == 0), "rsi"] = 50

    # VWAP resets for each trading date.
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    volume = df["volume"].fillna(0).clip(lower=0)
    trading_date = df["timestamp"].dt.date

    cumulative_pv = (typical_price * volume).groupby(trading_date).cumsum()
    cumulative_volume = volume.groupby(trading_date).cumsum()
    df["vwap"] = cumulative_pv / cumulative_volume.replace(0, np.nan)

    # Compare volume with the average of previous bars only.
    prior_volume_average = (
        volume.groupby(trading_date)
        .transform(lambda x: x.rolling(20, min_periods=5).mean().shift(1))
    )
    df["rvol"] = volume / prior_volume_average.replace(0, np.nan)

    df["ema_cross_up"] = (
        (df["ema9"] > df["ema20"])
        & (df["ema9"].shift(1) <= df["ema20"].shift(1))
    )
    df["ema_cross_down"] = (
        (df["ema9"] < df["ema20"])
        & (df["ema9"].shift(1) >= df["ema20"].shift(1))
    )

    rsi_ok = df["rsi"].between(min_rsi, max_rsi)
    volume_ok = (df["rvol"] >= min_rvol) if require_rvol else True

    # The signal is known only after this candle closes.
    df["long_signal"] = (
        df["ema_cross_up"]
        & (df["close"] > df["vwap"])
        & rsi_ok
        & volume_ok
    ).fillna(False)

    df["exit_signal"] = df["ema_cross_down"].fillna(False)

    # Simple explainable score; not a probability of profit.
    score = (
        (df["ema9"] > df["ema20"]).astype(int) * 30
        + (df["close"] > df["vwap"]).astype(int) * 25
        + (df["rsi"].between(40, 65)).astype(int) * 20
        + (df["rvol"] >= min_rvol).fillna(False).astype(int) * 15
        + df["ema_cross_up"].astype(int) * 10
    )
    df["score"] = score.clip(0, 100).astype(int)

    return df


# ----------------------------
# Backtesting engine
# ----------------------------

def run_backtest(
    source: pd.DataFrame,
    symbol: str,
    initial_capital: float,
    risk_pct: float,
    stop_pct: float,
    target_pct: float,
    cost_per_side_pct: float,
    min_rsi: float,
    max_rsi: float,
    min_rvol: float,
    require_rvol: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Long-only backtest:
    - Signal formed on a candle close.
    - Entry is at the next candle's open.
    - Stop/target checks use candle high/low.
    - If stop and target are both touched in one candle, assume stop first.
    - One open position at a time per symbol.
    - Positions are closed at the end of each trading date.
    """
    df = add_indicators(
        source, min_rsi, max_rsi, min_rvol, require_rvol
    )

    if len(df) < 30:
        return df, pd.DataFrame()

    trades = []
    position = None
    equity = float(initial_capital)

    for i in range(1, len(df)):
        row = df.iloc[i]
        previous = df.iloc[i - 1]
        current_date = row["timestamp"].date()
        previous_date = previous["timestamp"].date()
        is_last_bar_of_day = (
            i == len(df) - 1
            or df.iloc[i + 1]["timestamp"].date() != current_date
        )

        was_open_at_start = position is not None
        closed_this_bar = False

        # Close an existing position if the prior candle had an exit signal.
        # This uses the prior completed candle, never the future candle.
        if position is not None and bool(previous["exit_signal"]):
            exit_price = float(row["open"])
            reason = "EMA cross-down"
            trades.append(
                close_trade(
                    position, exit_price, row["timestamp"], reason,
                    symbol, cost_per_side_pct
                )
            )
            equity += trades[-1]["net_pnl"]
            position = None
            closed_this_bar = True

        # If still holding, process the current candle's stop and target.
        if position is not None:
            low = float(row["low"])
            high = float(row["high"])
            stop_hit = low <= position["stop_price"]
            target_hit = high >= position["target_price"]

            if stop_hit:
                # Conservative assumption if both levels are crossed.
                exit_price = position["stop_price"]
                reason = (
                    "Stop-loss (both levels touched; stop assumed first)"
                    if target_hit else "Stop-loss"
                )
            elif target_hit:
                exit_price = position["target_price"]
                reason = "Target"
            else:
                exit_price = None
                reason = ""

            if exit_price is not None:
                trades.append(
                    close_trade(
                        position, exit_price, row["timestamp"], reason,
                        symbol, cost_per_side_pct
                    )
                )
                equity += trades[-1]["net_pnl"]
                position = None
                closed_this_bar = True

        # Enter at this candle's open using the previous candle's signal.
        if (
            position is None
            and not was_open_at_start
            and not closed_this_bar
            and bool(previous["long_signal"])
        ):
            entry_price = float(row["open"])
            if entry_price > 0:
                risk_cash = equity * risk_pct / 100
                risk_per_share = entry_price * stop_pct / 100
                shares_by_risk = math.floor(risk_cash / risk_per_share)
                shares_by_cash = math.floor(equity / entry_price)
                quantity = min(shares_by_risk, shares_by_cash)

                if quantity >= 1:
                    position = {
                        "symbol": symbol,
                        "entry_time": row["timestamp"],
                        "entry_price": entry_price,
                        "quantity": int(quantity),
                        "stop_price": entry_price * (1 - stop_pct / 100),
                        "target_price": entry_price * (1 + target_pct / 100),
                    }

                    # Check the remainder of the entry candle's range.
                    stop_hit = float(row["low"]) <= position["stop_price"]
                    target_hit = float(row["high"]) >= position["target_price"]

                    if stop_hit or target_hit:
                        if stop_hit:
                            exit_price = position["stop_price"]
                            reason = (
                                "Stop-loss (both levels touched; stop assumed first)"
                                if target_hit else "Stop-loss"
                            )
                        else:
                            exit_price = position["target_price"]
                            reason = "Target"

                        trades.append(
                            close_trade(
                                position, exit_price, row["timestamp"],
                                reason, symbol, cost_per_side_pct
                            )
                        )
                        equity += trades[-1]["net_pnl"]
                        position = None
                        closed_this_bar = True

        # No overnight holding in this intraday strategy.
        if position is not None and is_last_bar_of_day:
            exit_price = float(row["close"])
            trades.append(
                close_trade(
                    position, exit_price, row["timestamp"],
                    "End-of-day exit", symbol, cost_per_side_pct
                )
            )
            equity += trades[-1]["net_pnl"]
            position = None

    # Close any remaining position at the final available close.
    if position is not None:
        last = df.iloc[-1]
        trades.append(
            close_trade(
                position, float(last["close"]), last["timestamp"],
                "End of available data", symbol, cost_per_side_pct
            )
        )

    trade_df = pd.DataFrame(trades)
    return df, trade_df


def close_trade(
    position: dict,
    exit_price: float,
    exit_time,
    reason: str,
    symbol: str,
    cost_per_side_pct: float,
) -> dict:
    entry = float(position["entry_price"])
    exit_ = float(exit_price)
    quantity = int(position["quantity"])

    entry_turnover = entry * quantity
    exit_turnover = exit_ * quantity
    gross_pnl = (exit_ - entry) * quantity

    # Approximate all-in cost on each leg. Set this in the UI.
    costs = (
        entry_turnover + exit_turnover
    ) * cost_per_side_pct / 100

    net_pnl = gross_pnl - costs

    return {
        "symbol": symbol,
        "entry_time": position["entry_time"],
        "exit_time": exit_time,
        "entry_price": round(entry, 2),
        "exit_price": round(exit_, 2),
        "quantity": quantity,
        "stop_price": round(position["stop_price"], 2),
        "target_price": round(position["target_price"], 2),
        "gross_pnl": round(gross_pnl, 2),
        "estimated_costs": round(costs, 2),
        "net_pnl": round(net_pnl, 2),
        "return_on_entry_%": round(
            100 * net_pnl / entry_turnover, 3
        ) if entry_turnover else 0,
        "exit_reason": reason,
    }


def calculate_metrics(trades: pd.DataFrame, initial_capital: float) -> dict:
    if trades.empty:
        return {
            "trades": 0, "win_rate": 0.0, "net_pnl": 0.0,
            "return_pct": 0.0, "profit_factor": None,
            "max_drawdown_pct": 0.0, "ending_equity": initial_capital,
        }

    net = trades["net_pnl"].astype(float)
    wins = net[net > 0]
    losses = net[net < 0]
    ending_equity = initial_capital + net.sum()

    equity_curve = pd.Series(
        [initial_capital] + list(initial_capital + net.cumsum())
    )
    running_peak = equity_curve.cummax()
    drawdown = (equity_curve - running_peak) / running_peak.replace(0, np.nan)
    max_drawdown_pct = abs(float(drawdown.min())) * 100 if len(drawdown) else 0

    profit_factor = (
        float(wins.sum() / abs(losses.sum()))
        if len(losses) and losses.sum() != 0
        else (None if len(wins) else 0.0)
    )

    return {
        "trades": len(trades),
        "win_rate": float((net > 0).mean() * 100),
        "net_pnl": float(net.sum()),
        "return_pct": float((ending_equity / initial_capital - 1) * 100),
        "profit_factor": profit_factor,
        "max_drawdown_pct": max_drawdown_pct,
        "ending_equity": float(ending_equity),
    }


# ----------------------------
# Sidebar controls
# ----------------------------

st.sidebar.header("1. Data source")

data_source = st.sidebar.radio(
    "Choose data source",
    ["Yahoo Finance", "Upload CSV"],
)

symbols_text = st.sidebar.text_input(
    "NSE Yahoo symbols",
    value=DEFAULT_SYMBOLS,
    help="Example: RELIANCE.NS, INFY.NS, TCS.NS",
)

interval = st.sidebar.selectbox(
    "Candle interval",
    ["5m", "15m", "30m"],
    index=0,
)

period = st.sidebar.selectbox(
    "Historical period",
    ["5d", "1mo", "2mo"],
    index=1,
    help="Yahoo intraday history is limited; available data may be shorter.",
)

uploaded_file = None
if data_source == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader(
        "Upload OHLCV CSV",
        type=["csv"],
        help="Required columns: timestamp/date, open, high, low, close, volume.",
    )

st.sidebar.header("2. Strategy settings")

initial_capital = st.sidebar.number_input(
    "Starting paper capital (₹)",
    min_value=1000.0,
    max_value=100000000.0,
    value=100000.0,
    step=10000.0,
)

risk_pct = st.sidebar.slider(
    "Capital risk budget per trade (%)",
    min_value=0.25,
    max_value=2.0,
    value=1.0,
    step=0.25,
)

stop_pct = st.sidebar.slider(
    "Stop-loss distance (%)",
    min_value=0.2,
    max_value=3.0,
    value=0.8,
    step=0.1,
)

target_pct = st.sidebar.slider(
    "Target distance (%)",
    min_value=0.2,
    max_value=6.0,
    value=1.6,
    step=0.1,
)

min_rsi = st.sidebar.slider(
    "Minimum RSI for entry",
    min_value=20,
    max_value=60,
    value=40,
)

max_rsi = st.sidebar.slider(
    "Maximum RSI for entry",
    min_value=40,
    max_value=85,
    value=65,
)

min_rvol = st.sidebar.slider(
    "Minimum relative volume",
    min_value=0.5,
    max_value=3.0,
    value=1.0,
    step=0.1,
)

require_rvol = st.sidebar.checkbox(
    "Require relative-volume confirmation",
    value=True,
)

cost_per_side_pct = st.sidebar.number_input(
    "Estimated all-in cost per side (%)",
    min_value=0.0,
    max_value=1.0,
    value=0.05,
    step=0.01,
    help=(
        "A user-adjustable approximation per buy or sell leg. "
        "Not a broker-specific fee calculation."
    ),
)

st.sidebar.caption(
    "Signals are generated at candle close and executed at the next candle open. "
    "No live orders are placed."
)


# ----------------------------
# Load and scan market data
# ----------------------------

symbols = list(dict.fromkeys(
    s.strip().upper()
    for s in symbols_text.split(",")
    if s.strip()
))

if min_rsi >= max_rsi:
    st.error("Maximum RSI must be greater than minimum RSI.")
    st.stop()

if stop_pct <= 0 or target_pct <= 0:
    st.error("Stop-loss and target must both be greater than zero.")
    st.stop()

if not symbols and data_source == "Yahoo Finance":
    st.warning("Enter at least one Yahoo Finance symbol.")
    st.stop()

market_data = {}
load_errors = []

if data_source == "Yahoo Finance":
    with st.spinner("Loading historical candles from Yahoo Finance..."):
        for symbol in symbols:
            try:
                candles = fetch_yahoo(symbol, period, interval)
                if candles.empty:
                    load_errors.append(
                        f"{symbol}: no candles returned for this period."
                    )
                else:
                    market_data[symbol] = candles
            except Exception as exc:
                load_errors.append(f"{symbol}: {exc}")

else:
    try:
        csv_data = load_uploaded_csv(uploaded_file)
        if not csv_data.empty:
            market_data["Uploaded CSV"] = csv_data
    except Exception as exc:
        st.error(f"CSV error: {exc}")

if load_errors:
    with st.expander("Data loading messages"):
        for message in load_errors:
            st.write(f"- {message}")

if not market_data:
    st.info(
        "No candles loaded yet. Choose Yahoo Finance or upload a valid OHLCV CSV."
    )
    st.stop()


# ----------------------------
# Tabs
# ----------------------------

scanner_tab, backtest_tab, notes_tab = st.tabs(
    ["📡 Scanner", "🧪 Historical backtest", "📘 Rules & limitations"]
)


with scanner_tab:
    st.subheader("Latest candle and indicator scan")
    scan_rows = []
    prepared_data = {}

    for symbol, candles in market_data.items():
        enriched = add_indicators(
            candles, min_rsi, max_rsi, min_rvol, require_rvol
        )
        prepared_data[symbol] = enriched
        latest = enriched.iloc[-1]

        trend = (
            "Bullish trend"
            if latest["ema9"] > latest["ema20"]
            else "Not bullish"
        )
        above_vwap = (
            bool(latest["close"] > latest["vwap"])
            if pd.notna(latest["vwap"])
            else False
        )

        if bool(latest["long_signal"]):
            status = "Long setup on latest candle"
        elif bool(latest["ema9"] > latest["ema20"]) and above_vwap:
            status = "Trend conditions only"
        else:
            status = "No complete setup"

        scan_rows.append({
            "Symbol": symbol,
            "Latest candle": latest["timestamp"],
            "Close": round(float(latest["close"]), 2),
            "EMA 9": round(float(latest["ema9"]), 2),
            "EMA 20": round(float(latest["ema20"]), 2),
            "VWAP": round(float(latest["vwap"]), 2)
                if pd.notna(latest["vwap"]) else np.nan,
            "RSI": round(float(latest["rsi"]), 1)
                if pd.notna(latest["rsi"]) else np.nan,
            "RVOL": round(float(latest["rvol"]), 2)
                if pd.notna(latest["rvol"]) else np.nan,
            "Score / 100": int(latest["score"]),
            "Trend": trend,
            "Setup status": status,
            "Candles": len(enriched),
        })

    scan_df = pd.DataFrame(scan_rows).sort_values(
        "Score / 100", ascending=False
    )
    st.dataframe(scan_df, use_container_width=True, hide_index=True)

    st.caption(
        "The score is a simple rule-based ranking, not a probability of success "
        "or a guaranteed buy/sell recommendation."
    )

    selected_symbol = st.selectbox(
        "Inspect a symbol",
        list(prepared_data.keys()),
    )

    chart_df = prepared_data[selected_symbol].tail(250).copy()
    st.line_chart(
        chart_df.set_index("timestamp")[["close", "ema9", "ema20", "vwap"]],
        use_container_width=True,
    )

    st.download_button(
        "Download scanner results CSV",
        data=scan_df.to_csv(index=False).encode("utf-8"),
        file_name="nse_scanner_results.csv",
        mime="text/csv",
    )


with backtest_tab:
    st.subheader("Test the strategy on historical candles")
    st.write(
        "The test runs separately for each loaded symbol. Each symbol starts "
        "with the selected starting capital; trades are long-only, and no "
        "overnight positions are intentionally held."
    )

    backtest_symbol = st.selectbox(
        "Choose symbol to backtest",
        list(market_data.keys()),
        key="backtest_symbol",
    )

    if st.button("Run historical backtest", type="primary"):
        with st.spinner("Running the historical simulation..."):
            tested_df, trades_df = run_backtest(
                source=market_data[backtest_symbol],
                symbol=backtest_symbol,
                initial_capital=initial_capital,
                risk_pct=risk_pct,
                stop_pct=stop_pct,
                target_pct=target_pct,
                cost_per_side_pct=cost_per_side_pct,
                min_rsi=min_rsi,
                max_rsi=max_rsi,
                min_rvol=min_rvol,
                require_rvol=require_rvol,
            )

        st.session_state["tested_df"] = tested_df
        st.session_state["trades_df"] = trades_df
        st.session_state["tested_symbol"] = backtest_symbol
        st.session_state["tested_capital"] = initial_capital

    if (
        "trades_df" in st.session_state
        and st.session_state.get("tested_symbol") == backtest_symbol
    ):
        tested_df = st.session_state["tested_df"]
        trades_df = st.session_state["trades_df"]
        metrics = calculate_metrics(
            trades_df,
            st.session_state["tested_capital"],
        )

        st.caption(
            f"Tested {len(tested_df):,} candles from "
            f"{tested_df['timestamp'].min()} to {tested_df['timestamp'].max()}."
        )

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Closed trades", f"{metrics['trades']}")
        m2.metric("Win rate", f"{metrics['win_rate']:.1f}%")
        m3.metric("Net P/L after estimated costs", f"₹{metrics['net_pnl']:,.2f}")
        m4.metric("Return on starting capital", f"{metrics['return_pct']:.2f}%")

        m5, m6, m7 = st.columns(3)
        pf = metrics["profit_factor"]
        m5.metric(
            "Profit factor",
            "N/A" if pf is None else f"{pf:.2f}",
        )
        m6.metric("Max realised drawdown", f"{metrics['max_drawdown_pct']:.2f}%")
        m7.metric("Ending paper equity", f"₹{metrics['ending_equity']:,.2f}")

        if trades_df.empty:
            st.warning(
                "No trades were triggered. This can happen with strict rules, "
                "limited history, or missing indicator values. Do not loosen "
                "rules just to force a profitable-looking result."
            )
        else:
            st.markdown("#### Closed trade log")
            display_trades = trades_df.copy()
            for col in ["entry_time", "exit_time"]:
                display_trades[col] = pd.to_datetime(display_trades[col])

            st.dataframe(
                display_trades.sort_values("entry_time", ascending=False),
                use_container_width=True,
                hide_index=True,
            )

            st.download_button(
                "Download trade log CSV",
                data=trades_df.to_csv(index=False).encode("utf-8"),
                file_name=f"{backtest_symbol.replace('.', '_')}_backtest_trades.csv",
                mime="text/csv",
            )

            equity_curve = pd.DataFrame({
                "Trade": range(0, len(trades_df) + 1),
                "Realised equity": [
                    st.session_state["tested_capital"]
                ] + list(
                    st.session_state["tested_capital"]
                    + trades_df["net_pnl"].cumsum()
                ),
            })
            st.markdown("#### Realised equity after each closed trade")
            st.line_chart(
                equity_curve.set_index("Trade"),
                use_container_width=True,
            )

        with st.expander("See indicator data used in this backtest"):
            st.dataframe(
                tested_df.tail(300),
                use_container_width=True,
                hide_index=True,
            )


with notes_tab:
    st.subheader("How the initial strategy works")

    st.markdown("""
**Entry conditions**

1. EMA 9 crosses above EMA 20.
2. The signal candle closes above VWAP.
3. RSI is within your selected minimum and maximum.
4. If enabled, relative volume must meet the selected threshold.

**Execution and exits**

- The entry is simulated at the *next candle's open*, not the signal candle's close.
- The stop-loss and target are calculated from the simulated entry price.
- If the same candle crosses both the stop and target, the backtester assumes the stop was hit first.
- An EMA cross-down confirmed on a completed candle exits at the next candle's open.
- Positions are closed at the end of the available trading day.
- Position quantity is limited by both your risk budget and available paper capital.

**Interpret the results cautiously**

- Costs are a user-adjustable approximation, not an exact Zerodha contract-note calculation.
- Slippage, bid-ask spread, rejected orders, liquidity, taxes and actual fills can differ from this simulation.
- Candle data cannot reveal the exact sequence of prices within each candle.
- Parameters are chosen by the user; repeatedly changing them until results look good can overfit historical data.
- A short historical period is not enough to establish a reliable trading edge.
- The app does not submit live orders or provide guaranteed trading advice.
""")

    st.markdown(
        "For current statutory levies, consult the "
        "[NSE investor charges and taxes guide]"
        "(https://www.nseindia.com/static/invest/first-time-investor-sebi-turnover-fees-stt-other-levies)."
    )

st.divider()
st.caption(
    "Educational prototype • Historical data may be delayed, incomplete or revised • "
    f"Session rendered {datetime.now().strftime('%Y-%m-%d %H:%M')} • "
    "No live trading or broker integration"
)

