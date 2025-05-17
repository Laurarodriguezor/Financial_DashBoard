"""
Home.py - Main page for the Stock Dashboard with summary metrics and price visualization

This is the main entry point for the Stock Dashboard application, providing an overview
of the stock's performance, key metrics, and visualization of price and returns.
"""

from datetime import date, timedelta
import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import yfinance as yf

# ------------------------- Page Set-up & Styling --------------------------- #

st.set_page_config(
    page_title="Stock Market Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

BLUE, ORANGE, NEUTRAL = "#3274a1", "#e1813c", "#909090"

st.markdown(f"""
<style>
[data-testid="stMetric"] div:first-child {{font-size:0.85rem;}}
[data-testid="stMetricValue"]        {{font-weight:600;}}
[data-testid="stMetricDeltaPositive"] {{color:{BLUE} !important;}}
[data-testid="stMetricDeltaNegative"] {{color:{ORANGE} !important;}}
.insight-box {{padding:10px;border-radius:5px;background:#f0f2f6;margin-bottom:12px;}}
footer {{visibility:hidden;}}
</style>""", unsafe_allow_html=True)

# ------------------------------ Sidebar Functions ------------------------------ #

def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure a single-level column index."""
    if isinstance(df.columns, pd.MultiIndex):
        if df.columns.nlevels == 2:
            df.columns = df.columns.get_level_values(0)
        else:
            df.columns = ["_".join(map(str, tup)).strip() for tup in df.columns]
    return df


def fetch(tkr: str, start: date, end: date) -> pd.DataFrame:
    """Download from Yahoo and clean column index."""
    if not tkr:
        return pd.DataFrame()
    try:
        data = yf.download(tkr, start=start, end=end, progress=False, auto_adjust=False)
    except Exception as exc:
        st.error(f"Yahoo download error for {tkr}: {exc}")
        return pd.DataFrame()
    return flatten_columns(data)


def pick_price_col(df: pd.DataFrame) -> str | None:
    """Return the preferred price column present in *df* (Adj Close > Close)."""
    for col in ("Adj Close", "Close"):
        if col in df.columns:
            return col
    return None


def annualised(total_ret: float, n_days: int) -> float:
    return (1 + total_ret) ** (365 / n_days) - 1 if n_days else np.nan


def compute_metrics(price: pd.Series, volume: pd.Series) -> dict:
    rets = price.pct_change().dropna()
    n    = len(price)

    running_max = price.cummax()
    max_dd = (price / running_max - 1).min()

    delta = price.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    rsi = 100 - 100 / (1 + gain.rolling(14).mean() / loss.rolling(14).mean())

    ema12 = price.ewm(span=12, adjust=False).mean()
    ema26 = price.ewm(span=26, adjust=False).mean()
    macd_hist = (ema12 - ema26 - (ema12 - ema26).ewm(span=9, adjust=False).mean()).iloc[-1]

    up, down = (rets > 0).sum(), (rets < 0).sum()

    return {
        "total": price.iloc[-1] / price.iloc[0] - 1,
        "ann":   annualised(price.iloc[-1] / price.iloc[0] - 1, n),
        "avg_up":   rets[rets>0].mean() if not rets[rets>0].empty else np.nan,
        "avg_down": rets[rets<0].mean() if not rets[rets<0].empty else np.nan,
        "vol":  rets.std()*math.sqrt(252),
        "mdd":  max_dd,
        "var":  rets.quantile(0.05),
        "cvar": rets[rets<=rets.quantile(0.05)].mean() if not rets.empty else np.nan,
        "rsi":  rsi.iloc[-1],
        "macd": macd_hist,
        "sharpe": (annualised(price.iloc[-1] / price.iloc[0] - 1, n) / (rets.std()*math.sqrt(252))) if rets.std()>0 else np.nan,
        "up_down": up/down if down else np.nan,
        "avg_vol": volume.mean(),
        "vol_trend": volume.tail(5).mean()/volume.mean()-1 if volume.mean()>0 else np.nan,
        "rets": rets,  # keep for later analysis
    }

pct = lambda v: "–" if pd.isna(v) else f"{v*100:,.2f}%"
num = lambda v: "–" if pd.isna(v) else f"{v:,.0f}"

# ------------------------------ Sidebar ------------------------------------- #

with st.sidebar:
    st.title("📊 Stock Dashboard")
    
    st.header("🔧 Controls")

    ticker    = st.text_input("Ticker", "AAPL").upper().strip()
    benchmark = st.text_input("Benchmark (optional)", "SPY").upper().strip()

    # --- Quick-range selector to cut complexity ---------------------------------
    st.subheader("📅 Time Range")
    quick_range = st.selectbox(
        "Quick Range",
        ("YTD", "1M", "3M", "1Y", "3Y", "Max"),
        index=1,
        help="Preset ranges minimise fiddly date picking and anchor expectations.",
    )

    today = date.today()
    if quick_range == "YTD":
        start_default = date(today.year, 1, 1)
    elif quick_range == "1M":
        start_default = today - timedelta(days=30)
    elif quick_range == "3M":
        start_default = today - timedelta(days=90)
    elif quick_range == "1Y":
        start_default = today - timedelta(days=365)
    elif quick_range == "3Y":
        start_default = today - timedelta(days=365 * 3)
    else:  # Max (≈10y)
        start_default = today - timedelta(days=365 * 10)

    # Progressive disclosure — manual dates hidden unless needed --------------
    with st.expander("Fine-tune dates", expanded=False):
        start_date = st.date_input("Start date", start_default)
        end_date   = st.date_input("End date", today)

        if start_date > end_date:
            st.error("Start date must be before end date")
            st.stop()

    # Notes about using fixed historical dates (for reliable data)
    st.info("For most accurate results, use dates before 2024.")

# ------------------------------ Main Page ------------------------------------- #

# Header and description
st.title("📈 Stock Market Dashboard")
st.markdown("""
This dashboard provides an interactive analysis of stock performance, focusing on key metrics
and visualizations to help understand historical trends and performance indicators.

Use the sidebar to select a stock ticker and time period, then explore the charts and metrics below.
For detailed daily return analysis, check the "Daily Returns" page in the sidebar.
""")

# ------------------------------ Data Load ------------------------------------- #

with st.spinner("Fetching data..."):
    df  = fetch(ticker, start_date, end_date)
    bdf = fetch(benchmark, start_date, end_date) if benchmark else pd.DataFrame()

price_col = pick_price_col(df)
if not price_col:
    st.error(f"No price series (Adj Close / Close) found for {ticker}.")
    st.stop()

price = df[price_col].dropna()
volume = df["Volume"].dropna() if "Volume" in df.columns else pd.Series(index=price.index, dtype=float)

metrics = compute_metrics(price, volume)
rets = metrics.pop("rets")  # daily returns series for charts/tables

# Benchmark return series (may be empty)
bench_price_col = pick_price_col(bdf)
bench_rets = pd.Series(dtype=float)
bench_total = None
if bench_price_col:
    bench_price = bdf[bench_price_col].dropna()
    bench_rets  = bench_price.pct_change().rename(f"{benchmark} Return").dropna()
    bench_total = bench_price.iloc[-1] / bench_price.iloc[0] - 1

# ------------------------------ Header & Price Plot ------------------------------ #

# Salience bias: headline summary banner
st.subheader(f"📈 {ticker} Performance")

summary_text = (
    f"Since **{start_date:%b %d %Y}**, **{ticker}** has returned **{pct(metrics['total'])}**"
)
if bench_total is not None:
    summary_text += f" vs **{benchmark}** {pct(bench_total)}"

st.info(summary_text)

# --- Price chart -----------------------------------------------------------
ma30, sd30 = price.rolling(30).mean(), price.rolling(30).std()
upper, lower = ma30 + 2*sd30, ma30 - 2*sd30

fig = go.Figure()
fig.add_trace(go.Scatter(x=price.index, y=price, name=ticker))
fig.add_trace(go.Scatter(x=ma30.index, y=ma30, name="30-day MA", line=dict(dash="dot")))
fig.add_trace(go.Scatter(x=upper.index, y=upper, name="Upper 2σ", line=dict(width=0.5)))
fig.add_trace(go.Scatter(x=lower.index, y=lower, name="Lower 2σ", line=dict(width=0.5), fill="tonexty", fillcolor="rgba(50,116,161,0.12)"))

if bench_price_col:
    bench_norm = bdf[bench_price_col] * (price.iloc[0] / bdf[bench_price_col].iloc[0])
    fig.add_trace(go.Scatter(x=bench_norm.index, y=bench_norm, name=benchmark, line=dict(color=NEUTRAL, dash="dash")))

fig.update_layout(hovermode="x unified", margin=dict(l=0,r=0,t=30,b=0), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
fig.update_xaxes(rangeslider_visible=True)

st.plotly_chart(fig, use_container_width=True)

# ------------------------------ Key Metrics ------------------------------------- #

# Using the column-based layout from our previous design
st.subheader("Key Performance Metrics")

# Create three columns for metrics display using cards
metrics_col1, metrics_col2, metrics_col3 = st.columns(3)

with metrics_col1:
    # Return metrics
    st.markdown("### Return Metrics")
    
    # Format total return with color
    total_return = metrics['total']
    total_return_color = "green" if total_return >= 0 else "red"
    st.markdown(f"**Total Return:** <span style='color:{total_return_color}'>{pct(total_return)}</span>", unsafe_allow_html=True)
    
    # Format annualized return with color
    annualized_return = metrics['ann']
    ann_return_color = "green" if annualized_return >= 0 else "red"
    st.markdown(f"**Annualized Return:** <span style='color:{ann_return_color}'>{pct(annualized_return)}</span>", unsafe_allow_html=True)
    
    # Last day return
    avg_up = metrics['avg_up']
    avg_down = metrics['avg_down']
    st.markdown(f"**Avg Up Day:** {pct(avg_up)}")
    st.markdown(f"**Avg Down Day:** {pct(avg_down)}")

with metrics_col2:
    # Risk metrics
    st.markdown("### Risk Metrics")
    
    # Daily volatility
    volatility = metrics['vol']
    st.markdown(f"**Annualized Volatility:** {pct(volatility)}")
    
    # Max drawdown 
    max_drawdown = metrics['mdd']
    st.markdown(f"**Maximum Drawdown:** <span style='color:red'>{pct(max_drawdown)}</span>", unsafe_allow_html=True)
    
    # Value at Risk
    var_95 = metrics['var']
    st.markdown(f"**Daily VaR (95%):** <span style='color:red'>{pct(var_95)}</span>", unsafe_allow_html=True)
    
    # CVaR
    cvar = metrics['cvar']
    st.markdown(f"**CVaR (95%):** <span style='color:red'>{pct(cvar)}</span>", unsafe_allow_html=True)

with metrics_col3:
    # Technical indicators
    st.markdown("### Technical Indicators")
    
    # RSI
    rsi = metrics['rsi']
    rsi_color = "red" if rsi > 70 else "green" if rsi < 30 else "black"
    st.markdown(f"**RSI:** <span style='color:{rsi_color}'>{rsi:.1f}</span>", unsafe_allow_html=True)
    
    # MACD
    macd = metrics['macd']
    macd_color = "green" if macd >= 0 else "red"
    st.markdown(f"**MACD Hist:** <span style='color:{macd_color}'>{macd:.2f}</span>", unsafe_allow_html=True)
    
    # Sharpe Ratio
    sharpe = metrics['sharpe']
    st.markdown(f"**Sharpe Ratio:** {sharpe:.2f}" if not pd.isna(sharpe) else "**Sharpe Ratio:** –")
    
    # Up/Down Ratio
    up_down = metrics['up_down']
    st.markdown(f"**Up/Down Ratio:** {up_down:.2f}" if not pd.isna(up_down) else "**Up/Down Ratio:** –")

# ------------------------------ Price Change Summary ------------------------------------- #

# Add the price change summary and trading statistics from previous design
st.subheader("Price Analysis")

# Calculate values for visual displays
start_price = price.iloc[0]
end_price = price.iloc[-1]
price_change = end_price - start_price
price_change_pct = (price_change / start_price) * 100

# Trading statistics
highest_price = price.max()
lowest_price = price.min()
highest_day_idx = price.idxmax()
lowest_day_idx = price.idxmin()
avg_volume = volume.mean()

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Price Change Summary")
    
    change_color = "green" if price_change >= 0 else "red"
    change_icon = "↗" if price_change >= 0 else "↘"
    
    st.markdown(f"""
    <div style="padding: 10px; border-radius: 5px; background-color: {'rgba(0, 128, 0, 0.1)' if price_change >= 0 else 'rgba(255, 0, 0, 0.1)'}">
        <h3 style="color: {change_color};">{change_icon} {price_change_pct:.2f}% ({price_change:.2f})</h3>
        <p>From ${start_price:.2f} on {price.index[0].strftime('%b %d, %Y')} to ${end_price:.2f} on {price.index[-1].strftime('%b %d, %Y')}</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("### Trading Statistics")
    
    st.markdown(f"""
    <div style="padding: 10px; border-radius: 5px; background-color: rgba(0, 0, 255, 0.05)">
        <p><strong>Highest Price:</strong> ${highest_price:.2f} on {highest_day_idx.strftime('%b %d, %Y')}</p>
        <p><strong>Lowest Price:</strong> ${lowest_price:.2f} on {lowest_day_idx.strftime('%b %d, %Y')}</p>
        <p><strong>Average Daily Trading Volume:</strong> {avg_volume:,.0f} shares</p>
    </div>
    """, unsafe_allow_html=True)

# ------------------------------ Return Distribution ------------------------------------- #

st.subheader("📉 Return Distribution")
c1, c2 = st.columns(2)

with c1:
    hist = px.histogram(rets, nbins=20, labels={"value":"Daily Return","count":"Freq"}, color_discrete_sequence=[BLUE])
    hist.add_vline(x=rets.mean(), line_dash="dash", line_color=NEUTRAL)
    hist.add_vline(x=0, line_color="black")
    st.plotly_chart(hist, use_container_width=True)
with c2:
    sorted_r = rets.sort_values()
    theo = np.quantile(np.random.normal(size=10000), np.linspace(0.01,0.99,len(sorted_r)))
    qq = go.Figure()
    qq.add_trace(go.Scatter(x=theo, y=sorted_r, mode='markers', marker=dict(color=BLUE)))
    minv, maxv = min(theo.min(), sorted_r.min()), max(theo.max(), sorted_r.max())
    qq.add_trace(go.Scatter(x=[minv,maxv], y=[minv,maxv], mode='lines', line=dict(color=NEUTRAL,dash='dash')))
    qq.update_layout(title='QQ Plot', xaxis_title='Theoretical Q', yaxis_title='Empirical Q', margin=dict(l=0,r=0,t=30,b=0))
    st.plotly_chart(qq, use_container_width=True)

# ------------------------------ Footer ------------------------------------- #

st.markdown("---")
st.caption("Data provided by Yahoo Finance | Check the 'Daily Returns' page for more detailed analysis")