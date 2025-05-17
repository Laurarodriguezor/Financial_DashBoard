"""
pages/1_📅_Daily_Returns.py - Revised layout with side-by-side visualizations

This page provides detailed visualizations of daily returns for the selected stock,
showing the data in multiple formats placed side by side for better accessibility.
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
    page_title="Daily Returns Analysis",
    page_icon="📅",
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

# ------------------------------ Helper Functions ------------------------------ #

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

# ------------------------------ Sidebar ------------------------------------- #

with st.sidebar:
    st.title("📅 Daily Returns")
    
    st.header("🔧 Controls")

    ticker = st.text_input("Ticker", "AAPL").upper().strip()

    # --- Quick-range selector to cut complexity ---------------------------------
    st.subheader("📅 Time Range")
    quick_range = st.selectbox(
        "Quick Range",
        ("YTD", "1M", "3M", "1Y", "3Y", "Max"),
        index=3,  # Default to 1Y for daily returns (more data to analyze)
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
    
    # Option to limit table rows
    max_rows = st.slider("Max Table Rows", min_value=10, max_value=100, value=30, 
                        help="Limit the number of rows in the returns table")

    # Notes about using fixed historical dates (for reliable data)
    st.info("For most accurate results, use dates before 2024.")

# ------------------------------ Main Page ------------------------------------- #

st.title(f"📅 Daily Returns Analysis: {ticker}")
st.markdown("""
This page provides detailed visualizations of daily returns for the selected stock.
All visualizations are displayed side by side to allow for comprehensive analysis without extensive scrolling.
""")

# ------------------------------ Data Load ------------------------------------- #

with st.spinner("Fetching data..."):
    df = fetch(ticker, start_date, end_date)

price_col = pick_price_col(df)
if not price_col:
    st.error(f"No price series (Adj Close / Close) found for {ticker}.")
    st.stop()

price = df[price_col].dropna()
rets = price.pct_change().dropna()

if len(rets) == 0:
    st.warning(f"No valid return data found for {ticker} in the selected date range. Try selecting a different date range.")
    st.stop()

# ------------------------------ Summary Statistics ------------------------------------- #

st.subheader("Daily Returns Summary Statistics")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Mean Daily Return", f"{rets.mean()*100:.2f}%")
    st.metric("Positive Days", f"{(rets > 0).sum()} ({(rets > 0).mean()*100:.1f}%)")

with col2:
    st.metric("Return Std Dev", f"{rets.std()*100:.2f}%")
    st.metric("Negative Days", f"{(rets < 0).sum()} ({(rets < 0).mean()*100:.1f}%)")

with col3:
    st.metric("Min Daily Return", f"{rets.min()*100:.2f}%")
    st.metric("Max Daily Return", f"{rets.max()*100:.2f}%")

# ------------------------------ Side by Side Layout ------------------------------------- #

# Create a 2-column layout: left for table, right for visualizations
left_col, right_col = st.columns([1, 1])  # Equal width

with left_col:
    # ------------------------------ Data Table ------------------------------------- #
    st.subheader("Daily Returns Table")
    tbl = rets.rename("Return").to_frame()
    tbl.index.name = "Date"
    max_abs = tbl["Return"].abs().max()
    
    # Limit the number of rows based on the sidebar slider
    tbl = tbl.head(max_rows)
    
    styled = (
        tbl.style
        .bar(subset=["Return"], align="mid", color=[ORANGE, BLUE], vmin=-max_abs, vmax=max_abs)
        .format("{:+.2%}")
    )
    st.write(styled.to_html(), unsafe_allow_html=True)

with right_col:
    # Create tabs for the visualizations
    tab1, tab2 = st.tabs(["Calendar Heatmap", "Bar Chart"])
    
    with tab1:
        # ------------------------------ Calendar Heatmap ------------------------------------- #
        st.subheader("Calendar Heatmap")
        hm_df = rets.reset_index()
        hm_df.columns = ["date", "ret"]
        hm_df['day']  = hm_df['date'].dt.day_name()
        hm_df['week'] = hm_df['date'].dt.isocalendar().week
        heat = px.density_heatmap(
            hm_df,
            x='day', y='week', z='ret',
            category_orders={'day':['Monday','Tuesday','Wednesday','Thursday','Friday']},
            color_continuous_scale=[[0,ORANGE],[0.5,'white'],[1,BLUE]],
            labels={'ret':'Return'}
        )
        heat.update_layout(height=500, margin=dict(l=0,r=0,t=20,b=0))
        st.plotly_chart(heat, use_container_width=True)
    
    with tab2:
        # ------------------------------ Bar Chart ------------------------------------- #
        st.subheader("Bar Chart of Daily Returns")
        bar_df = rets.reset_index()
        bar_df.columns = ["date", "ret"]
        bar_fig = go.Figure()
        bar_fig.add_trace(
            go.Bar(
                x=bar_df['date'],
                y=bar_df['ret'],
                marker_color=np.where(bar_df['ret']>=0, BLUE, ORANGE),
            )
        )
        bar_fig.update_layout(
            yaxis_tickformat='.2%',
            yaxis_title='Daily Return',
            hovermode='x unified',
            height=500,
            margin=dict(l=0,r=0,t=20,b=0)
        )
        st.plotly_chart(bar_fig, use_container_width=True)

# ------------------------------ Footer ------------------------------------- #

st.markdown("---")
st.caption(f"Data for {ticker} from {start_date} to {end_date} | Data provided by Yahoo Finance")
