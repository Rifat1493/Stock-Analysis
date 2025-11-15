from datetime import date, timedelta
import math
from pathlib import Path
from typing import List, Tuple
import pandas as pd
import plotly.express as px
import streamlit as st
import yfinance as yf

APP_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = APP_DIR / "dataset"

DEFAULT_TICKER_FILE = APP_DIR / "dataset" / "tsx_tickers_extracted.csv"

PAGE_SIZE = 10
ALWAYS_KEEP = "CNR.TO"
YEARS = 15

# ------------------------------ Helpers ------------------------------ #

def map_to_yahoo(symbol: str) -> str:
    s = (symbol or "").strip().upper()
    if not s:
        return s
    # Convert DBA:TSX / :TSXV into Yahoo suffix
    if s.endswith(":TSXV"):
        return s[:-5] + ".V"
    if s.endswith(":TSX"):
        return s[:-4] + ".TO"
    return s if "." in s else s + ".TO"

@st.cache_data(show_spinner=False)
def load_ticker_list(csv_path: str | Path = DEFAULT_TICKER_FILE) -> List[str]:
    if csv_path and Path(csv_path).exists():
        df = pd.read_csv(csv_path)
        col = None
        for c in df.columns:
            if str(c).strip().lower() in ("ticker", "symbol"):
                col = c
                break
        if col is None:
            st.warning("No 'Ticker' column in CSV; falling back to Big 6 banks.")
            return ["RY.TO","TD.TO","BNS.TO","BMO.TO","CM.TO","NA.TO"]
        base = (df[col].astype(str).str.strip()
                    .str.replace(r":TSXV?$", "", regex=True))
        tickers = sorted({map_to_yahoo(s) for s in base if s})
        return tickers
    # Fallback: Big 6
    return ["RY.TO","TD.TO","BNS.TO","BMO.TO","CM.TO","NA.TO"]

@st.cache_data(show_spinner=True)
def download_prices(tickers: Tuple[str, ...], start: date, end: date) -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame()
    data = yf.download(list(tickers), start=start, end=end, auto_adjust=False,
                       progress=False, group_by="ticker")
    # Extract Adj Close
    if isinstance(data.columns, pd.MultiIndex):
        cols = [c for c in data.columns if c[1] == "Adj Close"]
        out = data.loc[:, cols].copy()
        out.columns = [c[0] for c in cols]
    else:
        # Single ticker
        only = list(tickers)[0]
        out = data[["Adj Close"]].copy()
        out.columns = [only]
    out = out.sort_index().dropna(axis=1, how="all").ffill()
    return out

def tidy_from_wide(adj: pd.DataFrame) -> pd.DataFrame:
    t = adj.reset_index().melt(id_vars="Date", var_name="Ticker", value_name="Adj Close")
    t = t.dropna(subset=["Adj Close"])  # remove empty rows
    return t

# ------------------------------ UI ------------------------------ #

st.set_page_config(page_title="TSX Stocks Viewer", layout="wide")
st.title("TSX Daily Prices — 15 Years")

left, right = st.columns([2, 1])

with right:
    st.subheader("Settings")
    csv_path = st.text_input("Ticker CSV (optional)", value=str(DEFAULT_TICKER_FILE))
    years = st.slider("Years of history", 5, 20, YEARS, 1)
    filter_max = st.number_input("Drop tickers if max price >", min_value=0, value=1000)
    always_keep = st.text_input("Always-keep ticker", value=ALWAYS_KEEP).upper().strip()
    page_size = st.number_input("Tickers per page", 5, 25, PAGE_SIZE, 1)

with left:
    tickers_all = load_ticker_list(csv_path)
    if not tickers_all:
        st.stop()
    st.write(f"Loaded {len(tickers_all)} tickers.")

    # Date range
    end = pd.Timestamp.today().normalize().date()
    start = end - timedelta(days=years * 365)

    with st.spinner("Downloading prices from Yahoo Finance..."):
        adj = download_prices(tuple(tickers_all), start, end)

    if adj.empty:
        st.error("No data returned for the provided tickers.")
        st.stop()

    # Persist CSVs
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    wide_path = DATASET_DIR / "tsx_adj_close_15y_wide.csv"
    tidy_path = DATASET_DIR / "tsx_adj_close_15y_tidy.csv"
    adj.to_csv(wide_path, index_label="Date")
    tidy = tidy_from_wide(adj)
    tidy.to_csv(tidy_path, index=False)

    # Filter: remove tickers with any price > filter_max
    max_by_ticker = tidy.groupby("Ticker")["Adj Close"].max()
    to_remove = max_by_ticker[max_by_ticker > float(filter_max)].index.tolist()
    tidy_filtered = tidy[~tidy["Ticker"].isin(to_remove)].copy()

    st.write(f"Removed {len(to_remove)} tickers with max price > {filter_max}.")

    # Pagination list (alphabetical), keep ALWAYS_KEEP if present
    all_tickers = sorted(tidy_filtered["Ticker"].dropna().unique().tolist())
    has_keep = always_keep in all_tickers
    others = [t for t in all_tickers if t != always_keep]

    per_page = (page_size - 1) if has_keep else page_size
    total_pages = max(1, math.ceil(len(others) / max(1, per_page)))

    if "page" not in st.session_state:
        st.session_state.page = 1

    def select_page(page: int) -> List[str]:
        page = max(1, min(page, total_pages))
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        page_others = others[start_idx:end_idx]
        sel = ([always_keep] + page_others) if has_keep else page_others
        return sorted(sel)

    # Controls
    c1, c2, c3, c4 = st.columns([1, 1, 2, 8])
    with c1:
        if st.button("◀Prev"):
            st.session_state.page = max(1, st.session_state.page - 1)
    with c2:
        if st.button("Next▶"):
            st.session_state.page = min(total_pages, st.session_state.page + 1)
    with c3:
        st.markdown(f"**Page {st.session_state.page}/{total_pages}**")

    selected = select_page(st.session_state.page)

    # Nicely split tickers across lines and prevent intra-word splits
    def format_ticker_lines(items: List[str], per_line: int = 8) -> str:
        def span(t: str) -> str:
            return f'<span class="ticker-label">{t}</span>'

        lines: list[str] = []
        for i in range(0, len(items), per_line):
            seg = items[i:i+per_line]
            parts: list[str] = []
            for j, t in enumerate(seg):
                parts.append(span(t))
                if j != len(seg) - 1:
                    parts.append('<span class="ticker-sep">, </span>')
            lines.append("".join(parts))
        return "<br/>".join(lines)

    with c4:
        html_tickers = format_ticker_lines(selected, per_line=8)
        st.markdown(
            f"""
            <style>
            .ticker-list {{
                white-space: normal;
                word-break: normal;
                overflow-wrap: normal;
            }}
            .ticker-label {{
                display: inline-block;
                white-space: nowrap;
                font-size: 0.95rem;
            }}
            .ticker-sep {{
                font-size: 0.95rem;
            }}
            </style>
            <div class="ticker-list">
                <span style="font-size:0.9rem; color: var(--text-color, #666);">Tickers:</span><br/>
                {html_tickers}
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Plot
    plot_df = tidy_filtered[tidy_filtered["Ticker"].isin(selected)].copy()
    # Keep CNR.TO consistently red across pages
    color_map = {"CNR.TO": "red"}
    fig = px.line(
        plot_df,
        x="Date",
        y="Adj Close",
        color="Ticker",
        title=f"TSX daily adjusted close — page {st.session_state.page}/{total_pages}",
        color_discrete_map=color_map,
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Download data"):
        st.download_button("Download wide CSV", data=wide_path.read_bytes(), file_name=wide_path.name)
        st.download_button("Download tidy CSV", data=tidy_path.read_bytes(), file_name=tidy_path.name)

st.sidebar.info(
    "Run with: streamlit run stock/src/app.py --server.address 0.0.0.0 --server.port 8501\n"
    "Then open http://<10.0.0.156>:8501 on devices in the same Wi‑Fi."
)
