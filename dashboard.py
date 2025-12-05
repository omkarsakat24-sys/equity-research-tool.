import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import google.generativeai as genai
from pypdf import PdfReader
import requests

# --- PAGE CONFIG ---
st.set_page_config(layout="wide", page_title="Professional Equity Desk")

# --- FILE HANDLING ---
WATCHLIST_FILE = "watchlist.csv"


def load_watchlist():
    if os.path.exists(WATCHLIST_FILE):
        return pd.read_csv(WATCHLIST_FILE)["Ticker"].tolist()
    return ["RELIANCE.NS", "TCS.NS", "ITC.NS", "VEDL.NS", "ADANIENT.NS"]


def save_watchlist(tickers):
    pd.DataFrame(tickers, columns=["Ticker"]).to_csv(WATCHLIST_FILE, index=False)


# --- HELPER: YAHOO SEARCH ---
def search_yahoo_finance(query):
    try:
        url = f"https://query2.finance.yahoo.com/v1/finance/search?q={query}&quotesCount=10&newsCount=0"
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers)
        data = r.json()
        results = []
        if "quotes" in data:
            for q in data["quotes"]:
                sym = q.get("symbol", "")
                if sym.endswith(".NS"):
                    results.append(f"{sym} ({q.get('shortname', 'N/A')})")
        return results
    except:
        return []


# --- SIDEBAR ---
st.sidebar.title("💎 Equity Research")
api_key = st.sidebar.text_input("🔑 Google API Key", type="password")

with st.sidebar.expander("🔍 Find Symbol"):
    q = st.text_input("Type Name (e.g. Vedanta)")
    if q:
        found = search_yahoo_finance(q)
        if found:
            for f in found:
                st.code(f.split(" ")[0])
                st.caption(f)
        else:
            st.warning("No .NS stock found.")

st.sidebar.divider()
my_watchlist = load_watchlist()
selected_ticker = st.sidebar.selectbox("Select Ticker", my_watchlist)
c1, c2 = st.sidebar.columns(2)
new_t = st.sidebar.text_input("Add Ticker")
if c1.button("Add"):
    if new_t and new_t not in my_watchlist:
        my_watchlist.append(new_t)
        save_watchlist(my_watchlist)
        st.rerun()
if c2.button("Remove"):
    if selected_ticker in my_watchlist:
        my_watchlist.remove(selected_ticker)
        save_watchlist(my_watchlist)
        st.rerun()

ticker = selected_ticker


# --- FUNCTIONS ---
def calculate_rsi(data, window=14):
    delta = data["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


@st.cache_data(ttl=600)
def get_stock_data(symbol):
    stock = yf.Ticker(symbol)
    history = stock.history(period="1y")
    info = stock.info
    financials = stock.financials
    news = stock.news
    try:
        major_holders = stock.major_holders
        inst_holders = stock.institutional_holders
    except:
        major_holders = None
        inst_holders = None
    return history, info, financials, news, major_holders, inst_holders


@st.cache_data(ttl=300)
def get_nifty_data():
    return yf.Ticker("^NSEI").history(period="1y")["Close"]


@st.cache_data(ttl=3600)
def fetch_nse_insider_trading(symbol):
    try:
        sym = symbol.replace(".NS", "")
        headers = {"User-Agent": "Mozilla/5.0"}
        session = requests.Session()
        session.get("https://www.nseindia.com", headers=headers, timeout=5)
        url = f"https://www.nseindia.com/api/corporates-pit?index=equities&symbol={sym}"
        response = session.get(url, headers=headers, timeout=5)
        if response.status_code == 200:
            data = response.json()
            if "data" in data:
                df = pd.DataFrame(data["data"])
                cols = [
                    "personCategory",
                    "secType",
                    "acqMode",
                    "secAcq",
                    "secVal",
                    "date",
                ]
                rename = {
                    "personCategory": "Person",
                    "secType": "Security",
                    "acqMode": "Type",
                    "secAcq": "Shares",
                    "secVal": "Value",
                    "date": "Date",
                }
                final_cols = [c for c in cols if c in df.columns]
                return df[final_cols].rename(columns=rename)
    except:
        return None
    return None


def analyze_insider_activity(df):
    if df is None or df.empty:
        return "⚪ No Data", "No recent insider activity found."

    pledges = df[
        df["Type"].astype(str).str.contains("Pledge|Encumbrance", case=False, na=False)
    ]
    if not pledges.empty:
        return (
            "⚠️ CRITICAL RISK: PLEDGING DETECTED",
            "Promoters are pledging shares for loans. This is highly risky.",
        )

    promoters = df[
        df["Person"].astype(str).str.contains("Promoter", case=False, na=False)
    ]
    buys = promoters[
        promoters["Type"]
        .astype(str)
        .str.contains("Purchase|Acquisition", case=False, na=False)
    ]
    sells = promoters[
        promoters["Type"]
        .astype(str)
        .str.contains("Sale|Disposal", case=False, na=False)
    ]

    if not buys.empty:
        total = pd.to_numeric(buys["Shares"], errors="coerce").sum()
        return "🟢 BULLISH SIGNAL", f"Promoters bought ~{total:,.0f} shares."
    if not sells.empty:
        total = pd.to_numeric(sells["Shares"], errors="coerce").sum()
        return "🔴 BEARISH SIGNAL", f"Promoters sold ~{total:,.0f} shares."

    return "⚪ NEUTRAL", "No significant Promoter activity."


def summarize_news_sentiment(news_list, ticker_symbol):
    if not api_key:
        return "⚠️ Enter API Key in Sidebar first."
    headlines = []
    for n in news_list[:10]:
        t = n.get("title") or n.get("content", {}).get("title")
        if t:
            headlines.append(f"- {t}")
    if not headlines:
        return "No headlines."
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")
    prompt = f"Analyze these headlines for {ticker_symbol}:\n{chr(10).join(headlines)}\n1. Sentiment. 2. Summary."
    try:
        return model.generate_content(prompt).text
    except Exception as e:
        return f"Error: {e}"


def run_scanner(ticker_list):
    results = []
    nifty = get_nifty_data().tail(30)
    progress = st.progress(0)
    for i, t in enumerate(ticker_list):
        try:
            stock = yf.Ticker(t)
            hist = stock.history(period="1mo")
            info = stock.info
            if not hist.empty:
                rsi = calculate_rsi(hist).iloc[-1]
                s_ret = (hist["Close"].iloc[-1] / hist["Close"].iloc[0] - 1) * 100
                n_ret = (nifty.iloc[-1] / nifty.iloc[0] - 1) * 100
                rel = s_ret - n_ret
                results.append(
                    {
                        "Ticker": t,
                        "Price": info.get("currentPrice", 0),
                        "RSI": round(rsi, 2),
                        "vs Nifty": round(rel, 2),
                        "Signal": "🔥 Strong" if rel > 0 and rsi < 70 else "Wait",
                    }
                )
        except:
            pass
        progress.progress((i + 1) / len(ticker_list))
    return pd.DataFrame(results)


# --- DASHBOARD ---
try:
    hist, info, financials, news_list, major_holders, inst_holders = get_stock_data(
        ticker
    )
    clean_ticker = ticker.replace(".NS", "")

    col1, col2 = st.columns([3, 1])
    col1.title(info.get("longName", ticker))
    col2.metric(
        "Price", f"{info.get('currency','INR')} {info.get('currentPrice','N/A')}"
    )

    tabs = st.tabs(
        [
            "🔍 Scanner",
            "📈 Technicals",
            "🔗 Correlation",
            "📊 Fundamentals",
            "🧮 Valuation",
            "🕵️ Insider Radar",
            "📰 News",
            "📂 Docs",
        ]
    )

    with tabs[0]:
        if st.button("Run Scanner"):
            df = run_scanner(my_watchlist)

            def color(v):
                return f'color: {"green" if v > 0 else "red"}'

            st.dataframe(df.style.applymap(color, subset=["vs Nifty"]))

    with tabs[1]:
        if not hist.empty:
            hist["SMA_50"] = hist["Close"].rolling(window=50).mean()
            hist["RSI"] = calculate_rsi(hist)
            fig = make_subplots(
                rows=3, cols=1, shared_xaxes=True, row_width=[0.2, 0.2, 0.6]
            )
            fig.add_trace(
                go.Candlestick(
                    x=hist.index,
                    open=hist["Open"],
                    high=hist["High"],
                    low=hist["Low"],
                    close=hist["Close"],
                    name="Price",
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=hist.index,
                    y=hist["SMA_50"],
                    line=dict(color="orange"),
                    name="50-Day SMA",
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Bar(
                    x=hist.index,
                    y=hist["Volume"],
                    showlegend=False,
                    marker_color="teal",
                ),
                row=2,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=hist.index, y=hist["RSI"], line=dict(color="purple"), name="RSI"
                ),
                row=3,
                col=1,
            )
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
            fig.update_layout(
                height=600, template="plotly_dark", xaxis_rangeslider_visible=False
            )
            st.plotly_chart(fig, use_container_width=True)

    with tabs[2]:
        nifty = get_nifty_data()
        if not hist.empty and not nifty.empty:
            df = pd.DataFrame({"Stock": hist["Close"], "Nifty": nifty}).dropna()
            df_norm = (df / df.iloc[0] * 100) - 100
            st.line_chart(df_norm)

    with tabs[3]:
        if not financials.empty:
            fin = financials.T.sort_index()
            st.bar_chart(
                fin[["Total Revenue", "Net Income"]]
                if "Total Revenue" in fin.columns
                else fin
            )

    with tabs[4]:
        fcf = float(info.get("freeCashflow", 1000000000) or 1000000000)
        c1, c2 = st.columns(2)
        g = c1.slider("Growth", 0, 30, 12) / 100
        w = c2.slider("WACC", 5, 20, 10) / 100
        fv = (
            (fcf * (1 + g) ** 5 * 1.03)
            / (w - 0.03)
            / (info.get("sharesOutstanding", 1) or 1)
        )
        st.metric("Fair Value", f"₹{fv:,.2f}")

    with tabs[5]:
        st.subheader("🕵️ Insider Radar & Pledges")

        st.write("### 🔍 Check Official Data")
        col_link1, col_link2 = st.columns(2)

        # 1. SCREENER.IN LINK (Best for Pledges)
        with col_link1:
            st.link_button(
                "🔗 Open Screener.in (Best for Pledges)",
                f"https://www.screener.in/company/{clean_ticker}/",
            )

        # 2. NSE OFFICIAL LINK
        with col_link2:
            st.link_button(
                "🔗 Open NSE Official Site",
                f"https://www.nseindia.com/get-quotes/equity?symbol={clean_ticker}",
            )

        st.caption(
            "Tip: On Screener.in, look for the box that says 'Pledged Percentage'."
        )

        insider = fetch_nse_insider_trading(ticker)
        if insider is not None and not insider.empty:
            signal, msg = analyze_insider_activity(insider)
            st.info(f"**Signal:** {signal}\n\n{msg}")
            st.dataframe(insider)
        else:
            st.divider()
            st.warning("⚠️ Live Insider Transactions blocked by NSE firewall.")
            c1, c2 = st.columns(2)
            with c1:
                st.caption("Promoters Holding")
                if major_holders is not None:
                    st.dataframe(major_holders)
            with c2:
                st.caption("Institutional Holding")
                if inst_holders is not None:
                    st.dataframe(inst_holders)

    with tabs[6]:
        if st.button("Sentiment"):
            st.info(summarize_news_sentiment(news_list, ticker))
        for n in news_list[:5]:
            t = n.get("title") or n.get("content", {}).get("title")
            l = n.get("link") or n.get("clickThroughUrl", {}).get("url")
            st.markdown(f"➤ [{t}]({l})")

    with tabs[7]:
        f = st.file_uploader("Upload PDF", type="pdf")
        if f:
            t = "".join([p.extract_text() for p in PdfReader(f).pages])
            q = st.text_input("Q:", "Risks?")
            if st.button("Ask"):
                genai.configure(api_key=api_key)
                st.write(
                    genai.GenerativeModel("gemini-2.0-flash")
                    .generate_content(f"Context: {t[:30000]}\nQ: {q}")
                    .text
                )

except Exception as e:
    st.error(f"Error: {e}")
