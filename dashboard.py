import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import google.generativeai as genai
from pypdf import PdfReader
from datetime import datetime

# --- PAGE CONFIG ---
st.set_page_config(layout="wide", page_title="Professional Equity Desk")

# --- FILE HANDLING ---
WATCHLIST_FILE = "watchlist.csv"

def load_watchlist():
    if os.path.exists(WATCHLIST_FILE):
        return pd.read_csv(WATCHLIST_FILE)['Ticker'].tolist()
    # Default list if empty
    return ["RELIANCE.NS", "TCS.NS", "ITC.NS", "HDFCBANK.NS", "INFY.NS", "TATAMOTORS.NS"]

def save_watchlist(tickers):
    pd.DataFrame(tickers, columns=["Ticker"]).to_csv(WATCHLIST_FILE, index=False)

# --- SIDEBAR ---
st.sidebar.title("💎 Equity Research")
api_key = st.sidebar.text_input("🔑 Google API Key", type="password")

# Watchlist Logic
my_watchlist = load_watchlist()
selected_ticker = st.sidebar.selectbox("Select Ticker", my_watchlist)
new_ticker = st.sidebar.text_input("Add Ticker (e.g. TATASTEEL.NS)")
if st.sidebar.button("Add"):
    if new_ticker and new_ticker not in my_watchlist:
        my_watchlist.append(new_ticker)
        save_watchlist(my_watchlist)
        st.rerun()
if st.sidebar.button("Remove"):
    if selected_ticker in my_watchlist:
        my_watchlist.remove(selected_ticker)
        save_watchlist(my_watchlist)
        st.rerun()

ticker = selected_ticker

# --- HELPER FUNCTIONS ---
def get_metrics(t):
    try:
        stock = yf.Ticker(t)
        info = stock.info
        return {
            "Ticker": t,
            "Price": info.get('currentPrice', 'N/A'),
            "P/E": info.get('trailingPE', 'N/A'),
            "ROE (%)": info.get('returnOnEquity', 0) * 100 if info.get('returnOnEquity') else 'N/A'
        }
    except:
        return None

def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def analyze_document(text_content, user_question):
    if not api_key: return "⚠️ Enter API Key first."
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.0-flash')
    prompt = f"Answer strictly from the text provided:\n\nText: {text_content[:30000]}\n\nQuestion: {user_question}"
    try:
        return model.generate_content(prompt).text
    except Exception as e:
        return f"Error: {e}"

def summarize_news_sentiment(news_list, ticker_symbol):
    if not api_key: return "⚠️ Enter API Key in Sidebar first."
    headlines = []
    for n in news_list[:10]:
        title = n.get('title')
        if not title: title = n.get('content', {}).get('title')
        if title: headlines.append(f"- {title}")
    
    if not headlines: return "No headlines found."
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.0-flash')
    prompt = f"Analyze these headlines for {ticker_symbol}:\n{chr(10).join(headlines)}\n1. Sentiment (Bullish/Bearish). 2. Key Summary."
    try:
        return model.generate_content(prompt).text
    except Exception as e:
        return f"Error: {str(e)}"

# --- SCANNER FUNCTION (NEW) ---
def run_scanner(ticker_list):
    results = []
    nifty = yf.Ticker("^NSEI")
    hist_nifty = nifty.history(period="1mo")['Close']
    
    # Progress Bar
    progress_bar = st.progress(0)
    total = len(ticker_list)
    
    for i, t in enumerate(ticker_list):
        try:
            stock = yf.Ticker(t)
            hist = stock.history(period="1mo")
            info = stock.info
            
            if not hist.empty:
                # RSI
                rsi = calculate_rsi(hist).iloc[-1]
                
                # Rel Strength vs Nifty (Last 20 days)
                # Simple calculation: Stock Return - Nifty Return
                stock_ret = (hist['Close'].iloc[-1] / hist['Close'].iloc[0] - 1) * 100
                nifty_ret = (hist_nifty.iloc[-1] / hist_nifty.iloc[0] - 1) * 100
                rel_strength = stock_ret - nifty_ret
                
                results.append({
                    "Ticker": t,
                    "Price": info.get('currentPrice', 0),
                    "P/E": info.get('trailingPE', 0),
                    "RSI (14)": round(rsi, 2),
                    "vs Nifty (%)": round(rel_strength, 2),
                    "Signal": "🔥 Strong" if rel_strength > 0 and rsi < 70 else "Wait"
                })
        except:
            pass
        progress_bar.progress((i + 1) / total)
        
    return pd.DataFrame(results)

# --- MAIN DASHBOARD ---
try:
    stock = yf.Ticker(ticker)
    info = stock.info
    
    col1, col2 = st.columns([3,1])
    col1.title(info.get('longName', ticker))
    col2.metric("Price", f"{info.get('currency','INR')} {info.get('currentPrice','N/A')}")

    # TABS: Added "🔍 Scanner" at the start
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(["🔍 Scanner", "📈 Technicals", "🔗 Correlation", "📊 Fundamentals", "🏛️ Ownership", "🧮 Valuation", "📰 News", "📂 Document AI"])

    # --- TAB 1: SCANNER (NEW!) ---
    with tab1:
        st.subheader("Market Opportunity Scanner")
        st.caption("Scans your entire Watchlist for Relative Strength and Valuation setups.")
        
        if st.button("🚀 Run Scanner"):
            with st.spinner("Scanning the market... this takes a few seconds..."):
                df_scan = run_scanner(my_watchlist)
                
                # Styling the results
                st.write(f"**Scanned {len(df_scan)} stocks:**")
                
                # Highlight logic
                def highlight_opportunities(val):
                    color = 'red' if val < 0 else 'green'
                    return f'color: {color}'

                st.dataframe(df_scan.style.applymap(highlight_opportunities, subset=['vs Nifty (%)'])
                             .format({"Price": "{:.2f}", "P/E": "{:.2f}", "vs Nifty (%)": "{:+.2f}"}))
                
                st.info("Tip: Look for Positive 'vs Nifty' (Green) combined with RSI < 50 (Not Overbought).")
        else:
            st.write("Click the button above to start the scan.")

    # --- TAB 2: TECH CHART ---
    with tab2:
        hist = stock.history(period="1y")
        if not hist.empty:
            hist['SMA_50'] = hist['Close'].rolling(window=50).mean()
            hist['RSI'] = calculate_rsi(hist)

            fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_width=[0.2, 0.2, 0.6], 
                                subplot_titles=("Price", "Volume", "RSI (Momentum)"))
            fig.add_trace(go.Candlestick(x=hist.index, open=hist['Open'], high=hist['High'], low=hist['Low'], close=hist['Close'], name='Price'), row=1, col=1)
            fig.add_trace(go.Scatter(x=hist.index, y=hist['SMA_50'], line=dict(color='orange'), name='50-Day SMA'), row=1, col=1)
            fig.add_trace(go.Bar(x=hist.index, y=hist['Volume'], showlegend=False, marker_color='teal'), row=2, col=1)
            fig.add_trace(go.Scatter(x=hist.index, y=hist['RSI'], line=dict(color='purple'), name='RSI'), row=3, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
            fig.update_layout(height=600, template="plotly_dark", xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)

    # --- TAB 3: CORRELATION ---
    with tab3:
        st.subheader("Relative Strength vs NIFTY 50")
        nifty = yf.Ticker("^NSEI")
        hist_stock = stock.history(period="1y")['Close']
        hist_nifty = nifty.history(period="1y")['Close']
        if not hist_stock.empty and not hist_nifty.empty:
            df = pd.DataFrame({'Stock': hist_stock, 'Nifty': hist_nifty}).dropna()
            df_norm = (df / df.iloc[0] * 100) - 100
            correlation = df['Stock'].pct_change().corr(df['Nifty'].pct_change())
            
            c1, c2 = st.columns(2)
            c1.metric("Correlation", f"{correlation:.2f}")
            
            fig_corr = go.Figure()
            fig_corr.add_trace(go.Scatter(x=df_norm.index, y=df_norm['Stock'], name=ticker, line=dict(color='#00FF00')))
            fig_corr.add_trace(go.Scatter(x=df_norm.index, y=df_norm['Nifty'], name="NIFTY", line=dict(color='white', dash='dash')))
            fig_corr.update_layout(template="plotly_dark")
            st.plotly_chart(fig_corr, use_container_width=True)

    # --- TAB 4: FUNDAMENTALS ---
    with tab4:
        st.subheader("Financial Performance")
        fin = stock.financials
        if not fin.empty:
            fin = fin.T.sort_index()
            fig_fin = go.Figure()
            if 'Total Revenue' in fin.columns: fig_fin.add_trace(go.Bar(x=fin.index.year, y=fin['Total Revenue'], name='Revenue', marker_color='blue'))
            if 'Net Income' in fin.columns: fig_fin.add_trace(go.Bar(x=fin.index.year, y=fin['Net Income'], name='Net Income', marker_color='green'))
            fig_fin.update_layout(barmode='group', height=400, template="plotly_dark")
            st.plotly_chart(fig_fin, use_container_width=True)

    # --- TAB 5: OWNERSHIP ---
    with tab5:
        st.subheader("Holdings")
        try:
            holders = stock.institutional_holders
            if holders is not None and not holders.empty: st.dataframe(holders)
            else: st.info("No data available.")
        except: st.info("Data unavailable.")

    # --- TAB 6: VALUATION ---
    with tab6:
        st.subheader("Quick DCF")
        try: fcf = float(info.get('freeCashflow', 1000000000))
        except: fcf = 1000000000.0
        c1, c2 = st.columns(2)
        fcf_input = c1.number_input("FCF", value=fcf)
        growth = c2.slider("Growth (%)", 0, 30, 12) / 100
        wacc = c1.slider("WACC (%)", 5, 20, 10) / 100
        fair_value = (fcf_input * (1+growth)**5 * 1.03) / (wacc - 0.03) / info.get('sharesOutstanding', 1)
        st.metric("Fair Value", f"₹{fair_value:,.2f}")

    # --- TAB 7: NEWS ---
    with tab7:
        col1, col2 = st.columns([3, 1])
        with col1: st.subheader("News")
        with col2:
             if st.button("📢 Sentiment"):
                 with st.spinner("Analyzing..."):
                     st.info(summarize_news_sentiment(stock.news, ticker))
        news_list = stock.news
        if news_list:
            for n in news_list[:10]:
                title = n.get('title')
                if not title: title = n.get('content', {}).get('title')
                if not title: title = "News Report"
                link = n.get('link')
                if not link: link = n.get('clickThroughUrl', {}).get('url')
                st.markdown(f"➤ **[{title}]({link if link else '#'})**")

    # --- TAB 8: DOC AI ---
    with tab8:
        st.subheader("📄 Chat with Reports")
        uploaded_file = st.file_uploader("Upload PDF", type="pdf")
        if uploaded_file:
            reader = PdfReader(uploaded_file)
            text = "".join([p.extract_text() for p in reader.pages])
            st.success("Loaded.")
            q = st.text_input("Question:", "Key risks?")
            if st.button("Ask"): st.write(analyze_document(text, q))

except Exception as e:
    st.error(f"App Error: {e}")