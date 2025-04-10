import streamlit as st
import pandas as pd
import yfinance as yf

st.set_page_config(page_title="FundMentor", layout="wide")

# ---- ETF Universe ----
etf_list = [
    "VTI", "SPY", "QQQ", "IWM", "VEA", "VWO", "IEFA", "XLK", "XLV", "XLF",
    "VYM", "SCHD", "BND", "AGG", "TLT", "LQD", "BNDX", "SHV", "VGSH", "GLD",
    "TIP", "ARKK", "XLE", "XLY", "XLC", "XLI", "XLB", "XLRE", "XLU", "XBI",
    "HYG", "SHY", "MUB", "ICSH", "GOVT", "SCHO", "IAU", "USFR", "SPAB", "BSV",
    "SPDW", "VIG", "FDN", "IGSB", "VCSH", "EMB", "BIL", "JPST", "SPSB", "GDX"
]

# ---- Allocation Matrix ----
allocation_matrix = {
    ("Retirement", "Conservative"): {"Equity": 30, "Bonds": 55, "Cash": 15},
    ("Retirement", "Balanced"): {"Equity": 50, "Bonds": 40, "Cash": 10},
    ("Retirement", "Growth"): {"Equity": 70, "Bonds": 25, "Cash": 5},
    ("Income", "Conservative"): {"Equity": 20, "Bonds": 65, "Cash": 15},
    ("Income", "Balanced"): {"Equity": 40, "Bonds": 50, "Cash": 10},
    ("Income", "Growth"): {"Equity": 60, "Bonds": 30, "Cash": 10},
    ("Wealth Growth", "Conservative"): {"Equity": 40, "Bonds": 50, "Cash": 10},
    ("Wealth Growth", "Balanced"): {"Equity": 60, "Bonds": 35, "Cash": 5},
    ("Wealth Growth", "Growth"): {"Equity": 85, "Bonds": 10, "Cash": 5},
}

# ---- App Tabs ----
tab1, tab2 = st.tabs(["Portfolio Builder", "Live ETF Screener"])

# ---- Sidebar: Client Profile ----
with st.sidebar:
    st.header("Client Profile")
    goal = st.selectbox("Investment Goal", ["Retirement", "Income", "Wealth Growth"])
    risk = st.selectbox("Risk Tolerance", ["Conservative", "Balanced", "Growth"])
    horizon = st.slider("Investment Horizon (Years)", 1, 30, 10)
    age = st.slider("Age", 18, 80, 35)
    amount = st.number_input("Investment Amount ($)", min_value=1000, step=1000)
    client_name = st.text_input("Client Name")
    notes = st.text_area("Meeting Notes")

# ---- Portfolio Builder ----
with tab1:
    st.subheader("Personalized Investment Plan")
    st.markdown(f"**Client:** {client_name or 'N/A'} | **Goal:** {goal} | **Risk Profile:** {risk} | "
                f"**Horizon:** {horizon} years | **Investment Amount:** ${amount:,.2f}")

    st.markdown("### Strategic Asset Allocation")

    allocation = allocation_matrix.get((goal, risk), {"Equity": 50, "Bonds": 40, "Cash": 10})
    allocation_text = {
        "Equity": "Growth-oriented exposure to drive portfolio appreciation.",
        "Bonds": "Income and capital preservation through fixed income assets.",
        "Cash": "Liquidity buffer to reduce volatility and manage short-term needs."
    }

    for asset_class, pct in allocation.items():
        val = (pct / 100) * amount
        st.markdown(f"**{pct}% {asset_class}** – ${val:,.2f}")
        st.caption(allocation_text.get(asset_class, ""))

    st.info("This strategy reflects a diversified blend tailored to the client’s objective and risk tolerance.")

    tab_eq, tab_bd, tab_cash = st.tabs(["Equity", "Bonds", "Cash"])
    tab_map = {"Equity": tab_eq, "Bonds": tab_bd, "Cash": tab_cash}

    for asset_class, tab in tab_map.items():
        with tab:
            st.markdown(f"### Top {asset_class} ETFs")
            recommendations = []
            for ticker in etf_list:
                try:
                    etf = yf.Ticker(ticker)
                    info = etf.info
                    hist = etf.history(period="1y")
                    if hist.empty:
                        continue

                    name = info.get("shortName", "N/A")
                    summary = info.get("longBusinessSummary", "N/A")
                    category = info.get("category", "").lower()
                    price = info.get("regularMarketPrice", None)
                    aum = info.get("totalAssets", None)
                    expense = info.get("expenseRatio", None)
                    yield_pct = info.get("yield", None)

                    first = hist["Close"].iloc[0]
                    last = hist["Close"].iloc[-1]
                    ret = ((last - first) / first) * 100

                    # --- Mapping to asset class ---
                    if asset_class == "Equity" and "equity" not in category and "stock" not in category:
                        continue
                    if asset_class == "Bonds" and "bond" not in category:
                        continue
                    if asset_class == "Cash" and "short" not in category and "treasury" not in category:
                        continue

                    recommendations.append({
                        "Ticker": ticker,
                        "Name": name,
                        "Price": f"${price:.2f}" if price else "N/A",
                        "1Y Return": f"{ret:.1f}%",
                        "Expense Ratio": f"{expense * 100:.2f}%" if expense else "N/A",
                        "Yield": f"{yield_pct * 100:.2f}%" if yield_pct else "N/A",
                        "AUM": f"${aum / 1e9:.1f}B" if aum else "N/A",
                        "Rating": "★★★★☆" if ret > 5 else "★★★☆☆",
                        "Suitability": "Growth-Oriented" if ret > 5 else "Income-Oriented",
                        "Summary": summary[:300] + "..." if summary else "N/A",
                        "Why": "Strong performance and cost efficiency." if ret > 5 else "Stable performer suitable for core allocation."
                    })
                except:
                    continue

            top_etfs = sorted(recommendations, key=lambda x: float(x["1Y Return"].replace('%','')), reverse=True)[:10]

            for etf in top_etfs:
                st.markdown(f"""
                <div style='background:#f9f9f9; padding:15px; border-radius:10px; border:1px solid #ddd; margin-bottom:15px;'>
                    <b><a href='https://finance.yahoo.com/quote/{etf["Ticker"]}' target='_blank'>{etf["Ticker"]}: {etf["Name"]}</a></b> – {etf["Price"]}<br>
                    <b>1Y Return:</b> {etf["1Y Return"]} &nbsp; <b>Expense Ratio:</b> {etf["Expense Ratio"]} &nbsp; <b>Yield:</b> {etf["Yield"]}<br>
                    <b>AUM:</b> {etf["AUM"]} &nbsp; <b>Rating:</b> {etf["Rating"]} &nbsp; <b>Suitability:</b> {etf["Suitability"]}<br>
                    <div style='font-size:13px; margin-top:6px;'>{etf["Summary"]}</div>
                    <div style='margin-top:6px;'><b>Why this ETF?</b> {etf["Why"]}</div>
                </div>
                """, unsafe_allow_html=True)

            if recommendations:
                df = pd.DataFrame(recommendations)
                st.markdown("**ETF Comparison Table**")
                st.dataframe(df[["Ticker", "Name", "Price", "1Y Return", "Expense Ratio", "Yield", "AUM", "Rating", "Suitability"]], use_container_width=True)

# ---- Live ETF Screener ----
with tab2:
    st.subheader("Live ETF Screener")
    st.caption("A dynamic list of 50+ ETFs pulled from Yahoo Finance. Sorted by 1Y return.")
    results = []

    for ticker in etf_list:
        try:
            etf = yf.Ticker(ticker)
            info = etf.info
            hist = etf.history(period="1y")
            if hist.empty: continue
            price = info.get("regularMarketPrice", None)
            first = hist["Close"].iloc[0]
            last = hist["Close"].iloc[-1]
            ret = ((last - first) / first) * 100
            results.append({
                "Ticker": ticker,
                "Name": info.get("shortName", "N/A"),
                "Price": f"${price:.2f}" if price else "N/A",
                "1Y Return": f"{ret:.1f}%",
                "Expense Ratio": f"{info.get('expenseRatio', 0) * 100:.2f}%" if info.get("expenseRatio") else "N/A",
                "Yield": f"{info.get('yield', 0) * 100:.2f}%" if info.get("yield") else "N/A",
                "AUM": f"${info.get('totalAssets', 0) / 1e9:.1f}B" if info.get("totalAssets") else "N/A"
            })
        except:
            continue

    df = pd.DataFrame(results).sort_values(by="1Y Return", ascending=False)
    st.dataframe(df, use_container_width=True)
