import streamlit as st
import pandas as pd
import yfinance as yf

st.set_page_config(page_title="FundMentor", layout="wide")

# ---- Branding ----
st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
st.image("logo.png", width=120)
st.markdown("<h1 style='text-align: center; font-size: 38px;'>FundMentor</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: gray;'>Smarter ETF Portfolios. Built for Advisors & Investors.</h4>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# ---- Client Profile ----
st.subheader("Client Profile")
col1, col2 = st.columns(2)
with col1:
    goal = st.selectbox("Investment Goal", ["Retirement", "Income", "Wealth Growth"])
    age = st.slider("Client Age", 18, 80, 35)
    horizon = st.slider("Investment Horizon (Years)", 1, 30, 10)
with col2:
    risk = st.selectbox("Risk Tolerance", ["Conservative", "Balanced", "Growth"])
    amount = st.number_input("Investment Amount ($)", min_value=1000, step=1000)

st.subheader("Client Info (Optional)")
client_name = st.text_input("Client Name")
notes = st.text_area("Meeting Notes")

# ---- Load ETF Data ----
try:
    etf_data = pd.read_csv("etfs.csv")
except Exception:
    st.error("Missing 'etfs.csv' file. Please upload it to run the app.")
    st.stop()

# ---- Allocation Logic ----
allocation_matrix = {
    ("Retirement", "Conservative"): {"Equity": 30, "Bonds": 55, "Cash": 15},
    ("Retirement", "Balanced"):     {"Equity": 50, "Bonds": 40, "Cash": 10},
    ("Retirement", "Growth"):       {"Equity": 70, "Bonds": 25, "Cash": 5},
    ("Income", "Conservative"):     {"Equity": 20, "Bonds": 65, "Cash": 15},
    ("Income", "Balanced"):         {"Equity": 40, "Bonds": 50, "Cash": 10},
    ("Income", "Growth"):           {"Equity": 60, "Bonds": 30, "Cash": 10},
    ("Wealth Growth", "Conservative"): {"Equity": 40, "Bonds": 50, "Cash": 10},
    ("Wealth Growth", "Balanced"):     {"Equity": 60, "Bonds": 35, "Cash": 5},
    ("Wealth Growth", "Growth"):       {"Equity": 85, "Bonds": 10, "Cash": 5},
}
selected_portfolio = allocation_matrix.get((goal, risk), {"Equity": 50, "Bonds": 40, "Cash": 10})

# ---- Recommendation Output ----
if st.button("Generate Portfolio Recommendation"):
    st.subheader("Portfolio Recommendation")

    st.markdown(f"**Client Name:** {client_name or 'N/A'}")
    st.markdown(f"**Goal:** {goal}  |  **Risk Profile:** {risk}  |  **Horizon:** {horizon} years")
    st.markdown(f"**Investment Amount:** ${amount:,.2f}")
    if notes:
        st.info(f"**Advisor Notes:** {notes}")

    tab1, tab2, tab3 = st.tabs(["Equity", "Bonds", "Cash"])
    tab_map = {"equity": tab1, "bonds": tab2, "cash": tab3}

    for asset_class, pct in selected_portfolio.items():
        ac_lower = asset_class.lower()
        tab = tab_map.get(ac_lower)
        with tab:
            st.markdown(f"### {asset_class} – {pct}% (${(pct / 100) * amount:,.2f})")
            etfs = etf_data[etf_data["Asset Class"].str.strip().str.lower() == ac_lower]
            if etfs.empty:
                st.warning("No ETFs available for this asset class.")
                continue

            recommended = []
            for _, row in etfs.iterrows():
                ticker = row["Ticker"]
                try:
                    etf = yf.Ticker(ticker)
                    info = etf.info
                    hist = etf.history(period="1y")
                    if hist.empty or not info.get("regularMarketPrice"):
                        continue

                    price = info["regularMarketPrice"]
                    first_price = hist["Close"].iloc[0]
                    last_price = hist["Close"].iloc[-1]
                    return_pct = ((last_price - first_price) / first_price) * 100

                    recommended.append({
                        "Ticker": ticker,
                        "Name": row["Name"],
                        "Price": f"${price:.2f}",
                        "1Y Return": f"{return_pct:.1f}%",
                        "Expense Ratio": f"{info.get('expenseRatio', 0) * 100:.2f}%" if info.get("expenseRatio") else "N/A",
                        "Yield": f"{info.get('yield', 0) * 100:.2f}%" if info.get("yield") else "N/A",
                        "AUM": f"${info.get('totalAssets', 0) / 1e9:.1f}B" if info.get('totalAssets') else "N/A",
                        "Rating": "★★★★☆" if return_pct > 5 else "★★★☆☆",
                        "Suitability": "Growth-Oriented" if return_pct > 5 else "Income-Oriented",
                        "Summary": info.get("longBusinessSummary", row.get("Description", "")),
                        "Why": "Strong performance and cost efficiency." if return_pct > 5 else "Stable performer suitable for core allocation."
                    })
                except:
                    continue

            if not recommended:
                st.warning("No valid ETFs could be loaded for this asset class.")
                continue

            for etf in recommended:
                st.markdown(f"""
                <div style='background:#f9f9f9; padding:15px; border-radius:10px; border:1px solid #ddd; margin-bottom:15px;'>
                    <b><a href='https://finance.yahoo.com/quote/{etf["Ticker"]}' target='_blank'>{etf["Ticker"]}: {etf["Name"]}</a></b> – {etf["Price"]}<br>
                    <b>1Y Return:</b> {etf["1Y Return"]} &nbsp; <b>Expense Ratio:</b> {etf["Expense Ratio"]} &nbsp; <b>Yield:</b> {etf["Yield"]}<br>
                    <b>AUM:</b> {etf["AUM"]} &nbsp; <b>Rating:</b> {etf["Rating"]} &nbsp; <b>Suitability:</b> {etf["Suitability"]}<br>
                    <div style='font-size:13px; margin-top:6px;'>{etf["Summary"]}</div>
                    <div style='margin-top:6px;'><b>Why this ETF?</b> {etf["Why"]}</div>
                </div>
                """, unsafe_allow_html=True)

            df = pd.DataFrame(recommended)
            st.markdown("**ETF Comparison Table**")
            st.dataframe(df[["Ticker", "Name", "Price", "1Y Return", "Expense Ratio", "Yield", "AUM", "Rating", "Suitability"]], use_container_width=True)
