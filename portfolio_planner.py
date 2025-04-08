import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

st.set_page_config(page_title="FundMentor", layout="wide")

# ---- Logo + Brand Header ----
st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
st.image("logo.png", width=120)
st.markdown("<h1 style='text-align: center; font-size: 38px;'>📘 FundMentor</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: gray;'>Learn. Invest. Prosper.</h4>", unsafe_allow_html=True)

# ---- Glossary ----
with st.expander("🧠 Glossary / Metric Tooltips"):
    st.markdown("""
    - **Expense Ratio**: Annual fee as a percentage of investment.
    - **Yield**: Percentage income return.
    - **1Y Return**: Price performance over the past 12 months.
    - **ESG**: Environmental, Social, Governance ethics measure.
    - **Rating**: A simulated star rating (1-5) based on return.
    - **AUM**: Total Assets Under Management.
    - **Suitability Tags**: ETF usage like 🧓 Retirement-Friendly, 🚀 Growth-Oriented.
    """)

st.markdown("<hr>", unsafe_allow_html=True)

# ---- Client Inputs ----
st.subheader("📋 Client Profile")
col1, col2 = st.columns(2)
with col1:
    goal = st.selectbox("Client's Investment Goal", ["Retirement", "Income", "Wealth Growth"])
    age = st.slider("Client Age", 18, 80, 35)
    horizon = st.slider("Investment Horizon (Years)", 1, 30, 10)
with col2:
    risk = st.selectbox("Risk Tolerance", ["Conservative", "Balanced", "Growth"])
    amount = st.number_input("Investment Amount ($)", min_value=1000, step=1000)

st.subheader("🧑‍💼 Client Info (Optional)")
client_name = st.text_input("Client Name")
notes = st.text_area("Notes / Meeting Summary")

# ---- Load ETF Data ----
try:
    etf_data = pd.read_csv("etfs.csv")
except:
    st.error("Please ensure 'etfs.csv' exists in the folder.")
    st.stop()

# ---- Allocation Matrix ----
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

# ---- Portfolio Recommendation ----
if st.button("✨ Generate Portfolio Recommendation"):
    st.markdown("---")
    st.subheader("🧾 Portfolio Recommendation")

    st.markdown(f"""
    **Client Name:** {client_name or 'N/A'}  
    **Goal:** {goal}  
    **Risk Profile:** {risk}  
    **Horizon:** {horizon} years  
    **Investment Amount:** ${amount:,.2f}

    📊 **Recommended Allocation:**
    """)
    for k, v in selected_portfolio.items():
        st.markdown(f"- **{v}%** in {k}")
    st.markdown("📘 _This portfolio is designed to align with the client’s goals and risk profile._")

    # ---- Pie Chart ----
    labels = list(selected_portfolio.keys())
    sizes = list(selected_portfolio.values())
    colors = ["#6baed6", "#9ecae1", "#c6dbef"]
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors, textprops={'fontsize': 10})
    for text in texts: text.set_color('black')
    for autotext in autotexts: autotext.set_weight('bold')
    ax.axis('equal')
    st.pyplot(fig)

    # ---- Tabs for ETF Classes ----
    tab1, tab2, tab3 = st.tabs(["📈 Equity", "📉 Bonds", "💵 Cash"])
    tab_map = {"equity": tab1, "bonds": tab2, "cash": tab3}

    for asset_class, pct in selected_portfolio.items():
        ac_lower = asset_class.lower()
        with tab_map[ac_lower]:
            st.markdown(f"### {asset_class} – {pct}% (${(pct / 100) * amount:,.2f})")

            # 🧠 Educational Summary
            if ac_lower == "equity":
                st.info("📘 Equities aim for long-term growth and capital appreciation.")
                st.markdown("""
                #### 🔑 Key Considerations:
                - ✅ Suitable for clients with longer time horizons (7+ years)  
                - 📉 Higher short-term volatility  
                - 🌎 Global and sector exposure  
                - 🧠 Ideal for growth-focused investors
                """)
            elif ac_lower == "bonds":
                st.info("📘 Bonds help manage volatility and generate income.")
                st.markdown("""
                #### 🔑 Key Considerations:
                - ✅ Suitable for conservative or income-focused clients  
                - 🛡️ Helps balance equity risk  
                - 📈 Sensitive to interest rate changes  
                - 🧓 Common in retirement portfolios
                """)
            elif ac_lower == "cash":
                st.info("📘 Cash equivalents offer liquidity and short-term safety.")
                st.markdown("""
                #### 🔑 Key Considerations:
                - ✅ Preserves capital  
                - 📉 Lowest expected returns  
                - 💰 Highly liquid  
                - 🧠 Ideal for short-term goals
                """)

            # ---- ETF Recommendations ----
            etfs = etf_data[etf_data["Asset Class"].str.strip().str.lower() == ac_lower]
            comp_data = []

            for _, row in etfs.iterrows():
                ticker = row['Ticker']
                url = f"https://finance.yahoo.com/quote/{ticker}"
                try:
                    etf = yf.Ticker(ticker)
                    info = etf.info
                    price = info.get("regularMarketPrice", "N/A")
                    hist = etf.history(period="1y")
                    first = hist["Close"].iloc[0] if not hist.empty else None
                    last = hist["Close"].iloc[-1] if not hist.empty else None
                    ret = ((last - first) / first) * 100 if first and last else None
                    ret_str = f"✅ +{ret:.1f}%" if ret and ret > 0 else f"🔻 {ret:.1f}%" if ret else "N/A"
                    ret_color = "green" if ret and ret > 0 else "red" if ret else "gray"
                    summary = info.get("longBusinessSummary", row["Description"])
                except:
                    price, ret_str, ret_color, summary, ret = "N/A", "N/A", "gray", row["Description"], None

                er = info.get("expenseRatio", 0)
                yld = info.get("yield", 0)
                aum = info.get("totalAssets", None)
                rating = "★★★★☆" if ret and ret > 5 else "★★★☆☆"
                esg = "Low ESG Risk" if er < 0.1 else "Standard ESG"
                suit = "🧓 Retirement-Friendly" if yld > 0.02 else "🚀 Growth-Oriented" if ret and ret > 5 else "🧠 Core Holding"
                reason = "Strong performance with reasonable fees." if ret and ret > 5 else "Stable income-focused option."

                # 💡 ETF Card
                st.markdown(f"""
                <div style='background:#f9f9f9; padding:15px; border-radius:10px; border:1px solid #ddd; margin-bottom:15px;'>
                <b><a href='{url}' target='_blank'>{ticker}: {row['Name']}</a></b> – ${price}<br>
                <b>📈 1Y Return:</b> <span style='color:{ret_color}; font-weight:bold;'>{ret_str}</span> &nbsp;
                <b>💼 Expense Ratio:</b> {er*100:.2f}% &nbsp;
                <b>💰 Yield:</b> {yld*100:.2f}%<br>
                <b>📊 AUM:</b> {aum/1e9:.1f}B &nbsp;
                <b>⭐ Rating:</b> {rating} &nbsp;
                <b>🌿 ESG:</b> {esg}<br>
                <b>🔖 Suitability:</b> {suit}<br>
                <i>{summary}</i><br>
                💡 <b>Why this ETF?</b> {reason}
                </div>
                """, unsafe_allow_html=True)

                comp_data.append({
                    "Ticker": ticker,
                    "Name": row['Name'],
                    "Price": f"${price}",
                    "1Y Return": ret_str,
                    "Expense Ratio": f"{er*100:.2f}%",
                    "Yield": f"{yld*100:.2f}%",
                    "AUM": f"${aum/1e9:.1f}B" if aum else "N/A",
                    "Rating": rating,
                    "ESG": esg,
                    "Suitability": suit
                })

            # 📊 Comparison Table
            if comp_data:
                df = pd.DataFrame(comp_data)
                st.markdown("**ETF Comparison Table**")
                st.dataframe(df, use_container_width=True)
