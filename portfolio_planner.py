import streamlit as st
import pandas as pd

st.set_page_config(page_title="FundMentor", layout="wide")

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

# ---- Risk Filter Mapping ----
risk_filters = {
    "Conservative": ["Low"],
    "Balanced": ["Low", "Medium"],
    "Growth": ["Medium", "High"]
}

# ---- Goal Preference Filter (Updated Thresholds) ----
goal_preferences = {
    "Retirement": lambda df: df[pd.to_numeric(df["Annual Dividend Yield %"].str.replace("%", ""), errors="coerce") > 1.8],
    "Income": lambda df: df[pd.to_numeric(df["Annual Dividend Yield %"].str.replace("%", ""), errors="coerce") > 2.2],
    "Wealth Growth": lambda df: df[pd.to_numeric(df["1 Year"].str.replace("%", ""), errors="coerce") > 6]
}

# ---- Risk Classification Function ----
def classify_risk(row):
    def parse_percentage(val):
        try:
            return float(str(val).replace('%', '').replace(',', '').strip())
        except:
            return None

    def parse_aum(val):
        try:
            return float(str(val).replace('$', '').replace('B', '').replace(',', '').strip())
        except:
            return None

    ret = parse_percentage(row.get("1 Year", ""))
    er = parse_percentage(row.get("ER", ""))
    yield_pct = parse_percentage(row.get("Annual Dividend Yield %", ""))
    aum = parse_aum(row.get("Total Assets", ""))

    score = 0

    if ret is not None:
        if ret > 12: score += 2
        elif ret < 4: score -= 2

    if er is not None:
        if er > 0.6: score += 2
        elif er < 0.1: score -= 2

    if yield_pct is not None:
        if yield_pct < 1: score += 1
        elif yield_pct > 2.5: score -= 1

    if aum is not None:
        if aum < 1: score += 2
        elif aum > 10: score -= 1

    if score <= -2:
        return "Low"
    elif score <= 2:
        return "Medium"
    else:
        return "High"

# ---- App Tabs ----
tab1, tab2 = st.tabs(["Portfolio Builder", "ETF Screener"])

# ---- Load ETF Data ----
@st.cache_data
def load_etf_data():
    df = pd.read_csv("etf_asset_class_tagged.csv")
    df = df.rename(columns={"Total Assets ": "Total Assets"})
    df["Risk Level"] = df.apply(classify_risk, axis=1)
    return df

etf_df = load_etf_data()

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
    asset_mapping = {"Equity": "equity", "Bonds": "bond", "Cash": "cash"}

    for asset_class, tab in tab_map.items():
        with tab:
            st.markdown(f"### Top {asset_class} ETFs")
            class_key = asset_mapping[asset_class]

            filtered = etf_df[
                (etf_df["Simplified Asset Class"].str.lower() == class_key)
                & (etf_df["Risk Level"].isin(risk_filters[risk]))
            ].copy()

            # Apply goal preference filter (if applicable)
            if goal in goal_preferences:
                try:
                    filtered = goal_preferences[goal](filtered)
                except:
                    pass

            st.caption(f"{len(filtered)} ETFs match the filters for this asset class and goal.")

            filtered["1 Year"] = pd.to_numeric(filtered["1 Year"].astype(str).str.replace("%", ""), errors="coerce")
            top_etfs = filtered.sort_values(by="1 Year", ascending=False).head(10)

            for _, row in top_etfs.iterrows():
                st.markdown(f"""
                <div style='background:#f9f9f9; padding:15px; border-radius:10px; border:1px solid #ddd; margin-bottom:15px;'>
                    <b><a href='https://finance.yahoo.com/quote/{row['Symbol']}' target='_blank'>{row['Symbol']}: {row['ETF Name']}</a></b><br>
                    <b>1Y Return:</b> {row['1 Year']} &nbsp; <b>Expense Ratio:</b> {row['ER']} &nbsp; <b>Yield:</b> {row['Annual Dividend Yield %']}<br>
                    <b>AUM:</b> {row['Total Assets']} &nbsp; <b>Risk Level:</b> {row['Risk Level']}<br>
                </div>
                """, unsafe_allow_html=True)

            if not top_etfs.empty:
                st.markdown("**ETF Comparison Table**")
                st.dataframe(top_etfs[["Symbol", "ETF Name", "1 Year", "ER", "Annual Dividend Yield %", "Total Assets", "Risk Level"]]
                             .rename(columns={
                                 "1 Year": "1Y Return",
                                 "ER": "Expense Ratio",
                                 "Annual Dividend Yield %": "Yield",
                                 "Total Assets": "AUM"
                             }), use_container_width=True)
