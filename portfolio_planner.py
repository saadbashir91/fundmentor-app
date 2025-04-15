import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

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

# ---- Goal Preference Filter ----
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

def generate_rebalance_actions(df, threshold=5):
    if 'Weight (%)' not in df.columns or 'Target Weight (%)' not in df.columns:
        raise ValueError("Missing 'Weight (%)' or 'Target Weight (%)' columns")

    df = df.copy()
    df['Drift (%)'] = df['Weight (%)'] - df['Target Weight (%)']

    def classify_action(drift):
        if drift > threshold:
            return f"🔻 Reduce ({round(drift, 1)}% overweight)"
        elif drift < -threshold:
            return f"➕ Buy (add {abs(round(drift, 1))}%)"
        else:
            return f"✅ Hold ({round(drift, 1)}% in range)"

    df['Action'] = df['Drift (%)'].apply(classify_action)

    return df[['ETF', 'Asset Class', 'Weight (%)', 'Target Weight (%)', 'Drift (%)', 'Action']]

# ---- Load ETF Data ----
@st.cache_data
def load_etf_data():
    df = pd.read_csv("etf_asset_class_tagged.csv")
    df = df.rename(columns={"Total Assets ": "Total Assets"})

    # --- Mixed ETF Classification Logic ---
    mixed_keywords = ["balanced", "growth", "portfolio", "income", "allocation", "target", "vgro", "xgro", "zbla", "zbal", "xbal", "vbla", "vbal"]
    df["is_potential_mixed"] = df["ETF Name"].str.lower().apply(
        lambda name: any(keyword in name for keyword in mixed_keywords)
    )
    df["Simplified Asset Class"] = df.apply(
        lambda row: "Mixed" if row["is_potential_mixed"] and 
        (pd.isna(row["Simplified Asset Class"]) or str(row["Simplified Asset Class"]).strip().lower() in ["", "other"]) 
        else row["Simplified Asset Class"],
        axis=1
    )

    df["Risk Level"] = df.apply(classify_risk, axis=1)
    return df

etf_df = load_etf_data()

# ---- Multi-Factor ETF Scoring Engine ----
def normalize(series):
    return (series - series.min()) / (series.max() - series.min())

def get_factor_weights(goal):
    if goal == "Wealth Growth":
        return {"1Y": 0.30, "ER": 0.20, "AUM": 0.10, "Yield": 0.10, "TaxEff": 0.30}
    elif goal == "Retirement":
        return {"1Y": 0.20, "ER": 0.20, "AUM": 0.10, "Yield": 0.20, "TaxEff": 0.30}
    elif goal == "Income":
        return {"1Y": 0.10, "ER": 0.20, "AUM": 0.10, "Yield": 0.30, "TaxEff": 0.30}
    else:
        return {"1Y": 0.20, "ER": 0.20, "AUM": 0.10, "Yield": 0.20, "TaxEff": 0.30}

def rank_etfs(df, goal):
    df = df.copy()

    # Clean and convert
    df["1Y_clean"] = pd.to_numeric(df["1 Year"].astype(str).str.replace("%", ""), errors="coerce")
    df["ER_clean"] = pd.to_numeric(df["ER"].astype(str).str.replace("%", ""), errors="coerce")
    df["Yield_clean"] = pd.to_numeric(df["Annual Dividend Yield %"].astype(str).str.replace("%", ""), errors="coerce")
    df["AUM_clean"] = pd.to_numeric(df["Total Assets"].astype(str).str.replace("$", "").str.replace("B", "").str.replace(",", ""), errors="coerce")
    df["TaxEff_clean"] = 1 / (df["Yield_clean"] + df["ER_clean"])

    # Normalize
    df["1Y_score"] = normalize(df["1Y_clean"])
    df["ER_score"] = 1 - normalize(df["ER_clean"])
    df["AUM_score"] = normalize(df["AUM_clean"])
    df["Yield_score"] = normalize(df["Yield_clean"])
    df["TaxEff_score"] = normalize(df["TaxEff_clean"])

    weights = get_factor_weights(goal)
    df["Final Score"] = (
        weights["1Y"] * df["1Y_score"] +
        weights["ER"] * df["ER_score"] +
        weights["AUM"] * df["AUM_score"] +
        weights["Yield"] * df["Yield_score"] +
        weights["TaxEff"] * df["TaxEff_score"]
    )
    return df

# ---- App Tabs ----
tab1, tab2, tab3, tab4 = st.tabs(["Portfolio Builder", "ETF Screener", "Portfolio Analyzer", "Rebalancing Checker"])

# ---- Sidebar: Client Profile ----

with st.sidebar:
    # ---- Risk Questionnaire Logic ----
    if "show_risk_quiz" not in st.session_state:
        st.session_state["show_risk_quiz"] = False
    if "quiz_step" not in st.session_state:
        st.session_state["quiz_step"] = 1
    if "quiz_score" not in st.session_state:
        st.session_state["quiz_score"] = 0

    def reset_quiz():
        st.session_state["show_risk_quiz"] = False
        st.session_state["quiz_step"] = 1
        st.session_state["quiz_score"] = 0

    def get_risk_profile(score):
        if score <= 4:
            return "Conservative"
        elif score <= 7:
            return "Balanced"
        else:
            return "Growth"

    if st.button("Take Risk Questionnaire"):
        st.session_state["show_risk_quiz"] = True
        st.session_state["quiz_step"] = 1
        st.session_state["quiz_score"] = 0

    if st.session_state["show_risk_quiz"]:
        st.subheader("Risk Profile Questionnaire")

        if st.session_state["quiz_step"] == 1:
            q1 = st.radio("1. How would you feel if your $10,000 investment dropped to $9,000 in a month?",
                          ["Very uncomfortable", "Somewhat uncomfortable", "Not concerned"])
            if st.button("Next"):
                st.session_state["quiz_score"] += {"Very uncomfortable": 1, "Somewhat uncomfortable": 2, "Not concerned": 3}[q1]
                st.session_state["quiz_step"] += 1

        elif st.session_state["quiz_step"] == 2:
            q2 = st.radio("2. What’s your primary investment goal?",
                          ["Preserve capital", "Generate income", "Grow wealth"])
            if st.button("Next"):
                st.session_state["quiz_score"] += {"Preserve capital": 1, "Generate income": 2, "Grow wealth": 3}[q2]
                st.session_state["quiz_step"] += 1

        elif st.session_state["quiz_step"] == 3:
            q3 = st.radio("3. What’s your investment horizon?",
                          ["Less than 3 years", "3–10 years", "10+ years"])
            if st.button("See Result"):
                st.session_state["quiz_score"] += {"Less than 3 years": 1, "3–10 years": 2, "10+ years": 3}[q3]
                st.session_state["quiz_step"] += 1

        elif st.session_state["quiz_step"] == 4:
            final_profile = get_risk_profile(st.session_state["quiz_score"])
            st.success(f"Based on your answers, your suggested risk profile is: **{final_profile}**")

            if st.button("Use this profile"):
                st.session_state["Risk Tolerance"] = final_profile
                reset_quiz()

            if st.button("Retake Quiz"):
                reset_quiz()

            st.markdown("This result is a guide and should be discussed further with an advisor if unsure.")

    # ---- Country & Use Context ----
    st.header("Context")

    country = st.selectbox("Select Country", ["Canada", "USA"])

    # Dynamic options based on country
    use_context_options = {
        "Canada": ["TFSA", "RRSP", "RESP", "Non-Registered", "Institutional"],
        "USA": ["Roth IRA", "401(k)", "Traditional IRA", "Taxable", "Institutional"]
    }
    use_context = st.selectbox("Use Context", use_context_options[country])

    # Rules engine
    context_rules = {
        ("Canada", "TFSA"): {
            "avoid_us_dividends": True,
            "favor_growth": True,
            "note": "Recommendations adjusted for TFSA — tax-efficient growth prioritized."
        },
        ("Canada", "RRSP"): {
            "avoid_us_dividends": False,
            "favor_growth": True,
            "note": "Optimized for RRSP — U.S. ETFs allowed without withholding tax."
        },
        ("Canada", "Non-Registered"): {
            "avoid_us_dividends": True,
            "favor_tax_efficiency": True,
            "note": "Tax-efficient ETFs prioritized for non-registered account."
        },
        ("USA", "Roth IRA"): {
            "avoid_dividends": True,
            "favor_growth": True,
            "note": "Roth IRA: growth-focused with tax-free appreciation."
        },
        ("USA", "401(k)"): {
            "favor_low_fee": True,
            "note": "401(k): cost efficiency and retirement growth prioritized."
        },
        ("USA", "Taxable"): {
            "favor_tax_efficiency": True,
            "note": "Taxable account: ETFs ranked for tax-efficient income."
        },
        # Fallback
        ("Canada", "Institutional"): {"note": "Institutional use — no retail tax adjustments applied."},
        ("USA", "Institutional"): {"note": "Institutional use — no retail tax adjustments applied."},
        ("Canada", "RESP"): {"note": "🎓 RESP use — educational investment preferences apply."},
    }

    # Store in session state
    st.session_state["use_context"] = use_context
    st.session_state["country"] = country
    st.session_state["use_context_rules"] = context_rules.get((country, use_context), {})
    st.session_state["use_context_note"] = st.session_state["use_context_rules"].get("note", "")


    st.header("Client Profile")
    goal = st.selectbox("Investment Goal", ["Retirement", "Income", "Wealth Growth"])
    risk = st.selectbox("Risk Tolerance", ["Conservative", "Balanced", "Growth"])
    horizon = st.slider("Investment Horizon (Years)", 1, 30, 10)
    age = st.slider("Age", 18, 80, 35)
    amount = st.number_input("Investment Amount ($)", min_value=1000, step=1000)
    client_name = st.text_input("Client Name")
    notes = st.text_area("Meeting Notes")

# ---- Portfolio Builder Tab ----
with tab1:
    st.subheader("Personalized Investment Plan")
    st.markdown(f"**Client:** {client_name or 'N/A'} | **Goal:** {goal} | **Risk Profile:** {risk} | Horizon: {horizon} years | Investment Amount: ${amount:,.2f}")

    st.markdown("### Strategic Asset Allocation")
    allocation = allocation_matrix.get((goal, risk), {"Equity": 50, "Bonds": 40, "Cash": 10})
    allocation_text = {
        "Equity": "Growth-oriented exposure to drive portfolio appreciation.",
        "Bonds": "Income and capital preservation through fixed income assets.",
        "Cash": "Liquidity buffer to reduce volatility and manage short-term needs.",
        "Mixed": "Balanced exposure across equity, bonds, and cash through a single diversified ETF."
    }

    for asset_class, pct in allocation.items():
        val = (pct / 100) * amount
        st.markdown(f"**{pct}% {asset_class}** – ${val:,.2f}")
        st.caption(allocation_text.get(asset_class, ""))

if st.session_state.get("use_context_note"):
    st.caption(st.session_state["use_context_note"])

    st.info("This strategy reflects a diversified blend tailored to the client’s objective and risk tolerance.")


    # Add Mixed as a separate tab
    tab_eq, tab_bd, tab_cash, tab_mixed, tab_other = st.tabs(["Equity", "Bonds", "Cash", "Mixed", "Other"])
    tab_map = {
        "Equity": tab_eq,
        "Bonds": tab_bd,
        "Cash": tab_cash,
        "Mixed": tab_mixed,
        "Other": tab_other
    }
    asset_mapping = {
        "Equity": "equity",
        "Bonds": "bond",
        "Cash": "cash",
        "Mixed": "mixed",
        "Other": "other"
    }

        
for asset_class, tab in tab_map.items():
    with tab:
        st.markdown(f"### Top {asset_class} ETFs")
        class_key = asset_mapping[asset_class]

        filtered = etf_df[
            (etf_df["Simplified Asset Class"].str.lower() == class_key)
        ].copy()

        rules = st.session_state.get("use_context_rules", {})

        if rules.get("avoid_us_dividends"):
            filtered = filtered[~filtered["ETF Name"].str.contains("USD", case=False, na=False)]

        if rules.get("avoid_dividends"):
            filtered = filtered[pd.to_numeric(filtered["Annual Dividend Yield %"].str.replace("%", ""), errors="coerce") < 2]

        if rules.get("favor_tax_efficiency"):
            filtered = filtered[pd.to_numeric(filtered["Annual Dividend Yield %"].str.replace("%", ""), errors="coerce") < 2.5]
            filtered = filtered[pd.to_numeric(filtered["ER"].str.replace("%", ""), errors="coerce") < 0.3]

        if rules.get("favor_growth"):
            filtered = filtered[pd.to_numeric(filtered["1 Year"].str.replace("%", ""), errors="coerce") > 5]

        if rules.get("favor_low_fee"):
            filtered = filtered[pd.to_numeric(filtered["ER"].str.replace("%", ""), errors="coerce") < 0.25]

        # Apply standard filters (goal + risk)
        if asset_class != "Other":
            filtered = filtered[filtered["Risk Level"].isin(risk_filters[risk])]
            if goal in goal_preferences:
                try:
                    filtered = goal_preferences[goal](filtered)
                except:
                    pass

        st.caption(f"{len(filtered)} ETFs match the filters for this asset class and goal.")

        if st.session_state.get("use_context_note"):
            st.caption(st.session_state["use_context_note"])

        filtered["1 Year"] = pd.to_numeric(filtered["1 Year"].astype(str).str.replace("%", ""), errors="coerce")
        ranked_df = rank_etfs(filtered, goal)
        top_etfs = ranked_df.sort_values(by="Final Score", ascending=False).head(10)


        for _, row in top_etfs.iterrows():
            st.markdown(f"""
            <div style='background:#f9f9f9; padding:15px; border-radius:10px; border:1px solid #ddd; margin-bottom:15px;'>
                <b><a href='https://finance.yahoo.com/quote/{row['Symbol']}' target='_blank'>{row['Symbol']}: {row['ETF Name']}</a></b><br>
                <b>1Y Return:</b> {row['1 Year']} &nbsp; <b>Expense Ratio:</b> {row['ER']} &nbsp; <b>Yield:</b> {row['Annual Dividend Yield %']}<br>
                <b>AUM:</b> {row['Total Assets']} &nbsp; <b>Risk Level:</b> {row['Risk Level']}<br>
                <b>Score:</b> {row['Final Score']:.2f}
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

# ---- ETF Screener Tab ----
with tab2:
    st.subheader("ETF Screener")
    asset_class_options = ["All", "equity", "bond", "cash", "mixed", "other"]
    selected_asset_class = st.selectbox("Filter by Asset Class", asset_class_options)
    risk_options = ["All", "Low", "Medium", "High"]
    selected_risk = st.selectbox("Filter by Risk Level", risk_options)
    keyword = st.text_input("Search by Symbol or Name")

    screener_df = etf_df.copy()
    if selected_asset_class != "All":
        screener_df = screener_df[screener_df["Simplified Asset Class"].str.lower() == selected_asset_class.lower()]
    if selected_risk != "All":
        screener_df = screener_df[screener_df["Risk Level"] == selected_risk]
    if keyword:
        screener_df = screener_df[
            screener_df["ETF Name"].str.contains(keyword, case=False, na=False) |
            screener_df["Symbol"].str.contains(keyword, case=False, na=False)
        ]

    screener_df_display = screener_df[["Symbol", "ETF Name", "1 Year", "ER", "Annual Dividend Yield %", "Total Assets", "Risk Level"]].rename(columns={
        "1 Year": "1Y Return",
        "ER": "Expense Ratio",
        "Annual Dividend Yield %": "Yield",
        "Total Assets": "AUM"
    })

    st.dataframe(screener_df_display, use_container_width=True)

# ---- Portfolio Analyzer Tab ----
with tab3:
    st.subheader("Portfolio Analyzer")
    uploaded_file = st.file_uploader("Upload Portfolio CSV", type=["csv"])

    if uploaded_file is not None:
        user_df = pd.read_csv(uploaded_file)

        if "Symbol" not in user_df.columns:
            st.error("Your file must have a 'Symbol' column with ETF tickers.")
        else:
            # Merge with ETF metadata
            portfolio_df = etf_df[etf_df["Symbol"].isin(user_df["Symbol"])].copy()
            portfolio_df["Risk Level"] = portfolio_df.apply(classify_risk, axis=1)

            # Merge user data to get Quantity and Price (optional)
            merged = pd.merge(user_df, portfolio_df, on="Symbol", how="inner")

            # Handle market value calculation
            if "Market Value" not in merged.columns:
                if "Quantity" in merged.columns and "Price" in merged.columns:
                    merged["Market Value"] = merged["Quantity"] * merged["Price"]
                else:
                    st.warning("To calculate allocation, please provide Quantity and Price in your CSV.")
                    merged["Market Value"] = None

            if merged["Market Value"].isnull().all():
                st.warning("No valid Market Value could be calculated. Skipping allocation chart.")
            else:
                merged["Weight (%)"] = (merged["Market Value"] / merged["Market Value"].sum()) * 100

            if merged.empty:
                st.warning("No matching ETFs found in the dataset.")
            else:
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("#### Asset Class Allocation")
                    asset_counts = merged["Simplified Asset Class"].str.capitalize().value_counts()
                    fig1, ax1 = plt.subplots(figsize=(2.8, 2.8))
                    ax1.pie(asset_counts, labels=asset_counts.index, autopct="%1.0f%%", startangle=140, wedgeprops=dict(width=0.4))
                    st.pyplot(fig1)

                with col2:
                    st.markdown("#### Risk Level Distribution")
                    risk_counts = merged["Risk Level"].value_counts()
                    fig2, ax2 = plt.subplots(figsize=(2.8, 2.8))
                    ax2.pie(risk_counts, labels=risk_counts.index, autopct="%1.0f%%", startangle=140, wedgeprops=dict(width=0.4))
                    st.pyplot(fig2)

                # ---- Portfolio Scorecard ----
                st.markdown("### Portfolio Scorecard")
                col3, col4, col5, col6 = st.columns(4)

                def parse_metric(col):
                    if col == "Total Assets":
                        return pd.to_numeric(
                            merged[col].astype(str)
                            .str.replace("$", "", regex=False)
                            .str.replace("B", "", regex=False)
                            .str.replace(",", "", regex=False),
                            errors="coerce"
                        )
                    else:
                        return pd.to_numeric(merged[col].astype(str).str.replace("%", "", regex=False), errors="coerce")

                with col3:
                    st.metric("Average 1Y Return", f"{parse_metric('1 Year').mean():.2f}%")
                with col4:
                    st.metric("Avg. Expense Ratio", f"{parse_metric('ER').mean():.2f}%")
                with col5:
                    st.metric("Avg. Dividend Yield", f"{parse_metric('Annual Dividend Yield %').mean():.2f}%")
                with col6:
                    st.metric("Avg. AUM (B)", f"{parse_metric('Total Assets').mean():.2f}B")

                # ---- Portfolio Table ----
                st.markdown("### Portfolio Breakdown")
                display_cols = [
                    "Symbol", "ETF Name", "1 Year", "ER", "Annual Dividend Yield %", 
                    "Total Assets", "Simplified Asset Class", "Risk Level", "Market Value", "Weight (%)"
                ]
                display_renames = {
                    "1 Year": "1Y Return",
                    "ER": "Expense Ratio",
                    "Annual Dividend Yield %": "Yield",
                    "Total Assets": "AUM",
                    "Simplified Asset Class": "Asset Class"
                }

                existing_cols = [col for col in display_cols if col in merged.columns]
                st.dataframe(merged[existing_cols].rename(columns=display_renames), use_container_width=True)

with tab4:
    st.subheader("Rebalancing Checker")
    st.caption("Upload your portfolio CSV (with Symbol, Quantity, and Price). We’ll auto-match ETF metadata and calculate drift from your personalized FundMentor recommendation.")

    rebal_file = st.file_uploader("Upload Current Portfolio (CSV with 'Symbol', 'Quantity', and 'Price')", type=["csv"], key="rebal_upload")

    if rebal_file is not None:
        try:
            uploaded_portfolio = pd.read_csv(rebal_file)
            uploaded_portfolio["Symbol"] = uploaded_portfolio["Symbol"].astype(str)

            if not {"Symbol", "Quantity", "Price"}.issubset(uploaded_portfolio.columns):
                st.error("Your CSV must contain 'Symbol', 'Quantity', and 'Price' columns.")
            else:
                # Merge with ETF metadata
                merged = pd.merge(uploaded_portfolio, etf_df[["Symbol", "Simplified Asset Class"]], on="Symbol", how="left")
                merged["Market Value"] = merged["Quantity"] * merged["Price"]
                merged["Weight (%)"] = (merged["Market Value"] / merged["Market Value"].sum()) * 100
                merged["Simplified Asset Class"] = merged["Simplified Asset Class"].fillna("Other")
                merged["Market Value"] = merged["Quantity"] * merged["Price"]
                merged["Weight (%)"] = (merged["Market Value"] / merged["Market Value"].sum()) * 100
                merged["Simplified Asset Class"] = merged["Simplified Asset Class"].str.lower()

                # Notes
                mixed_weight = merged.loc[merged["Simplified Asset Class"] == "mixed", "Weight (%)"].sum()
                if mixed_weight > 0:
                    st.markdown(f"📊 **Note:** Mixed ETFs account for **{mixed_weight:.2f}%** of the portfolio. They are not included in drift calculations due to diversified composition.")

                if "other" in merged["Simplified Asset Class"].values:
                    other_weight = merged.loc[merged["Simplified Asset Class"] == "other", "Weight (%)"].sum()
                    st.markdown(f"📌 **Note:** 'Other' holdings account for **{other_weight:.2f}%** of the portfolio and are excluded from drift analysis.")

                if "Weight (%)" not in merged.columns or merged["Weight (%)"].isnull().all():
                    st.warning("⚠️ Cannot run rebalancing: missing or invalid weight data. Please check your file.")
                    st.stop()

                # Core asset class comparison
                filtered = merged[merged["Simplified Asset Class"].isin(["equity", "bond", "cash"])].copy()
                filtered["Weight (%)"] = (filtered["Market Value"] / filtered["Market Value"].sum()) * 100
                filtered["Market Value"] = filtered["Quantity"] * filtered["Price"]
                filtered["Weight (%)"] = (filtered["Market Value"] / filtered["Market Value"].sum()) * 100


                actual_alloc = filtered.groupby("Simplified Asset Class")["Weight (%)"].sum().reset_index()
                model_allocation = allocation_matrix.get((goal, risk), {})
                normalized_model = {k.lower(): v for k, v in model_allocation.items()}

                actual_alloc["Recommended (%)"] = actual_alloc["Simplified Asset Class"].map(normalized_model)
                actual_alloc["Drift (%)"] = actual_alloc["Weight (%)"] - actual_alloc["Recommended (%)"]

                def suggest_action(row):
                    drift = row["Drift (%)"]
                    asset_class = row["Simplified Asset Class"].capitalize()
                    if drift > 5:
                        return f"🔻 Reduce exposure to {asset_class} by {drift:.2f}%"
                    elif drift < -5:
                        return f"🔺 Increase exposure to {asset_class} by {abs(drift):.2f}%"
                    else:
                        return "✅ On Target"

                actual_alloc["Suggested Action"] = actual_alloc.apply(suggest_action, axis=1)

                def highlight_drift(val):
                    return "color: red;" if pd.notna(val) and abs(val) > 5 else ""

                st.markdown("### Rebalancing Summary with Suggested Actions")
                st.dataframe(
                    actual_alloc[["Simplified Asset Class", "Weight (%)", "Recommended (%)", "Drift (%)", "Suggested Action"]]
                    .style.applymap(highlight_drift, subset=["Drift (%)"]),
                    use_container_width=True
                )

                st.info("Red cells highlight over/underweighting greater than ±5%.")

                # ETF-Level rebalancing
                if "Weight (%)" not in filtered.columns or filtered["Weight (%)"].isnull().all():
                    st.warning("⚠️ Cannot run ETF-level rebalancing: weight data is missing or invalid.")
                else:
                    st.markdown("### ETF-Level Rebalancing Actions")

                    target_df = pd.DataFrame([
                        {"Asset Class": k.capitalize(), "Target Weight (%)": v}
                        for k, v in normalized_model.items()
                    ])

                    # Recalculate Weight (%)
                    filtered["Market Value"] = filtered["Quantity"] * filtered["Price"]
                    filtered["Weight (%)"] = (filtered["Market Value"] / filtered["Market Value"].sum()) * 100

                    # Prepare current portfolio
                    current_df = filtered[["Symbol", "Simplified Asset Class", "Weight (%)"]].copy()
                    current_df = current_df.rename(columns={
                        "Symbol": "ETF",
                        "Simplified Asset Class": "Asset Class"
                    })
                    current_df["Asset Class"] = current_df["Asset Class"].str.capitalize()

                    # Prepare target portfolio by splitting total target weight across ETFs in that class
                    target_df = pd.DataFrame([
                        {"Asset Class": k.capitalize(), "Target Weight (%)": v}
                        for k, v in normalized_model.items()
                    ])

                    # Merge target weights into current_df
                    current_df = current_df.merge(target_df, on="Asset Class", how="left")

                    # Now safely generate rebalancing actions
                    etf_rebalance_result = generate_rebalance_actions(current_df)
                    st.dataframe(etf_rebalance_result, use_container_width=True)

                    st.dataframe(etf_rebalance_result, use_container_width=True)

        except Exception as e:
            st.error(f"Error processing file: {e}")
    else:
        st.caption("Please upload a portfolio CSV to run the Rebalancing Checker.")

