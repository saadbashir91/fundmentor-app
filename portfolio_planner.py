import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import datetime as dt, pathlib  # A1 banner
import re                       # A4 screener toggle

# ==== DATA SOURCE SETTINGS ====
# ==== DATA SOURCE SETTINGS ====
USE_NEW_DATA = True  # keep using Excel files

# --- Point these to your actual files (keep the .xlsx extension) ---
US_DATA_PATH = "US ETFs.xlsx"   # e.g., /mnt/data/screener-etf-2025-08-15.xlsx if you don't rename
US_DATA_SHEET = "Export"        # change if your US file uses a different sheet name

CA_DATA_PATH = "CA ETFs.xlsx"   # e.g., /mnt/data/screener-etf-2025-08-15 (1).xlsx if you don't rename
CA_DATA_SHEET = "Export"        # change if your CA file uses a different sheet name

# Fallback defaults (used when no region picked)
DEFAULT_DATA_PATH = US_DATA_PATH
DEFAULT_DATA_SHEET = US_DATA_SHEET

# These two get set dynamically later based on your choice/country
NEW_DATA_PATH = DEFAULT_DATA_PATH
NEW_DATA_SHEET = DEFAULT_DATA_SHEET

OLD_DATA_PATH = "etf_asset_class_tagged.csv"  # keep legacy CSV fallback if needed

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
    # Helpers (keep them INDENTED inside classify_risk)
    def parse_percentage(val):
        try:
            x = float(str(val).replace('%', '').replace(',', '').strip())
            # if value looks like 0.05 (i.e., 5%), bump to percent
            if x < 1:
                x *= 100.0
            return x
        except Exception:
            return None

    def parse_aum(val):
        """
        Turn things like "$12.3B", "27,845,929,000", "500M", "850000000"
        into a plain number of dollars (float).
        """
        try:
            s = str(val).strip().replace('$', '').replace(',', '').lower()
            mult = 1.0
            if s.endswith('t'):
                mult, s = 1e12, s[:-1]
            elif s.endswith('b'):
                mult, s = 1e9, s[:-1]
            elif s.endswith('m'):
                mult, s = 1e6, s[:-1]
            elif s.endswith('k'):
                mult, s = 1e3, s[:-1]
            return float(s) * mult
        except Exception:
            return None

    # Read values
    ret = parse_percentage(row.get("1 Year", ""))
    er = parse_percentage(row.get("ER", ""))
    yield_pct = parse_percentage(row.get("Annual Dividend Yield %", ""))
    aum = parse_aum(row.get("Total Assets", ""))  # now defined!

    # Scoring
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

    # IMPORTANT: your Excel has AUM as raw dollars; compare to billions.
    if aum is not None:
        if aum < 1e9:   # <$1B is riskier
            score += 2
        elif aum > 1e10:  # >$10B is a bit safer
            score -= 1

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

def infer_listing_country(symbol, name="", tags=""):
    s = str(symbol or "").upper()
    # Obvious TSX/NEO/Canadian suffixes
    if any(s.endswith(suf) for suf in ["TO", "TSX", "TSXV", "NE", "NEO", "CN"]):
        return "Canada"
    # Heuristics from the text too
    txt = f"{name} {tags}".lower()
    if any(w in txt for w in ["tsx", "toronto", "canada"]):
        return "Canada"
    # Default fallback
    return "USA"

def get_country_policy(country: str, account_type: str, asset_class: str):
    """
    Returns dict with:
      - hard_include: set of Listing Country values to keep exclusively
      - hard_exclude: set to drop
      - score_boost: dict of { (listing_country, asset_class): factor }
    """
    country = (country or "").strip()
    acct = (account_type or "").strip()
    ac = asset_class  # "Equity", "Bonds", "Cash", ...

    policy = {"hard_include": set(), "hard_exclude": set(), "score_boost": {}}

    if country == "USA":
        # US investors: avoid PFIC headaches -> keep US only
        policy["hard_include"] = {"USA"}

    elif country == "Canada":
        if acct == "RRSP":
            if ac == "Equity":
                # allow both, but nudge US-listed for US equity in RRSP
                policy["score_boost"] = {("USA", "Equity"): 1.03}
        elif acct in {"TFSA", "RESP"}:
            if ac == "Equity":
                # avoid unrecoverable US withholding in TFSA/RESP
                policy["hard_exclude"] = {"USA"}
        elif acct == "Non-Registered":
            # allow both; tiny preference for CAD-listed equities
            policy["score_boost"] = {("Canada", "Equity"): 1.01}

    return policy

# ---- Load ETF Data ----
@st.cache_data
def load_etf_data(use_new=USE_NEW_DATA, _excel_mtime=None, _csv_mtime=None):
    """
    Loads ETF data from either the new Excel (preferred) or the legacy CSV,
    and normalizes column names to the legacy schema used throughout the app.
    """
    import pandas as pd

    # --- 1) Load file
    if use_new:
        try:
            df = pd.read_excel(NEW_DATA_PATH, sheet_name=NEW_DATA_SHEET)
        except Exception as e:
            st.warning(f"Could not read new Excel file ({NEW_DATA_PATH}). Falling back to legacy CSV. Error: {e}")
            df = pd.read_csv(OLD_DATA_PATH)
            df = df.rename(columns={"Total Assets ": "Total Assets"})
    else:
        df = pd.read_csv(OLD_DATA_PATH)
        df = df.rename(columns={"Total Assets ": "Total Assets"})

        # --- SAFETY: if the reads somehow failed, keep the app alive with an empty table
    if 'df' not in locals() or df is None:
        st.error("Failed to load any data source. Using an empty table so the app can still run.")
        df = pd.DataFrame(columns=[
            "Symbol","ETF Name","1 Year","ER","Annual Dividend Yield %","Total Assets",
            "Tags","Listing Country"
        ])

    # --- 2) Rename new-file columns to the old names your app already uses
    rename_map_new_to_legacy = {
        "Fund Name": "ETF Name",
        "Assets": "Total Assets",
        "Exp. Ratio": "ER",
        "Return 1Y": "1 Year",
        "Div. Yield": "Annual Dividend Yield %",
    }
    rename_actual = {k: v for k, v in rename_map_new_to_legacy.items() if k in df.columns}
    if rename_actual:
        df = df.rename(columns=rename_actual)

    # --- 3) Make sure the key columns exist (so the rest of your app works)
    required_cols = ["Symbol", "ETF Name", "1 Year", "ER", "Annual Dividend Yield %", "Total Assets"]
    for col in required_cols:
        if col not in df.columns:
            df[col] = None  # create empty column if missing

    # Standardize types as strings; later code strips %/$ anyway
    for col in ["1 Year", "ER", "Annual Dividend Yield %", "Total Assets"]:
        df[col] = df[col].astype(str)

    # --- 4) Build "Simplified Asset Class" if missing (the new file doesn’t have it)
    if "Simplified Asset Class" not in df.columns:
        sector_words = [
            "technology","healthcare","financial","energy","materials","industrials",
            "utilities","real estate","reit","consumer","communication","staples","discretionary"
        ]
        index_words = [
            "s&p","spdr","nasdaq","russell","msci","ftse","dow","tsx","total market","all cap",
            "small cap","mid cap","large cap","micro cap","value","growth","dividend","covered call"
        ]
        equity_words = ["equity","stock","shares","index","etf","capital"]
        bond_words   = ["bond","fixed income","treasury","t-bill","t bill","govt","government","corporate","credit","aggregate","muni","mortgage"]
        cash_words   = ["money market","cash","t-bill","t bill","ultra short","very short","savings"]
        mixed_words  = ["balanced","allocation","portfolio","target","asset allocation","vgro","vbal","xbal","xgro","zbal","zbla"]

        def classify_row(name, tags):
            name_l = str(name).lower()
            tags_l = str(tags).lower()

            if any(w in name_l for w in mixed_words) or any(w in tags_l for w in ["balanced","allocation","multi-asset","multi asset"]):
                return "mixed"
            if any(w in name_l for w in bond_words) or any(w in tags_l for w in ["treasury","bond","fixed income"]):
                return "bond"
            if any(w in name_l for w in cash_words) or any(w in tags_l for w in ["cash","t-bill","t bill","money market"]):
                return "cash"
            if any(w in name_l for w in sector_words + index_words + equity_words) \
               or any(w in tags_l for w in ["sector","equity","crypto","covered-calls","covered calls"]):
                return "equity"
            if "etf" in name_l:
                return "equity"
            return "other"

        df["Simplified Asset Class"] = df.apply(
            lambda r: classify_row(r.get("ETF Name", ""), r.get("Tags", "")),
            axis=1
        )

    # --- 5) Keep your Mixed re-tagging rule
    mixed_keywords = ["balanced", "growth", "portfolio", "income", "allocation", "target",
                      "vgro", "xgro", "zbla", "zbal", "xbal", "vbla", "vbal"]
    df["is_potential_mixed"] = df["ETF Name"].astype(str).str.lower().apply(
        lambda name: any(keyword in name for keyword in mixed_keywords)
    )
    df["Simplified Asset Class"] = df.apply(
        lambda row: "Mixed" if row["is_potential_mixed"] and
        (pd.isna(row["Simplified Asset Class"]) or str(row["Simplified Asset Class"]).strip().lower() in ["", "other"])
        else row["Simplified Asset Class"],
        axis=1
    )
    df["Simplified Asset Class"] = df["Simplified Asset Class"].astype(str).str.lower()

    # --- Numeric helper columns used across the app ---
    def _pct_series(s):
        n = pd.to_numeric(pd.Series(s).astype(str).str.replace('%','', regex=False).str.replace(',','', regex=False), errors='coerce')
        # If looks like 0.05 rather than 5, convert to percent scale
        return n.where(n > 1.5, n * 100)

    def _aum_to_billions(x):
        s = str(x).strip().lower().replace('$','').replace(',','')
        mult = 1.0
        if s.endswith('t'): mult, s = 1e12, s[:-1]
        elif s.endswith('b'): mult, s = 1e9, s[:-1]
        elif s.endswith('m'): mult, s = 1e6, s[:-1]
        try:
            return float(s) * mult / 1e9
        except:
            return None

    df["1Y_num"]    = _pct_series(df["1 Year"])
    df["ER_num"]    = _pct_series(df["ER"])
    df["Yield_num"] = _pct_series(df["Annual Dividend Yield %"])
    df["AUM_bil"]   = df["Total Assets"].apply(_aum_to_billions)

    
    

        # --- 4b) Add Listing Country using native columns first, then heuristics
    exch_to_country = {
        # USA exchanges
        "NYSE": "USA", "NASDAQ": "USA", "NYSEARCA": "USA", "BATS": "USA",
        "CBOE": "USA", "CBOE BZX": "USA", "CBOE BYX": "USA", "CBOE EDGX": "USA",
        # Canada exchanges
        "TSX": "Canada", "TSXV": "Canada", "TSX VENTURE": "Canada",
        "NEO": "Canada", "CBOE CANADA": "Canada", "CSE": "Canada",
        "CBOE NEO": "Canada"
    }

    # Start with an empty column
    df["Listing Country"] = None

    # 1) If there's a Country column, use it (normalize)
    if "Country" in df.columns:
        df["Listing Country"] = (
            df["Country"].astype(str)
              .str.strip()
              .str.replace(r"\.0$", "", regex=True)
              .map({"United States": "USA", "Canada": "Canada"})
              .fillna(df["Country"])  # keep other values as-is (e.g., China)
        )

    # 2) Fill any blanks using Exchange mapping
    if "Exchange" in df.columns:
        df["Listing Country"] = df["Listing Country"].where(
            df["Listing Country"].notna(),
            df["Exchange"].astype(str).str.strip().str.upper().map(exch_to_country)
        )

    # 3) Any remaining blanks: fall back to ticker/name heuristics
    def _infer(symbol, name="", tags=""):
        # Use your existing helper
        return infer_listing_country(symbol, name, tags)

    df["Listing Country"] = df["Listing Country"].where(
        df["Listing Country"].notna(),
        df.apply(lambda r: _infer(r.get("Symbol", ""), r.get("ETF Name", ""), r.get("Tags", "")), axis=1)
    )

    # 4) Final tidy: Unknown if still empty
    df["Listing Country"] = df["Listing Country"].fillna("Unknown")

    # --- 6) Compute Risk Level and return
    df["Risk Level"] = df.apply(classify_risk, axis=1)
    return df

def _mtime(path):
    try:
        return os.path.getmtime(path)
    except:
        return None



# --- Data banner helper ---
def _asof_label():
    src = NEW_DATA_PATH if USE_NEW_DATA else OLD_DATA_PATH
    try:
        ts = dt.datetime.fromtimestamp(os.path.getmtime(src))
        return f"{pathlib.Path(src).name} • {ts:%Y-%m-%d %H:%M} • {len(etf_df):,} rows"
    except:
        return f"{pathlib.Path(src).name} • {len(etf_df):,} rows"

# ---- Multi-Factor ETF Scoring Engine ----
def normalize(series):
    s = pd.to_numeric(series, errors="coerce")
    lo, hi = s.min(), s.max()
    # If all values are NaN, or the column is constant (hi == lo),
    # return a neutral 0.5 for every row to avoid divide-by-zero.
    if pd.isna(lo) or pd.isna(hi) or hi == lo:
        return pd.Series(0.5, index=s.index)
    return (s - lo) / (hi - lo)


def get_factor_weights(goal):
    override = st.session_state.get("weights_override")
    if override:
        return override  # already normalized from the sliders
    # default weights if no override is set
    if goal == "Wealth Growth":
        return {"1Y": 0.30, "ER": 0.20, "AUM": 0.10, "Yield": 0.10, "TaxEff": 0.30}
    elif goal == "Retirement":
        return {"1Y": 0.20, "ER": 0.20, "AUM": 0.10, "Yield": 0.20, "TaxEff": 0.30}
    elif goal == "Income":
        return {"1Y": 0.10, "ER": 0.20, "AUM": 0.10, "Yield": 0.30, "TaxEff": 0.30}
    else:
        return {"1Y": 0.20, "ER": 0.20, "AUM": 0.10, "Yield": 0.20, "TaxEff": 0.30}


def safe_goal_filter(df, goal):
    y = pd.to_numeric(df["Annual Dividend Yield %"].astype(str).str.replace("%", ""), errors="coerce")
    r = pd.to_numeric(df["1 Year"].astype(str).str.replace("%", ""), errors="coerce")
    # if numbers look like 0.05 instead of 5, turn them into percents
    if y.notna().any() and y.max() <= 1.5: y = y * 100
    if r.notna().any() and r.max() <= 1.5: r = r * 100

    if goal == "Retirement":   return df[y > 1.8]
    if goal == "Income":       return df[y > 2.2]
    if goal == "Wealth Growth":return df[r > 6]
    return df

def rank_etfs(df, goal):
    df = df.copy()

        # Prefer precomputed numeric columns if present
    df["1Y_clean"]    = df.get("1Y_num",    pd.to_numeric(df["1 Year"].astype(str).str.replace('%','', regex=False), errors="coerce"))
    df["ER_clean"]    = df.get("ER_num",    pd.to_numeric(df["ER"].astype(str).str.replace('%','', regex=False), errors="coerce"))
    df["Yield_clean"] = df.get("Yield_num", pd.to_numeric(df["Annual Dividend Yield %"].astype(str).str.replace('%','', regex=False), errors="coerce"))
    df["AUM_clean"]   = df.get("AUM_bil",   pd.to_numeric(
        df["Total Assets"].astype(str).str.replace("$","", regex=False).str.replace("B","", regex=False).str.replace(",","", regex=False),
        errors="coerce"
    ))

    # Zero-division safe tax-efficiency
    denom = (df["Yield_clean"].fillna(0) + df["ER_clean"].fillna(0)).replace(0, pd.NA)
    TaxEff = 1 / denom
    TaxEff = pd.to_numeric(TaxEff, errors="coerce")
    TaxEff.replace([float("inf"), float("-inf")], pd.NA, inplace=True)
    df["TaxEff_clean"] = TaxEff



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

# --- Helper: ranked candidates for one asset class (same logic as the UI tabs) ---
def get_ranked_for_class(asset_class, goal, risk, country, account_type, etf_df, risk_filters):
    """
    Returns a scored-and-sorted DataFrame of ETFs for a single asset class,
    applying the same filters/policies used in the Portfolio Builder UI.
    """
    # Map UI asset-class label to the normalized key used in your dataset
    class_key_map = {"Equity": "equity", "Bonds": "bond", "Cash": "cash", "Mixed": "mixed", "Other": "other"}
    class_key = class_key_map[asset_class]

    # Start from the master dataset for this asset class
    df = etf_df[etf_df["Simplified Asset Class"].str.lower() == class_key].copy()

    # Country/account policy (hard filters first)
    policy = get_country_policy(country, account_type, asset_class)
    if policy["hard_include"]:
        df = df[df["Listing Country"].isin(policy["hard_include"])]
    if policy["hard_exclude"]:
        df = df[~df["Listing Country"].isin(policy["hard_exclude"])]

    # Sidebar context rules (soft filters)
    rules = st.session_state.get("use_context_rules") or {}

    if rules.get("avoid_us_dividends"):
        df = df[df["Listing Country"].ne("USA")]

    if rules.get("avoid_dividends"):
        df = df[pd.to_numeric(df["Annual Dividend Yield %"].str.replace("%", ""), errors="coerce") < 2]

    if rules.get("favor_tax_efficiency"):
        df = df[
            (pd.to_numeric(df["Annual Dividend Yield %"].str.replace("%", ""), errors="coerce") < 2.5) &
            (pd.to_numeric(df["ER"].str.replace("%", ""), errors="coerce") < 0.3)
        ]

    if rules.get("favor_growth"):
        df = df[pd.to_numeric(df["1 Year"].str.replace("%", ""), errors="coerce") > 5]

    if rules.get("favor_low_fee"):
        df = df[pd.to_numeric(df["ER"].str.replace("%", ""), errors="coerce") < 0.25]

    # Risk/goal filtering (gentle on bonds & cash)
    if asset_class != "Other":
        allowed = set(risk_filters[risk])
        if asset_class in ["Bonds", "Cash"]:
            allowed.update({"Low", "Medium"})  # keep safer choices in fixed income/cash
        df = df[df["Risk Level"].isin(allowed)]

    if asset_class == "Equity":
        df = safe_goal_filter(df, goal)  # only applies goal thresholds to equities

    # Score with your engine
    ranked = rank_etfs(df, goal).sort_values("Final Score", ascending=False)

    # Soft preference boosts (after scoring)
    boosts = policy.get("score_boost", {})
    if boosts and "Listing Country" in ranked.columns:
        for (lst_country, ac), factor in boosts.items():
            if asset_class == ac:
                ranked.loc[ranked["Listing Country"] == lst_country, "Final Score"] *= float(factor)
        ranked = ranked.sort_values("Final Score", ascending=False)

    return ranked

# --- Helper: avoid picking two ETFs tracking the same index in one sleeve ---
# Uses simple name patterns (S&P 500, Nasdaq-100, Aggregate Bond, etc.)
import re  # already imported above; safe to re-import

_INDEX_BUCKETS = [
    ("sp500",        re.compile(r"\b(s&p(\s*500)?|500 index)\b", re.I)),
    ("total_us",     re.compile(r"\b(total\s*market|vti|schb|itot)\b", re.I)),
    ("nasdaq100",    re.compile(r"\b(nasdaq\s*100|qqq|ndx)\b", re.I)),
    ("russell2000",  re.compile(r"\b(russell\s*2000|iwo|iwn)\b", re.I)),
    ("intl_dev",     re.compile(r"\b(msci\s*eafe|eafe|developed\s*ex\s*us)\b", re.I)),
    ("emerging",     re.compile(r"\b(emerging|msci\s*em|vwo|iemg)\b", re.I)),
    ("agg_bond",     re.compile(r"\b(aggregate|aggg?|core bond|total bond|bnd|agg)\b", re.I)),
    ("treasury",     re.compile(r"\b(treasur(y|ies)|gov(ernment)? bond)\b", re.I)),
    ("corp_bond",    re.compile(r"\b(corporate|ig credit)\b", re.I)),
    ("t_bill_cash",  re.compile(r"\b(t[- ]?bill|money market|ultra short)\b", re.I)),
]

def _index_bucket(etf_name: str) -> str:
    name = str(etf_name or "")
    for bucket, pat in _INDEX_BUCKETS:
        if pat.search(name):
            return bucket
    return "other"

def keep_one_per_bucket(df, name_col="ETF Name", max_per_bucket=1):
    """
    Keeps the highest-scoring ETF per index bucket (e.g., only one S&P 500 fund).
    Increase max_per_bucket if you *do* want duplicates.
    """
    if df.empty or name_col not in df.columns:
        return df
    df = df.copy()
    df["_bucket"] = df[name_col].apply(_index_bucket)
    df = df.sort_values("Final Score", ascending=False)
    df = df.groupby("_bucket", as_index=False, group_keys=False).head(max_per_bucket)
    return df.drop(columns=["_bucket"], errors="ignore")

# ---- App Tabs ----
# ---- Add Custom ETF Lists Tab ----
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Portfolio Builder", "ETF Screener", "Portfolio Analyzer", "Rebalancing Checker", "Custom ETF Lists"
])

# --- Sidebar: Client Profile ---

with st.sidebar:
    st.header("Context")

    # Choose which Excel to use
    dataset_choice = st.radio(
        "Dataset file",
        ["Auto (match Country)", "US ETFs", "CA ETFs"],
        index=0,
        help="Auto uses the Country picker below to choose the file."
    )

    # Allow blank ("None") as default option for more flexibility
    country = st.selectbox("Select Country", ["", "Canada", "USA"], index=0)

    account_type_options = {
        "Canada": ["", "TFSA", "RRSP", "RESP", "Non-Registered", "Institutional"],
        "USA": ["", "Roth IRA", "401(k)", "Traditional IRA", "Taxable", "Institutional"]
    }
    account_type = st.selectbox("Account Type", account_type_options.get(country, [""]), index=0)

    # Rules engine with fallback behavior for blanks
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
        ("Canada", "Institutional"): {"note": "Institutional use — no retail tax adjustments applied."},
        ("USA", "Institutional"): {"note": "Institutional use — no retail tax adjustments applied."},
        ("Canada", "RESP"): {"note": "🎓 RESP use — educational investment preferences apply."},
    }

    rules_key = (country, account_type)
    rules_applied = context_rules.get(rules_key, {})

    st.session_state["use_context"] = account_type
    st.session_state["country"] = country
    st.session_state["use_context_rules"] = rules_applied
    st.session_state["use_context_note"] = rules_applied.get("note", "")

    # --- Quick data sanity check (optional) ---
    if st.checkbox("Show data sanity check", key="dbg_assetclass"):
        st.write("Row count:", len(etf_df))
        st.write(etf_df["Simplified Asset Class"].value_counts(dropna=False))

    # Risk questionnaire logic (unchanged)
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

    st.header("Client Profile")
    goal = st.selectbox("Investment Goal", ["Retirement", "Income", "Wealth Growth"])
    risk = st.selectbox("Risk Tolerance", ["Conservative", "Balanced", "Growth"])
    horizon = st.slider("Investment Horizon (Years)", 1, 30, 10)
    age = st.slider("Age", 18, 80, 35)
    amount = st.number_input("Investment Amount ($)", min_value=1000, step=1000)
    client_name = st.text_input("Client Name")
    notes = st.text_area("Meeting Notes")

    with st.expander("Advanced: scoring weights"):
        w_1y = st.slider("1Y performance", 0.0, 0.6, 0.30 if goal=="Wealth Growth" else 0.20, 0.05, key="w1y")
        w_er = st.slider("Expense ratio",   0.0, 0.6, 0.20, 0.05, key="wer")
        w_aum = st.slider("AUM size",       0.0, 0.6, 0.10, 0.05, key="waum")
        w_yld = st.slider("Dividend yield", 0.0, 0.6, 0.10 if goal=="Wealth Growth" else 0.20, 0.05, key="wyld")
        w_tax = st.slider("Tax efficiency", 0.0, 0.6, 0.30, 0.05, key="wtax")
        tot = max(w_1y + w_er + w_aum + w_yld + w_tax, 1e-9)  # normalize
        st.session_state["weights_override"] = {
            "1Y": w_1y/tot, "ER": w_er/tot, "AUM": w_aum/tot, "Yield": w_yld/tot, "TaxEff": w_tax/tot
        }

        # NEW: reset weights to FundMentor defaults
        if st.button("Reset weights to defaults", key="w_reset"):
            st.session_state["w1y"] = 0.30 if goal=="Wealth Growth" else 0.20
            st.session_state["wer"] = 0.20
            st.session_state["waum"] = 0.10
            st.session_state["wyld"] = 0.10 if goal=="Wealth Growth" else 0.20
            st.session_state["wtax"] = 0.30
            st.session_state.pop("weights_override", None)
            # If this errors on your Streamlit version, replace with st.rerun()
            st.experimental_rerun()

        # Decide which Excel to open based on dataset_choice + Country
        if dataset_choice == "US ETFs" or (dataset_choice.startswith("Auto") and country == "USA"):
            NEW_DATA_PATH = US_DATA_PATH
            NEW_DATA_SHEET = US_DATA_SHEET
        elif dataset_choice == "CA ETFs" or (dataset_choice.startswith("Auto") and country == "Canada"):
            NEW_DATA_PATH = CA_DATA_PATH
            NEW_DATA_SHEET = CA_DATA_SHEET
        else:
            NEW_DATA_PATH = DEFAULT_DATA_PATH
            NEW_DATA_SHEET = DEFAULT_DATA_SHEET

        # Now load the data for the selected file
        etf_df = load_etf_data(
            _excel_mtime=_mtime(NEW_DATA_PATH),   # cache-busts per file
            _csv_mtime=_mtime(OLD_DATA_PATH),
        )

        # Show the banner *after* loading, so it reports the correct file & row count
        st.sidebar.caption(f"Data as of: {_asof_label()}")


# ---- Portfolio Builder Tab ----
with tab1:
    st.subheader("Personalized Investment Plan")
    st.markdown(f"**Client:** {client_name or 'N/A'} | **Goal:** {goal} | **Risk Profile:** {risk} | Horizon: {horizon} years | Investment Amount: ${amount:,.2f}")

    st.markdown("### Strategic Asset Allocation")
    # --- Glidepath option ---
    base_alloc = allocation_matrix.get((goal, risk), {"Equity": 50, "Bonds": 40, "Cash": 10})
    use_glide = st.checkbox("Use age-based glidepath (≈ 120 − age)", value=False, help="Adjusts equity weight by age and scales Bonds/Cash proportionally.")
    if use_glide:
        # Equity target from a simple, explainable rule of thumb
        eq_target = max(30, min(90, 120 - age))  # clamp between 30% and 90%
        # --- Guardrails: risk capacity checks ---
        cap_by_risk = {"Conservative": 50, "Balanced": 70, "Growth": 90}
        reasons = []

        # 1) Cap equity by stated risk tolerance
        risk_cap = cap_by_risk.get(risk, 90)
        if eq_target > risk_cap:
            reasons.append(f"risk profile cap {risk_cap}%")
            eq_target = risk_cap

        # 2) Cap further for very short horizon (≤ 3 years)
        if horizon <= 3 and eq_target > 40:
            reasons.append("short-horizon cap 40%")
            eq_target = 40

        # 3) Cap for retirement-age clients (≥ 65)
        if age >= 65 and eq_target > 60:
            reasons.append("retirement-age cap 60%")
            eq_target = 60

        if reasons:
            st.warning("Glidepath guardrails applied: " + " • ".join(reasons))

        # Split the remaining between Bonds & Cash in the same proportion as the base mix
        non_eq = 100 - eq_target
        bonds_base = base_alloc.get("Bonds", 0)
        cash_base  = base_alloc.get("Cash", 0)
        base_non_eq = max(bonds_base + cash_base, 1e-9)
        bonds_target = non_eq * (bonds_base / base_non_eq)
        cash_target  = non_eq * (cash_base  / base_non_eq)
        allocation = {"Equity": round(eq_target, 1), "Bonds": round(bonds_target, 1), "Cash": round(cash_target, 1)}
        st.caption("🧭 Glidepath applied (age-aware).")
    else:
        allocation = base_alloc

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
    else:
        st.caption("📌 No account-specific filtering applied. Showing standard recommendations.")

    st.info("This strategy reflects a diversified blend tailored to the client’s objective and risk tolerance.")

    # --- Build a concrete model from the strategy ---
    st.markdown("### Build a concrete model (tickers + dollars)")
    etfs_per_class = st.slider("ETFs per asset class", 1, 3, 1, help="Pick top-N by score within each asset class.")
    min_score = st.slider("Min score to include", 0.00, 1.00, 0.00, 0.05, help="Skip very weak candidates.")
    include_mixed_as_core = st.checkbox("Allow a single Mixed fund to replace Equity/Bonds/Cash if it scores very high", value=False, help="If enabled and a top Mixed ETF scores ≥ chosen threshold, it will be used as a one-ticket core.")

    if st.button("Generate model portfolio"):
        rows = []

        # Optional: replace the whole core with one Mixed ETF if enabled and strong enough
        if include_mixed_as_core:
            mixed_ranked = get_ranked_for_class("Mixed", goal, risk, st.session_state.get("country",""), st.session_state.get("use_context",""), etf_df, risk_filters)
            if not mixed_ranked.empty and mixed_ranked.iloc[0]["Final Score"] >= min_score:
                sym, name, score = mixed_ranked.iloc[0]["Symbol"], mixed_ranked.iloc[0]["ETF Name"], mixed_ranked.iloc[0]["Final Score"]
                rows.append({"Asset Class":"Mixed","Symbol":sym,"ETF Name":name,"Weight %":100.0,"Score":round(score,3)})
            else:
                st.caption("No Mixed candidate passed the threshold; using 3-bucket approach.")
    
        if not rows:  # normal 3-bucket build
            for ac, pct in allocation.items():  # e.g., {"Equity":60,"Bonds":35,"Cash":5}
                ranked = get_ranked_for_class(ac, goal, risk, st.session_state.get("country",""), st.session_state.get("use_context",""), etf_df, risk_filters)
                ranked = ranked[ranked["Final Score"] >= min_score]
                ranked = keep_one_per_bucket(ranked)
                if ranked.empty:
                    continue
                take = ranked.head(etfs_per_class)
                for _, r in take.iterrows():
                    rows.append({
                        "Asset Class": ac,
                        "Symbol": r["Symbol"],
                        "ETF Name": r["ETF Name"],
                        "Weight %": round(pct / max(len(take),1), 2),
                        "Score": round(r["Final Score"], 3)
                    })

        if rows:
            model_df = pd.DataFrame(rows)
            model_df["Dollars"] = (model_df["Weight %"] / 100.0) * float(amount)

            # ---- Bring in ER and Yield from the master ETF table ----
            meta = etf_df[["Symbol", "ER", "Annual Dividend Yield %"]].copy()

            # Parse ER and Yield as numeric PERCENTS (e.g., 0.20 means 0.20%)
            meta["ER_num"] = pd.to_numeric(
                meta["ER"].astype(str).str.replace("%", "", regex=False),
                errors="coerce"
            )
            meta["Yield_num"] = pd.to_numeric(
                meta["Annual Dividend Yield %"].astype(str).str.replace("%", "", regex=False),
                errors="coerce"
            )

            # Merge onto your model
            model_df = model_df.merge(meta[["Symbol", "ER_num", "Yield_num"]], on="Symbol", how="left")

            # Safety: treat missing ER/Yield as 0
            model_df[["ER_num", "Yield_num"]] = model_df[["ER_num", "Yield_num"]].fillna(0)

            # ---- Compute dollars per year ----
            model_df["Fee $/yr"]    = (model_df["Dollars"] * (model_df["ER_num"] / 100)).round(2)
            model_df["Income $/yr"] = (model_df["Dollars"] * (model_df["Yield_num"] / 100)).round(2)

            # Nice column names for display
            display_cols = [
                "Asset Class", "Symbol", "ETF Name",
                "Weight %", "Dollars",
                "ER_num", "Yield_num",
                "Fee $/yr", "Income $/yr",
                "Score"
            ]
            display_renames = {"ER_num": "ER (%)", "Yield_num": "Yield (%)"}

            st.dataframe(
                model_df[display_cols].rename(columns=display_renames),
                use_container_width=True
            )

            # ---- Summary metrics ----
            total_fees   = float(model_df["Fee $/yr"].sum())
            total_income = float(model_df["Income $/yr"].sum())
            weighted_er  = (model_df["ER_num"]   * model_df["Dollars"]).sum() / max(model_df["Dollars"].sum(), 1e-9)
            weighted_yld = (model_df["Yield_num"]* model_df["Dollars"]).sum() / max(model_df["Dollars"].sum(), 1e-9)

            c1, c2, c3 = st.columns(3)
            with c1: st.metric("Total annual fees",   f"${total_fees:,.2f}")
            with c2: st.metric("Est. annual income",  f"${total_income:,.2f}")
            with c3: st.metric("Weighted ER / Yield", f"{weighted_er:.2f}% / {weighted_yld:.2f}%")
            
            st.session_state["model_df"] = model_df  # <— add this line to reuse in the explanation
            st.success("Model portfolio created.")

            # NEW: easy copy/download of tickers
            tickers_str = ",".join(model_df["Symbol"].astype(str).unique())
            st.text_input("Tickers (copy/paste)", tickers_str, key="mp_tickers")

            st.download_button(
                "Download tickers (.txt)",
                tickers_str.encode(),
                file_name="model_tickers.txt",
                mime="text/plain",
                key="mp_tickers_dl"
            )


                # ---- Optional: whole-share quantities (enter prices) ----
            with st.expander("Optional: enter ETF prices to get whole-share quantities"):
                st.caption("Enter prices for each ETF (or a default). Leave blank/0 to skip a row.")

                # Keep per-symbol prices in session
                if "model_prices" not in st.session_state:
                    st.session_state["model_prices"] = {}

                # Default price that fills any blank rows (optional)
                default_price = st.number_input(
                    "Default price for blank rows ($)",
                    min_value=0.0, value=0.0, step=0.01, key="default_price_model"
                )

                # Collect per-ETF prices
                price_inputs = []
                for _, r in model_df.iterrows():
                    sym = str(r["Symbol"])
                    key = f"price_{sym}"
                    # Pre-fill from saved state (if any)
                    preset = float(st.session_state["model_prices"].get(sym, 0.0))
                    price_val = st.number_input(f"{sym} price ($)", min_value=0.0, value=preset, step=0.01, key=key)
                    st.session_state["model_prices"][sym] = price_val
                    price_inputs.append(price_val)

                # Build price Series aligned to model_df
                price_series = pd.Series(price_inputs, index=model_df.index).replace(0, pd.NA)
                if default_price and default_price > 0:
                    price_series = price_series.fillna(default_price)

                model_df["Price ($)"] = price_series

                # Compute whole-share quantities only where price > 0
                qty_series = pd.to_numeric(
                    (model_df["Dollars"] / model_df["Price ($)"]).where(model_df["Price ($)"] > 0),
                    errors="coerce"
                ).fillna(0).astype(int)

                model_df["Qty (whole)"]  = qty_series
                model_df["Allocated $"]  = (model_df["Qty (whole)"] * model_df["Price ($)"]).round(2)
                model_df["Residual Cash"] = (model_df["Dollars"] - model_df["Allocated $"]).round(2)

                # Show a trade-ready table (adds to the earlier table; no need to remove it)
                st.markdown("### Trade-ready table (with quantities)")
                trade_cols = [
                    "Asset Class","Symbol","ETF Name",
                    "Weight %","Dollars",
                    "Price ($)","Qty (whole)","Allocated $","Residual Cash",
                    "ER_num","Yield_num","Fee $/yr","Income $/yr","Score"
                ]
                rename_cols = {"ER_num":"ER (%)","Yield_num":"Yield (%)"}
                st.dataframe(
                    model_df[trade_cols].rename(columns=rename_cols),
                    use_container_width=True
                )

                # Totals
                tot_alloc = float(model_df["Allocated $"].sum())
                tot_cash  = float(model_df["Residual Cash"].sum())
                st.caption(f"Allocated today: ${tot_alloc:,.2f}  •  Unallocated cash: ${tot_cash:,.2f}")


            # ---- Download CSV (with fees/income) ----
            csv_out = model_df.rename(columns=display_renames).to_csv(index=False).encode()
            st.download_button("Download model portfolio (CSV)", csv_out, "model_portfolio.csv", "text/csv")
        else:
            st.info("No qualifying ETFs for the current filters/thresholds.")

                # --- Explain my plan (client-friendly text) ---
    with st.expander("📝 Explain my plan"):
        # Friendly names for factor weights
        _friendly = {"1Y":"1-year performance","ER":"fees (expense ratio)","AUM":"fund size (AUM)",
                     "Yield":"dividend yield","TaxEff":"tax efficiency"}
        fw = get_factor_weights(goal)
        top_drivers = " + ".join([_friendly[k] for k, _ in sorted(fw.items(), key=lambda kv: kv[1], reverse=True)[:2]])

        # Allocation lines (percent and dollars)
        alloc_lines = []
        for k, v in allocation.items():
            try:
                dollars = (float(v)/100.0) * float(amount)
                alloc_lines.append(f"- **{k}**: **{v}%** (≈ **${dollars:,.0f}**)")
            except Exception:
                alloc_lines.append(f"- **{k}**: **{v}%**")

        # Context note (TFSA/RRSP/Taxable etc.), if any
        acct_note = st.session_state.get("use_context_note") or ""

        # Was a model built? If yes, list tickers in plain English
        model_df = st.session_state.get("model_df")
        if model_df is not None and not model_df.empty:
            lines_tickers = []
            for _, r in model_df.sort_values("Weight %", ascending=False).iterrows():
                lines_tickers.append(f"- **{r['Symbol']}** ({r['Asset Class']}) — {r['Weight %']}% (≈ ${float(r['Dollars']):,.0f})")
            tickers_block = "\n".join(lines_tickers)
        else:
            tickers_block = "_Generate a model portfolio above to see specific ETFs and dollar amounts._"

        # Build the narrative
        st.markdown(f"""
**Client:** {client_name or "N/A"}  
**Goal:** **{goal}** • **Risk:** **{risk}** • **Horizon:** **{horizon} yrs** • **Age:** **{age}**  
**Amount:** **${float(amount):,.0f}**

### Why this mix
- Your plan targets **{allocation.get('Equity',0)}% Equity**, **{allocation.get('Bonds',0)}% Bonds**, and **{allocation.get('Cash',0)}% Cash**{ " with an age-aware glidepath" if use_glide else "" }.
- ETF candidates are ranked primarily by **{top_drivers}** for this goal.

{f"**Account/Country note:** {acct_note}" if acct_note else ""}

### Allocation summary
{chr(10).join(alloc_lines)}

### How to implement (ETFs & dollars)
{tickers_block}

### Next steps / caveats
- This is a starting point, not personal advice. Markets change; rebalance and review annually or after big life changes.
- Fees and taxes matter. Lower fees and tax efficiency generally help long-term outcomes.
- If you prefer fewer moving parts, consider a **Mixed** (all-in-one) ETF option in the builder above.
""")


    # Tabs by asset class
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
            st.markdown(f"### {asset_class} ETF Recommendations")
            class_key = asset_mapping[asset_class]

            filtered = etf_df[
                (etf_df["Simplified Asset Class"].str.lower() == class_key)
            ].copy()


            # ---- Country/account policy (hard filters first) ----
            policy = get_country_policy(
                st.session_state.get("country", ""),
                st.session_state.get("use_context", ""),
                asset_class
            )

            if policy["hard_include"]:
                filtered = filtered[filtered["Listing Country"].isin(policy["hard_include"])]

            if policy["hard_exclude"]:
                filtered = filtered[~filtered["Listing Country"].isin(policy["hard_exclude"])]


            rules = st.session_state.get("use_context_rules") or {}

            if rules.get("avoid_us_dividends"):
                filtered = filtered[filtered["Listing Country"].ne("USA")]

            if rules.get("avoid_dividends"):
                filtered = filtered[pd.to_numeric(filtered["Annual Dividend Yield %"].str.replace("%", ""), errors="coerce") < 2]

            if rules.get("favor_tax_efficiency"):
                filtered = filtered[
                    (pd.to_numeric(filtered["Annual Dividend Yield %"].str.replace("%", ""), errors="coerce") < 2.5) &
                    (pd.to_numeric(filtered["ER"].str.replace("%", ""), errors="coerce") < 0.3)
                ]

            if rules.get("favor_growth"):
                filtered = filtered[pd.to_numeric(filtered["1 Year"].str.replace("%", ""), errors="coerce") > 5]

            if rules.get("favor_low_fee"):
                filtered = filtered[pd.to_numeric(filtered["ER"].str.replace("%", ""), errors="coerce") < 0.25]

            # ---- one risk/goal filter pass (gentle on bonds & cash) ----
            if asset_class != "Other":
                allowed = set(risk_filters[risk])
                if asset_class in ["Bonds", "Cash"]:
                    allowed.update({"Low", "Medium"})  # don't over-prune safe stuff
                filtered = filtered[filtered["Risk Level"].isin(allowed)]

            # goal thresholds are for equities; don’t shrink bonds/cash with them
            if asset_class == "Equity":
                filtered = safe_goal_filter(filtered, goal)
        

            st.caption(f"{len(filtered)} ETFs match the filters for this asset class and goal.")

            if st.session_state.get("use_context_note"):
                st.caption(st.session_state["use_context_note"])
            elif not st.session_state.get("country") or not st.session_state.get("use_context"):
                st.caption("📌 No account-specific filtering applied.")

            ranked_df = rank_etfs(filtered, goal)
            ranked_df = ranked_df.sort_values(by="Final Score", ascending=False)

            # ---- Soft preference nudges (apply after scoring) ----
            boosts = policy.get("score_boost", {})
            if boosts and "Listing Country" in ranked_df.columns:
                for (lst_country, ac), factor in boosts.items():
                    if asset_class == ac:
                        ranked_df.loc[ranked_df["Listing Country"] == lst_country, "Final Score"] *= float(factor)
                ranked_df = ranked_df.sort_values(by="Final Score", ascending=False)

            
            # If totally empty after filters, switch to a relaxed fallback and still show tiers
            if ranked_df.empty:
                relax = etf_df[etf_df["Simplified Asset Class"].str.lower() == class_key].copy()
                if not relax.empty:
                    ranked_df = rank_etfs(relax, goal).sort_values(by="Final Score", ascending=False).head(12)
                    st.warning("No ETFs matched after filters. Showing top options for this asset class without risk filtering.")

            if ranked_df.empty:
                st.info("No ETFs available after filtering.")
            else:
                
                # --- Dynamic tiers based on your data ---
                scores = ranked_df["Final Score"].dropna()

                if len(scores) >= 4:
                    # 75th and 50th percentiles – gives you a “top quartile” Tier 1
                    q75 = scores.quantile(0.75)
                    q50 = scores.quantile(0.50)
                else:
                    # Very small sets: use gentle fixed cutoffs
                    q75, q50 = 0.55, 0.45

                tier_1 = ranked_df[ranked_df["Final Score"] >= q75]
                tier_2 = ranked_df[(ranked_df["Final Score"] < q75) & (ranked_df["Final Score"] >= q50)]
                tier_3 = ranked_df[ranked_df["Final Score"] < q50]

                # Safety net: if Tier 1 ends up empty (e.g., all scores are very similar),
                # make sure we still show a few in Tier 1/2/3 by slicing.
                if tier_1.empty and not ranked_df.empty:
                    tier_1 = ranked_df.head(3)
                    tier_2 = ranked_df.iloc[3:8]
                    tier_3 = ranked_df.iloc[8:20]

                
                def _fmt_pct(x):
                    try:
                        return f"{float(x):.2f}%"
                    except:
                        return "—"

                def _fmt_bil(x):
                    try:
                        return f"${float(x):.2f}B"
                    except:
                        return "—"

                if not tier_1.empty:
                    st.markdown("### Tier 1 ETFs - (highest rated)")
                    top_3 = tier_1.head(3)
                    for _, row in top_3.iterrows():
                        one_year = row.get("1Y_clean", row.get("1 Year"))
                        er = row.get("ER_clean", row.get("ER"))
                        yld = row.get("Yield_clean", row.get("Annual Dividend Yield %"))
                        aum_bil = row.get("AUM_clean", None)

                        # --- WHY THIS ETF (top two contributors) ---
                        weights_here = get_factor_weights(goal)
                        contrib = {
                            "1Y":   weights_here["1Y"]   * row.get("1Y_score", 0),
                            "ER":   weights_here["ER"]   * row.get("ER_score", 0),
                            "AUM":  weights_here["AUM"]  * row.get("AUM_score", 0),
                            "Yield":weights_here["Yield"]* row.get("Yield_score", 0),
                            "Tax":  weights_here["TaxEff"]*row.get("TaxEff_score", 0),
                        }
                        top2 = sorted(contrib.items(), key=lambda kv: kv[1], reverse=True)[:2]
                        why = " + ".join([f"{k}↑" for k,_ in top2]) if top2 else "—"

                        st.markdown(f"""
                        <div style='background:#eef9f2; padding:15px; border-radius:10px; border:1px solid #b6e5c5; margin-bottom:15px;'>
                            <b><a href='https://finance.yahoo.com/quote/{row['Symbol']}' target='_blank'>{row['Symbol']}: {row['ETF Name']}</a></b><br>
                            <b>1Y Return:</b> {_fmt_pct(one_year)} &nbsp; <b>Expense Ratio:</b> {_fmt_pct(er)} &nbsp; <b>Yield:</b> {_fmt_pct(yld)}<br>
                            <b>AUM:</b> {_fmt_bil(aum_bil)} &nbsp; <b>Risk Level:</b> {row['Risk Level']}<br>
                            <b>Score:</b> {row['Final Score']:.2f} &nbsp; <b>Why:</b> {why}
                        </div>
                        """, unsafe_allow_html=True)


                    st.markdown("**Full Tier 1 Comparison Table**")
                    st.dataframe(
                        tier_1[["Symbol", "ETF Name", "1 Year", "ER", "Annual Dividend Yield %", "Total Assets", "Risk Level"]]
                        .rename(columns={
                            "1 Year": "1Y Return",
                            "ER": "Expense Ratio",
                            "Annual Dividend Yield %": "Yield",
                            "Total Assets": "AUM"
                        }),
                        use_container_width=True
                    )    

                if not tier_2.empty:
                    st.markdown("### Tier 2 ETFs (Strong Alternatives)")
                    st.dataframe(
                        tier_2[["Symbol", "ETF Name", "1 Year", "ER", "Annual Dividend Yield %", "Total Assets", "Risk Level"]]
                        .rename(columns={
                            "1 Year": "1Y Return",
                            "ER": "Expense Ratio",
                            "Annual Dividend Yield %": "Yield",
                            "Total Assets": "AUM"
                        }),
                        use_container_width=True
                    )
                    

                if not tier_3.empty:
                    st.markdown("### Tier 3 ETFs (Broader Exploration)")
                    st.dataframe(
                        tier_3[["Symbol", "ETF Name", "1 Year", "ER", "Annual Dividend Yield %", "Total Assets", "Risk Level"]]
                        .rename(columns={
                            "1 Year": "1Y Return",
                            "ER": "Expense Ratio",
                            "Annual Dividend Yield %": "Yield",
                            "Total Assets": "AUM"
                        }),
                        use_container_width=True
                    )

# ---- ETF Screener Tab ----
with tab2:
    st.subheader("ETF Screener")
    asset_class_options = ["All", "equity", "bond", "cash", "mixed", "other"]
    selected_asset_class = st.selectbox("Filter by Asset Class", asset_class_options, key="scr_asset_class")
    risk_options = ["All", "Low", "Medium", "High"]
    selected_risk = st.selectbox("Filter by Risk Level", risk_options, key="scr_risk")
    keyword = st.text_input("Search by Symbol or Name", key="scr_keyword")

    # NEW: one-click reset for screener filters
    if st.button("Reset screener filters", key="scr_reset"):
        st.session_state["scr_asset_class"] = "All"
        st.session_state["scr_risk"] = "All"
        st.session_state["scr_keyword"] = ""
        st.session_state["scr_exlev"] = True
        st.success("Screener filters reset.")

    # start from master df
    screener_df = etf_df.copy()

    # 5b(i) Safety: ensure "Listing Country" exists so filtering never crashes
    if "Listing Country" not in screener_df.columns:
        if "Listing Country" in etf_df.columns:
            screener_df["Listing Country"] = etf_df["Listing Country"]
        else:
            screener_df["Listing Country"] = screener_df.apply(
                lambda r: infer_listing_country(
                    r.get("Symbol", ""), r.get("ETF Name", ""), r.get("Tags", "")
                ),
                axis=1
            )
    screener_df["Listing Country"] = screener_df["Listing Country"].fillna("Unknown")

    # 5b(ii) Apply country filter from the sidebar only when chosen
    if country in ("USA", "Canada"):
        screener_df = screener_df[screener_df["Listing Country"] == country]
        st.caption(f"Active country filter: {country}")
    else:
        st.caption("Active country filter: All")

    # Asset class filter
    if selected_asset_class != "All":
        screener_df = screener_df[
            screener_df["Simplified Asset Class"].str.lower() == selected_asset_class.lower()
        ]

    # Risk filter
    if selected_risk != "All":
        screener_df = screener_df[screener_df["Risk Level"] == selected_risk]

    # Keyword search
    if keyword:
        screener_df = screener_df[
            screener_df["ETF Name"].str.contains(keyword, case=False, na=False) |
            screener_df["Symbol"].str.contains(keyword, case=False, na=False)
        ]
    
    # Optional safety screen
    ex_lev = st.checkbox("Exclude leveraged/inverse/ETNs/option overlays", value=True, key="scr_exlev")
    if ex_lev:
        bad = [
            "3x","2x","-3x","-2x",
            "2xbull","3xbull","bear -1x",   # NEW edge cases
            "leveraged","ultra","inverse","short","bear","etn",
            "covered call","buy-write","option income","option overlay"
        ]
        pat = re.compile("|".join(map(re.escape, bad)))
        screener_df = screener_df[~screener_df["ETF Name"].str.lower().str.contains(pat, na=False)]


    # Display
    screener_df_display = screener_df[
        ["Symbol", "ETF Name", "1 Year", "ER", "Annual Dividend Yield %", "Total Assets", "Risk Level"]
    ].rename(columns={
        "1 Year": "1Y Return",
        "ER": "Expense Ratio",
        "Annual Dividend Yield %": "Yield",
        "Total Assets": "AUM"
    })

    # NEW: choose columns and download results
    available_cols = list(screener_df_display.columns)
    cols_to_show = st.multiselect("Columns to show", available_cols, default=available_cols, key="scr_cols")
    screener_shown = screener_df_display[cols_to_show]

    st.download_button(
        "Download screener results (CSV)",
        screener_shown.to_csv(index=False).encode(),
        file_name="screener_results.csv",
        mime="text/csv",
        key="scr_dl"
    )



    if screener_df_display.empty:
        st.info("No ETFs match the current filters.")
    else:
        st.dataframe(screener_shown, use_container_width=True)



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
                # --- Concentration metrics ---
                w_series = pd.to_numeric(merged["Weight (%)"], errors="coerce").fillna(0) / 100.0
                if not w_series.empty and w_series.sum() > 0:
                    hhi = float((w_series**2).sum())
                    top_idx = w_series.idxmax()
                    top_sym = merged.loc[top_idx, "Symbol"]
                    top_w   = float(merged.loc[top_idx, "Weight (%)"])
                    c1, c2 = st.columns(2)
                    with c1: st.metric("Concentration (HHI)", f"{hhi:.3f}", help="0≈diversified, 1=single position")
                    with c2: st.metric("Largest position", f"{top_sym} • {top_w:.1f}%")

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

    # NEW: make drift threshold adjustable
    drift_thr = st.slider("Rebalance drift threshold (±%)", min_value=1, max_value=20, value=5, step=1,
                          help="How far an actual weight can deviate from target before recommending a trade.")


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
                    if drift > drift_thr:
                        return f"🔻 Reduce exposure to {asset_class} by {drift:.2f}%"
                    elif drift < -drift_thr:
                        return f"🔺 Increase exposure to {asset_class} by {abs(drift):.2f}%"
                    else:
                        return "✅ On Target"


                actual_alloc["Suggested Action"] = actual_alloc.apply(suggest_action, axis=1)
              
                def highlight_drift(val):
                    return "color: red;" if pd.notna(val) and abs(val) > drift_thr else ""


                st.markdown("### Rebalancing Summary with Suggested Actions")
                st.dataframe(
                    actual_alloc[["Simplified Asset Class", "Weight (%)", "Recommended (%)", "Drift (%)", "Suggested Action"]]
                    .style.map(highlight_drift, subset=["Drift (%)"]),
                    use_container_width=True
                )

                st.info(f"Red cells highlight over/underweighting greater than ±{drift_thr}%.")


                # ETF-Level rebalancing
                if "Weight (%)" not in filtered.columns or filtered["Weight (%)"].isnull().all():
                    st.warning("⚠️ Cannot run ETF-level rebalancing: weight data is missing or invalid.")
                else:
                    st.markdown("### ETF-Level Rebalancing Actions")

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
                    etf_rebalance_result = generate_rebalance_actions(current_df, threshold=drift_thr)
                    st.dataframe(etf_rebalance_result, use_container_width=True)

                    # --- Build trade list with dollar and share amounts ---
                    total_mv = filtered["Market Value"].sum()
                    reb = etf_rebalance_result.merge(
                        filtered.rename(columns={"Symbol": "ETF"})[["ETF","Price","Market Value"]],
                        on="ETF", how="left"
                    )
                    reb["Target MV"]  = total_mv * (reb["Target Weight (%)"] / 100.0)
                    reb["Current MV"] = total_mv * (reb["Weight (%)"] / 100.0)
                    reb["Trade $"]    = reb["Target MV"] - reb["Current MV"]
                    reb["Trade Qty"]  = (reb["Trade $"] / reb["Price"]).round().astype("Int64")

                    st.markdown("### Trade List")
                    st.dataframe(reb[["ETF","Asset Class","Price","Current MV","Target MV","Trade $","Trade Qty"]],
                                use_container_width=True)
                    
                    # NEW: total absolute trade dollars
                    total_trade_abs = reb["Trade $"].abs().sum()
                    st.metric("Total trade (absolute $)", f"${total_trade_abs:,.0f}")


                    csv_trades = reb.to_csv(index=False).encode()
                    st.download_button("Download trade list (CSV)", csv_trades, "rebalance_trades.csv", "text/csv")


        except Exception as e:
            st.error(f"Error processing file: {e}")
    else:
        st.caption("Please upload a portfolio CSV to run the Rebalancing Checker.")

with tab5:
    st.subheader("Custom ETF Lists (Session-Only)")
    st.caption("Upload one or more of your own ETF lists to filter, rank, and manage portfolios. No data is stored after the session ends.")

    if "custom_etf_lists" not in st.session_state:
        st.session_state["custom_etf_lists"] = {}

    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"], key="custom_etf_upload")
    list_name = st.text_input("Name this list", key="custom_etf_name")

    if uploaded_file and list_name:
        try:
            df = pd.read_csv(uploaded_file)
            df.columns = [col.strip() for col in df.columns]  # clean column names
            st.session_state["custom_etf_lists"][list_name] = df
            st.success(f"✅ List '{list_name}' uploaded successfully.")
        except Exception as e:
            st.error(f"❌ Upload failed: {e}")

    if st.session_state["custom_etf_lists"]:
        st.subheader("Your Uploaded Lists")
        selected_list = st.selectbox("Choose a list to view", list(st.session_state["custom_etf_lists"].keys()), key="custom_list_selector")

        if selected_list:
            df_view = st.session_state["custom_etf_lists"][selected_list]
            st.write(f"### {selected_list} — {len(df_view)} ETFs")

            if "Symbol" not in df_view.columns:
                st.warning("⚠️ Your file must include a 'Symbol' column to identify ETFs.")
            else:
                goal_input = goal
                risk_input = risk

                def get_weights(goal, available):
                    base = {
                        "1 Year": 0.3, "ER": 0.2, "Total Assets": 0.1,
                        "Annual Dividend Yield %": 0.2, "TaxEff": 0.2
                    }
                    if goal == "Income":
                        base["Annual Dividend Yield %"] = 0.3
                        base["1 Year"] = 0.1
                    elif goal == "Wealth Growth":
                        base["1 Year"] = 0.4
                        base["Annual Dividend Yield %"] = 0.1
                    return {k: v for k, v in base.items() if k in available}

                df_clean = df_view.copy()
                df_clean.columns = [col.strip() for col in df_clean.columns]

                numeric_cols = {
                    "1 Year": "%", "ER": "%", "Annual Dividend Yield %": "%", "Total Assets": "$"
                }

                for col, symbol in numeric_cols.items():
                    if col in df_clean.columns:
                        df_clean[col] = pd.to_numeric(
                            df_clean[col].astype(str).str.replace(symbol, "", regex=False).str.replace(",", ""),
                            errors="coerce"
                        )

                if "Annual Dividend Yield %" in df_clean.columns and "ER" in df_clean.columns:
                    df_clean["TaxEff"] = 1 / (df_clean["Annual Dividend Yield %"] + df_clean["ER"])

                # Ensure Risk Level is computed before filtering
                if "Risk Level" not in df_clean.columns:
                    df_clean["Risk Level"] = df_clean.apply(classify_risk, axis=1)

                available_cols = df_clean.columns.tolist()
                weights = get_weights(goal_input, available_cols)

                if not weights:
                    st.warning("⚠️ No usable columns found to apply scoring.")
                else:
                    def normalize(series):
                        s = pd.to_numeric(series, errors="coerce")
                        lo, hi = s.min(), s.max()
                        if pd.isna(lo) or pd.isna(hi) or hi == lo:
                            return pd.Series(0.5, index=s.index)
                        return (s - lo) / (hi - lo)


                    score = pd.Series(0, index=df_clean.index)
                    for col, weight in weights.items():
                        try:
                            score += normalize(df_clean[col]) * weight
                        except:
                            continue
                    df_clean["Final Score"] = score

                    # Auto-classify into simplified asset classes
                    if "ETF Name" in df_clean.columns:
                        keywords = {
                            "Bond": ["bond", "fixed income", "rate", "yield"],
                            "Equity": ["equity", "stock", "dividend", "shares", "growth", "capital"],
                            "Cash": ["money market", "cash", "ultra short", "treasury"],
                            "Mixed": ["balanced", "allocation", "portfolio", "vgro", "vbal", "xbal", "xgro", "zbal", "zbla"],
                        }

                        def classify_name(name):
                            name = str(name).lower()
                            for category, words in keywords.items():
                                if any(w in name for w in words):
                                    return category
                            return "Other"

                        df_clean["Simplified Asset Class"] = df_clean["ETF Name"].apply(classify_name)
                    else:
                        df_clean["Simplified Asset Class"] = "Other"

                    # Apply Goal Preferences (clean version)
                    goal_preferences_clean = {
                        "Retirement": lambda df: df[df["Annual Dividend Yield %"] > 1.8],
                        "Income": lambda df: df[df["Annual Dividend Yield %"] > 2.2],
                        "Wealth Growth": lambda df: df[df["1 Year"] > 6]
                    }

                    if goal_input in goal_preferences_clean:
                        try:
                            df_clean = goal_preferences_clean[goal_input](df_clean)
                        except Exception as e:
                            st.warning(f"⚠️ Could not apply goal filter: {e}")

                    # Apply Risk Filter
                    if risk_input in risk_filters:
                        df_clean = df_clean[df_clean["Risk Level"].isin(risk_filters[risk_input])]

                    # Apply Account Type Filters
                    rules = st.session_state.get("use_context_rules") or {}

    
                    if st.session_state.get("use_context_note"):
                        st.caption(f"Account-specific filtering applied: {st.session_state['use_context_note']}")
                    else:
                        st.caption("📌 No account-specific filtering applied.")

                    # Main Table
                    display_cols = ["Symbol", "ETF Name"] if "ETF Name" in df_clean.columns else ["Symbol"]
                    display_cols += list(weights.keys()) + ["Final Score", "Simplified Asset Class"]

                    st.dataframe(df_clean[display_cols].sort_values(by="Final Score", ascending=False), use_container_width=True)

                    # Tabs by Asset Class
                    tab_eq, tab_bd, tab_cash, tab_mixed, tab_other = st.tabs(["Equity", "Bonds", "Cash", "Mixed", "Other"])
                    tab_map = {
                        "Equity": tab_eq, "Bond": tab_bd, "Cash": tab_cash, "Mixed": tab_mixed, "Other": tab_other
                    }

                    for class_name, tab in tab_map.items():
                        with tab:
                            class_df = df_clean[df_clean["Simplified Asset Class"] == class_name]
                            st.markdown(f"### {class_name} ETFs — {len(class_df)}")
                            if not class_df.empty:
                                st.dataframe(class_df[display_cols].sort_values(by="Final Score", ascending=False), use_container_width=True)
