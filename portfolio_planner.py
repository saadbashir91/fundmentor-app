import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import datetime as dt, pathlib  # A1 banner
import re                       # A4 screener toggle

global etf_df, NEW_DATA_PATH, NEW_DATA_SHEET

# ——— SAFETY: make sure the name exists even before loading ———
etf_df = pd.DataFrame()

def _has_etf_data():
    return 'etf_df' in globals() and isinstance(etf_df, pd.DataFrame) and not etf_df.empty

# ==== DATA SOURCE SETTINGS ====
USE_NEW_DATA = True  # keep using Excel files

from pathlib import Path  # add this import if not already present

# Resolve the Excel files based on where THIS .py file lives
BASE = Path(__file__).parent

US_DATA_PATH = str((BASE / "US ETFs.xlsx").resolve())
US_DATA_SHEET = "Export"

CA_DATA_PATH = str((BASE / "CA ETFs.xlsx").resolve())
CA_DATA_SHEET = "Export"


# Fallback defaults (used when no region picked)
DEFAULT_DATA_PATH = US_DATA_PATH
DEFAULT_DATA_SHEET = US_DATA_SHEET

# These two get set dynamically later based on your choice/country
NEW_DATA_PATH = DEFAULT_DATA_PATH
NEW_DATA_SHEET = DEFAULT_DATA_SHEET

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

# Unified asset-class mapping used everywhere
ASSET_MAP = {
    "Equity": "equity",
    "Bonds": "bond",
    "Cash": "cash",
    "Mixed": "mixed",
    "Other": "other",
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
    """
    BEST-EFFORT fallback only.
    Detect common Canadian and US exchange *prefixes* (e.g., "TSX:"),
    then fall back to *strict* suffix checks like ".TO" or "-NE".
    Default = USA.
    """
    s_raw = str(symbol or "").upper().strip()

    # --- Prefix patterns (strongest signal) ---
    if s_raw.startswith(("TSX:", "TSXV:", "NEO:", "CSE:", "CBOE CANADA:", "TSE:")):
        return "Canada"
    if s_raw.startswith(("NYSE:", "NASDAQ:", "NYSEARCA:", "AMEX:", "CBOE:", "BATS:", "ARCA:")):
        return "USA"

    # --- Strict suffix patterns (require dot or hyphen before the code) ---
    # Examples: "VFV.TO", "SHOP.TSX", "ABC-NE", "XYZ-CN"
    import re
    if re.search(r'(\.(TO|TSX|TSXV|NE|NEO|CN)|-(NE|CN))$', s_raw):
        return "Canada"

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

# --- Account-aware tax scoring (simple rules) ---
def _tax_flag_and_score(row, country: str, account_type: str):
    """
    Returns (flag_text, score_float in [0,1]) based on simple, explainable rules.
    Higher score = more tax efficient for the chosen account.
    """
    ac = str(row.get("Simplified Asset Class", "")).lower()   # equity/bond/cash/mixed/other
    lst = str(row.get("Listing Country", "")).strip() or "Unknown"
    yld = pd.to_numeric(str(row.get("Annual Dividend Yield %", "")).replace("%",""), errors="coerce")
    er  = pd.to_numeric(str(row.get("ER","")).replace("%",""), errors="coerce")

    # normalize edge cases: if values look like 0.05 => 5%
    if pd.notna(yld) and yld <= 1.5: yld *= 100
    if pd.notna(er)  and er  <= 1.5: er  *= 100

    yld = 0.0 if pd.isna(yld) else float(yld)
    er  = 0.0 if pd.isna(er)  else float(er)

    # start from a neutral baseline that slightly rewards lower yield+fee drag
    # (keeps continuity with your previous proxy, but bounded to [0,1])
    base = 1.0 - min((yld*0.003 + er*0.0025), 0.6)   # gentle penalty, cap at 0.6
    score = max(0.0, min(1.0, base))
    flag  = []  # accumulate short notes

    # ---------- CANADA ----------
    if country == "Canada":
        if account_type in {"TFSA", "RESP"}:
            # US-listed distributions face unrecoverable withholding; avoid for income ETFs
            if lst == "USA" and yld >= 1.0:
                score *= 0.75
                flag.append("US withholding in TFSA/RESP")
            if ac == "bond" and yld >= 1.5:
                score *= 0.85
                flag.append("Interest fully taxed (shelter vs tfsa cap)")

        elif account_type == "RRSP":
            # US-listed equity dividends generally avoid 15% withholding when held in RRSP
            if ac == "equity" and lst == "USA":
                score *= 1.05
                flag.append("No US WHT in RRSP")
            # High-yield covered-call style still creates ordinary income; light nudge down
            if yld >= 4.0:
                score *= 0.95
                flag.append("High income drag in RRSP")

        elif account_type == "Non-Registered":
            # Interest is fully taxable; nudge bonds down
            if ac == "bond":
                score *= 0.85
                flag.append("Interest fully taxable")
            # High dividend yield increases annual tax drag
            if yld >= 3.0:
                score *= 0.90
                flag.append("High dividend tax drag")
            # Canadian-listed equity gets a small nod (eligible dividend mechanics)
            if ac == "equity" and lst == "Canada":
                score *= 1.03
                flag.append("Eligible dividends (CAD)")

    # ---------- USA ----------
    elif country == "USA":
        if account_type in {"Roth IRA", "Traditional IRA", "401(k)"}:
            # Prefer growth/low yield inside tax-advantaged; dividends are less valuable
            if yld >= 2.0:
                score *= 0.92
                flag.append("Less benefit to dividends inside IRA/401(k)")
            # Bonds are fine inside tax-advantaged, so remove most penalty
            if ac == "bond":
                score *= 1.05
                flag.append("Bond income sheltered")

        elif account_type in {"Taxable"}:
            # Prefer lower ongoing distributions in taxable
            if yld >= 3.0:
                score *= 0.88
                flag.append("High dividend tax drag (taxable)")
            if ac == "bond":
                score *= 0.85
                flag.append("Bond interest taxed at ordinary rate")

    # Mixed/cash: keep near neutral, slight nudge for cash-like in taxable (interest)
    if ac == "cash" and account_type in {"Taxable","Non-Registered"}:
        score *= 0.95
        flag.append("Interest taxed annually")

    # Bound result
    score = max(0.05, min(1.0, score))  # never zero, to avoid killing candidates entirely
    note  = " • ".join(flag) if flag else ""
    return note, score

def add_account_tax_scores(df: pd.DataFrame, country: str, account_type: str) -> pd.DataFrame:
    """Adds:
       - TaxEff_account (0..1)
       - TaxEff_note (short text for cards/tables)
       - Tax Flag      (same note kept for legacy columns)
    """
    if df.empty:
        out = df.copy()
        out["TaxEff_account"] = 0.5
        out["TaxEff_note"] = ""
        out["TaxEff_flag"] = ""
        out["Tax Flag"] = ""
        return out

    notes, scores = [], []
    for _, r in df.iterrows():
        n, s = _tax_flag_and_score(r, country, account_type)
        notes.append(n); scores.append(s)

    out = df.copy()

    # make a Series (so we can use .fillna) and align it to the DataFrame index
    scores_s = pd.Series(scores, index=out.index)
    scores_s = pd.to_numeric(scores_s, errors="coerce").fillna(0.5).astype(float)

    out["TaxEff_account"] = scores_s
    out["TaxEff_note"] = notes              # short human text for UI
    out["TaxEff_flag"] = ""                 # (kept for future short labels)
    out["Tax Flag"] = notes                 # legacy column some tables expect
    return out

# ---- SAFE EXCEL READER (avoids "Permission denied") ----
def safe_read_excel(path, sheet):
    import tempfile, shutil, pandas as pd
    try:
        return pd.read_excel(path, sheet_name=sheet)
    except PermissionError:
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
            shutil.copyfile(path, tmp.name)
            return pd.read_excel(tmp.name, sheet_name=sheet)


# ---- Load ETF Data ----
@st.cache_data
@st.cache_data
def load_etf_data(
    use_new=USE_NEW_DATA,
    _excel_mtime=None, _csv_mtime=None,
    _path=None, _sheet=None, _cache_key=None,
    force_country: str | None = None,   # NEW
):
    """
    Loads ETF data from either the new Excel (preferred) or the legacy CSV,
    and normalizes column names to the legacy schema used throughout the app.
    """
    import pandas as pd

    # Resolve which file/sheet to read for this call
    path  = _path  if _path  else NEW_DATA_PATH
    sheet = _sheet if _sheet else NEW_DATA_SHEET

    # --- 1) Load file
    if use_new:
        try:
            # Try the requested sheet; if it fails or doesn't exist, use the first sheet
            try:
                df = safe_read_excel(path, sheet)
            except Exception:
                df = safe_read_excel(path, 0)

        except Exception as e:
            st.error(f"Could not read Excel file ({path} @ {sheet}). Error: {e}")
            # keep the app alive with an empty table that has the columns your app expects
            df = pd.DataFrame(columns=[
                "Symbol","ETF Name","1 Year","ER","Annual Dividend Yield %","Total Assets",
                "Tags","Listing Country"
            ])

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
        # your exports use CAGR names, so map them:
        "CAGR 1Y": "1 Year",
        "Div. Yield": "Annual Dividend Yield %",  # if it ever appears
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

    # Keep a raw copy for country inference (preserve TSX:, TSXV:, NEO:, etc.)
    df["_SymbolRaw"] = df["Symbol"].astype(str)


    # --- 4b) Add Listing Country using native columns first, then heuristics
    def _exchange_to_country(x: str):
        s = str(x).strip().upper()
        # USA clues
        if any(k in s for k in ["NYSE", "NASDAQ", "NYSEARCA", "BATS", "CBOE", "EDGX", "BZX", "BYX"]):
            return "USA"
        # CANADA clues
        if any(k in s for k in ["TSX", "TSXV", "TSX VENTURE", "NEO", "CBOE CANADA", "CSE", "TMX", "TORONTO STOCK"]):
            return "Canada"
        return None

    # 1) If the file ALREADY has a proper "Listing Country" column, normalize and use it.
    if "Listing Country" in df.columns:
        df["Listing Country"] = (
            df["Listing Country"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
            .replace({"United States": "USA", "U.S.": "USA", "US": "USA", "Canada": "Canada"})
        )
    else:
        # 2) Start from Exchange if present
        if "Exchange" in df.columns:
            df["Listing Country"] = df["Exchange"].apply(_exchange_to_country)
        else:
            df["Listing Country"] = None

        # 3) Fallback: infer from the *raw* ticker (keeps TSX:/TSXV:/NEO: prefixes)
        df["Listing Country"] = df["Listing Country"].where(
            df["Listing Country"].notna(),
            df.apply(lambda r: infer_listing_country(r.get("_SymbolRaw", ""), r.get("ETF Name", ""), r.get("Tags", "")), axis=1)
        )

        # Finally normalize Symbol (remove exchange prefixes only after country is set)
        df["Symbol"] = (
            df["_SymbolRaw"]
            .astype(str)
            .str.replace(r"^[A-Z ]+?:", "", regex=True)
            .str.strip()
        )


    # --- Final tidy / country normalization ---
    df["Listing Country"] = df["Listing Country"].fillna("Unknown")

    # If the caller knows which file this came from, prefer that as a fallback
    if force_country in {"USA", "Canada"}:
        df["Listing Country"] = df["Listing Country"].replace({"United States": "USA", "US": "USA"})
        df["Listing Country"] = df["Listing Country"].where(df["Listing Country"].ne("Unknown"), force_country)

    # Safety: if tickers clearly look Canadian (suffix .TO/.TSX/.TSXV/-NE/-CN), correct mislabels
    cand_pat = re.compile(r'(\.(TO|TSX|TSXV|NE|NEO|CN)|-(NE|CN))$', re.I)
    mask_can_suffix = df["_SymbolRaw"].astype(str).str.upper().str.contains(cand_pat)
    df.loc[mask_can_suffix, "Listing Country"] = "Canada"


    # --- 6) Compute Risk Level and return
    df["Risk Level"] = df.apply(classify_risk, axis=1)
    return df

def _mtime(path):
    try:
        return os.path.getmtime(path)
    except:
        return None


# --- Data banner helper ---
def _asof_label(rowcount=None, label="Dataset", ts=None):
    """Build a simple banner label without exposing file paths."""
    rows = rowcount if rowcount is not None else (len(etf_df) if 'etf_df' in globals() else 0)
    if ts:
        return f"{label} • {ts:%Y-%m-%d %H:%M} • {rows:,} rows"
    return f"{label} • {rows:,} rows"


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

def _account_tax_multiplier(row, country, account_type):
    """
    Returns a multiplicative factor (e.g., 0.95 … 1.05) that nudges the Final Score
    based on simple account-aware tax rules.

    Keep it LIGHT: we already hard-include/exclude in get_country_policy().
    This only provides gentle preference where both choices remain.
    """
    try:
        lst_country = str(row.get("Listing Country", "") or "")
        asset_class = str(row.get("Simplified Asset Class", "") or "").lower()
        yld = float(row.get("Yield_clean", 0) or 0)   # already in percent scale
        er  = float(row.get("ER_clean", 0) or 0)

        # Default: neutral
        mult = 1.00

        # ---- CANADA rules ----
        if country == "Canada":
            # TFSA / RESP: distributions unrecoverably taxed from US-listed — but we already exclude most via policy.
            # Give a mild boost to CAD-listed equities that have modest yield (keeping taxes simple).
            if account_type in {"TFSA", "RESP"}:
                if asset_class == "equity" and lst_country == "Canada":
                    if yld <= 2.0:
                        mult += 0.02  # favor CAD-listed, low-ish distribution equities in TFSA/RESP

            # RRSP: US dividends from US-listed are generally treaty-exempt; nudge US-listed equity slightly
            elif account_type == "RRSP":
                if asset_class == "equity" and lst_country == "USA":
                    mult += 0.03

            # Non-Registered: favor lower distributions (tax drag) and lower ER
            elif account_type == "Non-Registered":
                if asset_class in {"equity", "bond"}:
                    if yld <= 2.0:  # lower distribution = less tax drag
                        mult += 0.02
                    if er <= 0.25:
                        mult += 0.01

        # ---- USA rules ----
        elif country == "USA":
            # Taxable: favor lower yield & lower ER (tax efficiency)
            if account_type == "Taxable":
                if asset_class in {"equity", "bond"}:
                    if yld <= 1.8:
                        mult += 0.02
                    if er <= 0.10:
                        mult += 0.01
            # Roth IRA / 401(k) / Traditional IRA: growth > distributions
            elif account_type in {"Roth IRA", "401(k)", "Traditional IRA"}:
                if asset_class == "equity":
                    if yld <= 1.5:
                        mult += 0.02  # prefer growth-oriented, low distribution

        # Clamp multiplier to a safe band so nudges stay small
        return max(0.90, min(1.08, mult))
    except Exception:
        return 1.00

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


    # Account-aware tax-efficiency (0..1), plus a short flag
    if st.session_state.get("use_account_tax", True):
        # Use the account-aware tax signal
        df = add_account_tax_scores(
            df,
            country=st.session_state.get("country", ""),
            account_type=st.session_state.get("use_context", "")
        )
        df["TaxEff_clean"] = df["TaxEff_account"]
    else:
        # Fall back to the old simple proxy (1 / (yield + ER))
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

    # ★ Apply account-aware tax nudges automatically (no checkbox)    
    country = st.session_state.get("country", "")
    acct    = st.session_state.get("use_context", "")
    if country and acct:
        df["Final Score"] = df.apply(
            lambda r: r["Final Score"] * _account_tax_multiplier(r, country, acct),
            axis=1
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
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Portfolio Builder", "ETF Screener", "Portfolio Analyzer", "Rebalancing Checker", "Custom ETF Lists", "Advisor Notes"
])

# --- Sidebar: Client Profile ---

with st.sidebar:
    st.header("Context")

    # Choose which Excel to use
    dataset_choice = st.radio(
        "Dataset file",
        ["Auto (match Country)", "US ETFs", "CA ETFs", "All (US+CA)"],
        index=0,
        help="Auto uses the Country picker below to choose the file."
    )

    # Keep country in session
    if "country" not in st.session_state:
        st.session_state["country"] = ""

    # Let the user pick Country when using "Auto"
    # (we keep it visible always — harmless if they also switch to US/CA radio)
    st.session_state["country"] = st.selectbox(
        "Country",
        ["", "Canada", "USA"],   # "" = none selected yet
        index=["", "Canada", "USA"].index(st.session_state.get("country",""))
    )


    # Auto-set the country when a single-region dataset is chosen
    # Do NOT auto-set/clear country based on dataset file.
    # Country is a user choice that drives Account Type and tax logic.
    # (All dataset logic below will just read the chosen files.)
    if dataset_choice == "All (US+CA)":
        # Optional: do not clear country; user may still want CA/US-specific account types.
        pass


        # 2C — quick existence checks (show errors if either file is missing)
        if not os.path.exists(US_DATA_PATH):
            st.error(f"Missing US data file: {US_DATA_PATH}")
        if not os.path.exists(CA_DATA_PATH):
            st.error(f"Missing CA data file: {CA_DATA_PATH}")

    country = st.session_state.get("country", "")



    # Allow blank ("None") as default option for more flexibility
    country_options = ["", "Canada", "USA"]

    account_type_options = {
        "Canada": ["", "TFSA", "RRSP", "RESP", "Non-Registered", "Institutional"],
        "USA": ["", "Roth IRA", "401(k)", "Traditional IRA", "Taxable", "Institutional"]
    }

    # Reset account type if the country changed (prevents US choices sticking when switching to Canada, and vice versa)
    prev_country = st.session_state.get("_prev_country")
    curr_country = st.session_state.get("country", "")
    if prev_country != curr_country:
        st.session_state["use_context"] = ""   # clear previous selection
    st.session_state["_prev_country"] = curr_country
    
    account_type = st.selectbox(
        "Account Type",
        account_type_options.get(st.session_state.get("country", ""), [""]),
        index=0
    )

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

    rules_key = (st.session_state.get("country", ""), account_type)
    rules_applied = context_rules.get(rules_key, {})

    st.session_state["use_context"] = account_type
    st.session_state["use_context_rules"] = rules_applied
    st.session_state["use_context_note"] = rules_applied.get("note", "")
    
    # Friendly notice so users know tax-aware ranking is active
    if st.session_state.get("country", "") and account_type:
        st.caption(f" Tax-aware ranking applied for **{st.session_state['country']} – {account_type}**.")
    else:
        st.caption(" Tax-aware ranking: set Country and Account Type to enable.")

    # --- Quick data sanity check (optional) ---
    if st.checkbox("Show data sanity check", key="dbg_assetclass"):
        if _has_etf_data():
            st.write("Row count:", len(etf_df))
            st.write(etf_df["Simplified Asset Class"].value_counts(dropna=False))
        else:
            st.info("Load data first.")


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

        st.session_state["use_account_tax"] = True

        w_1y = st.slider("1Y performance", 0.0, 0.6, 0.30 if goal=="Wealth Growth" else 0.20, 0.05, key="w1y")
        w_er = st.slider("Expense ratio",   0.0, 0.6, 0.20, 0.05, key="wer")
        w_aum = st.slider("AUM size",       0.0, 0.6, 0.10, 0.05, key="waum")
        w_yld = st.slider("Dividend yield", 0.0, 0.6, 0.10 if goal=="Wealth Growth" else 0.20, 0.05, key="wyld")
        w_tax = st.slider("Tax efficiency", 0.0, 0.6, 0.30, 0.05, key="wtax")
        tot = max(w_1y + w_er + w_aum + w_yld + w_tax, 1e-9)  # normalize
        st.session_state["weights_override"] = {
            "1Y": w_1y/tot, "ER": w_er/tot, "AUM": w_aum/tot, "Yield": w_yld/tot, "TaxEff": w_tax/tot
        }

    use_account_tax = st.checkbox(
        "Use account-aware tax scoring",
        value=True,
        help="Factor in listing country, account type, and yield to rank tax efficiency."
    )
    st.session_state["use_account_tax"] = use_account_tax

    # Decide which Excel(s) to load (no file path shown to the user)
    dataset_label, dataset_ts = "US ETFs", None

    if dataset_choice == "US ETFs" or (dataset_choice.startswith("Auto") and st.session_state.get("country", "") == "USA"):

        # Lock the "current" dataset pointers to the US file/sheet
        dataset_label = "US ETFs"
        NEW_DATA_PATH, NEW_DATA_SHEET = US_DATA_PATH, US_DATA_SHEET
        if not os.path.exists(NEW_DATA_PATH):
            st.error(f"Missing data file: {NEW_DATA_PATH}")
            etf_df = pd.DataFrame()
        else:
            etf_df = load_etf_data(
                _excel_mtime=_mtime(NEW_DATA_PATH),
                _path=NEW_DATA_PATH,
                _sheet=NEW_DATA_SHEET,
                _cache_key=f"US::{_mtime(US_DATA_PATH)}::{US_DATA_SHEET}",
                force_country="USA",
            )
        st.caption("Listing Country counts: " + str(dict(etf_df["Listing Country"].value_counts(dropna=False).to_dict())))


    elif dataset_choice == "CA ETFs" or (dataset_choice.startswith("Auto") and st.session_state.get("country", "") == "Canada"):

        # Lock the "current" dataset pointers to the CA file/sheet
        dataset_label = "CA ETFs"
        NEW_DATA_PATH, NEW_DATA_SHEET = CA_DATA_PATH, CA_DATA_SHEET

        if not os.path.exists(NEW_DATA_PATH):
            st.error(f"Missing data file: {NEW_DATA_PATH}")
            etf_df = pd.DataFrame()
        else:
            etf_df = load_etf_data(
                _excel_mtime=_mtime(NEW_DATA_PATH),
                _path=NEW_DATA_PATH,
                _sheet=NEW_DATA_SHEET,
                _cache_key=f"CA::{_mtime(CA_DATA_PATH)}::{CA_DATA_SHEET}",
                force_country="Canada",
            )


        # === Sanity: show rows loaded or the exact problem ===
        if etf_df is None or etf_df.empty:
            st.error(f"{dataset_label}: 0 rows loaded. Check file exists and sheet has data.\n"
                     f"Path: {NEW_DATA_PATH}\nSheet: {NEW_DATA_SHEET}")
        else:
            st.success(f"{dataset_label}: {len(etf_df):,} rows loaded.")


    elif dataset_choice == "All (US+CA)":
        dataset_label = "All (US+CA)"
        # Load both and combine; clear NEW_DATA_* so later reloads don't assume a single file
        us_df = load_etf_data(
            _excel_mtime=_mtime(US_DATA_PATH),
            _path=US_DATA_PATH,
            _sheet=US_DATA_SHEET,
            _cache_key=f"US::{_mtime(US_DATA_PATH)}::{US_DATA_SHEET}",
            force_country="USA",
        )
        st.info(f"US rows loaded: {len(us_df):,}")

        ca_df = load_etf_data(
            _excel_mtime=_mtime(CA_DATA_PATH),
            _path=CA_DATA_PATH,
            _sheet=CA_DATA_SHEET,
            _cache_key=f"CA::{_mtime(CA_DATA_PATH)}::{CA_DATA_SHEET}",
            force_country="Canada",
        )
        st.info(f"CA rows loaded: {len(ca_df):,}")

        etf_df = pd.concat([us_df, ca_df], ignore_index=True)
        st.info(f"Combined (before de-dupe): {len(etf_df):,}")

        if {"Symbol", "ETF Name"}.issubset(etf_df.columns):
            etf_df = etf_df.drop_duplicates(subset=["Symbol", "ETF Name", "Listing Country"])
            st.info(f"After de-dupe on Symbol+Name: {len(etf_df):,}")

            st.caption("Listing Country counts: " + str(dict(etf_df["Listing Country"].value_counts(dropna=False).to_dict())))

        else:
            st.warning("Cannot de-duplicate: missing 'Symbol' or 'ETF Name' column.")

        NEW_DATA_PATH, NEW_DATA_SHEET = None, None  # prevent later single-file reloads

        # banner timestamp = newer of the two files
        t_us, t_ca = _mtime(US_DATA_PATH), _mtime(CA_DATA_PATH)
        newer = max([t for t in [t_us, t_ca] if t is not None], default=None)
        dataset_label = "All (US+CA)"
        dataset_ts = dt.datetime.fromtimestamp(newer) if newer else None

        # === Sanity: show rows loaded or the exact problem ===
        if etf_df is None or etf_df.empty:
            st.error(f"{dataset_label}: 0 rows loaded. Check file exists and sheet has data.\n"
                     f"US Path: {US_DATA_PATH}\nCA Path: {CA_DATA_PATH}\nSheet: Export")
        else:
            st.success(f"{dataset_label}: {len(etf_df):,} rows loaded.")


    else:
        # Fallback: default to US dataset
        NEW_DATA_PATH, NEW_DATA_SHEET = US_DATA_PATH, US_DATA_SHEET
        etf_df = load_etf_data(
            _excel_mtime=_mtime(NEW_DATA_PATH),
            _path=NEW_DATA_PATH,
            _sheet=NEW_DATA_SHEET,
        )
        
    st.caption("Listing Country counts: " + str(dict(etf_df["Listing Country"].value_counts(dropna=False).to_dict())))

    if dataset_choice == "All (US+CA)":
        # keep the label/timestamp set earlier in that branch
        pass
    else:
        t = _mtime(NEW_DATA_PATH)
        dataset_label = dataset_label if 'dataset_label' in locals() else "US ETFs"
        dataset_ts = dt.datetime.fromtimestamp(t) if t else None

    # === Sanity: show rows loaded or the exact problem ===
    if etf_df is None or etf_df.empty:
        st.error(f"{dataset_label}: 0 rows loaded. Check file exists and sheet has data.\n"
                f"Path: {NEW_DATA_PATH}\nSheet: {NEW_DATA_SHEET}")
    else:
        st.success(f"{dataset_label}: {len(etf_df):,} rows loaded.")

    # Show the banner AFTER etf_df exists (no file name or path shown)
    st.caption(f"Data as of: {_asof_label(len(etf_df), label=dataset_label, ts=dataset_ts)}")
    st.sidebar.caption(f"Data as of: {_asof_label(len(etf_df), label=dataset_label, ts=dataset_ts)}")
    with st.expander("🔎 Debug — what did we actually load?", expanded=False):
        st.write({
            "dataset_choice": dataset_choice,
            "country_session": st.session_state.get("country"),
            "account_type": st.session_state.get("use_context"),
            "rows_loaded": len(etf_df),
            "first_8_cols": list(etf_df.columns)[:8]
        })
        st.caption(f"US path: {US_DATA_PATH}")
        st.caption(f"CA path: {CA_DATA_PATH}")
        try:
            st.write("Listing Country counts:", etf_df["Listing Country"].value_counts(dropna=False).to_dict())
        except Exception:
            st.write("No 'Listing Country' column yet.")
        if st.button("Clear data cache and reload"):
            st.cache_data.clear()
            st.rerun()

        # ✅ Not an expander — just a toggle to reveal the sample
        show_sample = st.checkbox("Show quick country sanity sample", value=False, key="dbg_country_sample")
        if show_sample:
            try:
                n = min(15, len(etf_df))
                if n > 0:
                    st.write(etf_df.sample(n)[["Symbol","ETF Name","Listing Country"]])
                else:
                    st.caption("No data loaded yet.")
            except Exception as e:
                st.caption(f"(Could not sample: {e})")


    try:
        mix = etf_df["Listing Country"].value_counts().to_dict()
        st.sidebar.caption("Loaded rows: " + f"{len(etf_df):,}" + " • Country mix: " + ", ".join(f"{k}:{v}" for k,v in mix.items()))
    except Exception:
        pass


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

    relax_builder = st.checkbox(
        "If a sleeve is empty, relax risk (keep country/account rules)",
        value=True
    )

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
                if ranked.empty and relax_builder:
                    # Rebuild base set = same asset class, with country/account rules applied
                    base = etf_df[etf_df["Simplified Asset Class"].str.lower() == ASSET_MAP[ac]].copy()

                    # Apply country/account policy (same as tabs)
                    policy = get_country_policy(
                        st.session_state.get("country", ""),
                        st.session_state.get("use_context", ""),
                        ac
                    )
                    if policy["hard_include"]:
                        base = base[base["Listing Country"].isin(policy["hard_include"])]
                    if policy["hard_exclude"]:
                        base = base[~base["Listing Country"].isin(policy["hard_exclude"])]

                    # Apply the same context rules (avoid_us_dividends, etc.)
                    rules = st.session_state.get("use_context_rules") or {}
                    if rules.get("avoid_us_dividends"):
                        base = base[base["Listing Country"].ne("USA")]
                    if rules.get("avoid_dividends"):
                        base = base[pd.to_numeric(base["Annual Dividend Yield %"].str.replace("%",""), errors="coerce") < 2]
                    if rules.get("favor_tax_efficiency"):
                        base = base[
                            (pd.to_numeric(base["Annual Dividend Yield %"].str.replace("%",""), errors="coerce") < 2.5) &
                            (pd.to_numeric(base["ER"].str.replace("%",""), errors="coerce") < 0.3)
                        ]
                    if rules.get("favor_growth"):
                        base = base[pd.to_numeric(base["1 Year"].str.replace("%",""), errors="coerce") > 5]
                    if rules.get("favor_low_fee"):
                        base = base[pd.to_numeric(base["ER"].str.replace("%",""), errors="coerce") < 0.25]

                    # Keep equity goal filter; relax only risk
                    if ac == "Equity":
                        base = safe_goal_filter(base, goal)

                    ranked = rank_etfs(base, goal).sort_values("Final Score", ascending=False)
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

    for asset_class, tab in tab_map.items():
        with tab:
            st.markdown(f"### {asset_class} ETF Recommendations")
            class_key = ASSET_MAP[asset_class]

            # If data isn't loaded, show a friendly note and skip this tab render
            if etf_df.empty:
                st.info("ETF data not loaded yet — pick dataset/country in the sidebar.")
                continue

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

            # Save a copy after country/account policy + context rules (before risk/goal filters)
            base_after_rules = filtered.copy()

            # ---- one risk/goal filter pass (STRICT) ----
            filtered_final = base_after_rules.copy()

            if asset_class != "Other":
                allowed = set(risk_filters[risk])
                if asset_class in ["Bonds", "Cash"]:
                    allowed.update({"Low", "Medium"})  # keep safer choices in FI/cash
                filtered_final = filtered_final[filtered_final["Risk Level"].isin(allowed)]

            # Goal thresholds apply to equities only
            if asset_class == "Equity":
                filtered_final = safe_goal_filter(filtered_final, goal)

            # Count AFTER strict filters (this is what we try to show)
            st.caption(f"{len(filtered_final)} ETFs match the current filters.")

            # Rank strict result
            ranked_df = rank_etfs(filtered_final, goal).sort_values(by="Final Score", ascending=False)

            # ---- Fallback: RELAX **RISK ONLY** (keep country/account rules & equity goal) ----
            if ranked_df.empty:
                relaxed = base_after_rules.copy()  # country/account rules still applied
                if asset_class == "Equity":
                    relaxed = safe_goal_filter(relaxed, goal)  # keep the equity goal filter
                if not relaxed.empty:
                    ranked_df = (
                        rank_etfs(relaxed, goal)
                        .sort_values(by="Final Score", ascending=False)
                        .head(12)
                    )
                    st.warning(
                        "No ETFs matched after the **risk** filter. "
                        "Showing top options with risk relaxed (country/account rules still applied)."
                    )



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
                                    "1Y":   weights_here["1Y"]    * row.get("1Y_score", 0),
                                    "ER":   weights_here["ER"]    * row.get("ER_score", 0),
                                    "AUM":  weights_here["AUM"]   * row.get("AUM_score", 0),
                                    "Yield":weights_here["Yield"] * row.get("Yield_score", 0),
                                    "Tax":  weights_here["TaxEff"]* row.get("TaxEff_score", 0),
                                }
                                top2 = sorted(contrib.items(), key=lambda kv: kv[1], reverse=True)[:2]
                                why = " + ".join([f"{k}↑" for k,_ in top2]) if top2 else "—"

                                # --- TAX FLAG / NOTE (comes from add_account_tax_scores) ---
                                tax_flag = str(row.get("TaxEff_flag", "") or "").strip()   # e.g., "TFSA-inefficient", "RRSP-friendly"
                                tax_note = str(row.get("TaxEff_note", "") or "").strip()   # short human text
                                tax_badge = f"<span style='background:#fff3cd; color:#8a6d3b; padding:2px 6px; border-radius:999px; font-size:12px; border:1px solid #f7e49c; margin-left:6px;'>{tax_flag}</span>" if tax_flag else ""

                                def _fmt_pct(x):
                                    try: return f"{float(x):.2f}%"
                                    except: return "—"
                                def _fmt_bil(x):
                                    try: return f"${float(x):.2f}B"
                                    except: return "—"

                                st.markdown(f"""
                                <div style='background:#eef9f2; padding:15px; border-radius:10px; border:1px solid #b6e5c5; margin-bottom:15px;'>
                                    <b><a href='https://finance.yahoo.com/quote/{row['Symbol']}' target='_blank'>{row['Symbol']}: {row['ETF Name']}</a></b>
                                    {tax_badge}<br>
                                    <b>1Y Return:</b> {_fmt_pct(one_year)} &nbsp; <b>Expense Ratio:</b> {_fmt_pct(er)} &nbsp; <b>Yield:</b> {_fmt_pct(yld)}<br>
                                    <b>AUM:</b> {_fmt_bil(aum_bil)} &nbsp; <b>Risk Level:</b> {row['Risk Level']}<br>
                                    <b>Score:</b> {row['Final Score']:.2f} &nbsp; <b>Why:</b> {why}
                                    {"<br><b>Tax note:</b> " + tax_note if tax_note else ""}
                                </div>
                                """, unsafe_allow_html=True)

                    st.markdown("**Full Tier 1 Comparison Table**")
                    cols = ["Symbol", "ETF Name", "1 Year", "ER", "Annual Dividend Yield %", "Total Assets", "Risk Level"]
                    # Append Tax note/flag if present
                    if "TaxEff_note" in tier_1.columns:
                        cols.append("TaxEff_note")
                    if "TaxEff_flag" in tier_1.columns and "TaxEff_flag" not in cols:
                        cols.append("TaxEff_flag")

                    table_1 = tier_1[ [c for c in cols if c in tier_1.columns] ].rename(columns={
                        "1 Year": "1Y Return",
                        "ER": "Expense Ratio",
                        "Annual Dividend Yield %": "Yield",
                        "Total Assets": "AUM",
                        "TaxEff_note": "Tax note",
                        "TaxEff_flag": "Tax flag"
                    })
                    st.dataframe(table_1, use_container_width=True)

                    st.caption("ℹ️ Tax badge & note are based on your Country + Account Type (e.g., TFSA vs RRSP, Roth vs Taxable) and the ETF’s listing & yield profile.")


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

    # one-click reset for screener filters
    if st.button("Reset screener filters", key="scr_reset"):
        st.session_state["scr_asset_class"] = "All"
        st.session_state["scr_risk"] = "All"
        st.session_state["scr_keyword"] = ""
        st.session_state["scr_exlev"] = True
        st.success("Screener filters reset.")

    # start from master df (MAKE THIS FIRST!)
    screener_df = etf_df.copy()
    raw_count = len(screener_df)

    # Ensure Listing Country exists
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

    # Single checkbox to apply country filter (uses the sidebar country)
    apply_country_filter = st.checkbox(
        "Filter by listing country (from sidebar)",
        value=(dataset_choice != "All (US+CA)"),
        key="scr_country_filter"
    )


    # Use the sidebar 'country' value ("USA" / "Canada" / "")
    country_current = st.session_state.get("country", "")
    if apply_country_filter and country_current in ("USA", "Canada"):
        screener_df = screener_df[screener_df["Listing Country"] == country_current]
        st.caption(f"Active country filter: {country_current}")
    else:
        st.caption("Active country filter: All")


    after_country_count = len(screener_df)


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
    ex_lev = st.checkbox(
        "Exclude leveraged/inverse/ETNs/option overlays",
        value=True,
        key="scr_exlev"
    )
    if ex_lev:
        bad = [
            "3x","2x","-3x","-2x",
            "2xbull","3xbull","bear -1x",
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

    # choose columns and download results
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

    # Row counts summary
    final_count = len(screener_shown)
    st.caption(f"Rows: raw {raw_count:,} → after country {after_country_count:,} → after other filters {final_count:,}")

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
                    if "AUM_bil" in merged.columns:
                        aum_b = pd.to_numeric(merged["AUM_bil"], errors="coerce").mean()
                    else:
                        aum_b = pd.to_numeric(
                            merged["Total Assets"].astype(str)
                                .str.replace("$","", regex=False)
                                .str.replace(",","", regex=False)
                                .str.replace("B","", regex=False),
                            errors="coerce"
                        ).mean()
                    aum_b = 0 if pd.isna(aum_b) else aum_b
                    st.metric("Avg. AUM (B)", f"{aum_b:.2f}B")


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

                    # If a model portfolio exists, use its per-ETF target weights
                    model_df = st.session_state.get("model_df")
                    if model_df is not None and not model_df.empty:
                        per_etf_target = (model_df.groupby("Symbol")["Weight %"].sum()
                                        .rename("Target Weight (%)"))
                        current_df = current_df.merge(per_etf_target, left_on="ETF", right_index=True, how="left")

                    # Fallback: split the sleeve target equally across the ETFs you currently hold
                    missing = current_df["Target Weight (%)"].isna()
                    if missing.any():
                        class_target = current_df.loc[missing, "Asset Class"].str.lower().map(normalized_model)  # normalized_model already defined above
                        counts = current_df.groupby("Asset Class")["ETF"].transform("count").clip(lower=1)
                        current_df.loc[missing, "Target Weight (%)"] = class_target / counts

                    # Generate actions with correct ETF targets
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

# ---- Advisor Notes Assistant Tab ----
with tab6:
    st.subheader("📝 Advisor Notes Assistant")
    country = st.session_state.get("country", "")
    acct = st.session_state.get("use_context", "")

    # Safety check
    if 'etf_df' not in globals() or etf_df.empty:
        st.warning("ETF data is not loaded yet. Please load data first.")
        st.stop()

    country = st.session_state.get("country", "")
    acct    = st.session_state.get("use_context", "")



    # Let the user pick which sleeve to analyze
    notes_asset_choice = st.selectbox(
        "Analyze notes for:",
        ["All", "Equity", "Bonds", "Cash", "Mixed", "Other"],
        index=0
    )

    # How many top candidates to scan
    top_n = st.slider("How many top candidates per sleeve to scan", 3, 20, 8, step=1)

    # Helper to safely build a ranked set for a sleeve without relying on outside variables
    def _rank_for(ac_label: str) -> pd.DataFrame:
        return get_ranked_for_class(
            asset_class=ac_label,
            goal=goal,
            risk=risk,
            country=st.session_state.get("country", ""),
            account_type=st.session_state.get("use_context", ""),
            etf_df=etf_df,
            risk_filters=risk_filters
        ).head(top_n)

    # Build the DataFrame to annotate
    if notes_asset_choice == "All":
        frames = []
        for ac in ["Equity", "Bonds", "Cash", "Mixed"]:
            try:
                df_part = _rank_for(ac)
                if not df_part.empty:
                    frames.append(df_part)
            except Exception:
                pass
        candidates_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    else:
        candidates_df = _rank_for(notes_asset_choice)

    if candidates_df.empty:
        st.info("No candidates to analyze for notes based on your current filters.")
        st.stop()

    # Make sure we have the numeric helpers used by your logic
    if "Yield_clean" not in candidates_df.columns or "ER_clean" not in candidates_df.columns:
        # Recompute minimal cleans just in case
        tmp = candidates_df.copy()
        tmp["Yield_clean"] = pd.to_numeric(
            tmp["Annual Dividend Yield %"].astype(str).str.replace("%",""), errors="coerce"
        )
        tmp["ER_clean"] = pd.to_numeric(
            tmp["ER"].astype(str).str.replace("%",""), errors="coerce"
        )
        # fix 0.05 => 5.0 style inputs
        if tmp["Yield_clean"].notna().any() and tmp["Yield_clean"].max() <= 1.5:
            tmp["Yield_clean"] = tmp["Yield_clean"] * 100
        if tmp["ER_clean"].notna().any() and tmp["ER_clean"].max() <= 1.5:
            tmp["ER_clean"] = tmp["ER_clean"] * 100
        candidates_df["Yield_clean"] = tmp["Yield_clean"]
        candidates_df["ER_clean"] = tmp["ER_clean"]


    notes = []
    for _, r in candidates_df.iterrows():
        sym = str(r.get("Symbol", "—"))
        ac  = str(r.get("Simplified Asset Class", "")).lower()
        lst = str(r.get("Listing Country", "") or "Unknown")
        yld = float(r.get("Yield_clean", 0) or 0.0)
        er  = float(r.get("ER_clean", 0) or 0.0)

        this_notes = []

        # Country/account-aware tax hints (simple, readable)
        if country == "Canada":
            if acct in {"TFSA", "RESP"}:
                if lst == "USA" and yld >= 1.0:
                    this_notes.append("US withholding may make dividends tax-inefficient in TFSA/RESP.")
            elif acct == "RRSP":
                if ac == "equity" and lst == "USA":
                    this_notes.append("US-listed equity dividends are generally treaty-exempt in RRSP.")
            elif acct == "Non-Registered":
                if ac == "bond":
                    this_notes.append("Bond interest is fully taxable in a non-registered account.")
                if ac == "equity" and yld >= 3.0:
                    this_notes.append("High dividend yield can increase annual tax drag in taxable.")
                if ac == "equity" and lst == "Canada":
                    this_notes.append("Canadian-listed equity may benefit from eligible dividend treatment.")
        elif country == "USA":
            if acct in {"Taxable"}:
                if ac == "bond":
                    this_notes.append("Bond interest is taxed at ordinary rates in taxable.")
                if yld >= 3.0:
                    this_notes.append("High dividend yield can increase annual tax drag in taxable.")
            elif acct in {"Roth IRA", "Traditional IRA", "401(k)"}:
                if ac == "equity" and yld <= 1.5:
                    this_notes.append("Low-yield/growth-tilted equity fits well in tax-advantaged accounts.")

        # General, non-tax hygiene notes
        if er > 0.50:
            this_notes.append("Expense ratio is relatively high.")
        if yld >= 6.0 and ac in {"equity", "mixed"}:
            this_notes.append("Very high yield — check if it’s an option-overlay or riskier strategy.")
        if lst == "Unknown":
            this_notes.append("Listing country unknown — double-check ticker/source.")

        # If we’ve computed Tax Flag/Note in your pipeline, surface it
        tflag = str(r.get("Tax Flag", "") or "").strip()
        tnote = str(r.get("TaxEff_note", "") or "").strip()
        if tflag:
            this_notes.append(f"Tax flag: {tflag}.")
        if tnote:
            this_notes.append(f"Tax note: {tnote}.")

        # Fall-back if nothing triggered
        if not this_notes:
            this_notes.append("No major issues flagged.")

        notes.append(f"- **{sym}** — " + " ".join(this_notes))

    st.markdown("### Suggested Advisor Notes")
    st.markdown("\n".join(notes))

    # Optional: show the table we annotated
    with st.expander("Show analyzed candidates table"):
        cols_show = ["Symbol", "ETF Name", "Listing Country", "Simplified Asset Class", "ER", "Annual Dividend Yield %", "Final Score"]
        cols_show = [c for c in cols_show if c in candidates_df.columns]
        st.dataframe(candidates_df[cols_show], use_container_width=True)
