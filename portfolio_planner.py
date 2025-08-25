import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import datetime as dt, pathlib  # A1 banner
import re                       # A4 screener toggle
from typing import Optional

global etf_df, NEW_DATA_PATH, NEW_DATA_SHEET

# ——— SAFETY: make sure the name exists even before loading ———
etf_df = pd.DataFrame()

def _has_etf_data():
    return 'etf_df' in globals() and isinstance(etf_df, pd.DataFrame) and not etf_df.empty

# --- Account Types allowed per Builder View and Country ---
ACCOUNT_TYPES = {
    "Wealth Management": {
        "Canada": ["", "RRSP", "TFSA", "Non-Registered", "RESP"],
        "USA":    ["", "Roth IRA", "401(k)", "Traditional IRA", "Taxable"],
        "":       [""]
    },
    "Asset Management": {
        "Canada": ["", "Pension Plan", "Endowment/Foundation", "Institutional Taxable", "Pooled Fund", "SMA/Model", "Insurance General Account"],
        "USA":    ["", "Pension Plan", "Endowment/Foundation", "Institutional Taxable", "SMA/Model", "Insurance General Account"],
        "":       [""]
    }
}

# ==== DATA SOURCE SETTINGS ====
USE_NEW_DATA = True  # keep using Excel files

from pathlib import Path  # add this import if not already present

# Resolve the Excel files based on where THIS .py file lives
try:
    BASE = Path(__file__).parent
except NameError:
    BASE = Path.cwd()

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

# ---- Risk Classification Function ----
def classify_risk(row):
    """
    Uses precomputed numeric columns to avoid double-parsing and % scale mistakes.
    Expected units:
      - 1Y_num, ER_num, Yield_num are in percent scale (e.g., 8.5 means 8.5%)
      - AUM_bil is in billions of dollars (float)
    """
    # Pull already-clean numbers created earlier in your pipeline
    ret   = row.get("1Y_num", None)      # 1-year return, % (e.g., 8.5)
    er    = row.get("ER_num", None)      # expense ratio, % (e.g., 0.20)
    yld   = row.get("Yield_num", None)   # dividend yield, % (e.g., 2.1)
    aum_b = row.get("AUM_bil", None)     # AUM in billions (e.g., 15.3)

    # Coerce to floats; missing values become None
    try: ret = float(ret)
    except: ret = None
    try: er = float(er)
    except: er = None
    try: yld = float(yld)
    except: yld = None
    try: aum_b = float(aum_b)
    except: aum_b = None

    score = 0

    # 1Y returns: very low can be defensive (nudge down risk); very high can be riskier (nudge up)
    if ret is not None:
        if ret > 12:
            score += 2
        elif ret < 4:
            score -= 2

    # Fees: high ER often means niche/complex exposures -> nudge risk up; very low ER -> nudge down
    if er is not None:
        if er > 0.60:
            score += 2
        elif er < 0.10:
            score -= 2

    # Yield: extremely low or extremely high yield can imply more risk (growth or overlays/credit)
    if yld is not None:
        if yld < 1.0:
            score += 1
        elif yld > 2.5:
            score += 1

    # AUM: very small funds carry liquidity/closure risk; very large are a little safer
    if aum_b is not None:
        if aum_b < 1.0:      # < $1B
            score += 2
        elif aum_b > 10.0:   # > $10B
            score -= 1

    # Treat option-income products as a bit riskier
    try:
        if bool(row.get("is_option_income", False)):
            score += 2
    except Exception:
        pass

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

# --- Eligibility (fit-to-present) rules by Institution → Account Type ---
def apply_eligibility(df: pd.DataFrame, builder_view: str, country: str, account_type: str, asset_class: str) -> pd.DataFrame:
    """
    Drops rows that are NOT presentable for the chosen Institution + Account Type.
    Uses the same field names your loader already prepares (AUM_bil, ER_num, Yield_num, Age_months, Listing Country).
    All thresholds below mirror the Master Playbook we agreed on.
    """
    if df.empty:
        return df

    view = (builder_view or "Wealth Management").strip()
    acct = (account_type or "").strip()
    ac   = (asset_class or "").strip()

    out = df.copy()

    # ---- Common safety screens (eligibility level) ----
    # Age: require at least 12 months history unless explicitly loosened later
    if "Age_months" in out.columns:
        out = out[pd.to_numeric(out["Age_months"], errors="coerce").fillna(0) >= 12]

    # Normalize numeric helpers (already created by load_etf_data, but be defensive)
    if "AUM_bil" not in out.columns:
        out["AUM_bil"] = pd.NA
    if "ER_num" not in out.columns:
        out["ER_num"] = pd.to_numeric(out["ER"].astype(str).str.replace("%","", regex=False), errors="coerce")
        if out["ER_num"].notna().any() and out["ER_num"].max() <= 1.5:
            out["ER_num"] *= 100
    if "Yield_num" not in out.columns:
        out["Yield_num"] = pd.to_numeric(out["Annual Dividend Yield %"].astype(str).str.replace("%","", regex=False), errors="coerce")
        if out["Yield_num"].notna().any() and out["Yield_num"].max() <= 1.5:
            out["Yield_num"] *= 100

    # Leverage/inverse are handled later too, but exclude here at eligibility unless toggled
    if not st.session_state.get("allow_leveraged", False):
        for col in ["is_leveraged", "is_inverse"]:
            if col in out.columns:
                out = out[~out[col]]

    # ---- WEALTH MANAGEMENT ----
    if view == "Wealth Management":
        # TFSA / RESP (Equity focus)
        if acct in {"TFSA", "RESP"}:
            # Country constraint: CAD-listed for equity to avoid unrecoverable US WHT
            if ac == "Equity" and "Listing Country" in out.columns:
                out = out[out["Listing Country"].eq("Canada")]
            # Basic quality bars
            out = out[pd.to_numeric(out["AUM_bil"], errors="coerce").fillna(0) > 0.10]   # > $100M
            out = out[pd.to_numeric(out["ER_num"],  errors="coerce").fillna(9e9) < 0.75]  # < 0.75%

        # RRSP
        elif acct == "RRSP":
            out = out[pd.to_numeric(out["AUM_bil"], errors="coerce").fillna(0) > 0.25]   # > $250M
            out = out[pd.to_numeric(out["ER_num"],  errors="coerce").fillna(9e9) < 0.50]  # < 0.50%

        # Non-Registered (Taxable)
        elif acct == "Non-Registered":
            out = out[pd.to_numeric(out["AUM_bil"], errors="coerce").fillna(0) > 0.10]   # > $100M
            out = out[pd.to_numeric(out["ER_num"],  errors="coerce").fillna(9e9) < 0.75]  # < 0.75%
            out = out[pd.to_numeric(out["Yield_num"], errors="coerce").fillna(9e9) < 3.0] # Yield < 3% to limit tax drag

        else:
            # If no account selected yet, keep it permissive (only common screens applied)
            pass

    # ---- ASSET MANAGEMENT ----
    else:
        # Pension / Endowment
        if acct in {"Pension Plan", "Endowment/Foundation", "Institutional Taxable"}:
            out = out[pd.to_numeric(out["AUM_bil"], errors="coerce").fillna(0) > 1.00]   # > $1B
            if "Avg Dollar Volume" in out.columns:
                out = out[pd.to_numeric(out["Avg Dollar Volume"], errors="coerce").fillna(0) > 5_000_000]  # > $5M/day
            out = out[pd.to_numeric(out["Age_months"], errors="coerce").fillna(0) > 36]  # > 36 months

        # Insurance General Account (primarily bonds)
        elif acct == "Insurance General Account":
            out = out[pd.to_numeric(out["AUM_bil"], errors="coerce").fillna(0) > 0.50]    # > $500M
            # Optional duration-fit if you add a column later:
            # If you create "Effective Duration" and a user input "Liability Duration",
            # you can enforce |ETF - Liability| < 1 here. For now, we skip if missing.

        # SMA / Model
        elif acct in {"SMA/Model", "Pooled Fund"}:
            out = out[pd.to_numeric(out["AUM_bil"], errors="coerce").fillna(0) > 0.25]   # > $250M
            out = out[pd.to_numeric(out["ER_num"],  errors="coerce").fillna(9e9) < 0.25]  # < 0.25% (index-pure bias)

        else:
            pass

    return out

# --- Human-readable eligibility summary for the current context ---
def _eligibility_summary(view: str, country: str, account_type: str, asset_class: str) -> str:
    view = (view or "Wealth Management").strip()
    acct = (account_type or "").strip()
    ac   = (asset_class or "").strip()
    lines = [f"**View:** {view}  •  **Country:** {country or '—'}  •  **Account:** {acct or '—'}  •  **Asset Class:** {ac or '—'}"]

    # Common safety
    lines.append("Age ≥ 12 months • Leveraged/Inverse excluded (unless explicitly enabled)")

    if view == "Wealth Management":
        if acct in {"TFSA", "RESP"}:
            if ac == "Equity":
                lines.append("CAD-listed only (avoid US WHT on equity dividends)")
            lines.append("AUM ≥ $100M  •  ER < 0.75%")
        elif acct == "RRSP":
            lines.append("AUM ≥ $250M  •  ER < 0.50%  •  US/CA listings allowed (treaty)")
        elif acct in {"Non-Registered", "Taxable"}:
            lines.append("Dividend Yield < 3%  •  AUM ≥ $100M  •  ER < 0.75%")
        else:
            lines.append("Default wealth guardrails apply")
    else:
        if acct in {"Pension Plan", "Endowment/Foundation", "Institutional Taxable"}:
            lines.append("AUM ≥ $1B  •  ADV > $5M/day (if available)  •  Age ≥ 36 months")
        elif acct == "Insurance General Account":
            lines.append("AUM ≥ $500M  •  (Optional) Duration fit within ±1 year")
        elif acct in {"SMA/Model", "Pooled Fund"}:
            lines.append("AUM ≥ $250M  •  ER < 0.25%  •  One ETF per index sleeve")
        else:
            lines.append("Default institutional guardrails apply")

    return " \n- ".join(["**Eligibility applied**:"] + lines)


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
    import tempfile, shutil, pandas as pd, os
    try:
        return pd.read_excel(path, sheet_name=sheet)
    except PermissionError:
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
            shutil.copyfile(path, tmp.name)
            tmp_path = tmp.name
        try:
            return pd.read_excel(tmp_path, sheet_name=sheet)
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass


# ---- Load ETF Data ----

# --- AUM parser: convert 'Total Assets' to billions (float) ---
def _aum_to_billions(x):
    """
    Returns AUM in **billions**.
    Accepts '$14.3B', '650M', '14,300,000,000', '14.3 bn', '900000000', '14.3' (assumed billions), etc.
    """
    try:
        s = str(x).strip()
        if not s or s.lower() in {"nan", "none", "—", "-"}:
            return np.nan

        # strip currency and separators
        s = (s.replace("US$", "").replace("C$", "").replace("$", "")
               .replace(",", "").replace("USD", "").replace("CAD", "").strip())
        sl = s.lower()

        # first number in the string
        m = re.search(r'[-+]?\d*\.?\d+(?:[eE][+-]?\d+)?', sl)
        if not m:
            return np.nan
        num = float(m.group(0))

        # unit right after the number (if any)
        unit = sl[m.end():].strip()

        # explicit units
        if unit.startswith(("tn", "trillion", "t")):   # e.g., 1.2T
            return num * 1000.0
        if unit.startswith(("bn", "billion", "b")):    # e.g., 14.3B, 14.3 bn
            return num
        if unit.startswith(("mn", "million", "m")):    # e.g., 650M
            return num / 1000.0
        if unit.startswith(("k", "thousand")):         # e.g., 900k
            return num / 1_000_000.0

        # no unit -> treat as dollars or already-billions
        if num >= 1e9:   # looks like raw dollars
            return num / 1e9
        if num >= 1e6:   # still raw dollars but smaller
            return num / 1e9

        # small number with no unit: assume it's already in billions (e.g., '14.3')
        return num
    except Exception:
        return np.nan


@st.cache_data
def load_etf_data(
    use_new=USE_NEW_DATA,
    _excel_mtime=None, _csv_mtime=None,
    _path=None, _sheet=None, _cache_key=None,
    force_country: Optional[str] = None,   # NEW
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
        "Dollar Vol.": "Avg Dollar Volume",
        "Avg. Volume": "Average Volume",       # optional, for info tables
        "Inception": "Inception Date",
        "Country": "Listing Country",          # present in US file; CA file lacks it (that’s OK)
    }
    rename_actual = {k: v for k, v in rename_map_new_to_legacy.items() if k in df.columns}
    if rename_actual:
        df = df.rename(columns=rename_actual)

    # --- 3) Make sure the key columns exist (so the rest of your app works)
    required_cols = ["Symbol", "ETF Name", "1 Year", "ER", "Annual Dividend Yield %", "Total Assets"]
    for col in required_cols:
        if col not in df.columns:
            df[col] = None  # create empty column if missing

    
    # --- Derive Issuer from ETF Name (simple, name-based) ---
    # --- Derive Issuer from ETF Name (canonical labels) ---
    ISSUER_CANON = {
        "vanguard": "Vanguard",
        "blackrock": "iShares",
        "ishares": "iShares",
        "ishares canada": "iShares",
        "state street": "SPDR",
        "spdr": "SPDR",
        "charles schwab": "Schwab",
        "schwab": "Schwab",
        "invesco": "Invesco",
        "bmo": "BMO",
        "horizon": "Horizons",
        "horizons": "Horizons",
        "td": "TD",
        "xtrackers": "Xtrackers",
        "first trust": "First Trust",
        "global x": "Global X",
        "jp morgan": "JPMorgan",
        "jpmorgan": "JPMorgan",
    }

    def _issuer_from_name(name):
        s = str(name).lower()
        for k, v in ISSUER_CANON.items():
            if k in s:
                return v
        return "Other"

    if "Issuer" in df.columns:
        df["Issuer"] = df["Issuer"].fillna(df["ETF Name"].apply(_issuer_from_name))
    else:
        df["Issuer"] = df["ETF Name"].apply(_issuer_from_name)


    


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
        lambda row: "mixed" if row["is_potential_mixed"] and
        (pd.isna(row["Simplified Asset Class"]) or str(row["Simplified Asset Class"]).strip().lower() in ["", "other"])
        else row["Simplified Asset Class"],
        axis=1
    )
    df["Simplified Asset Class"] = df["Simplified Asset Class"].astype(str).str.lower()

    # --- Hedged / Unhedged detection (name + tags)
    def _hedged_status(name, tags):
        s = (str(name) + " " + str(tags)).lower()
        if "unhedged" in s or "no hedge" in s:
            return "Unhedged"
        if "hedged" in s or "currency hedged" in s or "cad-hedged" in s or "cad hedged" in s or "usd-hedged" in s:
            return "Hedged"
        return "Unknown"

    df["Hedged Status"] = df.apply(lambda r: _hedged_status(r.get("ETF Name",""), r.get("Tags","")), axis=1)


    # --- Numeric helper columns used across the app ---
    def _pct_series(col, frac_cutoff=1.5):
        """
        Converts a column that may contain values like '2.3%', '2.3', or 0.023
        into **percent units** (e.g., 2.3 -> 2.3% and 0.023 -> 2.3%).
        Works per-value (not whole-column guess).
        """
        s = pd.Series(col).astype(str)
        out = []
        for x in s:
            raw = x.strip()
            if not raw or raw.lower() in {"nan", "none"}:
                out.append(np.nan); continue
            if "%" in raw:
                out.append(pd.to_numeric(raw.replace("%","").replace(",",""), errors="coerce")); continue
            v = pd.to_numeric(raw.replace(",",""), errors="coerce")
            # Treat small numbers as fractions-of-1 and scale to %
            out.append(v * 100 if pd.notna(v) and abs(v) <= frac_cutoff else v)
        return pd.to_numeric(out, errors="coerce")

    def _er_series(col):
        """
        Converts expense ratios into **percent units**.
        Accepts '0.03%', '0.03', or 0.0003 -> becomes 0.03%.
        Uses a conservative 5% guard for decimals.
        """
        s = pd.Series(col).astype(str)
        out = []
        for x in s:
            raw = x.strip()
            if not raw or raw.lower() in {"nan","none"}:
                out.append(np.nan); continue
            if "%" in raw:
                out.append(pd.to_numeric(raw.replace("%","").replace(",",""), errors="coerce")); continue
            v = pd.to_numeric(raw.replace(",",""), errors="coerce")
            # Most ERs are decimals (e.g., 0.003 = 0.3%)
            out.append(v * 100 if pd.notna(v) and v <= 0.05 else v)  # 5% upper guard
        return pd.to_numeric(out, errors="coerce")

    df["1Y_num"]    = _pct_series(df["1 Year"], frac_cutoff=3.0)        # returns can be large
    df["ER_num"]    = _er_series(df["ER"])
    df["Yield_num"] = _pct_series(df["Annual Dividend Yield %"], frac_cutoff=1.5)
    df["AUM_bil"]   = df["Total Assets"].apply(_aum_to_billions)

    # --- Risk proxies from existing columns (no API calls) ---
    def _risk_proxies(df_in: pd.DataFrame) -> pd.DataFrame:
        """
        Builds simple risk proxies using columns that commonly exist in your curated files.
        - Volatility proxy: |Beta (5Y)|  (higher beta ≈ higher volatility)
        - Drawdown proxy:   worst of [52W Low Chg, ATL Chg (%), ATH Chg (%)] (more negative = worse)
        - Sharpe proxy:     1Y return divided by volatility proxy (risk-adjusted return)
        All proxies fall back to NaN and will be scored neutrally if missing.
        """
        df_out = df_in.copy()

        # Volatility proxy (abs to avoid signs)
        if "Beta (5Y)" in df_out.columns:
            df_out["Risk_Vol_proxy"] = pd.to_numeric(df_out["Beta (5Y)"], errors="coerce").abs()
        else:
            df_out["Risk_Vol_proxy"] = np.nan

        # Drawdown proxy: most negative seasonal/ATH/ATL change we can find
        dd_cols = []
        for col in ["52W Low Chg", "ATL Chg (%)", "ATH Chg (%)"]:
            if col in df_out.columns:
                dd_cols.append(pd.to_numeric(df_out[col], errors="coerce"))
        if dd_cols:
            # Take the worst (most negative) across available columns
            df_out["Risk_Drawdown_proxy"] = pd.concat(dd_cols, axis=1).min(axis=1)
        else:
            df_out["Risk_Drawdown_proxy"] = np.nan

        # Sharpe proxy: 1Y % divided by volatility proxy (avoid divide-by-zero)
        r1y = pd.to_numeric(df_out.get("1Y_num", np.nan), errors="coerce")
        vol = pd.to_numeric(df_out.get("Risk_Vol_proxy", np.nan), errors="coerce").replace(0, np.nan)
        df_out["Risk_Sharpe_proxy"] = r1y / vol

        return df_out

    # Build risk proxies now so they’re available downstream
    df = _risk_proxies(df)


    # Keep a raw copy for country inference (preserve TSX:, TSXV:, NEO:, etc.)
    df["_SymbolRaw"] = df["Symbol"].astype(str)

    # === NEW: product-safety flags (from name/tags) ===
    _name = df["ETF Name"].astype(str)
    _tags = (df["Tags"] if "Tags" in df.columns else pd.Series("", index=df.index)).astype(str)
    name_tags = (_name + " " + _tags).str.lower()
    df["is_option_income"] = name_tags.str.contains(r"(?:option income|yieldmax)", regex=True, na=False)

    df["is_leveraged"] = name_tags.str.contains(r"\b(?:2x|3x|ultra|leveraged|geared)\b", regex=True, na=False)

    df["is_inverse"]   = name_tags.str.contains(r"\b(?:-1x|inverse|short|bear)\b", regex=True, na=False)


    # Common hedged markers (TSX funds often say "Hedged" or "CAD Hedged")
    df["is_hedged"] = name_tags.str.contains(r"\b(?:hedged|currency hedged|currency-hedged|cad hedged|cad-hedged)\b", regex=True, na=False)


    # Optional: flag covered-call style (can be tax-inefficient in some accounts)
    df["is_covered_call"] = name_tags.str.contains(r"covered[- ]?call|buy[- ]?write|enhanced income", regex=True)



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
    # Use non-capturing groups to avoid the pandas "match groups" warning.
    cand_pat = re.compile(r'(?:\.(?:TO|TSX|TSXV|NE|NEO|CN)|-(?:NE|CN))$', re.I)

    # Be explicit and resilient: treat NaNs as False (na=False)
    mask_can_suffix = df["_SymbolRaw"].astype(str).str.contains(cand_pat, na=False)

    df.loc[mask_can_suffix, "Listing Country"] = "Canada"

    # Inception date (if present) -> months since inception
    from datetime import datetime
    def _months_since(date_str):
        try:
            # try several common formats; fall back if parsing fails
            for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%d-%b-%Y", "%b %d, %Y"):
                try:
                    d = datetime.strptime(str(date_str).strip(), fmt)
                    break
                except:
                    d = None
            if d is None:
                d = pd.to_datetime(date_str, errors="coerce")
            if pd.isna(d):
                return None
            today = datetime.today()
            return (today.year - d.year) * 12 + (today.month - d.month)
        except:
            return None

    if "Inception Date" in df.columns:
        df["Age_months"] = df["Inception Date"].apply(_months_since)
    else:
        df["Age_months"] = None


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
    # 1) If the user is using the sliders, keep honoring them
    override = st.session_state.get("weights_override")
    if override:
        return override  # already normalized from the sliders

    view = st.session_state.get("builder_view", "Wealth Management")
    acct = st.session_state.get("account_type") or st.session_state.get("use_context", "")

    # ---------- Asset Management (institutional) ----------
    if view == "Asset Management":
        # Per-account fine-tuning
        if acct in {"Pension Plan", "Endowment/Foundation", "Institutional Taxable"}:
            return {"1Y": 0.10, "ER": 0.30, "AUM": 0.25, "Yield": 0.05, "TaxEff": 0.05, "ADV": 0.20, "Age": 0.05}
        if acct == "Insurance General Account":
            # DurationFit would be another factor if you add that column later.
            return {"1Y": 0.05, "ER": 0.30, "AUM": 0.20, "Yield": 0.05, "TaxEff": 0.05, "ADV": 0.15, "Age": 0.20}
        if acct in {"SMA/Model", "Pooled Fund"}:
            return {"1Y": 0.05, "ER": 0.35, "AUM": 0.20, "Yield": 0.05, "TaxEff": 0.05, "ADV": 0.20, "Age": 0.10}
        # Fallback institutional mix
        return {"1Y": 0.10, "ER": 0.30, "AUM": 0.25, "Yield": 0.10, "TaxEff": 0.05, "ADV": 0.15, "Age": 0.05}

    # ---------- Wealth Management (retail) ----------
    # Account-type specific weights
    if acct in {"TFSA", "RESP"}:
        # Tax efficiency dominates; penalize yield; ER still important
        return {"1Y": 0.10, "ER": 0.25, "AUM": 0.20, "Yield": 0.10, "TaxEff": 0.35}
    if acct == "RRSP":
        # Cost + scale + return; tax still matters, yield modest
        return {"1Y": 0.20, "ER": 0.30, "AUM": 0.25, "Yield": 0.10, "TaxEff": 0.15}
    if acct == "Non-Registered":
        # After-tax focus: penalize yield; cost matters
        return {"1Y": 0.10, "ER": 0.25, "AUM": 0.20, "Yield": 0.05, "TaxEff": 0.40}
    # If account not selected yet → use your existing goal-based defaults
    if goal == "Wealth Growth":
        return {"1Y": 0.25, "ER": 0.20, "AUM": 0.10, "Yield": 0.10, "TaxEff": 0.25, "Risk": 0.10}
    elif goal == "Retirement":
        return {"1Y": 0.20, "ER": 0.20, "AUM": 0.10, "Yield": 0.20, "TaxEff": 0.30}
    elif goal == "Income":
        return {"1Y": 0.10, "ER": 0.20, "AUM": 0.10, "Yield": 0.30, "TaxEff": 0.30}
    else:
        return {"1Y": 0.20, "ER": 0.20, "AUM": 0.10, "Yield": 0.20, "TaxEff": 0.30}

    



def safe_goal_filter(df, goal):
    # Prefer the precomputed percent-scale helpers if present
    if "Yield_num" in df.columns:
        y = pd.to_numeric(df["Yield_num"], errors="coerce")
    else:
        y = pd.to_numeric(
            df["Annual Dividend Yield %"].astype(str).str.replace("%", "", regex=False),
            errors="coerce"
        )
        if y.notna().any() and y.max() <= 1.5:  # values like 0.02 mean 2%
            y = y * 100

    if "1Y_num" in df.columns:
        r = pd.to_numeric(df["1Y_num"], errors="coerce")
    else:
        r = pd.to_numeric(
            df["1 Year"].astype(str).str.replace("%", "", regex=False),
            errors="coerce"
        )
        if r.notna().any() and r.max() <= 1.5:
            r = r * 100

    if goal == "Retirement":
        return df[y > 1.8]
    if goal == "Income":
        return df[y > 2.2]
    if goal == "Wealth Growth":
        return df[r > 6]
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
    # Normalize decimals like 0.05 → 5.0%
    if "ER_clean" in df.columns and df["ER_clean"].notna().any() and df["ER_clean"].max() <= 1.5:
        df["ER_clean"] = df["ER_clean"] * 100
    if df["Yield_clean"].notna().any() and df["Yield_clean"].max() <= 1.5:
        df["Yield_clean"] = df["Yield_clean"] * 100
    df["AUM_clean"] = (
        df["AUM_bil"]
        if "AUM_bil" in df.columns
        else pd.to_numeric(
            df["Total Assets"].astype(str)
            .str.replace("$","", regex=False)
            .str.replace("B","", regex=False)
            .str.replace(",", "", regex=False),
            errors="coerce"
        )
    )


    # Account-aware tax-efficiency (0..1), plus a short flag
    if st.session_state.get("use_account_tax", True):
        # Use the account-aware tax signal
        df = add_account_tax_scores(
            df,
            country=st.session_state.get("country", ""),
            account_type=st.session_state.get("account_type") or st.session_state.get("use_context", ""),
        )
        df["TaxEff_clean"] = df["TaxEff_account"]
    else:
        # Fall back to the old simple proxy (1 / (yield + ER))
        denom = (df["Yield_clean"].fillna(0) + df["ER_clean"].fillna(0)).replace(0, pd.NA)
        TaxEff = 1 / denom
        TaxEff = pd.to_numeric(TaxEff, errors="coerce")
        TaxEff.replace([float("inf"), float("-inf")], pd.NA, inplace=True)
        df["TaxEff_clean"] = TaxEff


        # Normalize core factors
    # Normalize core factors
    df["1Y_score"]  = normalize(df["1Y_clean"])
    df["ER_score"]  = 1 - normalize(df["ER_clean"])   # lower ER is better
    df["AUM_score"] = normalize(df["AUM_clean"])

    # Yield: direction depends on account type (tax drag vs. desired income)
    _yield_raw = normalize(df["Yield_clean"])
    acct    = st.session_state.get("use_context", "")
    country = st.session_state.get("country", "")

    # Accounts where HIGH yield generally hurts ranking:
    _penalize_yield = (
        (country == "Canada" and acct in {"TFSA", "RESP", "Non-Registered"}) or
        (country == "USA"    and acct in {"Taxable", "Roth IRA", "401(k)", "Traditional IRA"})
    )

    df["Yield_score"]  = (1 - _yield_raw) if _penalize_yield else _yield_raw
    df["TaxEff_score"] = normalize(df["TaxEff_clean"])


    # Optional: ADV and Age for Asset Mgmt view
    if "Avg Dollar Volume" in df.columns:
        df["ADV_clean"] = pd.to_numeric(df["Avg Dollar Volume"], errors="coerce")
        df["ADV_score"] = normalize(df["ADV_clean"])
    else:
        df["ADV_score"] = 0.5

    if "Age_months" in df.columns:
        df["Age_clean"] = pd.to_numeric(df["Age_months"], errors="coerce")
        df["Age_score"] = normalize(df["Age_clean"])
    else:
        df["Age_score"] = 0.5

    # Flexible scoring
    weights = get_factor_weights(goal)
    base = (
        weights.get("1Y", 0)     * df["1Y_score"] +
        weights.get("ER", 0)     * df["ER_score"] +
        weights.get("AUM", 0)    * df["AUM_score"] +
        weights.get("Yield", 0)  * df["Yield_score"] +
        weights.get("TaxEff", 0) * df["TaxEff_score"]
    )
    base += weights.get("ADV", 0) * df["ADV_score"]
    base += weights.get("Age", 0) * df["Age_score"]
    # --- Risk-aware add-on (proxies) ---
    # Normalize: higher Sharpe is good; lower vol/drawdown are good
    if "Risk_Sharpe_proxy" in df.columns:
        sharpe_s = normalize(df["Risk_Sharpe_proxy"])
    else:
        sharpe_s = pd.Series(0.5, index=df.index)

    vol_s = normalize(-pd.to_numeric(df.get("Risk_Vol_proxy", np.nan), errors="coerce"))
    dd_s  = normalize(-pd.to_numeric(df.get("Risk_Drawdown_proxy", np.nan), errors="coerce"))

    df["Risk_score"] = (sharpe_s + vol_s + dd_s) / 3.0

    # If the weights include "Risk", blend it; otherwise treat as 0 (no change)
    base += weights.get("Risk", 0.0) * df["Risk_score"]

    df["Final Score"] = base
    


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

    # Compute country/account policy once up front
    policy = get_country_policy(country, account_type, asset_class)
    if "Listing Country" in df.columns:
        if policy["hard_include"]:
            df = df[df["Listing Country"].isin(policy["hard_include"])]
        if policy["hard_exclude"]:
            df = df[~df["Listing Country"].isin(policy["hard_exclude"])]



    # --- SAFETY: enforce numeric helpers ---
    if "ER_clean" not in df.columns:
        df["ER_clean"] = pd.to_numeric(df.get("ER_num", np.nan), errors="coerce")

    if "Yield_clean" not in df.columns:
        df["Yield_clean"] = pd.to_numeric(df.get("Yield_num", np.nan), errors="coerce")

    if "AUM_bil" not in df.columns:
        df["AUM_bil"] = pd.to_numeric(df.get("AUM_bil", np.nan), errors="coerce")

    if "Age_months" not in df.columns:
        df["Age_months"] = pd.to_numeric(df.get("Age_months", 0), errors="coerce")


    # Yield_clean (percentage units)
    if "Yield_clean" not in df.columns:
        if "Yield_num" in df.columns:
            y = pd.to_numeric(df["Yield_num"], errors="coerce")
        elif "Annual Dividend Yield %".strip() in df.columns:
            y = pd.to_numeric(df["Annual Dividend Yield %"].astype(str).str.replace("%","", regex=False), errors="coerce")
        elif "Dividend Yield" in df.columns:
            y = pd.to_numeric(df["Dividend Yield"].astype(str).str.replace("%","", regex=False), errors="coerce")
        elif "Yield" in df.columns:
            y = pd.to_numeric(df["Yield"].astype(str).str.replace("%","", regex=False), errors="coerce")
        else:
            y = pd.Series(np.nan, index=df.index)

        if y.notna().any() and (y.max(skipna=True) <= 1.5):
            y = y * 100.0
        df["Yield_clean"] = y

    # AUM_bil (billions). If missing, try to derive from common columns.
    if "AUM_bil" not in df.columns:
        if "AUM (B$)" in df.columns:
            aum_b = pd.to_numeric(df["AUM (B$)"], errors="coerce")
        elif "AUM_billion" in df.columns:
            aum_b = pd.to_numeric(df["AUM_billion"], errors="coerce")
        elif "AUM" in df.columns:
            aum_raw = pd.to_numeric(df["AUM"], errors="coerce")
            # Heuristic: if values are > 1e3 assume millions, convert to billions
            aum_b = np.where(aum_raw > 1_000, aum_raw / 1_000.0, aum_raw)
        else:
            aum_b = pd.Series(np.nan, index=df.index)
        df["AUM_bil"] = pd.to_numeric(aum_b, errors="coerce")

    # Age_months fallback (kept conservative: if unknown -> 0 months)
    if "Age_months" not in df.columns:
        df["Age_months"] = 0

    # Listing Country fallback (if completely missing, keep as blank to avoid crashes)
    if "Listing Country" not in df.columns:
        df["Listing Country"] = ""


    # --- Show human-readable guardrails (UI) ---
    if st.session_state.get("show_eligibility", True) if "show_eligibility" in st.session_state else True:
        summary_txt = _eligibility_summary(
            st.session_state.get("builder_view", "Wealth Management"),
            country,
            account_type,
            asset_class
        )
        st.info(summary_txt)


    # NEW: apply eligibility (fit-to-present) first
    df = apply_eligibility(
        df,
        builder_view=st.session_state.get("builder_view", "Wealth Management"),
        country=country,
        account_type=account_type,
        asset_class=asset_class
    )


    if risk == "Conservative" and "is_option_income" in df.columns:
        df = df[~df["is_option_income"]]


    # --- Data quality screens (AUM + fund age) ---
    min_aum = st.session_state.get("min_aum_bil", 0.0)
    min_age = st.session_state.get("min_age_months", 0)


    # 1) Exclude leveraged/inverse unless explicitly allowed
    if not st.session_state.get("allow_leveraged", False):
        for col in ["is_leveraged", "is_inverse"]:
            if col in df.columns:
                df = df[~df[col]]

    # 2) Max ER screen (values in %)
    max_er = st.session_state.get("max_er_pct", None)
    if max_er is not None and "ER" in df.columns:
        er_tmp = pd.to_numeric(df["ER"].astype(str).str.replace("%", ""), errors="coerce")
        # If looks like 0.05 instead of 5, convert to % scale once
        if er_tmp.notna().any() and er_tmp.max() <= 1.5:
            er_tmp = er_tmp * 100
        df = df[er_tmp <= float(max_er)]

    # Hedging rule: filter or nudge later
    hp = st.session_state.get("hedge_pref", "Any")
    if hp == "Avoid hedged" and "Hedged Status" in df.columns:
        df = df[df["Hedged Status"] != "Hedged"]


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

    # Hedge preference (gentle nudge, avoids empty results)
    pref = st.session_state.get("hedge_pref", "Any")
    if pref != "Any" and "is_hedged" in ranked.columns:
        if pref == "Prefer hedged":
            ranked["Final Score"] = ranked["Final Score"] * np.where(ranked["is_hedged"], 1.03, 0.99)
        else:  # Avoid hedged
            ranked["Final Score"] = ranked["Final Score"] * np.where(ranked["is_hedged"], 0.97, 1.01)
        ranked = ranked.sort_values("Final Score", ascending=False)

    # Make country/account soft boosts available in this function
    policy = get_country_policy(country, account_type, asset_class)

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

_INDEX_BUCKETS = [
    ("sp500",        re.compile(r"\b(s&p(\s*500)?|500 index)\b", re.I)),
    ("total_us",     re.compile(r"\b(total\s*market|vti|schb|itot)\b", re.I)),
    ("nasdaq100",    re.compile(r"\b(nasdaq\s*100|qqq|ndx)\b", re.I)),
    ("russell2000",  re.compile(r"\b(russell\s*2000|iwo|iwn)\b", re.I)),
    ("intl_dev",     re.compile(r"\b(msci\s*eafe|eafe|developed\s*ex\s*us)\b", re.I)),
    ("emerging",     re.compile(r"\b(emerging|msci\s*em|vwo|iemg)\b", re.I)),
    ("intl_ex_us",   re.compile(r"\b(developed\s*ex\s*us|acwi\s*ex\s*us|ex[- ]us|xeus|xus ex)\b", re.I)),
    ("canada_broad", re.compile(r"\b(tsx\s*composite|s&p/tsx|canada\s*all\s*cap|ftse\s*canada)\b", re.I)),
    ("agg_bond",     re.compile(r"\b(aggregate|aggg?|core bond|total bond|bnd|agg)\b", re.I)),
    ("treasury",     re.compile(r"\b(treasur(y|ies)|gov(ernment)? bond)\b", re.I)),
    ("corp_bond",    re.compile(r"\b(corporate|ig credit)\b", re.I)),
    ("t_bill_cash",  re.compile(r"\b(t[- ]?bill|money market|ultra short)\b", re.I)),
]

# Quick sector/thematic detector (name-based)
_SECTOR_WORDS = ["technology","healthcare","financial","energy","materials","industrials",
                 "utilities","real estate","reit","consumer","communication","staples",
                 "discretionary","semiconductor","ai","cyber","cloud","clean","solar","uranium"]

def _is_sectorish(name: str) -> bool:
    s = str(name).lower()
    return any(w in s for w in _SECTOR_WORDS)


def _index_bucket(etf_name: str) -> str:
    name = str(etf_name or "")
    for bucket, pat in _INDEX_BUCKETS:
        if pat.search(name):
            return bucket
    return "other"

def yahoo_symbol(symbol, listing_country, raw_symbol=None):
    """
    Build a Yahoo-friendly symbol. For Canada, Yahoo often expects .TO (or other suffix).
    We only add .TO if the raw symbol looked Canadian (TSX/TSXV/NEO prefixes).
    """
    s = str(symbol or "")
    lc = str(listing_country or "").strip()
    raw = str(raw_symbol or "")
    if lc == "Canada":
        # if it had TSX/TSXV/NEO/CSE prefix originally and no suffix now, add .TO
        if (re.search(r'^(TSX:|TSXV:|NEO:|CSE:|CBOE CANADA:|TSE:)', raw, re.I)
            and not re.search(r'\.(TO|TSX|NE|CN)$', s, re.I)):
            return s + ".TO"
    return s

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
    st.markdown("### FundMentor")
    st.caption("Smarter Portfolios, Built for Advisors & Investors")
    st.write("---")  # divider

    st.header("Context")

    # remember the toggle state across reruns
    if "show_eligibility" not in st.session_state:
        st.session_state["show_eligibility"] = True


    # --- defaults ---


    # Keep country in session
    if "country" not in st.session_state:
        st.session_state["country"] = ""

    # When Country changes, snap dataset back to Auto safely (inside a callback)
    def _on_country_change():
        st.session_state["use_context"] = ""

    # Country picker (use a key + callback; DO NOT assign the return into session_state)
    st.selectbox(
        "Country",
        ["", "Canada", "USA"],
        index=["", "Canada", "USA"].index(st.session_state.get("country","")),
        key="country",
        on_change=_on_country_change
    )

    # --- Builder View: who are you building for? ---
    builder_view = st.radio(
        "Who are you building for?",
        ["Wealth Management", "Asset Management"],
        horizontal=True,
        key="builder_view"
    )



    country = st.session_state.get("country", "")



    # Allow blank ("None") as default option for more flexibility
    country_options = ["", "Canada", "USA"]

    # --- Account Type options depend on Builder View + Country ---
    _view    = st.session_state.get("builder_view", "Wealth Management")
    _country = st.session_state.get("country", "")

    acct_options = ACCOUNT_TYPES.get(_view, {}).get(_country, [""])

    # Reset stale selection if it no longer exists in the new options
    if st.session_state.get("account_type") not in acct_options:
        st.session_state["account_type"] = acct_options[0] if acct_options else ""

    account_type = st.selectbox(
        "Account Type",
        acct_options,
        index=acct_options.index(st.session_state.get("account_type", acct_options[0])) if acct_options else 0,
        key="account_type"
    )

    # Show/hide the eligibility summary box shown above each asset-class tab
    st.checkbox(
        "Show Eligibility Summary",
        value=st.session_state.get("show_eligibility", True),
        key="show_eligibility",
        help="Display the guardrails currently applied (account/country-specific eligibility)."
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

    # If Asset Management view is selected, use a generic institutional note
    if st.session_state.get("builder_view") == "Asset Management":
        rules_applied = {"note": "Institutional use — no retail tax adjustments applied."}


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
    # Toggle: apply age-based glidepath (store in session so main panel can read it)
    st.checkbox(
        "Adjust for time horizon (age-based glidepath)",
        key="use_glide"
    )

    
    st.checkbox(
        "Apply guardrails (cap equity by profile & horizon)",
        value=True,
        key="apply_glide_guardrails",
        help="If ON: limit equity by risk-profile cap, short-horizon cap, and retirement-age cap.",
        disabled=not st.session_state.get("use_glide", False)
    )

    # NEW: How to apply glidepath (radio) – label hidden, disabled if glidepath OFF
    st.radio(
        "Glidepath application",
        ["Override base", "Blend 50/50 with base"],
        index=0,
        horizontal=True,
        key="gp_mode",
        disabled=not st.session_state.get("use_glide", False),
        label_visibility="collapsed",
    )

    amount = st.number_input("Investment Amount ($)", min_value=1000, step=1000)
    client_name = st.text_input("Client Name")
    notes = st.text_area("Meeting Notes")

    # --- Advanced: data quality filters (AUM & Fund Age) ---
    with st.expander("Advanced: data quality filters"):
        min_aum_bil = st.slider(
            "Minimum AUM (billions $)",
            0.0, 50.0, 0.25, 0.05,
            help="Drop tiny funds to reduce closure/liquidity risk."
        )
        min_age_months = st.slider(
            "Minimum fund age (months)",
            0, 240, 12, 3,
            help="Avoid brand-new funds."
        )

    # Save to session for use in filters
    st.session_state["min_aum_bil"] = float(min_aum_bil)
    st.session_state["min_age_months"] = int(min_age_months)

    # --- Advanced: trading liquidity ---
    with st.expander("Advanced: trading liquidity"):    
        use_adv = st.checkbox(
            "Require minimum average daily $ volume",
            value=(st.session_state.get("builder_view") == "Asset Management")
        )
        adv_min = st.number_input(
            "Min ADV ($)", min_value=0,
            value=5_000_000 if st.session_state.get("builder_view") == "Asset Management" else 1_000_000,
            step=50_000,
            help="Skip thinly-traded ETFs (if ADV data exists)."
        )


    # store in session (same indent level as 'with st.expander(...)')
    st.session_state["use_adv"] = bool(use_adv)
    st.session_state["adv_min"] = int(adv_min)

    # === Product policy controls ===
    hedge_pref = st.selectbox(
        "Currency hedging",
        ["Any", "Prefer hedged", "Avoid hedged"],
        index=0,
        help="Prefer or avoid currency-hedged ETFs."
    )
    allow_leveraged = st.checkbox(
        "Allow leveraged/inverse ETFs",
        value=False,
        help="If OFF, leveraged and inverse ETFs are excluded."
    )
    max_er_pct = st.number_input(
        "Max Expense Ratio (%)",
        min_value=0.0, value=0.80, step=0.05,
        help="Filter out ETFs with ER above this threshold."
    )

    st.session_state["hedge_pref"] = hedge_pref
    st.session_state["allow_leveraged"] = bool(allow_leveraged)
    st.session_state["max_er_pct"] = float(max_er_pct)


    # ---- One-click reset for common inputs ----
    if st.button("Reset all inputs"):
        for k in [
            "country", "use_context", "use_context_rules", "use_context_note",
            "min_aum_bil", "min_age_months", "use_adv", "adv_min",
            "weights_override", "use_account_tax",
            "w1y", "wer", "waum", "wyld", "wtax",
            "model_df", "model_prices", "_prev_country",
            "Risk Tolerance", "quiz_step", "quiz_score", "show_risk_quiz"
        ]:
            if k in st.session_state:
                del st.session_state[k]
        st.cache_data.clear()
        st.success("Inputs reset. Re-run your selections.")
        st.rerun()

    
    with st.expander("Advanced: scoring weights"):

        w_1y  = st.slider("1Y performance", 0.0, 0.6, 0.30 if goal=="Wealth Growth" else 0.20, 0.05, key="w1y")
        w_er  = st.slider("Expense ratio",   0.0, 0.6, 0.20, 0.05, key="wer")
        w_aum = st.slider("AUM size",        0.0, 0.6, 0.10, 0.05, key="waum")
        w_yld = st.slider("Dividend yield",  0.0, 0.6, 0.10 if goal=="Wealth Growth" else 0.20, 0.05, key="wyld")
        w_tax = st.slider("Tax efficiency",  0.0, 0.6, 0.30, 0.05, key="wtax")

        # NEW: Risk slider (vol/drawdown/Sharpe proxies)
        w_risk = st.slider(
            "Risk (vol/drawdown/Sharpe)",
            0.0, 0.30, 0.00, 0.01, key="wrisk",
            help="Turn this up to include risk-adjusted quality in the score."
        )

        tot = max(w_1y + w_er + w_aum + w_yld + w_tax + w_risk, 1e-9)  # normalize

        st.session_state["weights_override"] = {
            "1Y":    w_1y  / tot,
            "ER":    w_er  / tot,
            "AUM":   w_aum / tot,
            "Yield": w_yld / tot,
            "TaxEff":w_tax / tot,
            "Risk":  w_risk/ tot,    
        }


    use_account_tax = st.checkbox(
        "Use account-aware tax scoring",
        value=True,
        help="Factor in listing country, account type, and yield to rank tax efficiency."
    )
    st.session_state["use_account_tax"] = use_account_tax


    # Always load US + CA together (master dataset)
    dataset_label, dataset_ts = "All (US+CA)", None

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
    else:
        st.warning("Cannot de-duplicate: missing 'Symbol' or 'ETF Name' column.")

    # Banner timestamp = newer of the two files
    t_us, t_ca = _mtime(US_DATA_PATH), _mtime(CA_DATA_PATH)
    newer = max([t for t in [t_us, t_ca] if t is not None], default=None)
    dataset_ts = dt.datetime.fromtimestamp(newer) if newer else None

    # Sanity: show rows loaded or the exact problem
    if etf_df is None or etf_df.empty:
        st.error(
            f"{dataset_label}: 0 rows loaded. Check files exist and sheets have data.\n"
            f"US Path: {US_DATA_PATH}\nCA Path: {CA_DATA_PATH}\nSheet: Export"
        )
    else:
        st.success(f"{dataset_label}: {len(etf_df):,} rows loaded.")

    st.caption("Listing Country counts: " + str(dict(etf_df["Listing Country"].value_counts(dropna=False).to_dict())))

    # Debug expander (now independent of any dataset_choice)
    with st.expander("🔎 Debug — what did we actually load?", expanded=False):
        st.write({
            "dataset_choice": "All (US+CA) (forced)",
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

    # quick sanity sample toggle
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

    # --- Strategic Asset Allocation (single glidepath implementation) ---
    base_alloc = allocation_matrix.get((goal, risk), {"Equity": 50, "Bonds": 40, "Cash": 10})

    # Read the sidebar toggle (set under the Age slider); default False if not found
    use_glide = bool(st.session_state.get("use_glide", False))

    # Start with base mix so 'allocation' always exists
    allocation = base_alloc.copy()

    if use_glide:
        gp_mode = st.session_state.get("gp_mode", "Override base")
        apply_guardrails = bool(st.session_state.get("apply_glide_guardrails", True))


        # Equity target from rule of thumb using sidebar Age
        GLIDE_CONST = 120
        base_eq = base_alloc.get("Equity", 0)
        eq_raw = GLIDE_CONST - age
        eq_target = max(30, min(90, eq_raw))  # clamp 30–90%
        if gp_mode == "Blend 50/50 with base":
            eq_target = round(0.5 * eq_target + 0.5 * base_eq, 1)

        # Guardrails (risk, short horizon, retirement age)
        if apply_guardrails:
            cap_by_risk = {"Conservative": 50, "Balanced": 70, "Growth": 90}
            reasons = []
            risk_cap = cap_by_risk.get(risk, 90)
            if eq_target > risk_cap:
                reasons.append(f"risk profile cap {risk_cap}%")
                eq_target = risk_cap
            if horizon <= 3 and eq_target > 40:
                reasons.append("short-horizon cap 40%")
                eq_target = 40
            if age >= 65 and eq_target > 60:
                reasons.append("retirement-age cap 60%")
                eq_target = 60
            if reasons:
                st.warning("Glidepath guardrails applied: " + " • ".join(reasons))
        else:
            st.caption("Guardrails OFF for glidepath. Equity target left unconstrained.")


        # Split remainder across Bonds/Cash in base proportions
        non_eq = 100 - eq_target
        bonds_base = base_alloc.get("Bonds", 0)
        cash_base  = base_alloc.get("Cash", 0)
        base_non_eq = max(bonds_base + cash_base, 1e-9)
        bonds_target = non_eq * (bonds_base / base_non_eq)
        cash_target  = non_eq * (cash_base  / base_non_eq)
        allocation = {
            "Equity": round(eq_target, 1),
            "Bonds":  round(bonds_target, 1),
            "Cash":   round(cash_target, 1),
        }

        # Minimum cash floor
        MIN_CASH = 2.0
        if allocation.get("Cash", 0) < MIN_CASH:
            need = round(MIN_CASH - allocation.get("Cash", 0), 1)
            take_bonds = min(need, allocation.get("Bonds", 0))
            allocation["Bonds"] = round(allocation["Bonds"] - take_bonds, 1)
            allocation["Cash"]  = round(allocation["Cash"] + take_bonds, 1)
            remain = round(need - take_bonds, 1)
            if remain > 0:
                allocation["Equity"] = round(allocation["Equity"] - remain, 1)

    # Normalize to exactly 100% by nudging Cash (works for both cases)
    total = round(sum(allocation.values()), 1)
    if abs(total - 100.0) >= 0.1:
        allocation["Cash"] = round(allocation.get("Cash", 0) + (100.0 - total), 1)

    st.caption(
        f"Equity set to {allocation['Equity']}% via {'glidepath' if use_glide else 'base mix'}."
    )


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

    # Limit sector/thematic concentration within Equity sleeve 
    limit_sector = st.checkbox("Limit sector/thematic exposure to 50% within Equity", value=True)
    etfs_per_class = st.slider("ETFs per asset class", 1, 3, 1, help="Pick top-N by score within each asset class.")
    min_score = st.slider("Min score to include", 0.00, 1.00, 0.00, 0.05, help="Skip very weak candidates.")
    st.session_state["min_score"] = float(min_score)
    include_mixed_as_core = st.checkbox("Allow a single Mixed fund to replace Equity/Bonds/Cash if it scores very high", value=False, help="If enabled and a top Mixed ETF scores ≥ chosen threshold, it will be used as a one-ticket core.")
    st.session_state["include_mixed_as_core"] = include_mixed_as_core

    # Re-normalize to exactly 100% by nudging Cash after Mixed carve
    total_after_mixed = round(sum(allocation.values()), 1)
    if abs(total_after_mixed - 100.0) >= 0.1:
        allocation["Cash"] = round(allocation.get("Cash", 0) + (100.0 - total_after_mixed), 1)

    # NEW: Optional fourth sleeve that does NOT replace the core
    include_mixed_sleeve = st.checkbox(
        "Include Mixed as a fourth sleeve (do not replace core)",
        value=False
    )
    mixed_sleeve_pct = st.slider(
        "Mixed sleeve (% of portfolio)",
        0, 30, 10, 1,
        help="Taken proportionally from Equity & Bonds (not from Cash)."
    )

    # Carve Mixed from Equity + Bonds proportionally (leave Cash untouched)
    if (not st.session_state.get("include_mixed_as_core", False)) and include_mixed_sleeve and mixed_sleeve_pct > 0:
        eq = float(allocation.get("Equity", 0.0))
        bd = float(allocation.get("Bonds", 0.0))
        non_cash = eq + bd
        take = float(mixed_sleeve_pct)
        if non_cash > 0 and take > 0:
            scale = max((non_cash - take) / non_cash, 0.0)
            allocation["Equity"] = round(eq * scale, 1)
            allocation["Bonds"]  = round(bd * scale, 1)
            allocation["Mixed"]  = round(take, 1)

    # Re-normalize to exactly 100% by nudging Cash after Mixed carve
    total_after_mixed = round(sum(allocation.values()), 1)
    if abs(total_after_mixed - 100.0) >= 0.1:
        allocation["Cash"] = round(allocation.get("Cash", 0) + (100.0 - total_after_mixed), 1)



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
                if mixed_ranked.empty or float(mixed_ranked.iloc[0].get("Final Score", 0)) < float(min_score):
                    st.caption("No Mixed candidate passed the threshold; using 3-bucket approach.")
                else:
                    sym  = str(mixed_ranked.iloc[0]["Symbol"])
                    name = str(mixed_ranked.iloc[0]["ETF Name"])
                    score = float(mixed_ranked.iloc[0]["Final Score"])
                    rows.append({"Asset Class": "Mixed", "Symbol": sym, "ETF Name": name, "Weight %": 100.0, "Score": round(score, 3)})
                    st.caption("Using a one-ticket Mixed core (replaces Equity/Bonds/Cash).")


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
                    is_equity = (ac == "Equity")
                    if rules.get("avoid_us_dividends"):
                        base = base[base["Listing Country"].ne("USA")]
                    if is_equity and rules.get("avoid_dividends"):
                        base = base[pd.to_numeric(base["Annual Dividend Yield %"].str.replace("%",""), errors="coerce") < 2]

                    if is_equity and rules.get("favor_growth"):
                        base = base[pd.to_numeric(base["1 Year"].str.replace("%",""), errors="coerce") > 5]


                    if rules.get("favor_low_fee"):
                        base = base[pd.to_numeric(base["ER"].str.replace("%",""), errors="coerce") < 0.25]

                    # === Hard product-quality screens ===

                    # 1) Exclude leveraged/inverse unless explicitly allowed
                    if not st.session_state.get("allow_leveraged", False):
                        for col in ["is_leveraged", "is_inverse"]:
                            if col in base.columns:
                                base = base[~base[col]]

                    # 2) Max ER screen (values in %)
                    max_er = st.session_state.get("max_er_pct", None)
                    if max_er is not None and "ER" in base.columns:
                        er_tmp = pd.to_numeric(base["ER"].astype(str).str.replace("%", ""), errors="coerce")
                        # If looks like 0.05 instead of 5, convert to % scale once
                        if er_tmp.notna().any() and er_tmp.max() <= 1.5:
                            er_tmp = er_tmp * 100
                        base = base[er_tmp <= float(max_er)]


                    # Keep equity goal filter; relax only risk
                    if ac == "Equity":
                        base = safe_goal_filter(base, goal)

                        # Keep the same data-quality/liquidity screens in fallback
                    if "AUM_bil" in base.columns:
                        base = base[pd.to_numeric(base["AUM_bil"], errors="coerce").fillna(0) >= float(st.session_state.get("min_aum_bil", 0.0))]

                    if "Age_months" in base.columns and st.session_state.get("min_age_months", 0) > 0:
                        base = base[pd.to_numeric(base["Age_months"], errors="coerce").fillna(0) >= int(st.session_state.get("min_age_months", 0))]

                    if st.session_state.get("use_adv") and "Avg Dollar Volume" in base.columns:
                        base = base[pd.to_numeric(base["Avg Dollar Volume"], errors="coerce").fillna(0) >= st.session_state.get("adv_min", 0)]


                    ranked = rank_etfs(base, goal).sort_values("Final Score", ascending=False)
                    ranked = keep_one_per_bucket(ranked)


                if ranked.empty:
                    st.caption(f"{ac}: no eligible ETFs after current filters; sleeve skipped.")
                    continue


                # Enforce simple issuer cap first
                issuer_cap = 1
                capped = (
                    ranked.assign(_rank=ranked.groupby("Issuer").cumcount())
                        .query("_rank < @issuer_cap")
                        .drop(columns="_rank", errors="ignore")
                )

                if ac == "Equity" and limit_sector:
                    # Allow at most half the slots to be sector/thematic (rounded down)
                    sector_cap = max(1, etfs_per_class // 2) if etfs_per_class > 1 else 1
                    sec_mask = capped["ETF Name"].apply(_is_sectorish)

                    core_part   = capped[~sec_mask].head(etfs_per_class)      # broad/core first
                    sector_part = capped[ sec_mask].head(sector_cap)           # limited sector slots

                    take = pd.concat([core_part, sector_part]).head(etfs_per_class)
                else:
                    take = capped.head(etfs_per_class)


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

            # Keep both spellings for downstream helpers
            model_df["Weight (%)"] = model_df["Weight %"]

            # Use the precomputed, percent-scale helpers if available (preferred)
            if {"ER_num", "Yield_num", "1Y_num"}.issubset(etf_df.columns):
                meta = etf_df[["Symbol", "ER_num", "Yield_num", "1Y_num"]].copy()
            else:
                # Fallback: parse and normalize once
                meta = etf_df[["Symbol", "ER", "Annual Dividend Yield %", "1 Year"]].copy()

                # ER → percent units
                meta["ER_num"] = pd.to_numeric(meta["ER"].astype(str).str.replace("%","", regex=False), errors="coerce")
                if meta["ER_num"].notna().any() and meta["ER_num"].max() <= 1.5:
                    meta["ER_num"] *= 100

                # Yield → percent units
                meta["Yield_num"] = pd.to_numeric(meta["Annual Dividend Yield %"].astype(str).str.replace("%","", regex=False), errors="coerce")
                if meta["Yield_num"].notna().any() and meta["Yield_num"].max() <= 1.5:
                    meta["Yield_num"] *= 100

                # 1Y return → percent units (guard small decimals)
                meta["1Y_num"] = pd.to_numeric(meta["1 Year"].astype(str).str.replace("%","", regex=False), errors="coerce")
                if meta["1Y_num"].notna().any() and meta["1Y_num"].max() <= 3.0:
                    meta["1Y_num"] *= 100

            # Merge onto your model
            model_df = model_df.merge(meta[["Symbol", "ER_num", "Yield_num", "1Y_num"]], on="Symbol", how="left")

            # --- Normalize ER/Yield to % if they were stored as tiny fractions (e.g., 0.0049 = 0.49%) ---
            for col in ["ER_num", "Yield_num"]:
                s = pd.to_numeric(model_df[col], errors="coerce")
                # If 90% of values are <= 1.5, they were almost certainly fractions of 1 → convert to %
                if s.notna().sum() and s.dropna().quantile(0.90) <= 1.5:
                    s = s * 100
                model_df[col] = s.fillna(0)


            # Safety: treat missing ER/Yield as 0
            model_df[["ER_num", "Yield_num"]] = model_df[["ER_num", "Yield_num"]].fillna(0)

            # ---- Compute dollars per year ----
            model_df["Fee $/yr"]    = (model_df["Dollars"] * (model_df["ER_num"] / 100)).round(2)
            model_df["Income $/yr"] = (model_df["Dollars"] * (model_df["Yield_num"] / 100)).round(2)

            # Join listing country & asset class to estimate after-tax income
            model_df = model_df.merge(
                etf_df[["Symbol","Listing Country","Simplified Asset Class"]],
                on="Symbol", how="left"
            )

            # Bring in AUM & Age for badges
            model_df = model_df.merge(
                etf_df[["Symbol","AUM_bil","Age_months"]],
                on="Symbol", how="left"
            )

            def _badges(row):
                tags = []

                # 1) Cost / size / history
                try:
                    if float(row.get("ER_num", 9e9)) <= 0.10:
                        tags.append("Low fee")
                except:
                    pass
                try:
                    if float(row.get("AUM_bil", 0)) >= 5:
                        tags.append("Large AUM")
                except:
                    pass
                try:
                    if float(row.get("Age_months", 0)) >= 36:
                        tags.append("3yr+ history")
                except:
                    pass

                # 2) Broad vs Sector/thematic
                try:
                    name = str(row.get("ETF Name", ""))
                    if callable(globals().get("_is_sectorish")) and _is_sectorish(name):
                        tags.append("Sector/theme")
                    else:
                        # If we can bucket the index (e.g., S&P 500, TSX, All-World), call it "Broad exposure"
                        if callable(globals().get("_index_bucket")):
                            b = _index_bucket(name)
                            if b != "other":
                                tags.append("Broad exposure")
                except:
                    pass

                # 3) Income tilt
                try:
                    if float(row.get("Yield_num", 0)) >= 3.0:
                        tags.append("Income tilt")
                except:
                    pass

                # 4) Currency hedge hint
                try:
                    nm = str(row.get("ETF Name", "")).lower()
                    if "hedged" in nm:
                        tags.append("Currency-hedged")
                except:
                    pass

                # 5) Account-aware, country-aware hints (TFSA/RRSP friendly)
                try:
                    country = st.session_state.get("country", "")
                    acct    = st.session_state.get("use_context", "")
                    lst     = str(row.get("Listing Country", "") or "")
                    ac_key  = str(row.get("Simplified Asset Class", "") or "").lower()

                    if country == "Canada" and ac_key in {"equity","mixed"}:
                        if acct in {"TFSA","RESP"} and lst == "Canada":
                            tags.append("TFSA-friendly")
                        if acct == "RRSP" and lst == "USA":
                            tags.append("RRSP treaty benefit")
                except:
                    pass

                # 6) Home-listed
                try:
                    if str(row.get("Listing Country","")) == st.session_state.get("country",""):
                        tags.append("Home-listed")
                except:
                    pass

                return " • ".join(tags)

            model_df["Badges"] = model_df.apply(_badges, axis=1)

            # ---- Explainability: Top drivers tags (Excel-only) ----
            def _norm(s):
                s = pd.to_numeric(s, errors="coerce")
                lo, hi = s.min(), s.max()
                if not np.isfinite(lo) or not np.isfinite(hi) or hi == lo:
                    return pd.Series(0.5, index=s.index)  # neutral if no spread
                return (s - lo) / (hi - lo)

            # Use weights from sidebar if present; else default by goal
            weights = st.session_state.get("weights_override") or get_factor_weights(goal)

            _factors = []  # (key, label, series, weight)

            # 1Y performance (higher is better)
            if "1Y_num" in model_df.columns and weights.get("1Y", 0) > 0:
                _factors.append(("1Y", "Recent performance", _norm(model_df["1Y_num"]), weights["1Y"]))

            # Expense ratio (lower is better) -> "Low fee"
            if "ER_num" in model_df.columns and weights.get("ER", 0) > 0:
                _factors.append(("ER", "Low fee", _norm(-pd.to_numeric(model_df["ER_num"], errors="coerce")), weights["ER"]))


            # AUM (higher is better) -> "Fund size / liquidity"
            if "AUM_bil" in model_df.columns and weights.get("AUM", 0) > 0:
                _factors.append(("AUM", "Fund size / liquidity", _norm(model_df["AUM_bil"]), weights["AUM"]))


            # Dividend yield (higher is better) -> "Income"
            if "Yield_num" in model_df.columns and weights.get("Yield", 0) > 0:
                _factors.append(("Yield", "Income", _norm(model_df["Yield_num"]), weights["Yield"]))


            # Tax efficiency proxy (if available) -> "Tax-efficient"
            if weights.get("TaxEff", 0) > 0:
                # Always create a Series indexed to model_df, even if the column is missing
                if "TaxEff_account" in model_df.columns:
                    tax_series = pd.to_numeric(model_df["TaxEff_account"], errors="coerce")
                else:
                    tax_series = pd.Series(np.nan, index=model_df.index)

                if tax_series.isna().all():
                    tax_series = pd.Series(0.5, index=model_df.index)

                _factors.append(("TaxEff", "Tax-efficient", _norm(tax_series), weights["TaxEff"]))

            # Risk score (if computed) -> "Risk-adjusted quality"
            if "Risk_score" in model_df.columns and weights.get("Risk", 0) > 0:
                _factors.append(("Risk", "Risk-adjusted quality", _norm(model_df["Risk_score"]), weights["Risk"]))

            def _drivers_row(i: int) -> str:
                conts = []
                for _k, _label, _series, _w in _factors:
                    try:
                        val = float(_series.iloc[i])
                        conts.append((_w * val, _label))
                    except Exception:
                        pass
                conts.sort(reverse=True, key=lambda x: x[0])
                top = [lbl for _, lbl in conts[:3]]
                return " • ".join(top)

            model_df["Drivers"] = [ _drivers_row(i) for i in range(len(model_df)) ]



            def _after_tax_yield_factor(listing_country, asset_class_key, country, acct):
                # Gentle, transparent rules of thumb — NOT tax advice.
                # asset_class_key is lower-case (equity/bond/cash/mixed/other)
                lc = str(listing_country or "")
                ac = str(asset_class_key or "").lower()
                c  = str(country or "")
                a  = str(acct or "")

                f = 1.00
                if c == "Canada":
                    if a in {"TFSA","RESP"} and lc == "USA" and ac in {"equity","mixed"}:
                        f = 0.85   # unrecoverable US WHT on distributions
                    elif a == "RRSP" and lc == "USA" and ac == "equity":
                        f = 1.00   # treaty exemption typical for US-listed equity
                    elif a == "Non-Registered" and ac == "bond":
                        f = 0.70   # interest fully taxable ~rough guide
                elif c == "USA":
                    if a == "Taxable" and ac == "bond":
                        f = 0.75   # interest at ordinary rate ~rough guide
                    elif a in {"Roth IRA","Traditional IRA","401(k)"}:
                        f = 1.00   # sheltered
                return f

            _country = st.session_state.get("country","")
            _acct    = st.session_state.get("use_context","")

            model_df["After-tax income $/yr"] = model_df.apply(
                lambda r: round(
                    float(r["Income $/yr"]) * _after_tax_yield_factor(
                        r.get("Listing Country",""),
                        r.get("Simplified Asset Class",""),
                        _country, _acct
                    ), 2
                ),
                axis=1
            )


            # Nice column names for display

            display_cols = [
                "Asset Class", "Symbol", "ETF Name",
                "Weight %", "Dollars",
                "ER_num", "Yield_num",
                "Fee $/yr", "Income $/yr",
                "Score", "Badges", "Drivers"
            ]


            display_renames = {"ER_num": "ER (%)", "Yield_num": "Yield (%)", "Badges": "Why it ranks", "Drivers": "Top drivers",}

            # Optional: add friendly labels for fields used elsewhere
            display_renames.update({
                "AUM_bil": "AUM (B$)",
                "Age_months": "Age (months)",
                "Risk_score": "Risk score (0–1)"
            })


            st.dataframe(
                model_df[display_cols].rename(columns=display_renames),
                use_container_width=True
            )

            # --- Advanced metrics view (pulled from the full dataset by Symbol) ---
            with st.expander("Show advanced metrics (optional)"):
                # columns we’ll try to show if present in your excel files
                extra_cols = [
                    "AUM_bil", "Age_months", "Beta (5Y)",
                    "52W Low Chg", "ATH Chg (%)", "ATL Chg (%)",
                    "Risk_Vol_proxy", "Risk_Drawdown_proxy", "Risk_Sharpe_proxy", "Risk_score"
                ]

                # Build a compact table: model symbols + the extra metrics from etf_df
                try:
                    have = ["Symbol", "ETF Name"] + [c for c in extra_cols if c in etf_df.columns]
                    adv_df = model_df[["Symbol", "ETF Name"]].merge(
                        etf_df[have].drop_duplicates(subset=["Symbol","ETF Name"]),
                        on=["Symbol","ETF Name"],
                        how="left"
                    )
                    # Only show the columns that actually exist
                    show_cols = [c for c in ["Symbol","ETF Name"] + extra_cols if c in adv_df.columns]
                    if len(show_cols) > 2:
                        # Nice headers for a few fields
                        adv_renames = {
                            "AUM_bil": "AUM (B$)",
                            "Age_months": "Age (months)",
                            "Risk_Vol_proxy": "Volatility proxy (|β 5Y|)",
                            "Risk_Drawdown_proxy": "Drawdown proxy (worst %)",
                            "Risk_Sharpe_proxy": "Sharpe proxy",
                            "Risk_score": "Risk score (0–1)"
                        }
                        st.dataframe(adv_df[show_cols].rename(columns=adv_renames), use_container_width=True)
                    else:
                        st.caption("No advanced columns available for this selection.")
                except Exception as e:
                    st.caption(f"(Could not build advanced metrics table: {e})")


            st.caption(
                "Quality gate active: leveraged/inverse excluded unless enabled • "
                f"Max ER: {st.session_state.get('max_er_pct', '—')}% • "
                f"Hedging: {st.session_state.get('hedge_pref','Any')}"
            )


            # Sector/thematic exposure check (Equity sleeve)
            if limit_sector and not model_df.empty:
                eq_slice = model_df[model_df["Asset Class"] == "Equity"].copy()
                if not eq_slice.empty:
                    eq_sector = eq_slice["ETF Name"].apply(_is_sectorish)
                    sector_weight = float(eq_slice.loc[eq_sector, "Weight %"].sum())
                    if sector_weight > 50:
                        st.warning(
                            f"Equity sleeve has {sector_weight:.1f}% sector/thematic exposure. "
                            f"Consider adding broad-market ETFs for diversification."
                        )

            # ---- Summary metrics ----
            total_fees   = float(model_df["Fee $/yr"].sum())
            total_income = float(model_df["Income $/yr"].sum())
            weighted_er  = (model_df["ER_num"]   * model_df["Dollars"]).sum() / max(model_df["Dollars"].sum(), 1e-9)
            weighted_yld = (model_df["Yield_num"]* model_df["Dollars"]).sum() / max(model_df["Dollars"].sum(), 1e-9)

            total_after_tax_income = float(model_df["After-tax income $/yr"].sum())

            c1, c2, c3, c4 = st.columns(4)
            with c1: st.metric("Total annual fees",        f"${total_fees:,.2f}")
            with c2: st.metric("Est. gross income",        f"${total_income:,.2f}")
            with c3: st.metric("Est. after-tax income",    f"${total_after_tax_income:,.2f}")
            with c4: st.metric("Weighted ER / Yield",      f"{weighted_er:.2f}% / {weighted_yld:.2f}%")

            
            st.session_state["model_df"] = model_df  # <— add this line to reuse in the explanation
            st.success("Model portfolio created.")

            with st.expander("See close alternatives for each pick"):
                try:
                    # Pre-score once per asset class for speed
                    scored_by_class = {}
                    for ac_lbl, ac_key in ASSET_MAP.items():
                        pool = etf_df[etf_df["Simplified Asset Class"].str.lower() == ac_key].copy()
                        if not pool.empty:
                            scored_by_class[ac_lbl] = rank_etfs(pool, goal).sort_values("Final Score", ascending=False)

                    for _, r in model_df.iterrows():
                        name = str(r["ETF Name"])
                        bucket = _index_bucket(name)
                        ac_lbl = str(r["Asset Class"])
                        pool = scored_by_class.get(ac_lbl)
                        if pool is None or pool.empty:
                            continue
                        pool["_bucket"] = pool["ETF Name"].apply(_index_bucket)
                        alts = pool[(pool["_bucket"] == bucket) & (pool["Symbol"] != r["Symbol"])].head(2)
                        if not alts.empty:
                            st.markdown(f"**{r['Symbol']}** alternatives (same index style):")
                            st.dataframe(
                                alts[["Symbol","ETF Name","ER","Annual Dividend Yield %","Total Assets"]]
                                .rename(columns={"Annual Dividend Yield %":"Yield","Total Assets":"AUM"}),
                                use_container_width=True
                            )
                except Exception as e:
                    st.caption(f"(Could not generate alternatives: {e})")


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
                for idx, r in model_df.iterrows():
                    sym = str(r["Symbol"])
                    key = f"price_{sym}_{idx}"  # unique per row to avoid duplicate keys
                    # Pre-fill from saved state (kept per symbol; last edit wins if duplicates)
                    preset = float(st.session_state["model_prices"].get(sym, 0.0))
                    price_val = st.number_input(f"{sym} price ($)", min_value=0.0, value=preset, step=0.01, key=key)

                    st.session_state["model_prices"][sym] = price_val
                    price_inputs.append(price_val)

                # Build price Series aligned to model_df
                price_series = pd.Series(price_inputs, index=model_df.index).replace(0, pd.NA)
                if default_price and default_price > 0:
                    price_series = price_series.fillna(default_price)

                model_df["Price ($)"] = price_series

                # Ensure Dollars/Price/Qty are numeric (handles "", None on Streamlit Cloud)
                model_df["Dollars"] = pd.to_numeric(model_df["Dollars"], errors="coerce").fillna(0.0)
                model_df["Price ($)"] = pd.to_numeric(model_df["Price ($)"], errors="coerce").fillna(0.0)

                # Compute whole-share quantities only where price > 0
                qty_series = pd.to_numeric(
                    (model_df["Dollars"] / model_df["Price ($)"]).where(model_df["Price ($)"] > 0),
                    errors="coerce"
                ).fillna(0).astype("Int64")  # nullable integer (safer on Cloud)

                model_df["Qty (whole)"] = qty_series

                # Now multiply with guaranteed numeric dtypes
                model_df["Allocated $"] = (
                    model_df["Qty (whole)"].astype("float64") * model_df["Price ($)"].astype("float64")
                ).round(2)

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

                # Tip if too much cash is left after whole-share rounding
                if tot_cash / max(float(amount), 1e-9) > 0.05:
                    st.info("Residual cash is above 5%. Consider adjusting prices or decreasing the ETF count to reduce leftover cash.")



            # ---- Download CSV (with fees/income) ----
            csv_out = model_df.rename(columns=display_renames).to_csv(index=False).encode()
            st.download_button("Download model portfolio (CSV)", csv_out, "model_portfolio.csv", "text/csv")
        else:
            st.info("No qualifying ETFs for the current filters/thresholds.")

                # --- Explain my plan (client-friendly text) ---
    with st.expander("📝 Explain my plan"):
        # Friendly names for factor weights
        _friendly = {
            "1Y": "1-year performance",
            "ER": "fees (expense ratio)",
            "AUM": "fund size (AUM)",
            "Yield": "dividend yield",
            "TaxEff": "tax efficiency",
            "Risk": "risk-adjusted quality"  # NEW
        }

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

            # Data quality screens to match the builder
            min_aum = st.session_state.get("min_aum_bil", 0.0)
            min_age = st.session_state.get("min_age_months", 0)

            if "AUM_bil" in filtered.columns:
                filtered = filtered[pd.to_numeric(filtered["AUM_bil"], errors="coerce").fillna(0) >= float(min_aum)]

            if "Age_months" in filtered.columns and min_age > 0:
                filtered = filtered[pd.to_numeric(filtered["Age_months"], errors="coerce").fillna(0) >= int(min_age)]


            # Optional ADV screen for tabs
            if st.session_state.get("use_adv") and "Avg Dollar Volume" in filtered.columns:
                filtered = filtered[
                    pd.to_numeric(filtered["Avg Dollar Volume"], errors="coerce").fillna(0)
                    >= st.session_state.get("adv_min", 0)
                ]



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
            is_equity = (asset_class == "Equity")

            if rules.get("avoid_us_dividends"):
                filtered = filtered[filtered["Listing Country"].ne("USA")]

            if is_equity and rules.get("avoid_dividends"):
                filtered = filtered[pd.to_numeric(filtered["Annual Dividend Yield %"].str.replace("%",""), errors="coerce") < 2]

            if rules.get("favor_tax_efficiency"):
                filtered = filtered[
                    (pd.to_numeric(filtered["Annual Dividend Yield %"].str.replace("%", ""), errors="coerce") < 2.5) &
                    (pd.to_numeric(filtered["ER"].str.replace("%", ""), errors="coerce") < 0.3)
                ]

            if is_equity and rules.get("favor_growth"):
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

            # --- Unified tab ranking: same as Builder + same relax-if-empty fallback ---

            # Strict pass (identical to the Builder’s normal sleeve scoring)
            strict_ranked = get_ranked_for_class(
                asset_class=asset_class,
                goal=goal,
                risk=risk,
                country=st.session_state.get("country",""),
                account_type=st.session_state.get("use_context",""),
                etf_df=etf_df,
                risk_filters=risk_filters
            )

            ranked_df = strict_ranked.copy()
            st.caption(f"{len(ranked_df)} ETFs match the current filters.")

            # If strict results are empty, mirror the Builder’s “relax risk if empty” behavior
            if ranked_df.empty and 'relax_builder' in locals() and relax_builder:
                st.info("Showing top options with risk relaxed to reach at least 6 results (country/account rules still applied).")

                base = etf_df[etf_df["Simplified Asset Class"].str.lower() == class_key].copy()

                # Apply country/account hard policy (same as Builder fallback)
                policy = get_country_policy(
                    st.session_state.get("country",""),
                    st.session_state.get("use_context",""),
                    asset_class
                )
                if policy["hard_include"]:
                    base = base[base["Listing Country"].isin(policy["hard_include"])]
                if policy["hard_exclude"]:
                    base = base[~base["Listing Country"].isin(policy["hard_exclude"])]

                # Context rules (avoid_us_dividends, favor_growth, etc.) — same as Builder fallback
                rules = st.session_state.get("use_context_rules") or {}
                is_equity = (asset_class == "Equity")

                if rules.get("avoid_us_dividends"):
                    base = base[base["Listing Country"].ne("USA")]
                if is_equity and rules.get("avoid_dividends"):
                    base = base[pd.to_numeric(base["Annual Dividend Yield %"].str.replace("%",""), errors="coerce") < 2]
                if is_equity and rules.get("favor_growth"):
                    base = base[pd.to_numeric(base["1 Year"].str.replace("%",""), errors="coerce") > 5]
                if rules.get("favor_low_fee"):
                    base = base[pd.to_numeric(base["ER"].str.replace("%",""), errors="coerce") < 0.25]

                # Quality/liquidity screens
                if "AUM_bil" in base.columns:
                    base = base[pd.to_numeric(base["AUM_bil"], errors="coerce").fillna(0) >= float(st.session_state.get("min_aum_bil", 0.0))]
                if "Age_months" in base.columns and st.session_state.get("min_age_months", 0) > 0:
                    base = base[pd.to_numeric(base["Age_months"], errors="coerce").fillna(0) >= int(st.session_state.get("min_age_months", 0))]
                if st.session_state.get("use_adv") and "Avg Dollar Volume" in base.columns:
                    base = base[pd.to_numeric(base["Avg Dollar Volume"], errors="coerce").fillna(0) >= st.session_state.get("adv_min", 0)]

                # Keep the equity goal fit (we relax only the Risk-Level screen)
                if is_equity:
                    base = safe_goal_filter(base, goal)

                # Score & sort same as Builder
                ranked_df = rank_etfs(base, goal).sort_values("Final Score", ascending=False)

                # Optional: mark ETFs already in the generated model
            try:
                in_model = set(st.session_state.get("model_df", pd.DataFrame()).get("Symbol", []))
                if in_model:
                    ranked_df["_in_model"] = ranked_df["Symbol"].isin(in_model)
                else:
                    ranked_df["_in_model"] = False
            except Exception:
                ranked_df["_in_model"] = False

            # --- Ensure a minimum number of recommendations in the tab ---
            MIN_TAB_RESULTS = 6  # change to taste (e.g., 6–10)

            if len(ranked_df) < MIN_TAB_RESULTS:
                # Relax **risk only** (keep country/account rules and equity goal filter)
                relaxed = base_after_rules.copy()
                if asset_class == "Equity":
                    relaxed = safe_goal_filter(relaxed, goal)

                relaxed_ranked = (
                    rank_etfs(relaxed, goal)
                    .sort_values(by="Final Score", ascending=False)
                )

                # Top up the strict set with relaxed picks (no duplicates)
                ranked_df = pd.concat([ranked_df, relaxed_ranked]).drop_duplicates(
                    subset=["Symbol", "ETF Name"], keep="first"
                ).drop_duplicates(subset=["Symbol", "ETF Name"], keep="first").head(MIN_TAB_RESULTS)

                st.info(
                    f"Showing top options with **risk relaxed** to reach at least {MIN_TAB_RESULTS} results "
                    f"(country/account rules still applied)."
                )


            # Explain why nothing matched (before relaxing risk)
            if ranked_df.empty and not base_after_rules.empty:
                reasons = []
                # Show which risk levels are allowed
                try:
                    allowed_risk = sorted(set(risk_filters[risk]))
                except Exception:
                    allowed_risk = []
                extra = " (+Low/Medium for FI/Cash)" if asset_class in ["Bonds", "Cash"] else ""
                reasons.append(f"Risk allowed: {', '.join(allowed_risk)}{extra}" if allowed_risk else "Risk filter applied")

                # Goal thresholds only apply to Equities
                if asset_class == "Equity":
                    reasons.append(f"Goal thresholds applied ({goal})")

                # Context & data-quality screens, if active
                if st.session_state.get('use_context_rules'):
                    reasons.append("Context rules active")
                if st.session_state.get("min_aum_bil", 0) > 0:
                    reasons.append(f"AUM ≥ {st.session_state['min_aum_bil']}B")
                if st.session_state.get("min_age_months", 0) > 0:
                    reasons.append(f"Age ≥ {st.session_state['min_age_months']} months")
                if st.session_state.get("use_adv") and "Avg Dollar Volume" in base_after_rules.columns:
                    reasons.append(f"ADV ≥ ${st.session_state.get('adv_min', 0):,}")

                st.caption("No matches after strict filters. Filters in effect: " + " • ".join(reasons))

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

                                ysym = yahoo_symbol(row['Symbol'], row.get('Listing Country'), row.get('_SymbolRaw'))

                                st.markdown(f"""
                                <div style='background:#eef9f2; padding:15px; border-radius:10px; border:1px solid #b6e5c5; margin-bottom:15px;'>
                                    <b><a href='https://finance.yahoo.com/quote/{ysym}' target='_blank'>{row['Symbol']}: {row['ETF Name']}</a></b>
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
        value=False,
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
                    is_equity = (ac == "Equity")

    
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
        ranked = get_ranked_for_class(
            asset_class=ac_label,
            goal=goal,
            risk=risk,
            country=st.session_state.get("country", ""),
            account_type=st.session_state.get("use_context", ""),
            etf_df=etf_df,
            risk_filters=risk_filters
        ).copy()

        # Apply the SAME threshold & uniqueness used by the builder
        ms = float(st.session_state.get("min_score", 0.0))
        if "Final Score" in ranked.columns:
            ranked = ranked[ranked["Final Score"] >= ms]

        # Prevent duplicates tracking the same index (same rule used on build)
        if callable(globals().get("keep_one_per_bucket")):
            ranked = keep_one_per_bucket(ranked)

        return ranked.head(top_n)
    
    # Build the DataFrame to annotate — prefer actual model picks if available
    _model = st.session_state.get("model_df")
    if _model is not None and not _model.empty:
        # Optional: respect the per‑sleeve dropdown in Notes
        if notes_asset_choice != "All":
            _model = _model[_model["Asset Class"] == notes_asset_choice]

        # Bring in full fund metadata for notes logic
        candidates_df = _model.merge(
            etf_df.drop_duplicates(subset=["Symbol"]),
            on="Symbol",
            how="left"
        )
    else:
        # Fall back to ranked candidates if no model has been generated yet
        if notes_asset_choice == "All":
            frames = []
            for ac in [k for k in ["Equity", "Bonds", "Cash", "Mixed"] if k in allocation and allocation[k] > 0]:
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

        # If Age_months is missing, try to compute from Inception Date
        if "Age_months" not in candidates_df.columns:
            if "Inception Date" in candidates_df.columns:
                _inc = pd.to_datetime(candidates_df["Inception Date"], errors="coerce")
                candidates_df["Age_months"] = ((_inc - _inc.min()).dt.days * 0 +  # keep index
                                       (pd.Timestamp.today().normalize() - _inc).dt.days / 30.44)
            else:
                candidates_df["Age_months"] = np.nan


        # If Issuer is missing, derive a simple label from the start of the fund name
        if "Issuer" not in candidates_df.columns and "ETF Name" in candidates_df.columns:
            candidates_df["Issuer"] = (
                candidates_df["ETF Name"]
                .astype(str)
                .str.extract(r"^(Vanguard|iShares|SPDR|BMO|Schwab|Invesco|Horizons|BlackRock)", expand=False)
                .fillna("Other")
            )



    notes = []
    # --- Portfolio-level diversification flags (added before per-ETF notes) ---
    div_flags = []

    # 1) Issuer concentration (by count share of picks)
    if "Issuer" in candidates_df.columns and not candidates_df.empty:
        issuer_share = candidates_df["Issuer"].value_counts(normalize=True)
        if len(issuer_share) > 0:
            top_issuer = str(issuer_share.index[0])
            top_share  = float(issuer_share.iloc[0])
            if top_share >= 0.50 and len(issuer_share) > 1:
                div_flags.append(f"Concentration: {int(round(top_share*100))}% of picks are from {top_issuer}.")

    # 2) Duplicate index trackers (e.g., multiple S&P 500 funds)
    if "ETF Name" in candidates_df.columns and callable(globals().get("_index_bucket")):
        buckets = candidates_df["ETF Name"].apply(_index_bucket)
        dup = buckets.value_counts()
        overlapped = [b for b, c in dup.items() if b != "other" and c > 1]
        if overlapped:
            div_flags.append("Multiple funds tracking the same index: " + ", ".join(overlapped))

    # Surface portfolio checks at the TOP of the notes list
    for f in div_flags:
        notes.append("**Portfolio Check** — " + f)

    for _, r in candidates_df.iterrows():
        sym = str(r.get("Symbol", "—"))
        ac  = str(r.get("Simplified Asset Class", "")).lower()
        lst = str(r.get("Listing Country", "") or "Unknown")
        yld = float(r.get("Yield_clean", 0) or 0.0)
        er  = float(r.get("ER_clean", 0) or 0.0)

        this_notes = []

        # --- Behavioral guardrails (ETF-level) ---
        aum_b = float(r.get("AUM_bil", 0) or 0.0)
        age_m = float(r.get("Age_months", 0) or 0.0)
        ret1  = float(r.get("1Y_clean", r.get("1Y_num", 0)) or 0.0)

        # Liquidity / scale
        if aum_b > 0 and aum_b < 0.05:
            this_notes.append("⚠️ Low AUM (<$50M) — potential liquidity risk.")

        # Cost
        if er >= 1.0:
            this_notes.append("⚠️ High expense ratio (>1.00%).")

        # Track record
        if age_m and age_m < 12:
            this_notes.append("⚠️ Limited track record (<12 months).")

        # Chasing momentum
        if ret1 and ret1 > 80:
            this_notes.append("🟡 Very high 1Y return — momentum risk; reassess fundamentals.")


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
