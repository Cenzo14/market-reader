# Market Reader — "Read Between the Lines" Stock Intelligence Model
# Built as the dream toolkit of a market analyst. Free data (Yahoo Finance via yfinance).
# Nothing here is a guarantee. It stacks edges, measures itself honestly, and tells you what the tape is saying.

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import streamlit.components.v1 as components
import json, os, base64
from datetime import datetime, timedelta

st.set_page_config(page_title="Market Reader", page_icon="📈", layout="wide")

DEFAULT_WATCHLIST = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "JPM", "XOM", "UNH", "COST", "AVGO", "LLY", "AMD", "NFLX"]
DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "market_reader_data.json")
GIST_DESC = "market-reader-data"
GIST_FILE = "market_reader_data.json"

# ------------------------------------------------------------------ helpers
def safe(d, key, default=np.nan):
    try:
        v = d.get(key, default)
        return default if v is None else v
    except Exception:
        return default

def clamp(x, lo=0, hi=100):
    return float(max(lo, min(hi, x)))

def pct_score(x, lo, hi):
    """Linear score 0-100 between lo (worst) and hi (best). Handles reversed ranges."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    if lo == hi:
        return 50.0
    return clamp((x - lo) / (hi - lo) * 100)

def nz(x, default=np.nan):
    try:
        return default if x is None or (isinstance(x, float) and np.isnan(x)) else float(x)
    except Exception:
        return default

# ------------------------------------------------------------------ saved data: watchlist + track record
# Permanent saving (free): add GITHUB_TOKEN (classic token, "gist" scope only) in the app's Secrets.
# Without it, data is kept in a local file that Streamlit wipes when the app sleeps or reboots.
def _secret(name):
    try:
        return st.secrets.get(name)
    except Exception:
        return None

def _gh_headers():
    tok = _secret("GITHUB_TOKEN")
    return {"Authorization": f"Bearer {tok}", "Accept": "application/vnd.github+json"} if tok else None

def _gist_load():
    h = _gh_headers()
    if not h:
        return None, None
    try:
        import requests
        gid = _secret("GIST_ID")
        if not gid:
            r = requests.get("https://api.github.com/gists?per_page=100", headers=h, timeout=10)
            gid = next((g["id"] for g in (r.json() if r.status_code == 200 else []) if g.get("description") == GIST_DESC), None)
        if not gid:
            return None, None
        r = requests.get(f"https://api.github.com/gists/{gid}", headers=h, timeout=10)
        if r.status_code != 200:
            return None, None
        return json.loads(r.json()["files"][GIST_FILE]["content"]), gid
    except Exception:
        return None, None

def _gist_save(data, gid):
    h = _gh_headers()
    if not h:
        return gid
    try:
        import requests
        files = {GIST_FILE: {"content": json.dumps(data, indent=1)}}
        if gid:
            requests.patch(f"https://api.github.com/gists/{gid}", headers=h, json={"files": files}, timeout=15)
            return gid
        r = requests.post("https://api.github.com/gists", headers=h,
                          json={"description": GIST_DESC, "public": False, "files": files}, timeout=15)
        return r.json().get("id") if r.status_code == 201 else None
    except Exception:
        return gid

@st.cache_resource(show_spinner=False)
def _store_holder():
    data, gid = _gist_load()
    if data is None:
        try:
            with open(DATA_FILE) as fh:
                data = json.load(fh)
        except Exception:
            data = {}
    data.setdefault("watchlist", list(DEFAULT_WATCHLIST))
    data.setdefault("calls", [])
    return {"data": data, "gist": gid}

def store():
    return _store_holder()["data"]

def storage_is_permanent():
    return bool(_secret("GITHUB_TOKEN"))

def save_store():
    h = _store_holder()
    try:
        with open(DATA_FILE, "w") as fh:
            json.dump(h["data"], fh)
    except Exception:
        pass
    h["gist"] = _gist_save(h["data"], h["gist"])

def clean_ticker(t):
    t = "".join(ch for ch in str(t).upper().strip() if ch.isalnum() or ch in ".-^=")
    return t[:12]

def add_to_watchlist(t):
    t = clean_ticker(t)
    s = store()
    if t and t not in s["watchlist"]:
        s["watchlist"].append(t)
        save_store()
    st.session_state["wl_text"] = ", ".join(s["watchlist"])

def remove_from_watchlist(t):
    s = store()
    if t in s["watchlist"]:
        s["watchlist"].remove(t)
        save_store()
    st.session_state["wl_text"] = ", ".join(s["watchlist"])

def save_watchlist_text():
    tickers = [clean_ticker(x) for x in st.session_state.get("wl_text", "").split(",")]
    store()["watchlist"] = list(dict.fromkeys(t for t in tickers if t))
    save_store()

def set_ticker(t):
    st.session_state["ticker_in"] = t

def call_direction(composite):
    return "LONG" if composite >= 58 else ("AVOID" if composite <= 35 else "NEUTRAL")

def log_call(ticker, bar_date, price, composite, verdict, conf, ml_p=None, horizon=None, regime_label="", source="viewed", save=True):
    """One call per ticker per trading day. Graded later against what the stock actually did."""
    s = store()
    bd = pd.Timestamp(bar_date).strftime("%Y-%m-%d")
    if any(c.get("bar_date") == bd and c.get("ticker") == ticker for c in s["calls"]):
        return False
    s["calls"].append(dict(bar_date=bd, logged=datetime.now().strftime("%Y-%m-%d %H:%M"), ticker=ticker,
                           price=round(float(price), 4), score=round(float(composite), 1), call=call_direction(composite),
                           verdict=verdict, confidence=conf, ml_p=None if ml_p is None else round(float(ml_p), 3),
                           horizon=horizon, regime=regime_label.split(" ")[0], source=source))
    if save:
        save_store()
    return True

@st.cache_data(ttl=1800, show_spinner=False)
def get_hist(ticker, period="5y"):
    try:
        h = yf.Ticker(ticker).history(period=period, auto_adjust=True)
        if h is None or h.empty:
            return pd.DataFrame()
        h = h[["Open", "High", "Low", "Close", "Volume"]].dropna()
        h.index = pd.to_datetime(h.index).tz_localize(None)
        return h
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=3600, show_spinner=False)
def get_info(ticker):
    try:
        return yf.Ticker(ticker).info or {}
    except Exception:
        return {}

@st.cache_data(ttl=3600, show_spinner=False)
def get_extras(ticker, with_options=True):
    t = yf.Ticker(ticker)
    out = {}
    for name in ["insider_transactions", "earnings_dates", "cashflow", "financials", "balance_sheet",
                 "upgrades_downgrades", "recommendations"]:
        try:
            out[name] = getattr(t, name)
        except Exception:
            out[name] = None
    if not with_options:
        out["options_exp"], out["chains"] = [], []
        return out
    try:
        exps = t.options
        out["options_exp"] = list(exps)[:3] if exps else []
        chains = []
        for e in out["options_exp"][:2]:
            try:
                c = t.option_chain(e)
                chains.append((e, c.calls, c.puts))
            except Exception:
                pass
        out["chains"] = chains
    except Exception:
        out["options_exp"], out["chains"] = [], []
    return out

# ------------------------------------------------------------------ technicals
def add_indicators(h):
    d = h.copy()
    c = d["Close"]
    for n in (10, 20, 50, 100, 200):
        d[f"SMA{n}"] = c.rolling(n).mean()
    delta = c.diff()
    up = delta.clip(lower=0).rolling(14).mean()
    dn = (-delta.clip(upper=0)).rolling(14).mean()
    d["RSI"] = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    ema12, ema26 = c.ewm(span=12).mean(), c.ewm(span=26).mean()
    d["MACD"] = ema12 - ema26
    d["MACDsig"] = d["MACD"].ewm(span=9).mean()
    tr = pd.concat([d["High"] - d["Low"], (d["High"] - c.shift()).abs(), (d["Low"] - c.shift()).abs()], axis=1).max(axis=1)
    d["ATR"] = tr.rolling(14).mean()
    d["ATRpct"] = d["ATR"] / c * 100
    d["Vol20"] = d["Volume"].rolling(20).mean()
    d["VolRatio"] = d["Volume"] / d["Vol20"]
    d["Ret1"] = c.pct_change()
    d["RealVol30"] = d["Ret1"].rolling(30).std() * np.sqrt(252) * 100
    d["High52"] = c.rolling(252).max()
    d["Low52"] = c.rolling(252).min()
    d["DistHigh52"] = (c / d["High52"] - 1) * 100
    # On-balance volume + accumulation/distribution
    d["OBV"] = (np.sign(c.diff()).fillna(0) * d["Volume"]).cumsum()
    mfm = ((c - d["Low"]) - (d["High"] - c)) / (d["High"] - d["Low"]).replace(0, np.nan)
    d["ADL"] = (mfm.fillna(0) * d["Volume"]).cumsum()
    d["BBmid"] = d["SMA20"]
    d["BBstd"] = c.rolling(20).std()
    d["BBpos"] = (c - (d["BBmid"] - 2 * d["BBstd"])) / (4 * d["BBstd"]).replace(0, np.nan)
    d["Mom12_1"] = c.shift(21) / c.shift(252) - 1
    d["Mom6"] = c / c.shift(126) - 1
    d["Mom3"] = c / c.shift(63) - 1
    d["Mom1"] = c / c.shift(21) - 1
    d["Drawdown"] = c / c.cummax() - 1
    return d

def slope(series, n=20):
    s = series.dropna().tail(n)
    if len(s) < 5:
        return np.nan
    x = np.arange(len(s))
    return np.polyfit(x, (s / s.iloc[0] if s.iloc[0] != 0 else s).values, 1)[0]

# ------------------------------------------------------------------ market regime
@st.cache_data(ttl=1800, show_spinner=False)
def market_regime():
    spy = get_hist("SPY", "3y")
    vix = get_hist("^VIX", "1y")
    tnx = get_hist("^TNX", "1y")   # 10y yield x10
    irx = get_hist("^IRX", "1y")   # 13w bill x10
    hyg = get_hist("HYG", "1y")
    lqd = get_hist("LQD", "1y")
    rsp = get_hist("RSP", "1y")    # equal-weight S&P (breadth proxy)
    r = {}
    if not spy.empty:
        s = add_indicators(spy)
        last = s.iloc[-1]
        r["spy_above_200"] = last["Close"] > last["SMA200"]
        r["spy_above_50"] = last["Close"] > last["SMA50"]
        r["sma50_above_200"] = last["SMA50"] > last["SMA200"]
        r["spy_1m"] = last["Mom1"] * 100
        r["spy_dd"] = last["Drawdown"] * 100
        r["spy_realvol"] = last["RealVol30"]
    r["vix"] = nz(vix["Close"].iloc[-1]) if not vix.empty else np.nan
    r["vix_20d_avg"] = nz(vix["Close"].tail(20).mean()) if not vix.empty else np.nan
    if not tnx.empty and not irx.empty:
        r["curve"] = nz(tnx["Close"].iloc[-1]) / 10 - nz(irx["Close"].iloc[-1]) / 10
    else:
        r["curve"] = np.nan
    if not hyg.empty and not lqd.empty:
        ratio = (hyg["Close"] / lqd["Close"]).dropna()
        r["credit_trend"] = slope(ratio, 40)
    else:
        r["credit_trend"] = np.nan
    if not rsp.empty and not spy.empty:
        rs = (rsp["Close"] / spy["Close"]).dropna()
        r["breadth_trend"] = slope(rs, 40)
    else:
        r["breadth_trend"] = np.nan

    score = 50
    if r.get("spy_above_200"): score += 15
    else: score -= 15
    if r.get("sma50_above_200"): score += 10
    else: score -= 10
    v = r["vix"]
    if not np.isnan(v):
        score += 10 if v < 16 else (0 if v < 22 else (-10 if v < 30 else -20))
    if not np.isnan(r["curve"]):
        score += 5 if r["curve"] > 0.5 else (-5 if r["curve"] < 0 else 0)
    if not np.isnan(r["credit_trend"]):
        score += 5 if r["credit_trend"] > 0 else -5
    if not np.isnan(r["breadth_trend"]):
        score += 5 if r["breadth_trend"] > 0 else -5
    r["score"] = clamp(score)
    if r["score"] >= 65: r["label"] = "RISK-ON (bull trend)"
    elif r["score"] >= 45: r["label"] = "NEUTRAL / CHOPPY"
    elif r["score"] >= 30: r["label"] = "CAUTION (weakening)"
    else: r["label"] = "RISK-OFF (bear / stress)"
    return r

# regime-dependent factor weights — the market decides what gets paid
def factor_weights(regime_label):
    if regime_label.startswith("RISK-ON"):
        return dict(Momentum=0.25, Trend=0.15, Growth=0.15, Quality=0.10, Value=0.05, Sentiment=0.15, SmartMoney=0.10, Risk=0.05)
    if regime_label.startswith("NEUTRAL"):
        return dict(Momentum=0.15, Trend=0.10, Growth=0.15, Quality=0.15, Value=0.15, Sentiment=0.10, SmartMoney=0.10, Risk=0.10)
    return dict(Momentum=0.05, Trend=0.10, Growth=0.10, Quality=0.25, Value=0.20, Sentiment=0.05, SmartMoney=0.10, Risk=0.15)

# ------------------------------------------------------------------ factor scoring
def factor_scores(d, info, extras, spy):
    last = d.iloc[-1]
    f, notes = {}, {}

    # Momentum (relative strength vs SPY matters more than raw)
    rel = np.nan
    if not spy.empty and len(spy) > 130:
        rel = (last["Close"] / d["Close"].iloc[-126] - 1) - (spy["Close"].iloc[-1] / spy["Close"].iloc[-126] - 1)
    m = np.nanmean([pct_score(nz(last["Mom12_1"]), -0.3, 0.6), pct_score(nz(last["Mom6"]), -0.25, 0.4),
                    pct_score(nz(last["Mom3"]), -0.2, 0.3), pct_score(rel, -0.2, 0.3)])
    f["Momentum"] = m
    notes["Momentum"] = f"12-1m {nz(last['Mom12_1'])*100:.0f}% | 6m {nz(last['Mom6'])*100:.0f}% | vs SPY 6m {nz(rel)*100:+.0f}%"

    # Trend
    t = 0
    for a, b in [("Close", "SMA50"), ("Close", "SMA200"), ("SMA50", "SMA200"), ("Close", "SMA20")]:
        t += 25 if nz(last[a]) > nz(last[b]) else 0
    t = t * 0.7 + pct_score(nz(last["DistHigh52"]), -40, 0) * 0.3
    f["Trend"] = t
    notes["Trend"] = f"{nz(last['DistHigh52']):.0f}% from 52w high | RSI {nz(last['RSI']):.0f}"

    # Value
    pe = nz(safe(info, "forwardPE", safe(info, "trailingPE")))
    ev = nz(safe(info, "enterpriseToEbitda"))
    fcf = nz(safe(info, "freeCashflow")); mcap = nz(safe(info, "marketCap"))
    fcfy = fcf / mcap * 100 if mcap and not np.isnan(fcf) and not np.isnan(mcap) and mcap > 0 else np.nan
    peg = nz(safe(info, "pegRatio"))
    f["Value"] = np.nanmean([pct_score(pe, 60, 8) if pe > 0 else 10, pct_score(ev, 40, 5) if ev > 0 else np.nan,
                             pct_score(fcfy, -2, 8), pct_score(peg, 4, 0.5) if peg > 0 else np.nan])
    notes["Value"] = f"Fwd P/E {pe:.1f} | EV/EBITDA {ev:.1f} | FCF yield {fcfy:.1f}% | PEG {peg:.2f}"

    # Quality
    roe = nz(safe(info, "returnOnEquity")) * 100
    gm = nz(safe(info, "grossMargins")) * 100
    om = nz(safe(info, "operatingMargins")) * 100
    de = nz(safe(info, "debtToEquity"))
    cr = nz(safe(info, "currentRatio"))
    f["Quality"] = np.nanmean([pct_score(roe, -10, 35), pct_score(gm, 10, 70), pct_score(om, -5, 35),
                               pct_score(de, 300, 0), pct_score(cr, 0.5, 2.5)])
    notes["Quality"] = f"ROE {roe:.0f}% | Gross margin {gm:.0f}% | Op margin {om:.0f}% | Debt/Eq {de:.0f}"

    # Growth
    rg = nz(safe(info, "revenueGrowth")) * 100
    eg = nz(safe(info, "earningsGrowth")) * 100
    eqg = nz(safe(info, "earningsQuarterlyGrowth")) * 100
    f["Growth"] = np.nanmean([pct_score(rg, -10, 40), pct_score(eg, -20, 60), pct_score(eqg, -20, 60)])
    notes["Growth"] = f"Revenue growth {rg:.0f}% | EPS growth {eg:.0f}% | Qtr EPS growth {eqg:.0f}%"

    # Sentiment / analysts / positioning
    rec = nz(safe(info, "recommendationMean"))  # 1 strong buy .. 5 sell
    tgt = nz(safe(info, "targetMeanPrice")); px = nz(last["Close"])
    upside = (tgt / px - 1) * 100 if not np.isnan(tgt) and px else np.nan
    si = nz(safe(info, "shortPercentOfFloat")) * 100
    f["Sentiment"] = np.nanmean([pct_score(rec, 4, 1.5), pct_score(upside, -20, 40),
                                 pct_score(si, 25, 2) if not np.isnan(si) else np.nan])
    notes["Sentiment"] = f"Analyst rec {rec:.2f}/5 | Target upside {upside:+.0f}% | Short % float {si:.1f}%"

    # Smart money: insiders + institutions + accumulation
    ins_score, ins_note = insider_read(extras.get("insider_transactions"))
    inst = nz(safe(info, "heldPercentInstitutions")) * 100
    adl_s = slope(d["ADL"], 40); px_s = slope(d["Close"], 40)
    accum = 70 if (not np.isnan(adl_s) and adl_s > 0 and nz(px_s) <= 0) else (60 if nz(adl_s) > 0 else 35)
    f["SmartMoney"] = np.nanmean([ins_score, pct_score(inst, 20, 85), accum])
    notes["SmartMoney"] = f"{ins_note} | Institutions {inst:.0f}%"

    # Risk (higher score = calmer / safer)
    beta = nz(safe(info, "beta"))
    f["Risk"] = np.nanmean([pct_score(nz(last["RealVol30"]), 80, 15), pct_score(beta, 2.5, 0.6),
                            pct_score(nz(last["Drawdown"]) * 100, -60, -5)])
    notes["Risk"] = f"Realized vol {nz(last['RealVol30']):.0f}% | Beta {beta:.2f} | Drawdown {nz(last['Drawdown'])*100:.0f}%"
    return f, notes

def insider_read(ins):
    """Cluster insider buying = one of the most reliable tells there is."""
    try:
        if ins is None or len(ins) == 0:
            return 50, "Insiders: no data"
        df = ins.copy()
        datecol = next((c for c in df.columns if "Date" in c), None)
        if datecol:
            df[datecol] = pd.to_datetime(df[datecol], errors="coerce")
            df = df[df[datecol] >= datetime.now() - timedelta(days=180)]
        text = df.get("Text", pd.Series([""] * len(df))).astype(str).str.lower()
        buys = df[text.str.contains("purchase|buy")]
        sells = df[text.str.contains("sale|sold")]
        bv = nz(buys.get("Value", pd.Series(dtype=float)).sum(), 0)
        sv = nz(sells.get("Value", pd.Series(dtype=float)).sum(), 0)
        nb, ns = len(buys), len(sells)
        note = f"Insiders 6m: {nb} buys (${bv/1e6:.1f}M) / {ns} sells (${sv/1e6:.1f}M)"
        if nb >= 3 and bv > sv:
            return 90, note + " — CLUSTER BUYING"
        if nb > 0 and bv > sv:
            return 70, note
        if ns > 5 and sv > 5 * max(bv, 1):
            return 25, note + " — heavy selling"
        return 50, note
    except Exception:
        return 50, "Insiders: unreadable"

# ------------------------------------------------------------------ read between the lines
def between_the_lines(d, info, extras, spy):
    """Divergences and tells that the headline numbers hide."""
    tells = []
    last = d.iloc[-1]

    # 1. Price vs volume divergence (accumulation while price flat/down = quiet buying)
    adl_s, px_s = slope(d["ADL"], 40), slope(d["Close"], 40)
    if not np.isnan(adl_s) and not np.isnan(px_s):
        if adl_s > 0 and px_s < 0:
            tells.append(("🟢", "Quiet accumulation", "Accumulation/Distribution rising while price falls — someone is buying the dip in size."))
        elif adl_s < 0 and px_s > 0:
            tells.append(("🔴", "Distribution into strength", "Price up but money flow down — rally is being sold into. Fragile."))

    # 2. Volume spike
    vr = nz(last["VolRatio"])
    if vr > 2:
        direction = "up" if nz(last["Ret1"]) > 0 else "down"
        tells.append(("🟡", f"Unusual volume ({vr:.1f}x avg)", f"Big players moved today ({direction} day). Follow-through in the next 1-3 sessions confirms or fails it."))

    # 3. RSI divergence
    c20, r20 = d["Close"].tail(40), d["RSI"].tail(40)
    if len(c20) == 40 and c20.iloc[-1] < c20.min() * 1.02 and r20.iloc[-1] > r20.min() + 5:
        tells.append(("🟢", "Bullish RSI divergence", "New price low but momentum is NOT making a new low — sellers are getting tired."))
    if len(c20) == 40 and c20.iloc[-1] > c20.max() * 0.98 and r20.iloc[-1] < r20.max() - 5:
        tells.append(("🔴", "Bearish RSI divergence", "New price high with weaker momentum — the move is losing fuel."))

    # 4. Short squeeze setup
    si = nz(safe(info, "shortPercentOfFloat")) * 100
    sr = nz(safe(info, "shortRatio"))
    if si > 15 and nz(last["Mom1"]) > 0.05:
        tells.append(("🟢", f"Squeeze fuel: {si:.0f}% of float short, {sr:.1f} days to cover", "Shorts are trapped if price keeps rising — squeezes are violent and fast."))
    elif si > 20:
        tells.append(("🟡", f"Heavy short interest ({si:.0f}%)", "Either the shorts know something, or this is a coiled spring. Check the fundamentals section."))

    # 5. Earnings quality: net income vs free cash flow (accruals tell)
    try:
        cf = extras.get("cashflow"); fin = extras.get("financials")
        fcf = float(cf.loc["Free Cash Flow"].iloc[0]); ni = float(fin.loc["Net Income"].iloc[0])
        if ni > 0 and fcf < 0.6 * ni:
            tells.append(("🔴", "Paper earnings", f"Net income ${ni/1e9:.2f}B but free cash flow only ${fcf/1e9:.2f}B — profits aren't converting to cash. Watch for write-downs."))
        elif fcf > 1.2 * ni and ni > 0:
            tells.append(("🟢", "Cash-rich earnings", f"FCF ${fcf/1e9:.2f}B beats net income ${ni/1e9:.2f}B — earnings are understated / high quality."))
    except Exception:
        pass

    # 6. Dilution vs buybacks
    try:
        bs = extras.get("balance_sheet")
        sh = bs.loc["Ordinary Shares Number"].dropna()
        if len(sh) >= 2:
            chg = (sh.iloc[0] / sh.iloc[1] - 1) * 100
            if chg < -1.5:
                tells.append(("🟢", f"Share count shrinking ({chg:.1f}%/yr)", "Buybacks — management thinks it's cheap and every share you own gets a bigger slice."))
            elif chg > 5:
                tells.append(("🔴", f"Share count growing ({chg:+.1f}%/yr)", "Dilution — they're paying bills with your ownership. Growth must outrun this."))
    except Exception:
        pass

    # 7. Analyst revisions
    try:
        ug = extras.get("upgrades_downgrades")
        if ug is not None and len(ug):
            ug = ug.copy(); ug.index = pd.to_datetime(ug.index, errors="coerce")
            recent = ug[ug.index >= datetime.now() - timedelta(days=60)]
            act = recent.get("Action", pd.Series(dtype=str)).astype(str).str.lower()
            ups, downs = int(act.str.contains("up").sum()), int(act.str.contains("down").sum())
            if ups >= 2 and ups > downs:
                tells.append(("🟢", f"{ups} upgrades / {downs} downgrades (60d)", "Sell-side is chasing — estimate revisions drive institutional flows for weeks."))
            elif downs >= 2 and downs > ups:
                tells.append(("🔴", f"{downs} downgrades / {ups} upgrades (60d)", "Sell-side is bailing — usually more downgrades follow the first ones."))
    except Exception:
        pass

    # 8. Earnings surprise streak
    try:
        ed = extras.get("earnings_dates")
        if ed is not None and "Surprise(%)" in ed.columns:
            s = ed["Surprise(%)"].dropna().head(6)
            if len(s) >= 4:
                beats = int((s > 0).sum())
                if beats == len(s):
                    tells.append(("🟢", f"Beat estimates {beats}/{len(s)} straight quarters", "Management sandbags guidance — the 'beat and raise' machine. Priced in partially, but streaks persist."))
                elif beats <= 1:
                    tells.append(("🔴", f"Missed {len(s)-beats} of last {len(s)} quarters", "Chronic misser — guidance is not trustworthy."))
    except Exception:
        pass

    # 9. Relative strength vs market on down days (hidden leadership)
    if not spy.empty:
        j = d[["Ret1"]].join(spy["Close"].pct_change().rename("SPY"), how="inner").tail(60)
        down = j[j["SPY"] < -0.005]
        if len(down) >= 5:
            avg = down["Ret1"].mean() - down["SPY"].mean()
            if avg > 0.004:
                tells.append(("🟢", "Holds up on red market days", f"Outperforms SPY by {avg*100:.2f}% on down days — institutions defend it. Leaders act like this."))
            elif avg < -0.006:
                tells.append(("🔴", "Gets hit hardest on red days", f"Underperforms SPY by {abs(avg)*100:.2f}% on down days — weak hands / high-beta dumping ground."))

    # 10. Implied vs realized vol (options market's fear vs reality)
    iv_note = options_read(extras, nz(last["Close"]), nz(last["RealVol30"]))
    if iv_note:
        tells.append(iv_note)

    # 11. Bollinger squeeze
    bw = (4 * d["BBstd"] / d["BBmid"]).dropna()
    if len(bw) > 120 and bw.iloc[-1] <= bw.tail(120).quantile(0.1):
        tells.append(("🟡", "Volatility squeeze", "Tightest range in 6 months — a big move is coming; direction usually follows the trend + volume break."))

    if not tells:
        tells.append(("⚪", "No strong tells", "Nothing hidden is screaming right now. Let the factor score and probabilities lead."))
    return tells

def options_read(extras, spot, realvol):
    try:
        chains = extras.get("chains") or []
        if not chains or np.isnan(spot):
            return None
        exp, calls, puts = chains[0]
        atm_c = calls.iloc[(calls["strike"] - spot).abs().argsort()[:1]]
        atm_p = puts.iloc[(puts["strike"] - spot).abs().argsort()[:1]]
        iv = np.nanmean([atm_c["impliedVolatility"].iloc[0], atm_p["impliedVolatility"].iloc[0]]) * 100
        pcr = nz(puts["volume"].sum()) / max(nz(calls["volume"].sum(), 1), 1)
        if np.isnan(iv) or np.isnan(realvol):
            return None
        prem = iv - realvol
        if prem > 15:
            return ("🟡", f"Options pricing fear: IV {iv:.0f}% vs realized {realvol:.0f}%",
                    f"Options are expensive (P/C {pcr:.2f}). Sell premium rather than buy it; big event likely priced in.")
        if prem < -5:
            return ("🟢", f"Options cheap: IV {iv:.0f}% vs realized {realvol:.0f}%",
                    f"Complacency — options are underpricing the actual movement (P/C {pcr:.2f}). Cheap insurance / cheap upside.")
        return ("⚪", f"Options fairly priced: IV {iv:.0f}% vs realized {realvol:.0f}%", f"Put/Call volume {pcr:.2f} ({'bearish lean' if pcr > 1.2 else 'bullish lean' if pcr < 0.6 else 'balanced'}).")
    except Exception:
        return None

# ------------------------------------------------------------------ composite, verdict, confidence, catalysts
def composite_of(sc, w):
    keys = [k for k in w if not np.isnan(nz(sc.get(k)))]
    den = sum(w[k] for k in keys)
    return float(sum(sc[k] * w[k] for k in keys) / den) if den else 50.0

def verdict_of(composite, green, red, regime_label):
    v = "STRONG SETUP — long bias" if composite >= 70 and green >= red else \
        "LEAN LONG" if composite >= 58 else \
        "AVOID / SHORT BIAS" if composite <= 35 else "NEUTRAL — wait for a better price or a catalyst"
    if regime_label.startswith("RISK-OFF") and composite >= 58:
        v += " (but the MARKET is risk-off — size smaller, tighter stops)"
    return v

PLAIN = dict(Momentum="price momentum", Trend="the price trend", Growth="sales & earnings growth",
             Quality="business quality (margins, debt)", Value="valuation (how cheap it is)",
             Sentiment="analyst & short-seller sentiment", SmartMoney="insider & big-fund buying", Risk="volatility / risk")

def confidence_read(composite, scores, weights, tells, regime_label, ml=None, catalysts=None):
    """High / Med / Low + the 2-3 plain-English reasons behind it."""
    long_side = composite >= 50
    lean = "bullish" if long_side else "bearish"
    pts, reasons = 0, []
    dist = abs(composite - 50)
    if dist >= 20:
        pts += 2; reasons.append((3.0, f"Score {composite:.0f}/100 is far from neutral — the factors clearly lean {lean}"))
    elif dist >= 10:
        pts += 1; reasons.append((2.0, f"Score {composite:.0f}/100 leans {lean}, but not by a lot"))
    else:
        reasons.append((3.0, f"Score {composite:.0f}/100 sits near the middle — no clear edge either way"))
    contrib = sorted(((nz(scores[k]) - 50) * weights[k], k) for k in weights if not np.isnan(nz(scores.get(k))))
    if contrib:
        if contrib[-1][0] > 3:
            k = contrib[-1][1]; reasons.append((2.2, f"Biggest plus: {PLAIN[k]} ({scores[k]:.0f}/100)"))
        if contrib[0][0] < -3:
            k = contrib[0][1]; reasons.append((2.1, f"Biggest drag: {PLAIN[k]} ({scores[k]:.0f}/100)"))
    green = sum(1 for t in tells if t[0] == "🟢"); red = sum(1 for t in tells if t[0] == "🔴")
    aligned = (green - red) if long_side else (red - green)
    if aligned >= 2:
        pts += 1; reasons.append((2.5, f"Hidden tells agree ({green} bullish vs {red} bearish)"))
    elif aligned <= -2:
        pts -= 1; reasons.append((2.5, f"Hidden tells disagree ({green} bullish vs {red} bearish)"))
    if ml:
        edge = ml["test_acc"] - max(ml["base_rate"], 1 - ml["base_rate"])
        p = ml["p_now"]
        if edge > 0.02 and abs(p - 0.5) >= 0.08:
            if (p > 0.5) == long_side:
                pts += 1; reasons.append((2.6, f"ML model agrees ({p*100:.0f}% chance up) and has beaten guessing on this stock"))
            else:
                pts -= 1; reasons.append((2.6, f"ML model disagrees ({p*100:.0f}% chance up)"))
    if long_side and regime_label.startswith("RISK-OFF"):
        pts -= 1; reasons.append((2.4, "Market is risk-off — a headwind for buying"))
    elif long_side and regime_label.startswith("RISK-ON") and composite >= 58:
        pts += 1; reasons.append((1.0, "Market is risk-on — tailwind"))
    elif not long_side and regime_label.startswith(("RISK-OFF", "CAUTION")):
        pts += 1; reasons.append((1.0, "Weak market backs up the bearish read"))
    if catalysts and any(c["kind"] == "earnings" for c in catalysts):
        pts -= 1; reasons.append((2.8, "Earnings within 7 days — the report can override everything above"))
    level = "High" if pts >= 3 else ("Med" if pts >= 1 else "Low")
    if dist < 8:
        level = "Low"
    reasons.sort(key=lambda r: -r[0])
    return level, [r[1] for r in reasons[:3]]

CONF_ICON = {"High": "🟢", "Med": "🟡", "Low": "⚪"}

def next_earnings(info, extras):
    now = datetime.now()
    try:
        e = extras.get("earnings_dates")
        idx = pd.to_datetime(e.index)
        idx = idx.tz_localize(None) if idx.tz is not None else idx
        fut = [x for x in idx if x >= now - timedelta(hours=12)]
        if fut:
            return min(fut)
    except Exception:
        pass
    for k in ("earningsTimestampStart", "earningsTimestamp"):
        ts = safe(info, k, None)
        try:
            if ts:
                dt = datetime.utcfromtimestamp(int(ts))
                if dt >= now - timedelta(hours=12):
                    return dt
        except Exception:
            pass
    return None

@st.cache_data(ttl=1800, show_spinner=False)
def get_news(ticker):
    out = []
    try:
        raw = yf.Ticker(ticker).news or []
    except Exception:
        raw = []
    for n in raw[:25]:
        try:
            c = n.get("content", n) if isinstance(n, dict) else {}
            title = c.get("title") or n.get("title")
            ts = c.get("pubDate") or c.get("displayTime") or n.get("providerPublishTime")
            t = datetime.utcfromtimestamp(ts) if isinstance(ts, (int, float)) else pd.to_datetime(ts, utc=True).tz_localize(None).to_pydatetime()
            link = (c.get("canonicalUrl") or {}).get("url") or (c.get("clickThroughUrl") or {}).get("url") or n.get("link")
            if title:
                out.append((t, title, link))
        except Exception:
            continue
    out.sort(key=lambda z: z[0], reverse=True)
    return out

def catalyst_flags(ticker, info, extras, days=7, with_news=True):
    """Events in the next 7 days that can make any model's read unreliable."""
    flags, today = [], datetime.now().date()
    nxt = next_earnings(info, extras)
    if nxt is not None:
        dd = (nxt.date() - today).days
        if 0 <= dd <= days:
            when = "TODAY" if dd == 0 else ("tomorrow" if dd == 1 else f"in {dd} days")
            flags.append(dict(kind="earnings", icon="📅", text=f"Earnings {when} ({nxt:%a %b %d}) — expect a big move either way"))
    exd = safe(info, "exDividendDate", None)
    try:
        if exd:
            xd = datetime.utcfromtimestamp(int(exd)).date()
            dd = (xd - today).days
            if 0 <= dd <= days:
                flags.append(dict(kind="dividend", icon="💵", text=f"Ex-dividend {xd:%a %b %d} — price drops by the dividend that morning"))
    except Exception:
        pass
    if with_news:
        fresh = [n for n in get_news(ticker) if n[0] >= datetime.utcnow() - timedelta(hours=48)]
        if len(fresh) >= 1:
            flags.append(dict(kind="news", icon="📰", text=f"{len(fresh)} headline(s) in the last 48h — latest: “{fresh[0][1]}”", link=fresh[0][2]))
    return flags

# ------------------------------------------------------------------ ML probability engine
def ml_features(d, spy):
    x = pd.DataFrame(index=d.index)
    c = d["Close"]
    x["rsi"] = d["RSI"]
    x["macd"] = d["MACD"] - d["MACDsig"]
    x["dist20"] = c / d["SMA20"] - 1
    x["dist50"] = c / d["SMA50"] - 1
    x["dist200"] = c / d["SMA200"] - 1
    x["mom1"] = d["Mom1"]; x["mom3"] = d["Mom3"]; x["mom6"] = d["Mom6"]
    x["volratio"] = d["VolRatio"]
    x["atrpct"] = d["ATRpct"]
    x["bbpos"] = d["BBpos"]
    x["dd"] = d["Drawdown"]
    x["dh52"] = d["DistHigh52"]
    x["adl_slope"] = d["ADL"].rolling(20).apply(lambda s: np.polyfit(np.arange(len(s)), s, 1)[0] / (abs(s.mean()) + 1e-9), raw=True)
    if not spy.empty:
        sp = spy["Close"].reindex(d.index).ffill()
        x["rel1m"] = c / c.shift(21) - sp / sp.shift(21)
        x["spy_dist200"] = sp / sp.rolling(200).mean() - 1
    return x

@st.cache_data(ttl=3600, show_spinner=False)
def train_ml(ticker, horizon=20):
    d = add_indicators(get_hist(ticker, "10y"))
    spy = get_hist("SPY", "10y")
    if len(d) < 600:
        return None
    X = ml_features(d, spy)
    y = (d["Close"].shift(-horizon) / d["Close"] - 1)
    df = X.join(y.rename("fwd")).dropna()
    df["label"] = (df["fwd"] > 0).astype(int)
    from sklearn.ensemble import GradientBoostingClassifier
    feats = [c for c in df.columns if c not in ("fwd", "label")]
    # walk-forward: train on everything up to 1 year ago, test on the last year (honest, no peeking)
    split = len(df) - 250
    Xtr, ytr = df[feats].iloc[:split], df["label"].iloc[:split]
    Xte, yte = df[feats].iloc[split:], df["label"].iloc[split:]
    model = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=7)
    model.fit(Xtr, ytr)
    p_te = model.predict_proba(Xte)[:, 1]
    acc = float(((p_te > 0.5).astype(int) == yte).mean())
    base = float(yte.mean())
    conf_mask = (p_te > 0.6) | (p_te < 0.4)
    conf_acc = float(((p_te[conf_mask] > 0.5).astype(int) == yte[conf_mask]).mean()) if conf_mask.sum() > 10 else np.nan
    # retrain on all data for today's call
    model.fit(df[feats], df["label"])
    X_now = X[feats].dropna().iloc[[-1]]
    p_now = float(model.predict_proba(X_now)[:, 1][0])
    imp = pd.Series(model.feature_importances_, index=feats).sort_values(ascending=False)
    return dict(p_now=p_now, test_acc=acc, base_rate=base, conf_acc=conf_acc, n_conf=int(conf_mask.sum()),
                importance=imp, horizon=horizon, n_train=len(df))

def monte_carlo(d, days=63, sims=5000, seed=7):
    r = d["Ret1"].dropna().tail(750).values
    if len(r) < 100:
        return None
    rng = np.random.default_rng(seed)
    # bootstrap real returns (keeps fat tails) — no bell-curve fantasy
    paths = np.cumprod(1 + rng.choice(r, size=(sims, days), replace=True), axis=1)
    spot = d["Close"].iloc[-1]
    return spot * paths

# ------------------------------------------------------------------ backtest of the composite score
def quick_backtest(d, spy):
    """Does 'trend + momentum + accumulation' actually beat buy-and-hold on THIS name? Honest check."""
    x = d.dropna(subset=["SMA200", "SMA50", "Mom6"]).copy()
    if len(x) < 300:
        return None
    sig = ((x["Close"] > x["SMA200"]) & (x["SMA50"] > x["SMA200"]) & (x["Mom6"] > 0)).astype(int).shift(1).fillna(0)
    strat = (1 + x["Ret1"] * sig).cumprod()
    bh = (1 + x["Ret1"]).cumprod()
    yrs = len(x) / 252
    def cagr(s): return (s.iloc[-1] ** (1 / yrs) - 1) * 100
    def mdd(s): return ((s / s.cummax()) - 1).min() * 100
    return dict(strat=strat, bh=bh, strat_cagr=cagr(strat), bh_cagr=cagr(bh), strat_mdd=mdd(strat), bh_mdd=mdd(bh),
                time_in=float(sig.mean() * 100))

# ------------------------------------------------------------------ watchlist screen (cached, light — no options chains)
@st.cache_data(ttl=3 * 3600, show_spinner=False)
def quick_read(tk, regime_label):
    try:
        h = get_hist(tk, "3y")
        if h.empty:
            return None
        dd = add_indicators(h)
        inf = get_info(tk); ex = get_extras(tk, with_options=False)
        spy_ = get_hist("SPY", "5y")
        w = factor_weights(regime_label)
        sc, _ = factor_scores(dd, inf, ex, spy_)
        comp = composite_of(sc, w)
        tl = between_the_lines(dd, inf, ex, spy_)
        g = sum(1 for t in tl if t[0] == "🟢"); r = sum(1 for t in tl if t[0] == "🔴")
        cats = catalyst_flags(tk, inf, ex, with_news=False)
        conf, reasons = confidence_read(comp, sc, w, tl, regime_label, None, cats)
        return dict(Ticker=tk, Score=round(comp), Call=call_direction(comp), Verdict=verdict_of(comp, g, r, regime_label),
                    Confidence=conf, Reasons=reasons, Catalyst=" · ".join(f"{c['icon']} {c['text'].split(' — ')[0]}" for c in cats),
                    Momentum=round(nz(sc["Momentum"], 50)), Quality=round(nz(sc["Quality"], 50)), Value=round(nz(sc["Value"], 50)),
                    Growth=round(nz(sc["Growth"], 50)), SmartMoney=round(nz(sc["SmartMoney"], 50)), Bull_tells=g, Bear_tells=r,
                    Price=round(float(dd["Close"].iloc[-1]), 2), From52wHigh=f"{nz(dd['DistHigh52'].iloc[-1]):.0f}%",
                    bar_date=str(dd.index[-1].date()))
    except Exception:
        return None

# ------------------------------------------------------------------ track record grading
@st.cache_data(ttl=1800, show_spinner=False)
def grade_calls(calls_json):
    calls = json.loads(calls_json)
    spy_ = get_hist("SPY", "2y")
    rows = []
    for c in calls:
        row = dict(c)
        for n in (1, 5, 20):
            row[f"ret{n}"] = np.nan; row[f"spy{n}"] = np.nan
        h = get_hist(c["ticker"], "2y")
        anchor = pd.Timestamp(c["bar_date"])
        if not h.empty:
            i = h.index.searchsorted(anchor)
            if i < len(h) and h.index[i].date() == anchor.date():
                base = h["Close"].iloc[i]
                for n in (1, 5, 20):
                    if i + n < len(h):
                        row[f"ret{n}"] = (h["Close"].iloc[i + n] / base - 1) * 100
            if not spy_.empty:
                j = spy_.index.searchsorted(anchor)
                if j < len(spy_) and spy_.index[j].date() == anchor.date():
                    for n in (1, 5, 20):
                        if j + n < len(spy_):
                            row[f"spy{n}"] = (spy_["Close"].iloc[j + n] / spy_["Close"].iloc[j] - 1) * 100
        rows.append(row)
    return pd.DataFrame(rows)

def win_of(call, ret):
    if ret is None or np.isnan(ret) or call == "NEUTRAL":
        return np.nan
    return 1.0 if (ret > 0) == (call == "LONG") else 0.0

# ------------------------------------------------------------------ Drag Shot (runs in the page; $0 — no paid AI)
DS_JS = r"""
(function(){
var W=window, D=document;
function ctx(){ try{ return JSON.parse(W.__dsCtx||'{}'); }catch(e){ return {}; } }
var css=D.createElement('style');
css.textContent='#ds-btn{position:fixed;right:16px;bottom:52px;z-index:100000;padding:7px 13px;border-radius:999px;border:1px solid rgba(255,255,255,.25);background:#1f6feb;color:#fff;font:600 13px system-ui,sans-serif;cursor:pointer;box-shadow:0 2px 10px rgba(0,0,0,.35)}'+
'#ds-btn:hover{background:#388bfd}'+
'#ds-ov{position:fixed;inset:0;z-index:100001;cursor:crosshair;background:rgba(0,0,0,.12);display:none}'+
'#ds-box{position:fixed;border:2px dashed #1f6feb;background:rgba(31,111,235,.12);z-index:100002;display:none;pointer-events:none}'+
'#ds-hint{position:fixed;top:10px;left:50%;transform:translateX(-50%);z-index:100003;font:12px system-ui,sans-serif;background:#111;color:#fff;padding:4px 10px;border-radius:6px;display:none}'+
'#ds-pop{position:fixed;right:16px;bottom:96px;width:min(430px,calc(100vw - 32px));max-height:72vh;overflow:auto;z-index:100004;background:#fff;color:#111;border-radius:12px;box-shadow:0 10px 32px rgba(0,0,0,.4);font:13px system-ui,sans-serif;display:none;padding:12px;box-sizing:border-box}'+
'@media (prefers-color-scheme:dark){#ds-pop{background:#1d1f27;color:#e8e8ee}}'+
'#ds-pop textarea{width:100%;height:110px;box-sizing:border-box;font:12px ui-monospace,Consolas,monospace;border:1px solid rgba(128,128,128,.45);border-radius:8px;padding:6px;background:transparent;color:inherit;resize:vertical}'+
'#ds-pop img{max-width:100%;max-height:120px;border-radius:6px;margin-top:6px;border:1px solid rgba(128,128,128,.3)}'+
'.ds-h{display:flex;justify-content:space-between;align-items:center;font-weight:700;margin-bottom:4px}'+
'.ds-s{font-size:11px;opacity:.65;margin:2px 0 6px}'+
'.ds-bar{display:flex;gap:6px;margin:8px 0 4px}'+
'.ds-bar input{flex:1;min-width:0;padding:8px 10px;border-radius:8px;border:1px solid rgba(128,128,128,.5);font:13px system-ui,sans-serif;background:transparent;color:inherit}'+
'.ds-bar button,.ds-chip{padding:6px 10px;border-radius:999px;border:1px solid rgba(128,128,128,.45);background:transparent;color:inherit;cursor:pointer;font:12px system-ui,sans-serif}'+
'.ds-bar button.go{background:#1f6feb;color:#fff;border-color:#1f6feb}'+
'.ds-chips{display:flex;flex-wrap:wrap;gap:6px;margin-top:6px}'+
'.ds-chip:hover{background:rgba(31,111,235,.15)}'+
'.ds-x{border:none;background:transparent;color:inherit;font-size:18px;cursor:pointer;line-height:1}'+
'.ds-note{border-top:1px solid rgba(128,128,128,.25);padding:6px 0;font-size:12px;white-space:pre-wrap}'+
'.ds-toast{position:fixed;left:50%;transform:translateX(-50%);top:14px;z-index:100005;background:#111;color:#fff;font:12px system-ui,sans-serif;padding:6px 10px;border-radius:6px}';
D.head.appendChild(css);
function el(tag,attrs,txt){ var e=D.createElement(tag); for(var k in (attrs||{})) e.setAttribute(k,attrs[k]); if(txt!=null) e.textContent=txt; return e; }
var btn=el('button',{id:'ds-btn',title:'Drag a box over anything on screen (or hold Ctrl and drag)'},'✂ Drag Shot');
var ov=el('div',{id:'ds-ov'}), box=el('div',{id:'ds-box'}), hint=el('div',{id:'ds-hint'},'Drag a box over what you want · Esc to cancel'), pop=el('div',{id:'ds-pop'});
D.body.appendChild(btn); D.body.appendChild(ov); D.body.appendChild(box); D.body.appendChild(hint); D.body.appendChild(pop);
var start=null, shot={text:'',img:null};
function toast(m){ var t=el('div',{'class':'ds-toast'},m); D.body.appendChild(t); setTimeout(function(){t.remove();},1800); }
function arm(){ ov.style.display='block'; hint.style.display='block'; }
function disarm(){ ov.style.display='none'; box.style.display='none'; hint.style.display='none'; start=null; }
function rectFrom(e){ var x=Math.min(start.x,e.clientX), y=Math.min(start.y,e.clientY); return {x:x,y:y,w:Math.abs(e.clientX-start.x),h:Math.abs(e.clientY-start.y)}; }
function draw(e){ var r=rectFrom(e); box.style.left=r.x+'px'; box.style.top=r.y+'px'; box.style.width=r.w+'px'; box.style.height=r.h+'px'; }
btn.addEventListener('click',function(e){ e.stopPropagation(); arm(); });
ov.addEventListener('mousedown',function(e){ start={x:e.clientX,y:e.clientY}; draw(e); box.style.display='block'; e.preventDefault(); });
D.addEventListener('mousedown',function(e){
  if(e.ctrlKey && e.button===0 && !start && !pop.contains(e.target) && e.target!==btn){
    e.preventDefault(); e.stopPropagation(); arm(); start={x:e.clientX,y:e.clientY}; draw(e); box.style.display='block';
  }
},true);
D.addEventListener('mousemove',function(e){ if(start) draw(e); },true);
D.addEventListener('mouseup',function(e){ if(!start) return; var r=rectFrom(e); disarm(); if(r.w<8||r.h<8) return; capture(r); },true);
D.addEventListener('keydown',function(e){ if(e.key==='Escape'){ disarm(); pop.style.display='none'; } });
function textIn(r){
  var items=[], wk=D.createTreeWalker(D.body,NodeFilter.SHOW_TEXT,null), n;
  while((n=wk.nextNode())){
    var t=n.nodeValue; if(!t||!t.trim()) continue;
    var p=n.parentElement; if(!p||p.closest('#ds-pop,#ds-btn,#ds-hint,script,style,noscript')) continue;
    var rg=D.createRange(); rg.selectNodeContents(n); var rs=rg.getClientRects();
    for(var i=0;i<rs.length;i++){ var b=rs[i]; if(!b.width||!b.height) continue; var cx=b.left+b.width/2, cy=b.top+b.height/2;
      if(cx>=r.x&&cx<=r.x+r.w&&cy>=r.y&&cy<=r.y+r.h){ items.push({x:b.left,y:b.top,t:t.trim()}); break; } }
  }
  items.sort(function(a,b){ return Math.abs(a.y-b.y)<6 ? a.x-b.x : a.y-b.y; });
  var lines=[], lastY=null;
  items.forEach(function(o){ if(lastY===null||Math.abs(o.y-lastY)>=6){ lines.push([o.t]); lastY=o.y; } else lines[lines.length-1].push(o.t); });
  return lines.map(function(l){ return l.join('  '); }).join('\n');
}
var STOP={};('A I AN THE AND OR OF TO IN ON AT BY FOR UP VS NO OK ATR RSI MACD SMA EMA EPS FCF ROE PE EV EBITDA PEG IV OI ML CSV USD ETF VIX CAGR ATM BB AD LONG SHORT AVOID NEUTRAL RISK OFF HIGH MED LOW BUY SELL YTD AI CEO US AM PM TODAY NEW MAX MIN P C SPX NYSE OTC SEC FOMC CPI GDP IPO API UI TOP ALL OUT ABOVE BELOW BIAS LEAN SETUP STRONG WAIT MARKET SCORE NOT THIS OBV ADL JSON PNG PDF DAYS DAY WEEK CALL CALLS PUT PUTS RIGHT WRONG YES GO NOW KEY').split(' ').forEach(function(w){STOP[w]=1;});
function tickers(txt){
  var c=ctx(), known={}; (c.watchlist||[]).concat(c.ticker?[c.ticker]:[]).forEach(function(t){known[t]=1;});
  var found=[], re=/(\$?)\b([A-Z]{1,5}(?:[.\-][A-Z])?)\b/g, m;
  while((m=re.exec(txt))){ var t=m[2]; if(found.indexOf(t)>=0) continue; if(m[1]||known[t]||(t.length>=2&&!STOP[t])) found.push(t); }
  found.sort(function(a,b){ return (known[b]?1:0)-(known[a]?1:0); });
  return found.slice(0,6);
}
function go(q){ W.location.search='?'+q; }
function notes(){ try{ return JSON.parse(W.localStorage.getItem('mr_ds_notes')||'[]'); }catch(e){ return []; } }
function setNotes(a){ try{ W.localStorage.setItem('mr_ds_notes',JSON.stringify(a)); }catch(e){} }
function copy(t){ try{ navigator.clipboard.writeText(t).then(function(){toast('Copied');},fallback); }catch(e){ fallback(); }
  function fallback(){ var ta=el('textarea'); ta.value=t; D.body.appendChild(ta); ta.select(); try{D.execCommand('copy'); toast('Copied');}catch(e){} ta.remove(); } }
function askClaude(cmd,txt){ var q=(cmd?cmd+'\n\n':'')+'Context from my Market Reader stock app:\n'+txt; W.open('https://claude.ai/new?q='+encodeURIComponent(q.slice(0,3500)),'_blank'); }
function loadH2C(cb){ if(W.html2canvas) return cb(); var s=D.createElement('script'); s.src='https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js'; s.onload=cb; s.onerror=function(){}; D.head.appendChild(s); }
function capture(r){
  shot={text:textIn(r),img:null}; render();
  loadH2C(function(){ try{ W.html2canvas(D.body,{x:r.x+W.scrollX,y:r.y+W.scrollY,width:r.w,height:r.h,logging:false,useCORS:true,
    ignoreElements:function(e){ return e.id&&e.id.indexOf('ds-')===0; }}).then(function(cv){ shot.img=cv.toDataURL('image/png'); render(true); }); }catch(e){} });
}
function render(keep){
  var ta=pop.querySelector('textarea'); if(keep&&ta) shot.text=ta.value;
  var cmdVal=keep&&pop.querySelector('.ds-bar input')?pop.querySelector('.ds-bar input').value:'';
  pop.innerHTML=''; pop.style.display='block';
  var h=el('div',{'class':'ds-h'}); h.appendChild(el('span',null,'✂ Drag Shot')); var x=el('button',{'class':'ds-x',title:'Close'},'×'); x.onclick=function(){pop.style.display='none';}; h.appendChild(x); pop.appendChild(h);
  pop.appendChild(el('div',{'class':'ds-s'},'Edit the text, then pick where it goes.'));
  var t=el('textarea'); t.value=shot.text||'(No text found in that box — the image below still has it.)'; pop.appendChild(t);
  if(shot.img){ var im=el('img',{src:shot.img,alt:'capture'}); pop.appendChild(im); }
  var bar=el('div',{'class':'ds-bar'}), inp=el('input',{placeholder:'Ticker, “watch TSLA”, “note”, or ask anything'}); inp.value=cmdVal;
  var SR=W.SpeechRecognition||W.webkitSpeechRecognition;
  if(SR){ var mic=el('button',{title:'Speak'},'🎤'); mic.onclick=function(){ try{ var rec=new SR(); rec.lang='en-US'; rec.onresult=function(ev){ inp.value=ev.results[0][0].transcript; inp.focus(); }; rec.start(); toast('Listening…'); }catch(e){} }; bar.appendChild(mic); }
  var g=el('button',{'class':'go'},'Go'); bar.appendChild(inp); bar.appendChild(g); pop.appendChild(bar);
  function run(){ var v=inp.value.trim(), txt=t.value, m;
    if(!v){ return; }
    if((m=v.match(/^(?:watch|add)\s+\$?([A-Za-z][A-Za-z.\-]{0,6})$/i))) return go('watch='+encodeURIComponent(m[1].toUpperCase()));
    if((m=v.match(/^note\b\s*([\s\S]*)$/i))){ var a=notes(); a.unshift({ts:new Date().toLocaleString(),text:(m[1]?m[1]+'\n':'')+txt}); setNotes(a); toast('Saved to Notes'); return render(true); }
    if(/^copy$/i.test(v)) return copy(txt);
    if((m=v.match(/^(?:analy[sz]e|open|show|read)?\s*\$?([A-Za-z]{1,5}(?:[.\-][A-Za-z])?)$/i)) && !/\s/.test(v.replace(/^(analy[sz]e|open|show|read)\s+/i,''))) return go('ticker='+encodeURIComponent(m[1].toUpperCase()));
    askClaude(v,txt);
  }
  g.onclick=run; inp.addEventListener('keydown',function(e){ if(e.key==='Enter') run(); });
  var ch=el('div',{'class':'ds-chips'});
  function chip(label,fn,title){ var b=el('button',{'class':'ds-chip',title:title||label},label); b.onclick=fn; ch.appendChild(b); }
  tickers(t.value).forEach(function(tk){ chip('📈 Analyze '+tk,function(){go('ticker='+encodeURIComponent(tk));}); chip('⭐ Watch '+tk,function(){go('watch='+encodeURIComponent(tk));}); });
  chip('📝 Save note',function(){ var a=notes(); a.unshift({ts:new Date().toLocaleString(),text:t.value}); setNotes(a); toast('Saved to Notes'); render(true); });
  chip('📋 Copy text',function(){ copy(t.value); });
  if(shot.img) chip('🖼 Save image',function(){ var a=el('a',{href:shot.img,download:'drag-shot.png'}); D.body.appendChild(a); a.click(); a.remove(); });
  chip('✨ Ask Claude',function(){ askClaude(inp.value.trim(),t.value); },'Opens Claude with this text');
  var ns=notes(); chip('📒 Notes ('+ns.length+')',function(){ nb.style.display=nb.style.display==='none'?'block':'none'; });
  pop.appendChild(ch);
  var nb=el('div',{style:'display:none;margin-top:8px'});
  if(!ns.length) nb.appendChild(el('div',{'class':'ds-s'},'No notes yet.'));
  ns.forEach(function(n,i){ var row=el('div',{'class':'ds-note'}); row.appendChild(el('div',{'class':'ds-s'},n.ts)); row.appendChild(el('div',null,n.text));
    var cp=el('button',{'class':'ds-chip'},'Copy'); cp.onclick=function(){copy(n.text);}; var dl=el('button',{'class':'ds-chip'},'Delete'); dl.onclick=function(){ var a=notes(); a.splice(i,1); setNotes(a); render(true); };
    row.appendChild(cp); row.appendChild(dl); nb.appendChild(row); });
  pop.appendChild(nb);
}
})();
"""

def drag_shot(ctx):
    payload = json.dumps(json.dumps(ctx))
    code = json.dumps(DS_JS).replace("</", "<\\/")
    components.html(
        "<script>(function(){var P=window.parent;try{P.__dsCtx=" + payload + ";"
        "if(!P.document.getElementById('ds-script')){var s=P.document.createElement('script');s.id='ds-script';"
        "s.textContent=" + code + ";P.document.body.appendChild(s);}}catch(e){}})();</script>",
        height=0)

# ------------------------------------------------------------------ UI
st.markdown("""<style>
.mr-footer{position:fixed;left:0;right:0;bottom:0;z-index:999;padding:3px 12px;font-size:11px;line-height:1.4;text-align:center;
  background:rgba(128,128,128,.14);backdrop-filter:blur(6px);-webkit-backdrop-filter:blur(6px);opacity:.9}
.block-container{padding-bottom:3.2rem}
.mr-why{font-size:0.9rem;margin:0.1rem 0 0.6rem 0}
</style>
<div class="mr-footer">⚠️ Not financial advice · For research & education only · Data delayed (Yahoo Finance) · Past results don't guarantee future results</div>""",
            unsafe_allow_html=True)

# links from Drag Shot / shared URLs: ?ticker=NVDA opens a stock, ?watch=TSLA adds it to the watchlist
_qp = st.query_params
if "ticker" in _qp:
    st.session_state["ticker_in"] = clean_ticker(_qp["ticker"])
    del st.query_params["ticker"]
if "watch" in _qp:
    _w = clean_ticker(_qp["watch"])
    del st.query_params["watch"]
    if _w:
        add_to_watchlist(_w)
        st.toast(f"⭐ {_w} added to your watchlist")
st.session_state.setdefault("ticker_in", "AAPL")
st.session_state.setdefault("wl_text", ", ".join(store()["watchlist"]))

st.title("📈 Market Reader")
st.caption("Reads between the lines: regime → factors → hidden tells → probabilities → honest backtest. Free data, no guarantees, stacked edges.")

with st.sidebar:
    st.header("Controls")
    ticker = clean_ticker(st.text_input("Ticker", key="ticker_in"))
    horizon = st.selectbox("Prediction horizon (trading days)", [5, 10, 20, 63], index=2)
    run_ml = st.checkbox("Run ML probability engine (slower)", True)
    st.divider()
    st.subheader("⭐ Your watchlist")
    st.text_area("Tickers (comma-separated)", key="wl_text", height=110)
    st.button("💾 Save watchlist", on_click=save_watchlist_text, use_container_width=True)
    st.caption("Saved permanently ☁️" if storage_is_permanent() else "Saved until the app sleeps — see Track Record tab to make it permanent")
    st.divider()
    st.subheader("Position sizing")
    account = st.number_input("Account size ($)", 1000, 10_000_000, 10000, step=500)
    risk_pct = st.slider("Risk per trade (%)", 0.25, 5.0, 1.0, 0.25)

regime = market_regime()
weights = factor_weights(regime["label"])

tab_today, tab_over, tab_lines, tab_prob, tab_earn, tab_track, tab_bt, tab_screen, tab_regime = st.tabs(
    ["Today", "Verdict", "Between the Lines", "Probability Engine", "Earnings & Options", "Track Record", "Backtest", "Screener", "Market Regime"])

# ---------------- Today: top 5 signals from the watchlist (auto-logged to the track record)
watchlist = store()["watchlist"]
with tab_today:
    st.subheader(f"Today's Top 5 — from your {len(watchlist)}-stock watchlist")
    results = []
    todo = [tk for tk in watchlist]
    prog = st.empty()
    for i, tk in enumerate(todo):
        if len(todo) > 3:
            prog.progress((i + 1) / len(todo), text=f"Reading {tk}…")
        r = quick_read(tk, regime["label"])
        if r:
            results.append(r)
    prog.empty()
    if not results:
        st.info("Add tickers to your watchlist in the left sidebar, then press 💾 Save watchlist.")
    else:
        ranked = sorted(results, key=lambda r: -r["Score"])
        top5 = ranked[:5]
        cols = st.columns(len(top5))
        for col, r in zip(cols, top5):
            with col.container(border=True):
                st.markdown(f"### {r['Ticker']}")
                st.metric("Score", f"{r['Score']}/100", f"${r['Price']:.2f}", delta_color="off")
                st.markdown(f"**{r['Verdict'].split(' (')[0].split(' — ')[0]}**")
                st.markdown(f"{CONF_ICON[r['Confidence']]} Confidence: **{r['Confidence']}**")
                if r["Catalyst"]:
                    st.caption(r["Catalyst"])
                st.button("Open →", key=f"open_{r['Ticker']}", on_click=set_ticker, args=(r["Ticker"],), use_container_width=True)
        weakest = [r for r in ranked[5:] if r["Score"] <= 35][-3:]
        if weakest:
            st.caption("Weakest on your list: " + " · ".join(f"{r['Ticker']} ({r['Score']})" for r in weakest))
        new_logs = sum(log_call(r["Ticker"], r["bar_date"], r["Price"], r["Score"], r["Verdict"], r["Confidence"],
                                regime_label=regime["label"], source="top5", save=False) for r in top5)
        if new_logs:
            save_store()
        st.caption(f"Market: {regime['label']} · Top 5 are logged to the Track Record each day so the app grades itself.")

with st.spinner(f"Reading {ticker}..."):
    hist = get_hist(ticker, "5y")
    spy = get_hist("SPY", "5y")

if hist.empty:
    with tab_over:
        st.error(f"No data for “{ticker}”. Check the symbol (e.g. AAPL, MSFT, BRK-B).")
    st.stop()

d = add_indicators(hist)
info = get_info(ticker)
extras = get_extras(ticker)
scores, notes = factor_scores(d, info, extras, spy)
composite = composite_of(scores, weights)
last = d.iloc[-1]
tells = between_the_lines(d, info, extras, spy)
green = sum(1 for t in tells if t[0] == "🟢"); red = sum(1 for t in tells if t[0] == "🔴")
verdict = verdict_of(composite, green, red, regime["label"])
catalysts = catalyst_flags(ticker, info, extras)
ml = None
if run_ml:
    with st.spinner(f"Training the ML model on {ticker}'s history..."):
        try:
            ml = train_ml(ticker, horizon)
        except Exception:
            ml = None
conf, conf_reasons = confidence_read(composite, scores, weights, tells, regime["label"], ml, catalysts)
log_call(ticker, d.index[-1], last["Close"], composite, verdict, conf, ml["p_now"] if ml else None, horizon, regime["label"], "viewed")
drag_shot({"ticker": ticker, "watchlist": watchlist})

# ---------------- Verdict
with tab_over:
    for c in catalysts:
        msg = f"{c['icon']} **Catalyst alert:** {c['text']}"
        if c["kind"] == "earnings":
            st.warning(msg + " — the model's read is least reliable right now.")
        elif c.get("link"):
            st.info(msg + f" [Read]({c['link']})")
        else:
            st.info(msg)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"{ticker} price", f"${last['Close']:.2f}", f"{nz(last['Ret1'])*100:+.2f}% today")
    c2.metric("Composite score", f"{composite:.0f}/100")
    c3.metric("Market regime", regime["label"].split(" ")[0], f"{regime['score']:.0f}/100")
    c4.metric("Hidden tells", f"{green} bullish / {red} bearish")

    st.subheader(f"Verdict: {verdict}")
    st.markdown(f"#### {CONF_ICON[conf]} Confidence: {conf}")
    st.markdown("<div class='mr-why'>" + "<br>".join(f"• {r}" for r in conf_reasons) + "</div>", unsafe_allow_html=True)
    wcol1, wcol2 = st.columns([1, 3])
    if ticker in watchlist:
        wcol1.button(f"✖ Remove {ticker} from watchlist", on_click=remove_from_watchlist, args=(ticker,))
    else:
        wcol1.button(f"⭐ Add {ticker} to watchlist", on_click=add_to_watchlist, args=(ticker,))
    st.write(f"**{safe(info,'longName',ticker)}** · {safe(info,'sector','')} / {safe(info,'industry','')} · Mkt cap ${nz(safe(info,'marketCap'))/1e9:,.1f}B")

    # radar of factors
    cats = list(weights.keys())
    vals = [nz(scores[k], 50) for k in cats]
    fig = go.Figure(go.Scatterpolar(r=vals + [vals[0]], theta=cats + [cats[0]], fill="toself", name=ticker))
    fig.update_layout(polar=dict(radialaxis=dict(range=[0, 100])), height=380, margin=dict(l=20, r=20, t=20, b=20), showlegend=False)
    colA, colB = st.columns([1, 1.2])
    colA.plotly_chart(fig, use_container_width=True)
    with colB:
        st.markdown(f"**Factor weights are set by the regime** ({regime['label']}): the market pays momentum in bull markets and quality/value in bear markets.")
        tbl = pd.DataFrame({"Factor": cats, "Score": [f"{nz(scores[k],50):.0f}" for k in cats],
                            "Weight": [f"{weights[k]*100:.0f}%" for k in cats], "Detail": [notes[k] for k in cats]})
        st.dataframe(tbl, hide_index=True, use_container_width=True)

    # price chart
    p = d.tail(300)
    fig2 = go.Figure()
    fig2.add_trace(go.Candlestick(x=p.index, open=p["Open"], high=p["High"], low=p["Low"], close=p["Close"], name="Price"))
    for n, col in [(20, "orange"), (50, "dodgerblue"), (200, "purple")]:
        fig2.add_trace(go.Scatter(x=p.index, y=p[f"SMA{n}"], name=f"SMA{n}", line=dict(width=1, color=col)))
    fig2.update_layout(height=420, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig2, use_container_width=True)

    # trade plan
    atr = nz(last["ATR"])
    stop = last["Close"] - 2 * atr
    risk_dollars = account * risk_pct / 100
    shares = int(risk_dollars / max(last["Close"] - stop, 0.01))
    st.markdown("**Trade plan (if you take it)**")
    st.write(f"- Entry ≈ ${last['Close']:.2f} · Stop ≈ ${stop:.2f} (2× ATR) · Target 1 ≈ ${last['Close'] + 3*atr:.2f} · Target 2 ≈ ${last['Close'] + 5*atr:.2f}")
    st.write(f"- Size: **{shares} shares** (${shares*last['Close']:,.0f}) risks ${risk_dollars:,.0f} = {risk_pct}% of account")

# ---------------- Between the lines
with tab_lines:
    st.subheader("What the headline numbers hide")
    for icon, title, body in tells:
        st.markdown(f"{icon} **{title}** — {body}")
    st.divider()
    st.markdown("**Money flow vs price (40 days)** — when the blue line rises while price falls, big money is buying quietly.")
    q = d.tail(120)
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=q.index, y=q["Close"], name="Price", yaxis="y1"))
    fig3.add_trace(go.Scatter(x=q.index, y=q["ADL"], name="Accum/Dist", yaxis="y2", line=dict(color="dodgerblue")))
    fig3.update_layout(height=340, yaxis=dict(title="Price"), yaxis2=dict(title="A/D", overlaying="y", side="right", showgrid=False),
                       margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h"))
    st.plotly_chart(fig3, use_container_width=True)
    ins = extras.get("insider_transactions")
    if ins is not None and len(ins):
        st.markdown("**Recent insider transactions**")
        st.dataframe(ins.head(15), use_container_width=True, hide_index=True)

# ---------------- Probability engine
with tab_prob:
    st.subheader(f"Odds for the next {horizon} trading days")
    mc = monte_carlo(d, days=horizon)
    if mc is not None:
        end = mc[:, -1]; spot = last["Close"]
        p5, p25, p50, p75, p95 = np.percentile(end, [5, 25, 50, 75, 95])
        cA, cB, cC, cD = st.columns(4)
        cA.metric("Chance price is UP", f"{(end > spot).mean()*100:.0f}%")
        cB.metric("Median outcome", f"${p50:.2f}", f"{(p50/spot-1)*100:+.1f}%")
        cC.metric("Bad case (5%)", f"${p5:.2f}", f"{(p5/spot-1)*100:+.1f}%")
        cD.metric("Good case (95%)", f"${p95:.2f}", f"{(p95/spot-1)*100:+.1f}%")
        tgt = st.number_input("Probability of touching this price (any time in the window)", value=float(round(spot * 1.1, 2)))
        touched = (mc.max(axis=1) >= tgt).mean() if tgt >= spot else (mc.min(axis=1) <= tgt).mean()
        st.write(f"- Chance of touching **${tgt:.2f}**: **{touched*100:.0f}%** (bootstrapped from {ticker}'s own real returns — fat tails included)")
        figm = go.Figure()
        for i in range(60):
            figm.add_trace(go.Scatter(y=mc[i], mode="lines", line=dict(width=0.7), opacity=0.35, showlegend=False))
        figm.add_trace(go.Scatter(y=np.median(mc, axis=0), mode="lines", line=dict(width=3, color="black"), name="Median"))
        figm.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Trading days ahead", yaxis_title="Price")
        st.plotly_chart(figm, use_container_width=True)

    if run_ml:
        st.divider()
        st.subheader("Machine-learning direction model (gradient boosting, walk-forward tested)")
        if ml is None:
            st.info("Not enough history to train (needs ~3 years).")
        else:
            edge = ml["test_acc"] - max(ml["base_rate"], 1 - ml["base_rate"])
            c1, c2, c3, c4 = st.columns(4)
            c1.metric(f"P(up in {horizon}d)", f"{ml['p_now']*100:.0f}%")
            c2.metric("Out-of-sample accuracy (last yr)", f"{ml['test_acc']*100:.0f}%", f"{edge*100:+.0f}% vs just guessing the trend")
            c3.metric("Accuracy when confident (>60/<40)", f"{ml['conf_acc']*100:.0f}%" if not np.isnan(ml["conf_acc"]) else "n/a", f"{ml['n_conf']} days")
            c4.metric("Training days", f"{ml['n_train']:,}")
            st.markdown("**Only trust it when it's confident AND the out-of-sample accuracy beats the base rate.** If the edge is ≤ 0%, this ticker is not predictable with technicals — lean on the fundamentals and tells instead.")
            imp = ml["importance"].head(8)
            figi = go.Figure(go.Bar(x=imp.values[::-1], y=imp.index[::-1], orientation="h"))
            figi.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10), title="What the model is looking at")
            st.plotly_chart(figi, use_container_width=True)

# ---------------- Earnings & options
with tab_earn:
    st.subheader("Earnings")
    ed = extras.get("earnings_dates")
    nxt = None
    try:
        e = ed.copy(); e.index = pd.to_datetime(e.index).tz_localize(None)
        fut = e[e.index >= datetime.now()]
        nxt = fut.index.min() if len(fut) else None
        past = e[e.index < datetime.now()].head(8)
    except Exception:
        past = None
    if nxt is not None:
        days_to = (nxt - datetime.now()).days
        st.write(f"- Next earnings: **{nxt.date()}** ({days_to} days). Inside 10 days = event risk; options price the move (see below).")
    else:
        st.write("- Next earnings date: not available.")
    # historical post-earnings moves
    try:
        moves = []
        for dt in past.index:
            loc = d.index.searchsorted(dt)
            if 1 <= loc < len(d) - 1:
                moves.append((dt.date(), (d["Close"].iloc[loc + 1] / d["Close"].iloc[loc - 1] - 1) * 100, past.loc[dt, "Surprise(%)"]))
        if moves:
            mv = pd.DataFrame(moves, columns=["Report date", "2-day move %", "EPS surprise %"])
            st.dataframe(mv.round(2), hide_index=True, use_container_width=True)
            st.write(f"- Average absolute earnings move: **{mv['2-day move %'].abs().mean():.1f}%** · Avg move after a beat: **{mv[mv['EPS surprise %']>0]['2-day move %'].mean():+.1f}%**")
    except Exception:
        pass

    st.divider()
    st.subheader("Options market")
    chains = extras.get("chains") or []
    if chains:
        exp, calls, puts = chains[0]
        spot = last["Close"]
        try:
            atm_c = calls.iloc[(calls["strike"] - spot).abs().argsort()[:1]].iloc[0]
            atm_p = puts.iloc[(puts["strike"] - spot).abs().argsort()[:1]].iloc[0]
            straddle = nz(atm_c["lastPrice"]) + nz(atm_p["lastPrice"])
            implied_move = straddle / spot * 100
            pcr_v = nz(puts["volume"].sum()) / max(nz(calls["volume"].sum(), 1), 1)
            pcr_oi = nz(puts["openInterest"].sum()) / max(nz(calls["openInterest"].sum(), 1), 1)
            # max pain
            strikes = sorted(set(calls["strike"]).union(puts["strike"]))
            pain = []
            for k in strikes:
                loss = ((k - calls["strike"]).clip(lower=0) * calls["openInterest"]).sum() + ((puts["strike"] - k).clip(lower=0) * puts["openInterest"]).sum()
                pain.append((k, loss))
            max_pain = min(pain, key=lambda z: z[1])[0]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric(f"Implied move by {exp}", f"±{implied_move:.1f}%", f"±${straddle:.2f}")
            c2.metric("ATM implied vol", f"{nz(atm_c['impliedVolatility'])*100:.0f}%", f"realized {nz(last['RealVol30']):.0f}%")
            c3.metric("Put/Call (volume)", f"{pcr_v:.2f}", f"OI {pcr_oi:.2f}")
            c4.metric("Max pain", f"${max_pain:.0f}")
            st.markdown("- Price tends to drift toward **max pain** into expiration. Put/Call > 1.2 = crowd is scared (contrarian bullish at extremes); < 0.6 = crowd is greedy.")
        except Exception:
            st.info("Options chain incomplete for this ticker.")
    else:
        st.info("No options data for this ticker.")

# ---------------- Backtest
with tab_bt:
    st.subheader("Does the trend model actually work on this name?")
    bt = quick_backtest(d, spy)
    if bt:
        c1, c2, c3 = st.columns(3)
        c1.metric("Model CAGR", f"{bt['strat_cagr']:.1f}%", f"buy & hold {bt['bh_cagr']:.1f}%")
        c2.metric("Model max drawdown", f"{bt['strat_mdd']:.0f}%", f"buy & hold {bt['bh_mdd']:.0f}%")
        c3.metric("Time invested", f"{bt['time_in']:.0f}%")
        figb = go.Figure()
        figb.add_trace(go.Scatter(x=bt["strat"].index, y=bt["strat"], name="Trend model (in only when price>200d, 50>200, 6m mom>0)"))
        figb.add_trace(go.Scatter(x=bt["bh"].index, y=bt["bh"], name="Buy & hold"))
        figb.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h"))
        st.plotly_chart(figb, use_container_width=True)
        st.markdown("- The point isn't beating buy-and-hold on return every time — it's **cutting the drawdown**. Smaller drawdowns = you stay in the game and size up with confidence.")
    else:
        st.info("Not enough history to backtest.")

# ---------------- Screener
with tab_screen:
    st.subheader("Your watchlist ranked by composite score (regime-weighted)")
    if results:
        res = pd.DataFrame(results).sort_values("Score", ascending=False)
        cols_show = ["Ticker", "Score", "Call", "Confidence", "Catalyst", "Momentum", "Quality", "Value", "Growth", "SmartMoney",
                     "Bull_tells", "Bear_tells", "Price", "From52wHigh"]
        st.dataframe(res[cols_show], hide_index=True, use_container_width=True)
        st.download_button("Download CSV", res[cols_show].to_csv(index=False), "screener.csv")
    else:
        st.write("Add tickers to your watchlist in the left sidebar.")

# ---------------- Track record
with tab_track:
    st.subheader("Track Record — every call, graded against what the stock actually did")
    calls = store()["calls"]
    if not calls:
        st.info("No calls yet. Every stock you open (and each day's Top 5) is logged automatically.")
    else:
        g = grade_calls(json.dumps(calls))
        for n in (1, 5, 20):
            g[f"win{n}"] = [win_of(c, r) for c, r in zip(g["call"], g[f"ret{n}"])]
            g[f"alpha{n}"] = g[f"ret{n}"] - g[f"spy{n}"]
        directional = g[g["call"] != "NEUTRAL"]
        def rate(col):
            s = directional[col].dropna()
            return (f"{s.mean()*100:.0f}%", f"{len(s)} graded") if len(s) else ("—", "waiting on prices")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Calls logged", f"{len(g)}", f"{len(directional)} long/avoid calls", delta_color="off")
        v, sub = rate("win5"); m2.metric("Right after 5 days", v, sub, delta_color="off")
        v, sub = rate("win20"); m3.metric("Right after 20 days", v, sub, delta_color="off")
        longs = g[g["call"] == "LONG"]["alpha20"].dropna()
        m4.metric("LONG calls vs S&P 500 (20d)", f"{longs.mean():+.1f}%" if len(longs) else "—",
                  f"{len(longs)} graded" if len(longs) else "waiting on prices", delta_color="off")
        st.markdown("**Does confidence mean anything?** (the real test — High should beat Low)")
        rows = []
        for lvl in ["High", "Med", "Low"]:
            sub_ = directional[directional["confidence"] == lvl]
            w5, w20 = sub_["win5"].dropna(), sub_["win20"].dropna()
            rows.append({"Confidence": f"{CONF_ICON[lvl]} {lvl}", "Calls": len(sub_),
                         "Right after 5d": f"{w5.mean()*100:.0f}% ({len(w5)})" if len(w5) else "—",
                         "Right after 20d": f"{w20.mean()*100:.0f}% ({len(w20)})" if len(w20) else "—"})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

        def fmt(x):
            return "" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:+.1f}%"
        def result(row):
            w = row["win20"] if not np.isnan(row["win20"]) else row["win5"]
            if row["call"] == "NEUTRAL":
                return "— (neutral)"
            if np.isnan(w):
                return "⏳ pending"
            return "✅ right" if w == 1 else "❌ wrong"
        tbl = pd.DataFrame({
            "Date": g["bar_date"], "Ticker": g["ticker"], "Call": g["call"],
            "Confidence": [f"{CONF_ICON.get(c, '')} {c}" for c in g["confidence"]], "Score": g["score"],
            "Price then": g["price"].map(lambda x: f"${x:,.2f}"),
            "1 day": g["ret1"].map(fmt), "5 days": g["ret5"].map(fmt), "20 days": g["ret20"].map(fmt),
            "vs S&P (20d)": g["alpha20"].map(fmt), "Result": g.apply(result, axis=1), "From": g["source"]})
        st.dataframe(tbl.iloc[::-1], hide_index=True, use_container_width=True)
        st.caption("A LONG call is right if the stock went up; an AVOID call is right if it went down. Neutral calls aren't graded.")

    st.divider()
    b1, b2 = st.columns(2)
    b1.download_button("⬇️ Download backup", json.dumps(store(), indent=1), "market_reader_backup.json", use_container_width=True)
    up = b2.file_uploader("Restore a backup", type=["json"], label_visibility="collapsed")
    if up is not None and st.session_state.get("restored") != up.name:
        try:
            data = json.load(up)
            s_ = store()
            have = {(c["bar_date"], c["ticker"]) for c in s_["calls"]}
            s_["calls"] += [c for c in data.get("calls", []) if (c.get("bar_date"), c.get("ticker")) not in have]
            s_["watchlist"] = list(dict.fromkeys(s_["watchlist"] + data.get("watchlist", [])))
            save_store()
            st.session_state["restored"] = up.name
            st.success("Backup restored.")
        except Exception:
            st.error("That file isn't a Market Reader backup.")
    if storage_is_permanent():
        st.caption("☁️ Saving permanently to your private GitHub gist.")
    else:
        st.warning("⚠️ Saving is temporary — Streamlit wipes it when the app sleeps. Add a free GitHub token to make it permanent (2 minutes).")
        with st.expander("Make saving permanent (free)"):
            st.markdown(
                "1. Open https://github.com/settings/tokens/new?scopes=gist&description=Market%20Reader\n"
                "2. **Expiration** dropdown → **No expiration**\n"
                "3. Scroll to the bottom → green **Generate token** button → click the copy icon next to the new token\n"
                "4. Open https://share.streamlit.io → click the **⋮** icon at the right end of the market-reader row → **Settings** → **Secrets** tab\n"
                "5. Paste this in the box, swapping in your token, then click **Save**:\n\n"
                "```\nGITHUB_TOKEN = \"paste-token-here\"\n```")

# ---------------- Regime
with tab_regime:
    st.subheader(f"Market regime: {regime['label']} ({regime['score']:.0f}/100)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("VIX (fear)", f"{regime['vix']:.1f}", f"20d avg {regime['vix_20d_avg']:.1f}")
    c2.metric("Yield curve (10y-3m)", f"{regime['curve']:+.2f}%", "inverted = recession warning" if nz(regime["curve"], 1) < 0 else "normal")
    c3.metric("SPY 1-month", f"{nz(regime.get('spy_1m')):+.1f}%", f"drawdown {nz(regime.get('spy_dd')):.1f}%")
    c4.metric("SPY vs 200d", "ABOVE" if regime.get("spy_above_200") else "BELOW", "50d > 200d" if regime.get("sma50_above_200") else "50d < 200d (death cross)")
    st.markdown(f"- Credit (junk vs investment grade): **{'improving — risk appetite healthy' if nz(regime['credit_trend']) > 0 else 'deteriorating — credit is sniffing out trouble'}**")
    st.markdown(f"- Breadth (equal-weight vs cap-weight): **{'broad participation — healthy' if nz(regime['breadth_trend']) > 0 else 'narrow — a few megacaps carrying the index (fragile)'}**")
    st.markdown("**How the regime changes the model:**")
    st.dataframe(pd.DataFrame({"Factor": list(weights.keys()), "Weight now": [f"{v*100:.0f}%" for v in weights.values()]}), hide_index=True)
    st.markdown("- Risk-on: ride momentum, buy strength. Neutral: balanced, buy pullbacks in quality. Risk-off: only high quality + cheap + low vol; cash is a position.")

st.caption("Data: Yahoo Finance (delayed). For research only — not financial advice. Every edge here is probabilistic; size positions so no single trade can hurt you.")
