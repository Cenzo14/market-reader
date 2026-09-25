# Market Reader v6 — "Read Between the Lines" Market Intelligence
# The whole market → sectors → news → one stock → what to do, explained in plain English.
# Free data (Yahoo Finance). Nothing here is a guarantee. It stacks edges and measures itself honestly.

import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta

def _favicon():
    try:
        from PIL import Image, ImageDraw
        im = Image.new("RGBA", (64, 64), (0, 0, 0, 0)); dr = ImageDraw.Draw(im)
        dr.rounded_rectangle([0, 0, 63, 63], radius=16, fill=(33, 37, 91, 255))
        for x, y, a in [(14, 34, 150), (25, 26, 200), (36, 19, 240)]:
            dr.rounded_rectangle([x, y, x + 7, 50], radius=2, fill=(255, 255, 255, a))
        dr.line([(13, 29), (24, 21), (34, 24), (51, 12)], fill=(184, 237, 253, 255), width=4)
        dr.ellipse([47, 8, 55, 16], fill=(184, 237, 253, 255))
        return im
    except Exception:
        return None

st.set_page_config(page_title="Market Reader", page_icon=_favicon(), layout="wide", initial_sidebar_state="collapsed")

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

def drop_partial_bar(h):
    """Drop today's still-forming bar while the market is open (it breaks volume ratio, RSI, Ret1)."""
    if h is None or h.empty:
        return h
    now = pd.Timestamp.now(tz="America/New_York")
    market_open = now.weekday() < 5 and (now.hour, now.minute) >= (9, 30) and now.hour < 16
    if market_open and h.index[-1].date() == now.date():
        return h.iloc[:-1]
    return h

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
                 "upgrades_downgrades", "recommendations", "earnings_estimate", "revenue_estimate"]:
        try:
            out[name] = getattr(t, name)
        except Exception:
            out[name] = None
    if not with_options:
        out["options_exp"], out["chains"] = [], []
        return out
    try:
        exps = t.options or []
        today = pd.Timestamp.now().normalize()
        # skip 0-6 DTE weeklies: Yahoo IV and prices are unreliable there
        out["options_exp"] = [e for e in exps if (pd.Timestamp(e) - today).days >= 7][:3]
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
    up = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()      # Wilder smoothing (matches brokers)
    dn = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
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
    d["CMF20"] = (mfm.fillna(0) * d["Volume"]).rolling(20).sum() / d["Volume"].rolling(20).sum()  # bounded [-1, 1]
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
    if "spy_above_200" in r:
        score += 15 if r["spy_above_200"] else -15
    if "sma50_above_200" in r:
        score += 10 if r["sma50_above_200"] else -10
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
def value_pe(pe):
    if np.isnan(pe):
        return np.nan                                 # unknown -> excluded from the average
    return pct_score(pe, 60, 8) if pe > 0 else 10     # negative earnings -> penalized

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
    peg = nz(safe(info, "pegRatio", safe(info, "trailingPegRatio")))   # pegRatio is often missing
    f["Value"] = np.nanmean([value_pe(pe), pct_score(ev, 40, 5) if ev > 0 else np.nan,
                             pct_score(fcfy, -2, 8), pct_score(peg, 4, 0.5) if peg > 0 else np.nan])
    notes["Value"] = f"Fwd P/E {pe:.1f} | EV/EBITDA {ev:.1f} | FCF yield {fcfy:.1f}% | PEG {peg:.2f}"

    # Quality
    roe = nz(safe(info, "returnOnEquity")) * 100
    gm = nz(safe(info, "grossMargins")) * 100
    om = nz(safe(info, "operatingMargins")) * 100
    de = nz(safe(info, "debtToEquity"))
    cr = nz(safe(info, "currentRatio"))
    if safe(info, "sector", "") == "Financial Services":
        # margins, debt/equity and current ratio are meaningless for banks/insurers
        f["Quality"] = np.nanmean([pct_score(roe, -10, 20), pct_score(nz(safe(info, "returnOnAssets")) * 100, 0, 2)])
    else:
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
        text = (df["Text"] if "Text" in df else pd.Series("", index=df.index)).astype(str).str.lower()
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
            tells.append(("🟢", "Quiet accumulation", "Accumulation/Distribution rising while price falls — often read as dip-buying. Built from daily bars, so it can't see who is trading; check the hit rate below."))
        elif adl_s < 0 and px_s > 0:
            tells.append(("🔴", "Distribution into strength", "Price up but money flow down — often read as a rally being sold into. Check the hit rate below."))

    # 2. Volume spike
    vr = nz(last["VolRatio"])
    if vr > 2:
        direction = "up" if nz(last["Ret1"]) > 0 else "down"
        tells.append(("🟡", f"Unusual volume ({vr:.1f}x avg)", f"Big players moved today ({direction} day). Follow-through in the next 1-3 sessions confirms or fails it."))

    # 3. RSI divergence
    c40, r40 = d["Close"].tail(40), d["RSI"].tail(40)
    if len(c40) == 40 and c40.iloc[-1] < c40.min() * 1.02 and r40.iloc[-1] > r40.min() + 5:
        tells.append(("🟢", "Bullish RSI divergence", "New price low but momentum is NOT making a new low — sellers are getting tired."))
    if len(c40) == 40 and c40.iloc[-1] > c40.max() * 0.98 and r40.iloc[-1] < r40.max() - 5:
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
                    tells.append(("🟢", f"Beat estimates {beats}/{len(s)} straight quarters", "Consistent beats — mostly reflects analysts setting a low bar. Streaks tend to persist, but are usually priced in."))
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

def clean_iv(v):
    """Yahoo regularly returns junk IVs; ignore values below 3% or above 300%."""
    v = nz(v)
    return not np.isnan(v) and 0.03 <= v <= 3.0

def mid_price(r):
    b, a = nz(r.get("bid"), 0), nz(r.get("ask"), 0)
    return (b + a) / 2 if b > 0 and a > 0 else nz(r.get("lastPrice"))

def options_read(extras, spot, realvol):
    try:
        chains = extras.get("chains") or []
        if not chains or np.isnan(spot):
            return None
        exp, calls, puts = chains[0]
        atm_c = calls.iloc[(calls["strike"] - spot).abs().argsort()[:1]]
        atm_p = puts.iloc[(puts["strike"] - spot).abs().argsort()[:1]]
        ivs = [v for v in (atm_c["impliedVolatility"].iloc[0], atm_p["impliedVolatility"].iloc[0]) if clean_iv(v)]
        if not ivs:
            return None
        iv = float(np.mean(ivs)) * 100
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
    x["cmf20"] = d["CMF20"]   # bounded money flow (cumulative ADL level depends on download length)
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
    # Walk-forward over up to 4 yearly folds. PURGE a gap of `horizon` days between train and test:
    # the last training labels would otherwise be computed from prices inside the test window.
    n, test_len = len(df), 250
    accs, edges, briers, briers_naive, fold_rows = [], [], [], [], []
    all_p, all_y = [], []
    for k in range(4):
        te_end = n - k * test_len
        te_start = te_end - test_len
        tr_end = te_start - horizon
        if tr_end < 500:
            break
        m = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=7)
        ytr = df["label"].iloc[:tr_end]
        m.fit(df[feats].iloc[:tr_end], ytr)
        p = m.predict_proba(df[feats].iloc[te_start:te_end])[:, 1]
        y = df["label"].iloc[te_start:te_end].values
        train_up = float(ytr.mean())
        naive_pred = 1 if train_up >= 0.5 else 0            # honest baseline: training-period majority, no hindsight
        acc = float(((p > 0.5).astype(int) == y).mean())
        naive_acc = float((y == naive_pred).mean())
        accs.append(acc); edges.append(acc - naive_acc)
        briers.append(float(((p - y) ** 2).mean())); briers_naive.append(float(((train_up - y) ** 2).mean()))
        all_p.append(p); all_y.append(y)
        fold_rows.append(dict(Fold=f"{df.index[te_start].date()} → {df.index[te_end-1].date()}",
                              Accuracy=f"{acc*100:.0f}%", Baseline=f"{naive_acc*100:.0f}%", Edge=f"{(acc-naive_acc)*100:+.0f}%"))
    if not accs:
        return None
    p_te, yte = np.concatenate(all_p), np.concatenate(all_y)
    conf_mask = (p_te > 0.6) | (p_te < 0.4)
    conf_acc = float(((p_te[conf_mask] > 0.5).astype(int) == yte[conf_mask]).mean()) if conf_mask.sum() > 10 else np.nan
    model = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=7)
    # retrain on all data for today's call
    model.fit(df[feats], df["label"])
    X_now = X[feats].dropna().iloc[[-1]]
    p_now = float(model.predict_proba(X_now)[:, 1][0])
    imp = pd.Series(model.feature_importances_, index=feats).sort_values(ascending=False)
    return dict(p_now=p_now, test_acc=float(np.mean(accs)), edge=float(np.mean(edges)), edge_min=float(min(edges)),
                edge_max=float(max(edges)), brier=float(np.mean(briers)), brier_naive=float(np.mean(briers_naive)),
                folds=pd.DataFrame(fold_rows), n_indep=int(len(yte) / horizon),
                conf_acc=conf_acc, n_conf=int(conf_mask.sum()), importance=imp, horizon=horizon, n_train=len(df))

def monte_carlo(d, days=63, sims=5000, seed=7, block=10, demean=True):
    """Block bootstrap of real returns: keeps fat tails AND volatility clustering.
    demean=True removes the past drift so a stock that ran up isn't auto-labeled 'likely up'."""
    r = d["Ret1"].dropna().tail(750).values
    if len(r) < 100:
        return None
    if demean:
        r = r - r.mean()
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(days / block))
    starts = rng.integers(0, len(r) - block, size=(sims, nb))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(sims, -1)[:, :days]
    paths = np.cumprod(1 + r[idx], axis=1)
    return d["Close"].iloc[-1] * paths

# ------------------------------------------------------------------ event study: do the tells actually work here?
def tell_signals(d):
    """Historical daily version of each technical tell, using only data available on that day."""
    adl_s = d["ADL"].rolling(40).apply(lambda s: np.polyfit(np.arange(40), s, 1)[0], raw=True)
    px_s = d["Close"].rolling(40).apply(lambda s: np.polyfit(np.arange(40), s, 1)[0], raw=True)
    bw = 4 * d["BBstd"] / d["BBmid"]
    return {
        "Quiet accumulation": (adl_s > 0) & (px_s < 0),
        "Distribution into strength": (adl_s < 0) & (px_s > 0),
        "Unusual volume, up day": (d["VolRatio"] > 2) & (d["Ret1"] > 0),
        "Unusual volume, down day": (d["VolRatio"] > 2) & (d["Ret1"] < 0),
        "Bullish RSI divergence": (d["Close"] < d["Close"].rolling(40).min() * 1.02) & (d["RSI"] > d["RSI"].rolling(40).min() + 5),
        "Bearish RSI divergence": (d["Close"] > d["Close"].rolling(40).max() * 0.98) & (d["RSI"] < d["RSI"].rolling(40).max() - 5),
        "Volatility squeeze": bw <= bw.rolling(120).quantile(0.1),
    }

def tell_event_study(d, horizon=20):
    fwd = d["Close"].shift(-horizon) / d["Close"] - 1
    base = fwd.dropna()
    rows = []
    for name, sig in tell_signals(d).items():
        on = fwd[sig.fillna(False).astype(bool)].dropna()
        rows.append({"Tell": name, "Times fired": len(on),
                     "Up after": f"{(on > 0).mean()*100:.0f}%" if len(on) else "n/a",
                     "Up (all days)": f"{(base > 0).mean()*100:.0f}%",
                     f"Avg {horizon}d return": f"{on.mean()*100:+.1f}%" if len(on) else "n/a",
                     "Avg (all days)": f"{base.mean()*100:+.1f}%",
                     "Enough data?": "yes" if len(on) >= 30 else "too few — ignore"})
    return pd.DataFrame(rows)

# ------------------------------------------------------------------ backtest of the composite score
def quick_backtest(d, spy):
    """Does 'trend + momentum + accumulation' actually beat buy-and-hold on THIS name? Honest check."""
    x = d.dropna(subset=["SMA200", "SMA50", "Mom6"]).copy()
    if len(x) < 300:
        return None
    sig = ((x["Close"] > x["SMA200"]) & (x["SMA50"] > x["SMA200"]) & (x["Mom6"] > 0)).astype(int).shift(1).fillna(0)
    cost = 0.0005  # 5 bps per side (spread + slippage)
    switches = sig.diff().abs().fillna(0)
    sret = x["Ret1"] * sig - switches * cost
    strat = (1 + sret).cumprod()
    bh = (1 + x["Ret1"]).cumprod()
    yrs = len(x) / 252
    def cagr(s): return (s.iloc[-1] ** (1 / yrs) - 1) * 100
    def mdd(s): return ((s / s.cummax()) - 1).min() * 100
    def sharpe(r): return float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else np.nan
    return dict(strat=strat, bh=bh, strat_cagr=cagr(strat), bh_cagr=cagr(bh), strat_mdd=mdd(strat), bh_mdd=mdd(bh),
                time_in=float(sig.mean() * 100), trades=int(switches.sum() / 2), strat_sharpe=sharpe(sret),
                bh_sharpe=sharpe(x["Ret1"].dropna()))


# ================================================================== MACRO / GLOBAL ENGINE
# Everything below is free Yahoo data. Rules are transparent and every claim is labeled as a tendency, not a promise.

UNIVERSE = {
    # symbol: (display name, group, plain-English "what it tells you")
    "SPY":  ("S&P 500", "US Stocks", "The 500 biggest US companies. The scoreboard for 'the market'."),
    "QQQ":  ("Nasdaq 100", "US Stocks", "Big tech and growth. Leads when investors feel bold."),
    "IWM":  ("Small caps", "US Stocks", "Small US companies. Most sensitive to interest rates and the economy."),
    "DIA":  ("Dow 30", "US Stocks", "30 blue-chip industrial names. Old-economy health."),
    "RSP":  ("S&P equal-weight", "US Stocks", "Every stock counts the same. If this lags SPY, only a few giants are carrying the market."),
    "EFA":  ("Europe/Japan", "World", "Developed markets outside the US."),
    "EEM":  ("Emerging markets", "World", "China, India, Brazil, etc. Falls when the dollar rises."),
    "^N225": ("Japan (Nikkei)", "World", "Japan's main index."),
    "^FTSE": ("UK (FTSE 100)", "World", "London's main index."),
    "^GDAXI": ("Germany (DAX)", "World", "Europe's industrial engine."),
    "^HSI": ("Hong Kong (Hang Seng)", "World", "China-linked stocks."),
    "^VIX": ("VIX (fear gauge)", "Risk", "How much traders are paying for protection. Under 16 = calm, over 25 = scared."),
    "^TNX": ("10-year yield", "Rates", "The interest rate that sets mortgages and stock valuations. Rising = pressure on growth stocks."),
    "^IRX": ("3-month yield", "Rates", "Tracks the Fed's rate. When 10-year is BELOW this, the curve is 'inverted' (recession warning)."),
    "TLT":  ("Long bonds", "Rates", "20-year Treasury bonds. Rises when rates fall or investors flee to safety."),
    "HYG":  ("Junk bonds", "Credit", "Risky company debt. If this drops while stocks rise, credit is sniffing out trouble."),
    "LQD":  ("Quality bonds", "Credit", "Safe company debt. Used as the comparison for junk bonds."),
    "DX-Y.NYB": ("US Dollar", "FX", "Strong dollar hurts emerging markets, commodities and US exporters."),
    "EURUSD=X": ("Euro / Dollar", "FX", "Euro strength vs the dollar."),
    "USDJPY=X": ("Dollar / Yen", "FX", "A fast-falling yen can force global funds to sell stocks (the 'carry trade')."),
    "CL=F": ("Crude oil", "Commodities", "Fuel and inflation. Spikes hurt airlines and consumers, help energy stocks."),
    "GC=F": ("Gold", "Commodities", "Fear and inflation hedge. Rises when trust in paper money falls."),
    "HG=F": ("Copper", "Commodities", "'Dr. Copper' — rises when the world economy is building things."),
    "NG=F": ("Natural gas", "Commodities", "Heating/power fuel. Weather and storage driven."),
    "BTC-USD": ("Bitcoin", "Crypto", "Pure risk appetite. Often moves before stocks on weekends."),
    "ETH-USD": ("Ethereum", "Crypto", "Second-largest crypto; higher beta than Bitcoin."),
}

SECTORS = {
    "XLK": "Technology", "XLC": "Communication", "XLY": "Consumer (wants)", "XLP": "Consumer (needs)",
    "XLF": "Financials", "XLV": "Health care", "XLI": "Industrials", "XLE": "Energy",
    "XLB": "Materials", "XLU": "Utilities", "XLRE": "Real estate",
}
SECTOR_STYLE = {  # what kind of market each sector tends to lead in
    "XLK": "risk-on / growth", "XLC": "risk-on / growth", "XLY": "risk-on / consumer strong",
    "XLF": "rising rates / healthy economy", "XLI": "economic expansion", "XLB": "global growth / inflation",
    "XLE": "inflation / oil shock", "XLV": "defensive", "XLP": "defensive", "XLU": "defensive / falling rates",
    "XLRE": "falling rates",
}

@st.cache_data(ttl=900, show_spinner=False)
def batch_hist(symbols, period="1y"):
    """One download for many symbols. Returns dict symbol -> Close series (and Volume where present)."""
    out = {}
    try:
        raw = yf.download(list(symbols), period=period, auto_adjust=True, progress=False, group_by="ticker", threads=True)
    except Exception:
        return out
    for s in symbols:
        try:
            df = raw[s] if isinstance(raw.columns, pd.MultiIndex) else raw
            c = df["Close"].dropna()
            c.index = pd.to_datetime(c.index).tz_localize(None)
            if len(c):
                out[s] = dict(close=c, volume=df["Volume"].dropna() if "Volume" in df else None)
        except Exception:
            continue
    return out

def chg(c, n):
    try:
        return (c.iloc[-1] / c.iloc[-1 - n] - 1) * 100
    except Exception:
        return np.nan

def ytd(c):
    try:
        y0 = c[c.index.year == c.index[-1].year]
        return (c.iloc[-1] / y0.iloc[0] - 1) * 100
    except Exception:
        return np.nan

@st.cache_data(ttl=900, show_spinner=False)
def global_snapshot():
    data = batch_hist(list(UNIVERSE.keys()) + list(SECTORS.keys()), "1y")
    rows = []
    for s, (name, grp, why) in UNIVERSE.items():
        if s not in data: continue
        c = data[s]["close"]
        rows.append(dict(Symbol=s, Name=name, Group=grp, Last=float(c.iloc[-1]), D1=chg(c, 1), W1=chg(c, 5), M1=chg(c, 21),
                         M3=chg(c, 63), YTD=ytd(c), Why=why, spark=c.tail(30).values))
    snap = pd.DataFrame(rows)
    return snap, data

def sector_table(data):
    spy = data.get("SPY", {}).get("close")
    rows = []
    for s, name in SECTORS.items():
        if s not in data or spy is None: continue
        c = data[s]["close"]
        rs = (c / spy.reindex(c.index).ffill()).dropna()
        rs_norm = rs / rs.rolling(60).mean() * 100          # relative strength ratio (100 = in line with SPY)
        rs_mom = rs_norm / rs_norm.shift(10) * 100           # momentum of relative strength
        rows.append(dict(Symbol=s, Sector=name, D1=chg(c, 1), W1=chg(c, 5), M1=chg(c, 21), M3=chg(c, 63), YTD=ytd(c),
                         RS=float(rs_norm.iloc[-1]), RSmom=float(rs_mom.iloc[-1]),
                         RS_trail=rs_norm.tail(8).values, RSmom_trail=rs_mom.tail(8).values,
                         Style=SECTOR_STYLE[s]))
    df = pd.DataFrame(rows)
    if df.empty: return df
    def quad(r):
        if r["RS"] >= 100 and r["RSmom"] >= 100: return "Leading"
        if r["RS"] >= 100 and r["RSmom"] < 100: return "Weakening"
        if r["RS"] < 100 and r["RSmom"] >= 100: return "Improving"
        return "Lagging"
    df["Quadrant"] = df.apply(quad, axis=1)
    return df

# ---------------------------------------------------------------- alerts: what changed that matters
def build_alerts(regime, snap, sectors, data):
    a = []
    g = snap.set_index("Symbol") if not snap.empty else pd.DataFrame()
    def get(sym, col):
        try: return float(g.loc[sym, col])
        except Exception: return np.nan
    vix, vix1 = nz(regime.get("vix")), get("^VIX", "D1")
    if vix1 > 15:
        a.append(("🔴", f"Fear spiked: VIX +{vix1:.0f}% today to {vix:.1f}",
                  "Traders are rushing to buy protection. Big down days cluster together.",
                  "Don't buy dips today. Reduce size. Wait for VIX to stop rising for 2 sessions."))
    elif vix1 < -12:
        a.append(("🟢", f"Fear collapsed: VIX {vix1:.0f}% today to {vix:.1f}",
                  "Relief. Money that was hiding comes back into stocks over the next days.",
                  "Buying strong names on the first pullback usually works here."))
    if vix >= 30:
        a.append(("🔴", f"VIX at {vix:.0f} — stress regime", "Above 30 means panic-level pricing. Swings of 2-3% a day are normal here.",
                  "Cash is a position. If you must trade, cut size in half."))
    if "spy_above_200" in regime:
        spy = data.get("SPY", {}).get("close")
        if spy is not None and len(spy) > 205:
            s200 = spy.rolling(200).mean()
            above_now, above_5 = spy.iloc[-1] > s200.iloc[-1], spy.iloc[-6] > s200.iloc[-6]
            if above_now and not above_5:
                a.append(("🟢", "S&P 500 reclaimed its 200-day average this week", "The most-watched line in the market. Above it, big funds are allowed to buy.",
                          "Bias flips to buying pullbacks."))
            if not above_now and above_5:
                a.append(("🔴", "S&P 500 broke below its 200-day average this week", "Trend-following funds sell mechanically below this line.",
                          "Stop adding risk. Tighten stops on everything."))
    curve = nz(regime.get("curve"))
    if not np.isnan(curve) and curve < 0:
        a.append(("🟡", f"Yield curve inverted ({curve:+.2f}%)", "Short rates above long rates. Every US recession in 50 years was preceded by this — but the lag is 6-18 months.",
                  "Not a sell signal today. It IS a reason to favor quality and keep cash."))
    if nz(regime.get("credit_trend")) < 0 and regime.get("spy_above_50"):
        a.append(("🟡", "Credit is weakening while stocks are up", "Bond investors are more careful than stock investors — when they disagree, bonds are usually right.",
                  "Be skeptical of this rally. Take partial profits into strength."))
    oil1 = get("CL=F", "D1")
    if abs(oil1) > 4:
        d = "spiked" if oil1 > 0 else "dropped"
        a.append(("🟡" if oil1 > 0 else "🟢", f"Oil {d} {oil1:+.1f}% today",
                  "Oil is a tax on consumers and a gift to energy companies." if oil1 > 0 else "Cheaper oil = more money in consumers' pockets, lower inflation, room for the Fed to cut.",
                  "Energy (XLE) up, airlines/cruise/retail down." if oil1 > 0 else "Airlines (JETS), consumer (XLY), transports benefit; XLE lags."))
    dxy1 = get("DX-Y.NYB", "W1")
    if abs(dxy1) > 1.5:
        a.append(("🟡", f"Dollar moved {dxy1:+.1f}% this week", "A fast-rising dollar squeezes emerging markets, gold and US companies that sell abroad." if dxy1 > 0 else "A falling dollar boosts gold, emerging markets, commodities and multinationals.",
                  "Favor domestic US names; avoid EEM." if dxy1 > 0 else "Gold (GLD), emerging markets (EEM), copper (FCX) tend to benefit."))
    tnx = data.get("^TNX", {}).get("close")
    if tnx is not None and len(tnx) > 6:
        d5 = (tnx.iloc[-1] - tnx.iloc[-6]) / 10
        if d5 > 0.2:
            a.append(("🔴", f"10-year yield jumped {d5:+.2f}% this week to {tnx.iloc[-1]/10:.2f}%", "Higher long rates are gravity for stock prices, especially growth and small caps.",
                      "Growth/tech and small caps (IWM) get hit; banks hold up better."))
        elif d5 < -0.2:
            a.append(("🟢", f"10-year yield fell {d5:+.2f}% this week to {tnx.iloc[-1]/10:.2f}%", "Falling rates lift valuations and help rate-sensitive groups.",
                      "Homebuilders (XHB), real estate (XLRE), small caps (IWM), long bonds (TLT) tend to benefit."))
    if not sectors.empty:
        hot = sectors[sectors["W1"] > 3.5]
        for _, r in hot.iterrows():
            a.append(("🟢", f"{r['Sector']} ({r['Symbol']}) up {r['W1']:+.1f}% this week", f"Money is rotating in. This sector tends to lead in a {r['Style']} market.",
                      "Look for the strongest stocks inside it (use the Screener)."))
        cold = sectors[sectors["W1"] < -3.5]
        for _, r in cold.iterrows():
            a.append(("🔴", f"{r['Sector']} ({r['Symbol']}) down {r['W1']:+.1f}% this week", "Money is leaving. Falling sectors keep falling more often than they bounce.",
                      "Avoid catching the knife; wait for a week of stabilizing."))
    btc = get("BTC-USD", "D1")
    if abs(btc) > 6:
        a.append(("🟡", f"Bitcoin {btc:+.1f}% today", "Crypto is the purest risk-appetite gauge and often leads stocks by a day.",
                  "Big BTC drops warn of risk-off in tech; big pops hint at risk-on."))
    if not a:
        a.append(("⚪", "Quiet tape — nothing is screaming", "No stress signals and no violent moves. These are the days to prepare, not chase.", "Build your watchlist and set alerts."))
    order = {"🔴": 0, "🟡": 1, "🟢": 2, "⚪": 3}
    return sorted(a, key=lambda z: order[z[0]])

# ---------------------------------------------------------------- news → theme → plays
# Each rule: keywords, what it means in plain English, who tends to win/lose, and the honest caveat.
NEWS_RULES = [
    dict(theme="Fed cuts / dovish", kw=["rate cut", "cuts rates", "cut rates", "dovish", "easing", "lower rates", "fed pivot", "pause"],
         means="Cheaper money. Lifts valuations, helps borrowers, weakens the dollar.",
         winners=["IWM", "XLRE", "XHB", "TLT", "GLD", "QQQ"], losers=["UUP", "XLF"],
         caveat="If the cut is because the economy is breaking, stocks can fall anyway ('bad news cut')."),
    dict(theme="Fed hikes / hawkish", kw=["rate hike", "hikes rates", "raise rates", "hawkish", "higher for longer", "tightening"],
         means="Expensive money. Pressures growth stocks, small caps and real estate. Helps the dollar and bank margins.",
         winners=["XLF", "UUP", "XLE"], losers=["QQQ", "IWM", "XLRE", "TLT", "XHB", "ARKK"],
         caveat="Markets price hikes in advance — the first reaction is often already over by the open."),
    dict(theme="Inflation hot", kw=["inflation rises", "hotter than expected", "cpi rises", "prices rise", "inflation surges", "sticky inflation", "ppi"],
         means="The Fed can't cut. Rates stay up, bonds and rate-sensitive stocks suffer, real assets hold value.",
         winners=["XLE", "GLD", "XLB", "UUP"], losers=["TLT", "XLRE", "IWM", "XLU", "QQQ"],
         caveat="One hot print rarely changes the trend; three in a row does."),
    dict(theme="Inflation cooling", kw=["inflation cools", "cooler than expected", "inflation slows", "disinflation", "prices fall", "inflation eases"],
         means="Opens the door to rate cuts. Everything rate-sensitive breathes.",
         winners=["IWM", "XLRE", "TLT", "XHB", "QQQ"], losers=["UUP"],
         caveat="Already-expected cooling is priced in; the surprise is what moves prices."),
    dict(theme="Jobs strong", kw=["jobs report", "payrolls beat", "unemployment falls", "strong hiring", "jobless claims fall", "labor market strong"],
         means="Economy healthy, but the Fed has less reason to cut. Good for cyclicals, mixed for growth.",
         winners=["XLI", "XLF", "XLY"], losers=["TLT", "XLU"],
         caveat="'Good news is bad news' when the market is obsessed with rate cuts."),
    dict(theme="Jobs weak / recession fear", kw=["layoffs", "unemployment rises", "payrolls miss", "recession", "job cuts", "jobless claims rise", "contraction", "slowdown"],
         means="Growth scare. Money runs to safety: bonds, utilities, staples, gold.",
         winners=["TLT", "XLU", "XLP", "GLD", "XLV"], losers=["IWM", "XLY", "XLF", "XLI", "HYG"],
         caveat="Recession scares happen 3-4 times a year; most don't become recessions."),
    dict(theme="Oil shock / Middle East", kw=["oil surges", "crude jumps", "opec", "oil prices", "strait of hormuz", "iran", "israel", "middle east", "pipeline", "oil spike"],
         means="Energy costs up → inflation up → consumers and airlines squeezed, energy producers rewarded.",
         winners=["XLE", "XOP", "OXY", "LMT"], losers=["JETS", "DAL", "CCL", "XLY", "EEM"],
         caveat="Geopolitical spikes in oil usually fade within weeks unless supply is actually cut."),
    dict(theme="Tariffs / trade war", kw=["tariff", "trade war", "import duties", "export controls", "sanctions", "trade deal", "trade talks"],
         means="Costs rise for importers, retaliation hits exporters and chips. Domestic-focused companies are safer.",
         winners=["IWM", "GLD", "XLU"], losers=["SMH", "AAPL", "EEM", "FXI", "XLB", "TSLA"],
         caveat="Headlines reverse fast — 'deal' rumors can flip everything in an hour."),
    dict(theme="Geopolitics / war", kw=["war", "missile", "invasion", "military strike", "attack", "troops", "escalation", "ceasefire"],
         means="Fear trade: defense, gold, oil and the dollar up; risk assets down. Ceasefire = the reverse.",
         winners=["ITA", "LMT", "RTX", "NOC", "GLD", "XLE"], losers=["EEM", "JETS", "QQQ"],
         caveat="Markets have historically recovered from geopolitical shocks within 1-3 months."),
    dict(theme="AI / chips boom", kw=["artificial intelligence", " ai ", "nvidia", "chips", "semiconductor", "data center", "gpu", "openai"],
         means="Capex flows to chips, servers, and the power to run them.",
         winners=["NVDA", "SMH", "AVGO", "TSM", "VRT", "VST", "CEG"], losers=[],
         caveat="Crowded trade — when it cracks it cracks hard. Watch NVDA's reaction to its own earnings."),
    dict(theme="Bank / credit stress", kw=["bank failure", "credit crunch", "default", "bankruptcy", "bank run", "deposits", "loan losses", "regional bank"],
         means="Plumbing problem. Financials and junk bonds sell, safe bonds and gold catch a bid.",
         winners=["TLT", "GLD", "XLU"], losers=["KRE", "XLF", "HYG", "XLRE"],
         caveat="The Fed usually backstops fast — the panic low is often within days."),
    dict(theme="China stimulus / growth", kw=["china stimulus", "beijing", "pboc", "china economy", "china growth", "china rebound"],
         means="China buying = commodities, emerging markets and luxury goods bid.",
         winners=["FXI", "KWEB", "EEM", "FCX", "XLB", "CAT"], losers=["UUP"],
         caveat="China stimulus announcements have disappointed repeatedly — fade the second-day chase."),
    dict(theme="Housing / mortgage rates", kw=["mortgage rates", "housing starts", "home sales", "homebuilder", "housing market"],
         means="Falling mortgage rates unlock buyers; builders and home-improvement move first.",
         winners=["XHB", "DHI", "LEN", "HD", "LOW"], losers=[],
         caveat="Builders move on the DIRECTION of rates, not the level."),
    dict(theme="Crypto momentum", kw=["bitcoin", "crypto", "ethereum", "btc", "coinbase", "stablecoin"],
         means="Risk-appetite barometer. Crypto strength usually spills into speculative tech.",
         winners=["COIN", "MSTR", "HOOD", "ARKK"], losers=[],
         caveat="Crypto trades 24/7 — weekend moves show up in stocks on Monday."),
    dict(theme="Government shutdown / debt ceiling", kw=["shutdown", "debt ceiling", "government funding", "budget deadline", "default on debt"],
         means="Noise more than signal historically — brief volatility, then resolution.",
         winners=["GLD"], losers=["ITA", "LMT", "IWM"],
         caveat="Every past shutdown has been followed by higher stocks within 3 months."),
    dict(theme="Earnings / guidance", kw=["earnings", "beats estimates", "misses estimates", "guidance", "raises outlook", "cuts outlook", "revenue"],
         means="Company-specific. The move after the first hour tells you how the big money interpreted it.",
         winners=[], losers=[],
         caveat="A stock that falls on GOOD earnings was over-owned — that's a tell."),
]

def _news_items(raw):
    """Normalize yfinance news across old/new formats."""
    items = []
    for n in raw or []:
        try:
            c = n.get("content", n)
            title = c.get("title") or n.get("title") or ""
            pub = c.get("provider", {}).get("displayName") if isinstance(c.get("provider"), dict) else n.get("publisher", "")
            link = (c.get("canonicalUrl") or {}).get("url") if isinstance(c.get("canonicalUrl"), dict) else n.get("link", "")
            ts = c.get("pubDate") or n.get("providerPublishTime")
            when = pd.to_datetime(ts, unit="s" if isinstance(ts, (int, float)) else None, utc=True, errors="coerce")
            summary = c.get("summary", "") or ""
            if title:
                items.append(dict(title=title, publisher=pub or "", link=link or "", when=when, summary=summary))
        except Exception:
            continue
    return items

@st.cache_data(ttl=900, show_spinner=False)
def get_news(symbols):
    seen, out = set(), []
    for s in symbols:
        try:
            for it in _news_items(yf.Ticker(s).news):
                key = it["title"].lower()[:80]
                if key in seen: continue
                seen.add(key); it["for"] = s; out.append(it)
        except Exception:
            continue
    out.sort(key=lambda z: (z["when"] if pd.notna(z["when"]) else pd.Timestamp(0, tz="UTC")), reverse=True)
    return out

def classify_news(items):
    """Tag each headline with themes. Returns (tagged items, theme counts)."""
    tagged, counts = [], {}
    for it in items:
        text = f" {it['title']} {it.get('summary','')} ".lower()
        themes = [r["theme"] for r in NEWS_RULES if any(k in text for k in r["kw"])]
        it = dict(it, themes=themes)
        tagged.append(it)
        for t in themes: counts[t] = counts.get(t, 0) + 1
    return tagged, counts

def plays_for_themes(theme_counts, data, extra_syms_data):
    """For each active theme: expected winners/losers + is the market ALREADY moving that way today?"""
    rules = {r["theme"]: r for r in NEWS_RULES}
    out = []
    for theme, n in sorted(theme_counts.items(), key=lambda z: -z[1]):
        r = rules[theme]
        if not r["winners"] and not r["losers"]:
            out.append(dict(theme=theme, n=n, means=r["means"], caveat=r["caveat"], winners=[], losers=[], confirm="n/a"))
            continue
        def move(s):
            src = data.get(s) or extra_syms_data.get(s)
            return chg(src["close"], 1) if src else np.nan
        w = [(s, move(s)) for s in r["winners"]]
        l = [(s, move(s)) for s in r["losers"]]
        wm = np.nanmean([m for _, m in w]) if w else np.nan
        lm = np.nanmean([m for _, m in l]) if l else np.nan
        spread = nz(wm, 0) - nz(lm, 0)
        confirm = "CONFIRMED — market is already trading this way" if spread > 0.6 else ("NOT YET — market hasn't reacted (early, or it doesn't care)" if spread > -0.6 else "OPPOSITE — market is trading against the headline (priced in / disbelieved)")
        out.append(dict(theme=theme, n=n, means=r["means"], caveat=r["caveat"], winners=w, losers=l, confirm=confirm, spread=spread))
    return out


# ======================================================================================================
# PRESENTATION LAYER — everything below only DISPLAYS engine output. No financial logic lives here.
# ======================================================================================================
P = dict(bg="#F7F9FC", text="#101828", text2="#667085", muted="#98A2B3", blue="#2563EB", blue_l="#EAF2FF",
         up="#12B76A", up_d="#067647", up_bg="#ECFDF3", down="#F04438", down_d="#B42318", down_bg="#FEF3F2",
         warn="#F79009", warn_d="#B54708", warn_bg="#FFFAEB", navy="#0B1B3F", border="rgba(16,24,40,0.07)")
PLOT = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, 'SF Pro Text', -apple-system, 'Segoe UI', sans-serif", color=P["text2"], size=12),
            margin=dict(l=4, r=4, t=10, b=4),
            xaxis=dict(gridcolor="rgba(16,24,40,.05)", zeroline=False, showline=False, tickfont=dict(color=P["muted"])),
            yaxis=dict(gridcolor="rgba(16,24,40,.05)", zeroline=False, showline=False, tickfont=dict(color=P["muted"])),
            hoverlabel=dict(bgcolor="#FFFFFF", bordercolor="#EAECF0", font=dict(color=P["text"], family="Inter")),
            legend=dict(orientation="h", y=-0.18, bgcolor="rgba(0,0,0,0)", font=dict(color=P["text2"])))
PCFG = dict(displayModeBar=False, responsive=True)

def _clean(t):
    """Strip indentation so Markdown never turns raw HTML into a code block."""
    return "\n".join(l.strip() for l in t.split("\n") if l.strip())

def html(t):
    st.markdown(_clean(t), unsafe_allow_html=True)

ICON = {
    "up": '<polyline points="22 7 13.5 15.5 8.5 10.5 2 17"/><polyline points="16 7 22 7 22 13"/>',
    "down": '<polyline points="22 17 13.5 8.5 8.5 13.5 2 7"/><polyline points="16 17 22 17 22 11"/>',
    "warn": '<path d="M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>',
    "neu": '<line x1="6" y1="12" x2="18" y2="12"/>',
    "check": '<polyline points="20 6 9 17 4 12"/>',
    "arrow_up": '<line x1="12" y1="19" x2="12" y2="5"/><polyline points="5 12 12 5 19 12"/>',
    "arrow_down": '<line x1="12" y1="5" x2="12" y2="19"/><polyline points="19 12 12 19 5 12"/>',
    "search": '<circle cx="11" cy="11" r="7"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>',
    "crop": '<path d="M6 2v14a2 2 0 0 0 2 2h14"/><path d="M18 22V8a2 2 0 0 0-2-2H2"/>',
}
def svg(name, size=16, color="currentColor", stroke=2):
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="{stroke}" '
            f'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" style="flex:0 0 auto">{ICON[name]}</svg>')

STATUS = {"🟢": ("up", "Bullish"), "🔴": ("down", "Bearish"), "🟡": ("warn", "Watch"), "⚪": ("neu", "Neutral")}

def logo_mark(size=28):
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 32 32" aria-label="Market Reader">'
            '<defs><linearGradient id="mrlg" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#0B1B3F"/><stop offset="1" stop-color="#2563EB"/></linearGradient></defs>'
            '<rect width="32" height="32" rx="9" fill="url(#mrlg)"/>'
            '<rect x="7" y="17" width="3.4" height="8" rx="1.2" fill="#fff" opacity=".55"/><rect x="12.3" y="13" width="3.4" height="12" rx="1.2" fill="#fff" opacity=".75"/>'
            '<rect x="17.6" y="9.5" width="3.4" height="15.5" rx="1.2" fill="#fff" opacity=".92"/>'
            '<path d="M6.5 14.5 L12 10.5 L17 12 L25.5 6" stroke="#9CC2FF" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round"/>'
            '<circle cx="25.5" cy="6" r="2" fill="#9CC2FF"/></svg>')

# ------------------------------------------------------------------------------------------------ CSS
def inject_global_css():
    search_svg = ("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='18' height='18' viewBox='0 0 24 24' fill='none' "
                  "stroke='%2398A2B3' stroke-width='2' stroke-linecap='round'><circle cx='11' cy='11' r='7'/><line x1='21' y1='21' x2='16.65' y2='16.65'/></svg>")
    css = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;650;700&display=swap" rel="stylesheet">
<style>
:root { --bg:#F7F9FC; --text:#101828; --text2:#667085; --muted:#98A2B3; --blue:#2563EB; --blue-l:#EAF2FF; --up:#12B76A; --up-d:#067647; --up-bg:#ECFDF3;
  --down:#F04438; --down-d:#B42318; --down-bg:#FEF3F2; --warn:#F79009; --warn-d:#B54708; --warn-bg:#FFFAEB; --navy:#0B1B3F; --border:rgba(16,24,40,0.07);
  --font: Inter, "SF Pro Display", "SF Pro Text", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
html, body, .stApp, .stApp p, .stApp label, .stApp input, .stApp textarea, .stApp button { font-family: var(--font); }
.stApp { color: var(--text); overflow-x: hidden;
  background: radial-gradient(1100px 540px at 90% -10%, #E6F0FF 0%, rgba(247,249,252,0) 62%),
              radial-gradient(900px 520px at -8% 38%, #F0F5FF 0%, rgba(247,249,252,0) 58%), var(--bg); }
/* remove Streamlit chrome */
header[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stStatusWidget"],
#MainMenu, footer, .stDeployButton, [data-testid="stAppDeployButton"], section[data-testid="stSidebar"],
[data-testid="collapsedControl"], [data-testid="stSidebarCollapsedControl"] { display: none !important; }
.block-container { max-width: 1500px !important; margin: 0 auto !important; padding: 6px 44px 64px 44px !important; }
[data-testid="stVerticalBlock"] { gap: 0.6rem; }
.element-container:has(iframe[height="0"]) { display: none !important; }
.stApp p { line-height: 1.55; }
a { color: var(--blue); }
/* ---------- navigation */
.mr-nav { display: flex; align-items: center; gap: 30px; height: 60px; border-bottom: 1px solid var(--border); }
.mr-brand { display: flex; align-items: center; gap: 10px; font-weight: 650; font-size: 16px; letter-spacing: -0.01em; color: var(--text); text-decoration: none; }
.mr-links { display: flex; gap: 26px; margin-left: 14px; }
.mr-links a { color: var(--text2); font-size: 14px; font-weight: 500; text-decoration: none; cursor: pointer; transition: color .18s ease; }
.mr-links a:hover { color: var(--text); }
.mr-navr { margin-left: auto; display: flex; align-items: center; gap: 12px; }
.mr-status { display: flex; align-items: center; gap: 7px; font-size: 12.5px; color: var(--text2); white-space: nowrap; }
.mr-live { width: 7px; height: 7px; border-radius: 50%; background: var(--muted); }
.mr-live.on { background: var(--up); box-shadow: 0 0 0 4px rgba(18,183,106,.15); }
.mr-iconbtn { width: 36px; height: 36px; border-radius: 50%; display: inline-flex; align-items: center; justify-content: center; color: var(--text);
  background: rgba(255,255,255,.8); border: 1px solid var(--border); cursor: pointer; transition: transform .18s ease, box-shadow .18s ease; }
.mr-cta { background: var(--navy); color: #fff !important; padding: 9px 18px; border-radius: 999px; font-size: 13.5px; font-weight: 600; text-decoration: none;
  cursor: pointer; white-space: nowrap; transition: transform .18s ease, box-shadow .18s ease; }
.mr-cta:hover, .mr-iconbtn:hover { transform: translateY(-1px); box-shadow: 0 8px 22px rgba(11,27,63,.18); }
.mr-cta:focus-visible, .mr-links a:focus-visible, .mr-iconbtn:focus-visible { outline: 2px solid var(--blue); outline-offset: 3px; }
/* ---------- stock header */
.mr-id { display: flex; align-items: center; gap: 12px; margin-top: 18px; }
.mr-co { width: 46px; height: 46px; border-radius: 13px; background-color: #fff; background-position: center; background-size: 26px 26px; background-repeat: no-repeat;
  border: 1px solid var(--border); box-shadow: 0 6px 16px rgba(16,24,40,.06); display: flex; align-items: center; justify-content: center; font-weight: 700; color: var(--text); flex: 0 0 auto; }
.mr-tk { font-size: clamp(30px, 2.7vw, 40px); font-weight: 700; letter-spacing: -0.02em; line-height: 1; color: var(--text); }
.mr-meta { font-size: 13px; color: var(--text2); margin-top: 4px; }
.mr-pricerow { display: flex; align-items: baseline; gap: 14px; flex-wrap: wrap; margin: 14px 0 2px 0; }
.mr-price { font-size: clamp(44px, 4.6vw, 66px); font-weight: 500; letter-spacing: -2px; line-height: 1; color: var(--text); font-variant-numeric: tabular-nums; }
.mr-chg { display: inline-flex; align-items: center; gap: 5px; font-size: 14.5px; font-weight: 600; padding: 4px 10px; border-radius: 999px; }
.mr-chg.up { color: var(--up-d); background: var(--up-bg); } .mr-chg.down { color: var(--down-d); background: var(--down-bg); }
/* search */
[data-testid="stTextInput"] label { display: none; }
[data-testid="stTextInput"] input { height: 48px; border-radius: 999px !important; padding-left: 46px !important; font-size: 14.5px; color: var(--text);
  background: rgba(255,255,255,.9) url("SEARCH_SVG") 17px center / 18px 18px no-repeat !important; border: 1px solid rgba(16,24,40,.08) !important;
  box-shadow: 0 8px 24px rgba(16,24,40,.04); transition: box-shadow .18s ease, border-color .18s ease; }
[data-testid="stTextInput"] input:focus { border-color: rgba(37,99,235,.45) !important; box-shadow: 0 0 0 4px rgba(37,99,235,.10) !important; }
[data-testid="stTextInput"] > div > div { border: 0 !important; background: transparent !important; }
[data-testid="InputInstructions"] { display: none; }
/* popover + buttons */
[data-testid="stPopover"] button, [data-testid="stPopoverButton"] { height: 48px; border-radius: 999px !important; background: rgba(255,255,255,.9) !important;
  border: 1px solid rgba(16,24,40,.08) !important; color: var(--text) !important; font-weight: 600 !important; }
.stButton > button, .stDownloadButton > button { background: var(--navy); color: #fff; border: 0; border-radius: 999px; padding: 9px 20px; font-weight: 600;
  transition: transform .18s ease, box-shadow .18s ease; }
.stButton > button:hover, .stDownloadButton > button:hover { color: #fff; transform: translateY(-1px); box-shadow: 0 10px 24px rgba(11,27,63,.2); }
.stButton > button:focus-visible { outline: 2px solid var(--blue); outline-offset: 3px; }
/* ---------- tabs (SaaS underline navigation) */ .mr-nav a { text-decoration: none !important; } .mr-brand { color: var(--text) !important; } .stTabs [role="tablist"] { gap: 30px; border-bottom: 1px solid var(--border); overflow-x: auto; flex-wrap: nowrap; scrollbar-width: none; } .stTabs [data-testid="stTab"] p { font-size: 14.5px !important; font-weight: 500 !important; color: var(--muted) !important; white-space: nowrap; } .stTabs [data-testid="stTab"][aria-selected="true"] p { color: var(--text) !important; font-weight: 600 !important; } .stTabs .react-aria-SelectionIndicator { background-color: var(--blue) !important; height: 2px !important; }
.stTabs [data-baseweb="tab-list"] { gap: 30px; border-bottom: 1px solid var(--border); background: transparent; overflow-x: auto; scrollbar-width: none; flex-wrap: nowrap; }
.stTabs [data-baseweb="tab-list"]::-webkit-scrollbar { display: none; }
.stTabs [data-baseweb="tab"] { height: 46px; padding: 0 !important; background: transparent !important; white-space: nowrap; }
.stTabs [data-baseweb="tab"] p { font-size: 14.5px !important; font-weight: 500 !important; color: var(--muted) !important; transition: color .18s ease; }
.stTabs [data-baseweb="tab"]:hover p { color: var(--text2) !important; }
.stTabs [aria-selected="true"] p { color: var(--text) !important; font-weight: 600 !important; }
.stTabs [data-baseweb="tab-highlight"] { background: var(--blue) !important; height: 2px !important; border-radius: 2px; transition: all .25s ease; }
.stTabs [data-baseweb="tab-border"] { display: none; }
.stTabs [data-baseweb="tab-panel"] { padding-top: 22px; }
/* ---------- cards + type */
.mr-card { background: rgba(255,255,255,.70); border: 1px solid rgba(16,24,40,.06); border-radius: 22px; box-shadow: 0 20px 60px rgba(16,24,40,.05);
  backdrop-filter: blur(20px); -webkit-backdrop-filter: blur(20px); padding: 20px 22px; transition: transform .18s ease, box-shadow .18s ease; min-width: 0; }
.mr-card:hover { transform: translateY(-2px); box-shadow: 0 26px 70px rgba(16,24,40,.08); }
.mr-lbl { font-size: 12.5px; color: var(--text2); font-weight: 500; }
.mr-num { font-size: clamp(32px, 3.1vw, 46px); font-weight: 500; letter-spacing: -1.5px; line-height: 1.05; color: var(--text); font-variant-numeric: tabular-nums; margin: 6px 0 4px; }
.mr-num small { font-size: .42em; color: var(--muted); letter-spacing: 0; margin-left: 2px; font-weight: 500; }
.mr-num.sm { font-size: clamp(24px, 2.2vw, 32px); letter-spacing: -1px; }
.mr-foot { font-size: 12.5px; color: var(--text2); line-height: 1.45; }
.mr-badge { display: inline-flex; align-items: center; gap: 5px; font-size: 12.5px; font-weight: 600; padding: 3px 9px; border-radius: 999px; }
.mr-badge.up { color: var(--up-d); background: var(--up-bg); } .mr-badge.down { color: var(--down-d); background: var(--down-bg); }
.mr-badge.warn { color: var(--warn-d); background: var(--warn-bg); } .mr-badge.neu { color: #344054; background: #F2F4F7; } .mr-badge.blue { color: #1849A9; background: var(--blue-l); }
.mr-grid2 { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; }
.mr-grid3 { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 14px; }
.mr-grid4 { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 14px; }
.mr-h2 { font-size: clamp(18px, 1.5vw, 21px); font-weight: 650; letter-spacing: -0.01em; color: var(--text); margin: 22px 0 2px 0; }
.mr-sub { font-size: 14px; color: var(--text2); margin: 0 0 10px 0; }
.mr-verdict { font-size: 15px; color: var(--text); margin: 14px 0 0 0; display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
/* ---------- signal rows */
.mr-sig { display: flex; gap: 14px; padding: 15px 0; border-bottom: 1px solid var(--border); }
.mr-sig:last-child { border-bottom: 0; }
.mr-dot { width: 28px; height: 28px; border-radius: 50%; flex: 0 0 auto; display: flex; align-items: center; justify-content: center; margin-top: 1px; }
.mr-dot.up { background: var(--up-bg); color: var(--up); } .mr-dot.down { background: var(--down-bg); color: var(--down); }
.mr-dot.warn { background: var(--warn-bg); color: var(--warn); } .mr-dot.neu { background: #F2F4F7; color: var(--text2); }
.mr-sig .t { font-weight: 600; font-size: 15px; color: var(--text); line-height: 1.35; }
.mr-sig .st { font-size: 11.5px; font-weight: 600; margin-left: 8px; letter-spacing: .02em; text-transform: uppercase; }
.mr-sig .st.up { color: var(--up-d); } .mr-sig .st.down { color: var(--down-d); } .mr-sig .st.warn { color: var(--warn-d); } .mr-sig .st.neu { color: var(--text2); }
.mr-sig .b { font-size: 14px; color: var(--text2); line-height: 1.55; margin-top: 3px; }
.mr-sig .c { font-size: 12.5px; color: var(--muted); margin-top: 5px; }
/* factor bars */
.mr-fac { display: grid; grid-template-columns: 110px 1fr 34px; align-items: center; gap: 12px; padding: 7px 0; }
.mr-fac .n { font-size: 13.5px; font-weight: 500; color: var(--text); }
.mr-fac .bar { height: 6px; border-radius: 6px; background: #EEF2F7; overflow: hidden; }
.mr-fac .bar > div { height: 6px; border-radius: 6px; }
.mr-fac .v { font-size: 13.5px; font-weight: 600; text-align: right; font-variant-numeric: tabular-nums; }
.mr-fac-note { font-size: 12px; color: var(--muted); margin: -4px 0 4px 122px; }
/* scenario cards */
.mr-scn .k { font-size: 12px; font-weight: 650; letter-spacing: .08em; text-transform: uppercase; display: flex; align-items: center; gap: 6px; }
.mr-scn.bull .k { color: var(--up-d); } .mr-scn.base .k { color: #1849A9; } .mr-scn.bear .k { color: var(--down-d); }
.mr-scn .rng { font-size: 15px; font-weight: 600; color: var(--text); font-variant-numeric: tabular-nums; }
.mr-scn ul { margin: 10px 0 0 0; padding-left: 16px; } .mr-scn li { font-size: 13px; color: var(--text2); line-height: 1.5; }
/* misc streamlit surfaces */
[data-testid="stPlotlyChart"] { background: transparent; }
div[data-testid="stExpander"] { background: rgba(255,255,255,.7); border: 1px solid rgba(16,24,40,.06); border-radius: 18px; }
div[data-testid="stExpander"] summary p { font-weight: 600; font-size: 14px; }
div[data-testid="stDataFrame"] { border: 1px solid rgba(16,24,40,.06); border-radius: 16px; overflow: hidden; }
[data-testid="stNumberInput"] input, [data-testid="stTextArea"] textarea, [data-baseweb="select"] > div { border-radius: 12px !important; }

/* ================= MOTION SYSTEM ================= */
@keyframes mrUp { from { opacity: 0; transform: translateY(14px); filter: blur(4px); } to { opacity: 1; transform: none; filter: none; } }
@keyframes mrFade { from { opacity: 0; } to { opacity: 1; } }
@keyframes mrSlideIn { from { opacity: 0; transform: translateX(-10px); } to { opacity: 1; transform: none; } }
@keyframes mrPanel { from { opacity: 0; transform: translateY(18px) scale(.985); } to { opacity: 1; transform: none; } }
@keyframes mrDraw { from { stroke-dashoffset: var(--len, 1200); } to { stroke-dashoffset: 0; } }
@keyframes mrDrift { from { transform: translateX(0); } to { transform: translateX(-50%); } }
@keyframes mrSpin { from { transform: translateZ(calc(var(--R) * -1)) rotateY(0deg); } to { transform: translateZ(calc(var(--R) * -1)) rotateY(-360deg); } }
@keyframes mrTape { from { transform: translateX(0); } to { transform: translateX(-50%); } }
@keyframes mrPulse { 0%,100% { box-shadow: 0 0 0 0 rgba(18,183,106,.35); } 50% { box-shadow: 0 0 0 6px rgba(18,183,106,0); } }
@keyframes mrShine { from { transform: translateX(-120%) skewX(-18deg); } to { transform: translateX(220%) skewX(-18deg); } }
@keyframes mrRing { from { stroke-dashoffset: var(--c); } }
.mr-nav { animation: mrFade .6s ease both; }
.mr-id { animation: mrUp .7s cubic-bezier(.2,.7,.2,1) .05s both; }
.mr-pricerow { animation: mrUp .7s cubic-bezier(.2,.7,.2,1) .15s both; }
.mr-spark { animation: mrFade .8s ease .3s both; }
.mr-spark path.l { stroke-dasharray: var(--len); animation: mrDraw 1.8s cubic-bezier(.3,.6,.2,1) .35s both; }
.stTabs [data-baseweb="tab-list"] { animation: mrFade .6s ease .25s both; }
.stTabs [data-baseweb="tab-panel"] > div { animation: mrPanel .55s cubic-bezier(.2,.7,.2,1) both; }
.mr-card { animation: mrUp .7s cubic-bezier(.2,.7,.2,1) both; position: relative; overflow: hidden; }
.mr-grid2 > .mr-card:nth-child(2), .mr-grid3 > .mr-card:nth-child(2), .mr-grid4 > .mr-card:nth-child(2) { animation-delay: .08s; }
.mr-grid3 > .mr-card:nth-child(3), .mr-grid4 > .mr-card:nth-child(3) { animation-delay: .16s; }
.mr-grid4 > .mr-card:nth-child(4) { animation-delay: .24s; }
.mr-card::after { content: ""; position: absolute; top: 0; bottom: 0; width: 40%; left: 0; pointer-events: none;
  background: linear-gradient(90deg, rgba(255,255,255,0), rgba(255,255,255,.55), rgba(255,255,255,0)); transform: translateX(-120%) skewX(-18deg); }
.mr-card:hover::after { animation: mrShine .9s ease; }
.mr-sig { animation: mrSlideIn .55s cubic-bezier(.2,.7,.2,1) both; }
.mr-sig:nth-child(2) { animation-delay: .07s; } .mr-sig:nth-child(3) { animation-delay: .14s; } .mr-sig:nth-child(4) { animation-delay: .21s; }
.mr-sig:nth-child(5) { animation-delay: .28s; } .mr-sig:nth-child(n+6) { animation-delay: .35s; }
.mr-fac .bar > div { transform-origin: left; animation: mrGrow 1.1s cubic-bezier(.2,.7,.2,1) .2s both; }
@keyframes mrGrow { from { transform: scaleX(0); } to { transform: scaleX(1); } }
.mr-live.on { animation: mrPulse 2s ease-in-out infinite; }
[data-testid="stPlotlyChart"] { animation: mrFade .9s ease .15s both; }
/* ticker tape */
.mr-tape { position: relative; overflow: hidden; height: 38px; border-bottom: 1px solid var(--border);
  -webkit-mask-image: linear-gradient(90deg, transparent, #000 6%, #000 94%, transparent); mask-image: linear-gradient(90deg, transparent, #000 6%, #000 94%, transparent); }
.mr-tape-track { display: flex; width: max-content; gap: 34px; align-items: center; height: 38px; animation: mrTape 60s linear infinite; }
.mr-tape:hover .mr-tape-track { animation-play-state: paused; }
.mr-tk-item { display: inline-flex; gap: 8px; align-items: baseline; font-size: 12.5px; white-space: nowrap; color: var(--text2); }
.mr-tk-item b { color: var(--text); font-weight: 600; } .mr-tk-item .u { color: var(--up-d); font-weight: 600; } .mr-tk-item .dn { color: var(--down-d); font-weight: 600; }
/* hero spark */
.mr-herorow { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1.1fr); gap: 18px; align-items: end; }
.mr-spark { width: 100%; height: 86px; }
/* score ring inside composite card */
.mr-ringwrap { display: flex; align-items: center; gap: 16px; }
.mr-ringwrap svg circle.p { animation: mrRing 1.6s cubic-bezier(.2,.7,.2,1) .2s both; }
/* coverflow carousel (continuous, like a product showcase) */
.mr-stage { position: relative; height: 540px; perspective: 1400px; overflow: hidden; border-radius: 26px;
  background: radial-gradient(70% 60% at 50% 55%, rgba(234,242,255,.95) 0%, rgba(247,249,252,0) 70%);
  -webkit-mask-image: radial-gradient(130% 110% at 50% 45%, #000 62%, transparent 100%); mask-image: radial-gradient(130% 110% at 50% 45%, #000 62%, transparent 100%); }
.mr-flow { position: absolute; left: 50%; top: 50%; width: 220px; height: 310px; margin: -170px 0 0 -110px; transform-style: preserve-3d; }
.mr-stage:hover .mr-face { animation-play-state: paused; }
@keyframes mrFlow {
  0%, 12.6%  { transform: translateX(230%) scale(.66) rotateY(-34deg); opacity: 0; z-index: 1; }
  16.6%, 29.3% { transform: translateX(112%) scale(.84) rotateY(-24deg); opacity: .8; z-index: 2; }
  33.3%, 46% { transform: translateX(0) scale(1) rotateY(0deg); opacity: 1; z-index: 3; }
  50%, 62.6% { transform: translateX(-112%) scale(.84) rotateY(24deg); opacity: .8; z-index: 2; }
  66.6%, 79.3% { transform: translateX(-230%) scale(.66) rotateY(34deg); opacity: 0; z-index: 1; }
  83.3%, 96% { transform: translateX(230%) scale(.66) rotateY(-34deg); opacity: 0; z-index: 0; }
  100% { transform: translateX(230%) scale(.66) rotateY(-34deg); opacity: 0; z-index: 1; } }
.mr-face { position: absolute; inset: 0; border-radius: 24px; padding: 22px 20px; display: flex; flex-direction: column; justify-content: space-between;
  background: linear-gradient(160deg, rgba(255,255,255,.94) 0%, rgba(255,255,255,.50) 100%); border: 1px solid rgba(255,255,255,.95);
  box-shadow: 0 30px 70px rgba(16,24,40,.12), inset 0 1px 0 #fff; backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px);
  animation: mrFlow 36s cubic-bezier(.65,0,.35,1) infinite; animation-delay: var(--dl); opacity: 0; }
.mr-face::before { content: ""; position: absolute; inset: -40px; border-radius: 60px; z-index: -1; filter: blur(26px);
  background: radial-gradient(circle at 50% 38%, var(--glow) 0%, rgba(255,255,255,0) 64%); }
.mr-face .ic { width: 44px; height: 44px; border-radius: 14px; display: flex; align-items: center; justify-content: center; background: var(--icbg); color: var(--iccol); }
.mr-face .n { font-size: 44px; font-weight: 500; letter-spacing: -1.5px; color: var(--text); line-height: 1; font-variant-numeric: tabular-nums; }
.mr-face .n small { font-size: 15px; color: var(--muted); letter-spacing: 0; margin-left: 2px; }
.mr-face .t { font-size: 15px; font-weight: 600; color: var(--text); margin-top: 8px; }
.mr-face .d { font-size: 12.5px; color: var(--text2); line-height: 1.45; margin-top: 4px; }
.mr-face .mini { width: 100%; height: 34px; margin-top: 10px; }
.mr-dots { position: absolute; bottom: 26px; left: 0; right: 0; text-align: center; font-size: 12px; color: var(--text2); letter-spacing: .02em; }
.mr-dots b { color: var(--text); font-weight: 500; }
@media (max-width: 767px) { .mr-dots .x { display: none; } }
.mr-stage .mr-wave { position: absolute; left: 0; top: 0; width: 200%; height: 100%; animation: mrDrift 38s linear infinite; }
.mr-stage .mr-wave path { stroke-dasharray: 2200; animation: mrDraw 3.2s cubic-bezier(.3,.6,.2,1) both; --len: 2200; }
.mr-floor { position: absolute; bottom: 58px; left: 50%; width: 640px; height: 110px; transform: translateX(-50%); border-radius: 50%;
  background: radial-gradient(closest-side, rgba(126,167,232,.28), rgba(126,167,232,0)); filter: blur(8px); }
@media (prefers-reduced-motion: reduce) { *, *::before, *::after { animation: none !important; transition: none !important; } }
@media (max-width: 1199px) { .mr-stage { height: 460px; } .mr-flow { width: 180px; height: 262px; margin: -150px 0 0 -90px; } .mr-face .n { font-size: 36px; } .mr-face .d { display: none; } }
@media (max-width: 767px) {
  .mr-herorow { grid-template-columns: 1fr; gap: 6px; } .mr-spark { height: 64px; }
  .mr-stage { height: 330px; perspective: 1000px; } .mr-flow { width: 150px; height: 214px; margin: -120px 0 0 -75px; } .mr-dots { bottom: 14px; }
  .mr-face { padding: 14px 12px; border-radius: 18px; } .mr-face .n { font-size: 28px; } .mr-face .t { font-size: 12.5px; } .mr-face .ic { width: 34px; height: 34px; border-radius: 11px; }
  .mr-face .mini { height: 26px; } .mr-floor { width: 360px; bottom: 30px; } .mr-tape { height: 34px; } .mr-tape-track { height: 34px; }
}

/* ---------- responsive */
@media (max-width: 1199px) {
  .block-container { padding: 6px 28px 52px 28px !important; }
  .mr-links { gap: 18px; } .mr-status { display: none; }
  .mr-grid4 { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
@media (max-width: 767px) {
  .block-container { padding: 4px 16px 44px 16px !important; }
  .mr-links, .mr-status, .mr-iconbtn { display: none; }
  .mr-nav { height: 52px; gap: 10px; }
  .mr-id { margin-top: 12px; } .mr-tk { font-size: 28px; } .mr-co { width: 40px; height: 40px; border-radius: 11px; }
  .mr-price { font-size: 42px; letter-spacing: -1.5px; } .mr-pricerow { margin: 10px 0 0 0; }
  [data-testid="stHorizontalBlock"] { flex-wrap: wrap !important; gap: .6rem !important; }
  [data-testid="stHorizontalBlock"] > div { width: 100% !important; flex: 1 1 100% !important; min-width: 100% !important; }
  .mr-grid3 { grid-template-columns: 1fr; }
  .mr-grid4 { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .mr-card { padding: 16px; border-radius: 18px; }
  .mr-num { font-size: 32px; }
  .stTabs [data-baseweb="tab-list"] { gap: 22px; }
  .stTabs [data-baseweb="tab-panel"] { padding-top: 16px; }
  .mr-fac { grid-template-columns: 92px 1fr 30px; } .mr-fac-note { margin-left: 104px; }
}
@media (max-width: 429px) {
  .mr-cta { padding: 8px 13px; font-size: 12.5px; }
  .mr-brand span { font-size: 15px; }
  .mr-num { font-size: 30px; }
}
</style>
""".replace("SEARCH_SVG", search_svg)
    st.markdown(_clean(css), unsafe_allow_html=True)

# ------------------------------------------------------------------------------------------------ components
def render_navigation(mkt_open, now_et):
    html(f"""
    <div class="mr-nav">
      <a class="mr-brand" data-mr-tab="Verdict" href="#">{logo_mark(28)}<span>Market Reader</span></a>
      <nav class="mr-links" aria-label="Main">
        <a data-mr-tab="Verdict" href="#">Home</a>
        <a data-mr-tab="Between the Lines" href="#">Stocks</a>
        <a data-mr-tab="Screener" href="#">Watchlist</a>
        <a data-mr-tab="Market Insights" href="#">Insights</a>
      </nav>
      <div class="mr-navr">
        <div class="mr-status"><span class="mr-live {'on' if mkt_open else ''}"></span>{'Market open' if mkt_open else 'Market closed'} · {now_et.strftime('%b %d, %I:%M %p ET')}</div>
        <a class="mr-iconbtn" data-mr-focus="search" href="#" aria-label="Search">{svg('search', 17)}</a>
        <a class="mr-cta" data-mr-focus="search" href="#">Get Started</a>
      </div>
    </div>""")

def render_mobile_navigation():
    """Wires nav links to the real tabs and Search / Get Started to the ticker search (desktop + mobile)."""
    st.components.v1.html("""<script>
    (function(){ const d = window.parent.document;
      function wire(){
        d.querySelectorAll('[data-mr-tab]').forEach(a => { if (a.dataset.w) return; a.dataset.w = 1;
          a.addEventListener('click', e => { e.preventDefault();
            const t = [...d.querySelectorAll('.stTabs [data-baseweb="tab"]')].find(x => x.innerText.trim() === a.dataset.mrTab);
            if (t) { t.click(); t.scrollIntoView({behavior: 'smooth', block: 'center'}); } }); });
        d.querySelectorAll('[data-mr-focus]').forEach(a => { if (a.dataset.w) return; a.dataset.w = 1;
          a.addEventListener('click', e => { e.preventDefault(); const i = d.querySelector('[data-testid="stTextInput"] input');
            if (i) { i.scrollIntoView({behavior: 'smooth', block: 'center'}); i.focus(); i.select(); } }); });
      }
      function countUp(){
        d.querySelectorAll('.mr-count:not([data-done])').forEach(el => { el.setAttribute('data-done', '1');
          const raw = el.textContent.replace(/[^0-9.\-]/g, ''); const num = parseFloat(raw); if (isNaN(num)) return;
          const dec = parseInt(el.dataset.dec || '0'); const pre = el.dataset.pre || ''; const t0 = performance.now(), dur = 1100;
          const fmt = v => pre + v.toLocaleString(undefined, {minimumFractionDigits: dec, maximumFractionDigits: dec});
          (function step(t){ const p = Math.min(1, (t - t0) / dur), e = 1 - Math.pow(1 - p, 4); el.textContent = fmt(num * e); if (p < 1) requestAnimationFrame(step); })(t0);
        });
      }
      function all(){ wire(); countUp(); }
      all(); new MutationObserver(all).observe(d.body, {childList: true, subtree: true});
    })();</script>""", height=0)

def company_logo(info, tk):
    site = str(safe(info, "website", "") or "")
    dom = site.replace("https://", "").replace("http://", "").replace("www.", "").split("/")[0]
    if dom:
        return f'<div class="mr-co" style="background-image:url(https://www.google.com/s2/favicons?domain={dom}&sz=128)" role="img" aria-label="{tk} logo"></div>'
    return f'<div class="mr-co">{tk[:1]}</div>'

def render_stock_header(ticker, info, last, d_close):
    r1 = nz(last["Ret1"]) * 100
    kind = "up" if r1 >= 0 else "down"
    exch = {"NMS": "NASDAQ", "NGM": "NASDAQ", "NCM": "NASDAQ", "NYQ": "NYSE", "ASE": "NYSE American", "PCX": "NYSE Arca"}.get(str(safe(info, "exchange", "")), str(safe(info, "exchange", "")))
    meta = " • ".join(x for x in [str(safe(info, "longName", "") or ""), exch] if x and x != "nan")
    html(f"""
    <div class="mr-id">{company_logo(info, ticker)}<div><div class="mr-tk">{ticker}</div><div class="mr-meta">{meta}</div></div></div>
    <div class="mr-herorow"><div class="mr-pricerow"><div class="mr-price"><span class="mr-count" data-dec="2" data-pre="$">${last['Close']:,.2f}</span></div>
      <span class="mr-chg {kind}">{svg('arrow_up' if r1 >= 0 else 'arrow_down', 14, stroke=2.5)}{abs(r1):.2f}% today</span></div>
      {spark_svg(d_close, 'mr-spark', 600, 86)}</div>""")


def spark_svg(series, cls, w=600, h=86, color=None):
    """Animated area sparkline (draws itself in). Real closing prices."""
    v = np.asarray(series, dtype=float); v = v[~np.isnan(v)]
    if len(v) < 5:
        return ""
    lo, hi = v.min(), v.max(); rng_ = (hi - lo) or 1
    xs = np.linspace(0, w, len(v)); ys = h - 6 - (v - lo) / rng_ * (h - 14)
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
    up = v[-1] >= v[0]; col = color or (P["up"] if up else P["down"])
    seg = np.sqrt(np.diff(xs) ** 2 + np.diff(ys) ** 2).sum()
    gid = f"g{abs(hash((cls, len(v), round(float(v[-1]), 4)))) % 10**8}"
    return (f'<svg class="{cls}" viewBox="0 0 {w} {h}" preserveAspectRatio="none" aria-hidden="true" style="--len:{seg:.0f}">'
            f'<defs><linearGradient id="{gid}" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="{col}" stop-opacity=".18"/><stop offset="1" stop-color="{col}" stop-opacity="0"/></linearGradient></defs>'
            f'<path d="M0,{h} L{pts.replace(" ", " L")} L{w},{h} Z" fill="url(#{gid})" style="animation:mrFade 1.2s ease .9s both"/>'
            f'<path class="l" d="M{pts.replace(" ", " L")}" fill="none" stroke="{col}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round" vector-effect="non-scaling-stroke"/>'
            f'<circle cx="{xs[-1]:.1f}" cy="{ys[-1]:.1f}" r="3.5" fill="{col}" style="animation:mrFade .4s ease 2s both"/></svg>')

def ring_svg(value, color, size=64, stroke=6):
    value = float(clamp(nz(value, 0))); r = (size - stroke) / 2; c = 2 * np.pi * r; off = c * (1 - value / 100)
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 {size} {size}" aria-hidden="true" style="flex:0 0 auto">'
            f'<circle cx="{size/2}" cy="{size/2}" r="{r}" stroke="#EEF2F7" stroke-width="{stroke}" fill="none"/>'
            f'<circle class="p" cx="{size/2}" cy="{size/2}" r="{r}" stroke="{color}" stroke-width="{stroke}" fill="none" stroke-linecap="round" '
            f'transform="rotate(-90 {size/2} {size/2})" stroke-dasharray="{c:.1f}" stroke-dashoffset="{off:.1f}" style="--c:{c:.1f}"/></svg>')

def render_ticker_tape(snap):
    """Seamless infinite tape of live global markets (real snapshot data)."""
    if snap is None or snap.empty:
        return
    items = []
    for _, r in snap.iterrows():
        v = r["Last"]; sym = r["Symbol"]
        val = f"{v/10:.2f}%" if sym in ("^TNX", "^IRX") else (f"{v:,.3f}" if "=X" in sym else (f"{v:,.0f}" if v > 1000 else f"{v:,.2f}"))
        ch = nz(r["D1"], 0); cls = "u" if ch >= 0 else "dn"
        items.append(f'<span class="mr-tk-item"><b>{r["Name"]}</b>{val}<span class="{cls}">{"▲" if ch >= 0 else "▼"} {abs(ch):.2f}%</span></span>')
    row = "".join(items)
    html(f'<div class="mr-tape" aria-label="Global markets"><div class="mr-tape-track">{row}{row}</div></div>')

def render_tabs():
    return st.tabs(["Verdict", "Between the Lines", "Probability Engine", "Earnings & Options", "Backtest", "Screener", "Market Insights"])

def render_metric_card(label, value, foot="", kind=None, small=False, badge=None):
    col = {"up": P["up_d"], "down": P["down_d"], "warn": P["warn_d"], "blue": P["blue"]}.get(kind, P["text"])
    b = f'<span class="mr-badge {badge[1]}">{badge[0]}</span>' if badge else ""
    return (f'<div class="mr-card"><div class="mr-lbl">{label}</div><div class="mr-num{" sm" if small else ""}" style="color:{col}">{value}</div>'
            f'<div class="mr-foot">{b} {foot}</div></div>')

def render_signal_row(code, title, body, confidence=None):
    kind, word = STATUS.get(code, ("neu", "Neutral"))
    c = f'<div class="c">{confidence}</div>' if confidence else ""
    return (f'<div class="mr-sig"><div class="mr-dot {kind}">{svg(kind, 15, stroke=2.4)}</div><div style="min-width:0">'
            f'<div class="t">{title}<span class="st {kind}">{word}</span></div><div class="b">{body}</div>{c}</div></div>')

def render_section(title, sub=None):
    html(f'<div class="mr-h2">{title}</div>' + (f'<div class="mr-sub">{sub}</div>' if sub else ""))

def wave_svg(seed_series=None, w=900, h=360, n=28):
    """Reusable abstract data wave: many thin Bézier paths forming a silk-like probability field."""
    rng_ = np.random.default_rng(11)
    paths = []
    for i in range(n):
        t = i / (n - 1)
        y0 = h * (0.30 + 0.45 * t)
        c1 = y0 - 110 + 60 * np.sin(t * 3.1)
        c2 = y0 + 90 - 70 * np.cos(t * 2.4)
        y1 = h * (0.55 - 0.25 * t) + rng_.normal(0, 4)
        op = 0.10 + 0.32 * np.sin(np.pi * t)
        paths.append(f'<path d="M -40 {y0:.1f} C {w*0.28:.0f} {c1:.1f}, {w*0.62:.0f} {c2:.1f}, {w+40} {y1:.1f}" stroke="url(#mrwg)" stroke-width="0.9" fill="none" opacity="{op:.2f}"/>')
    return (f'<svg class="mr-wave" viewBox="0 0 {w} {h}" preserveAspectRatio="none" aria-hidden="true">'
            '<defs><linearGradient id="mrwg" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="#DCEAFF"/><stop offset=".45" stop-color="#7EA7E8"/>'
            '<stop offset=".75" stop-color="#B9D3FF"/><stop offset="1" stop-color="#FFFFFF"/></linearGradient></defs>' + "".join(paths) + "</svg>")

def render_glass_intelligence_visual(ctx):
    """Auto-rotating 3D carousel of glass intelligence cards over a drifting data wave. Every value is live model output."""
    d, comp, reg = ctx["d"], ctx["composite"], ctx["regime"]
    close = d["Close"].tail(60).values
    faces = [
        ("down", P["down_bg"], P["down"], "rgba(240,68,56,.30)", f'{ctx["red"]}', "Bearish Momentum", "Signals working against the stock right now.", None),
        ("up", P["up_bg"], P["up"], "rgba(18,183,106,.30)", f'{ctx["green"]}', "Hidden Opportunities", "Signals the headline numbers don't show.", None),
        ("check", P["blue_l"], P["blue"], "rgba(37,99,235,.30)", f'{comp:.0f}<small>/100</small>', "Clearer Decisions", ctx["verdict"].split(" —")[0].split(" (")[0].title(), None),
        ("up" if ctx["p_up"] >= 50 else "down", P["blue_l"], P["blue"], "rgba(126,167,232,.35)", f'{ctx["p_up"]:.0f}<small>%</small>', f"Chance higher · {ctx['hname']}", "Neutral Monte Carlo, 5,000 paths.", None),
        ("up" if reg["score"] >= 50 else "down", P["up_bg"] if reg["score"] >= 50 else P["down_bg"], P["up"] if reg["score"] >= 50 else P["down"], "rgba(18,183,106,.22)" if reg["score"] >= 50 else "rgba(240,68,56,.22)",
         f'{reg["score"]:.0f}<small>/100</small>', f'Regime · {reg["label"].split(" ")[0]}', "Trend, VIX, curve, credit and breadth.", None),
        ("neu", "#F2F4F7", P["text2"], "rgba(152,162,179,.28)", f'{nz(ctx["last"]["RealVol30"]):.0f}<small>%</small>', "Realized volatility", "Annualized, last 30 sessions.", close),
    ]
    cards = []
    for i, (ic, icbg, iccol, glow, n, t, desc, mini) in enumerate(faces):
        m = spark_svg(mini, "mini", 200, 34) if mini is not None else ""
        cards.append(f'<div class="mr-face" style="--dl:{-((12 - i * 36 / len(faces)) % 36):.2f}s;--glow:{glow};--icbg:{icbg};--iccol:{iccol}">'
                     f'<div class="ic">{svg(ic, 20, stroke=2.2)}</div><div><div class="n">{n}</div><div class="t">{t}</div><div class="d">{desc}</div>{m}</div></div>')
    html(f"""<div class="mr-stage" role="img" aria-label="{ctx['red']} bearish signals, {ctx['green']} bullish signals, composite {comp:.0f} of 100, {ctx['p_up']:.0f}% chance higher">
      {wave_svg(w=1800)}<div class="mr-floor"></div><div class="mr-flow">{''.join(cards)}</div>
      <div class="mr-dots"><b>Turn data into confidence.</b><span class="x"> · live model output · hover to pause</span></div></div>""")

def render_factor_bars(scores, notes, weights):
    rows = []
    for k in weights:
        v = nz(scores.get(k), np.nan)
        if np.isnan(v):
            continue
        col = P["up"] if v >= 60 else P["down"] if v <= 40 else P["blue"]
        name = "Smart money" if k == "SmartMoney" else k
        rows.append(f'<div class="mr-fac"><div class="n">{name}</div><div class="bar"><div style="width:{v:.0f}%;background:{col}"></div></div>'
                    f'<div class="v" style="color:{col}">{v:.0f}</div></div><div class="mr-fac-note">{notes.get(k, "")} · weight {weights[k]*100:.0f}%</div>')
    html('<div class="mr-card">' + "".join(rows) + "</div>")

def render_probability_chart(end, spot, lo_t, hi_t):
    """Smooth distribution of simulated end prices, shaded Bear / Base / Bull."""
    counts, edges = np.histogram(end, bins=140)
    x = (edges[:-1] + edges[1:]) / 2
    k = np.exp(-0.5 * (np.arange(-6, 7) / 2.2) ** 2); k /= k.sum()
    y = np.convolve(counts, k, mode="same"); y = y / y.max()
    fig = go.Figure()
    for lo, hi, col, fill, name in [(x.min(), lo_t, P["down"], "rgba(240,68,56,.12)", "Bear"), (lo_t, hi_t, P["blue"], "rgba(37,99,235,.10)", "Base"),
                                    (hi_t, x.max(), P["up"], "rgba(18,183,106,.12)", "Bull")]:
        m = (x >= lo) & (x <= hi)
        if m.sum() < 2:
            continue
        fig.add_trace(go.Scatter(x=x[m], y=y[m], mode="lines", line=dict(color=col, width=1.6, shape="spline"), fill="tozeroy", fillcolor=fill, name=name,
                                 hovertemplate="$%{x:.2f}<extra>" + name + "</extra>"))
    fig.add_vline(x=spot, line=dict(color=P["text"], width=1, dash="dot"))
    fig.add_annotation(x=spot, y=1.04, text=f"Today ${spot:,.2f}", showarrow=False, font=dict(size=11, color=P["text2"]), yref="y")
    lay = dict(PLOT); lay["yaxis"] = dict(PLOT["yaxis"], visible=False); lay["xaxis"] = dict(PLOT["xaxis"], tickprefix="$", showgrid=False)
    fig.update_layout(height=300, showlegend=False, **lay)
    st.plotly_chart(fig, use_container_width=True, config=PCFG)

def price_chart(d):
    p = d.tail(250)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=p.index, y=p["Close"], mode="lines", name="Price", line=dict(color=P["navy"], width=2),
                             fill="tozeroy", fillcolor="rgba(37,99,235,.05)", hovertemplate="$%{y:.2f}<extra></extra>"))
    for n, col in [(50, P["blue"]), (200, "#98A2B3")]:
        fig.add_trace(go.Scatter(x=p.index, y=p[f"SMA{n}"], mode="lines", name=f"{n}-day average", line=dict(color=col, width=1.2, dash="dot" if n == 200 else None)))
    lay = dict(PLOT); lay["yaxis"] = dict(PLOT["yaxis"], tickprefix="$", range=[p["Low"].min() * 0.97, p["High"].max() * 1.02])
    fig.update_layout(height=320, **lay)
    st.plotly_chart(fig, use_container_width=True, config=PCFG)

# ------------------------------------------------------------------------------------------------ pages
def render_verdict(ctx):
    left, right = st.columns([55, 45], gap="large")
    with left:
        comp, reg = ctx["composite"], ctx["regime"]
        ck = "up" if comp >= 58 else "down" if comp <= 35 else "blue"
        rk = "up" if reg["label"].startswith("RISK-ON") else "down" if reg["label"].startswith("RISK-OFF") else "blue"
        html('<div class="mr-grid2">'
             + (f'<div class="mr-card"><div class="mr-lbl">Composite Score</div><div class="mr-ringwrap">'
                f'{ring_svg(comp, P["up"] if ck == "up" else P["down"] if ck == "down" else P["blue"])}'
                f'<div class="mr-num" style="color:{P["up_d"] if ck == "up" else P["down_d"] if ck == "down" else P["blue"]}"><span class="mr-count" data-dec="0">{comp:.0f}</span><small>/100</small></div></div>'
                f'<div class="mr-foot">Regime-weighted blend of 8 factors</div></div>')
             + render_metric_card("Market Regime", reg["label"].split(" ")[0], "", rk,
                                  badge=(f"{svg('arrow_up' if reg['score'] >= 50 else 'arrow_down', 12, stroke=2.5)} {reg['score']:.0f}/100", "up" if rk == "up" else "down" if rk == "down" else "blue"))
             + "</div>")
        vk = ctx["vkind"]
        html(f'<div class="mr-verdict"><span class="mr-badge {"up" if vk == "up" else "down" if vk == "down" else "warn"}">'
             f'{svg("up" if vk == "up" else "down" if vk == "down" else "warn", 13)} {ctx["verdict"]}</span></div>')
        render_section("What the headline numbers hide", "The strongest hidden signals in this stock right now.")
        tells = ctx["tells"]
        ranked = sorted(tells, key=lambda t: {"🔴": 0, "🟢": 0, "🟡": 1, "⚪": 2}.get(t[0], 3))
        html('<div class="mr-card" style="padding:4px 22px">' + "".join(render_signal_row(c, t, b) for c, t, b in ranked[:4]) + "</div>")
        if len(tells) > 4:
            html(f'<div class="mr-foot" style="margin-top:6px">{len(tells) - 4} more signal{"s" if len(tells) - 4 != 1 else ""} in <b>Between the Lines</b>.</div>')
    with right:
        render_glass_intelligence_visual(ctx)
    c1, c2 = st.columns([55, 45], gap="large")
    with c1:
        render_section("Price & trend", "Last 12 months with the 50-day and 200-day averages.")
        price_chart(ctx["d"])
    with c2:
        render_section("Factor breakdown", f"Weights set by the current regime ({ctx['regime']['label']}).")
        render_factor_bars(ctx["scores"], ctx["notes"], ctx["weights"])
    render_trade_plan(ctx)

def render_trade_plan(ctx):
    d, last, comp = ctx["d"], ctx["last"], ctx["composite"]
    atr = nz(last["ATR"]); stop = last["Close"] - 2 * atr
    risk_dollars = ctx["account"] * ctx["risk_pct"] / 100
    shares = int(risk_dollars / max(last["Close"] - stop, 0.01))
    capped = shares > int(ctx["account"] / last["Close"]); shares = min(shares, int(ctx["account"] / last["Close"]))
    render_section("Trade plan", "Position size, stop and targets from the model's volatility (2× ATR stop).")
    if comp < 58:
        html('<div class="mr-card" style="padding:4px 22px">' + render_signal_row("🟡", "No long setup right now",
             f"Composite is {comp:.0f}/100 — below the 58 threshold. The plan stays hidden until enough edges stack up.") + "</div>")
        return
    html('<div class="mr-grid4">'
         + render_metric_card("Entry", f"${last['Close']:,.2f}", "Last close", small=True)
         + render_metric_card("Stop", f"${stop:,.2f}", f"−{2*atr/last['Close']*100:.1f}% · 2× ATR", "down", small=True)
         + render_metric_card("Targets", f"${last['Close']+3*atr:,.0f} / ${last['Close']+5*atr:,.0f}", "3× and 5× ATR", "up", small=True)
         + render_metric_card("Size", f"{shares:,} sh", f"Risks ${shares*(last['Close']-stop):,.0f}" + (" · capped at account" if capped else ""), small=True)
         + "</div>")

def tell_confidence(d, tells, horizon=20):
    """Map live tells to their historical hit rate on this ticker (engine: tell_event_study)."""
    try:
        study = tell_event_study(d, horizon).set_index("Tell")
    except Exception:
        return {}
    keymap = {"Quiet accumulation": "Quiet accumulation", "Distribution into strength": "Distribution into strength",
              "Bullish RSI divergence": "Bullish RSI divergence", "Bearish RSI divergence": "Bearish RSI divergence", "Volatility squeeze": "Volatility squeeze"}
    out = {}
    last = d.iloc[-1]
    for code, title, _ in tells:
        k = next((v for kk, v in keymap.items() if title.startswith(kk)), None)
        if title.startswith("Unusual volume"):
            k = "Unusual volume, up day" if nz(last["Ret1"]) > 0 else "Unusual volume, down day"
        if k and k in study.index:
            r = study.loc[k]
            if int(r["Times fired"]) >= 10:
                out[title] = (f"History on this ticker: price was higher {horizon} days later {r['Up after']} of the time after this signal "
                              f"(n={int(r['Times fired'])}) vs {r['Up (all days)']} on an average day"
                              + ("" if r["Enough data?"] == "yes" else " · small sample"))
    return out

def render_between_lines(ctx):
    d, tells = ctx["d"], ctx["tells"]
    left, right = st.columns([55, 45], gap="large")
    with left:
        render_section("What the headline numbers hide", "Every hidden signal the model found, with its track record on this ticker when available.")
        conf = tell_confidence(d, tells)
        html('<div class="mr-card" style="padding:4px 22px">' + "".join(render_signal_row(c, t, b, conf.get(t)) for c, t, b in tells) + "</div>")
    with right:
        render_section("Money flow vs price", "When money flow rises while price falls, buyers may be accumulating quietly.")
        q = d.tail(120)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=q.index, y=q["Close"], name="Price", line=dict(color=P["navy"], width=2), yaxis="y1"))
        fig.add_trace(go.Scatter(x=q.index, y=q["ADL"], name="Accumulation / Distribution", line=dict(color=P["blue"], width=1.6, dash="dot"), yaxis="y2"))
        lay = dict(PLOT); lay["yaxis2"] = dict(overlaying="y", side="right", showgrid=False, visible=False)
        fig.update_layout(height=300, **lay)
        st.plotly_chart(fig, use_container_width=True, config=PCFG)
        render_section("Signal track record", "How each technical signal has performed on this ticker (next 20 trading days).")
        st.dataframe(tell_event_study(d, 20), hide_index=True, use_container_width=True)
        ins = ctx["extras"].get("insider_transactions")
        if ins is not None and len(ins):
            with st.expander("Recent insider transactions"):
                st.dataframe(ins.head(15), use_container_width=True, hide_index=True)

def render_probability_engine(ctx):
    d, last, horizon, ticker = ctx["d"], ctx["last"], ctx["horizon"], ctx["ticker"]
    hname = {5: "1 week", 10: "2 weeks", 20: "1 month", 63: "3 months"}[horizon]
    render_section("Probability Engine", f"Scenario-based probability analysis for the next {hname} — 5,000 simulated paths built from {ticker}'s own daily moves.")
    use_drift = st.toggle("Assume the past trend continues", False, help="Off = neutral odds (past drift removed). On = extrapolates the last 3 years' trend.")
    mc = monte_carlo(d, days=horizon, demean=not use_drift)
    if mc is None:
        st.info("Not enough history to simulate.")
        return
    spot = float(last["Close"]); end = mc[:, -1]; ret = end / spot - 1
    t = 0.5 * float(np.std(ret))
    lo_t, hi_t = spot * (1 - t), spot * (1 + t)
    buckets = {"bull": end[end > hi_t], "base": end[(end >= lo_t) & (end <= hi_t)], "bear": end[end < lo_t]}
    probs = {k: len(v) / len(end) * 100 for k, v in buckets.items()}
    rng_txt = {k: (f"${np.percentile(v, 10):,.0f}–${np.percentile(v, 90):,.0f}" if len(v) > 20 else "—") for k, v in buckets.items()}
    contrib = sorted([(k, (nz(ctx["scores"][k], 50) - 50) * ctx["weights"][k]) for k in ctx["weights"] if not np.isnan(nz(ctx["scores"][k], np.nan))], key=lambda z: z[1])
    name = lambda k: "Smart money" if k == "SmartMoney" else k
    bull_dr = [f"{name(k)} score {ctx['scores'][k]:.0f}/100" for k, v in contrib[::-1][:2] if v > 0] + [t_[1] for t_ in ctx["tells"] if t_[0] == "🟢"][:1]
    bear_dr = [f"{name(k)} score {ctx['scores'][k]:.0f}/100" for k, v in contrib[:2] if v < 0] + [t_[1] for t_ in ctx["tells"] if t_[0] == "🔴"][:1]
    base_dr = [f"Market regime: {ctx['regime']['label']}", f"30-day realized volatility {nz(last['RealVol30']):.0f}%"]
    def scn(cls, label, icon, key, drivers):
        lis = "".join(f"<li>{x}</li>" for x in drivers) or "<li>No strong drivers</li>"
        return (f'<div class="mr-card mr-scn {cls}"><div class="k">{svg(icon, 13, stroke=2.4)} {label}</div>'
                f'<div class="mr-num">{probs[key]:.0f}<small>%</small></div><div class="rng">{rng_txt[key]}</div><ul>{lis}</ul></div>')
    html('<div class="mr-grid3">' + scn("bull", "Bull case", "up", "bull", bull_dr) + scn("base", "Base case", "neu", "base", base_dr) + scn("bear", "Bear case", "down", "bear", bear_dr) + "</div>")
    html(f'<div class="mr-foot" style="margin-top:8px">Bull = finishes above ${hi_t:,.2f} (+{t*100:.1f}%) · Bear = below ${lo_t:,.2f} (−{t*100:.1f}%) · ranges show the middle 80% of each scenario.</div>')
    render_probability_chart(end, spot, lo_t, hi_t)
    p5, p95 = np.percentile(ret, [5, 95])
    ml = st.session_state.get("ml_" + ticker + str(horizon))
    cards = (render_metric_card("Expected Return", f"{np.mean(ret)*100:+.1f}<small>%</small>", "Mean of all simulated outcomes" + ("" if use_drift else " (drift removed)"), "up" if np.mean(ret) >= 0 else "down", small=True)
             + render_metric_card("Downside Risk", f"{p5*100:+.1f}<small>%</small>", f"1-in-20 bad case · ${spot*(1+p5):,.2f}", "down", small=True)
             + render_metric_card("Upside Potential", f"{p95*100:+.1f}<small>%</small>", f"1-in-20 good case · ${spot*(1+p95):,.2f}", "up", small=True))
    if ml is not None:
        cards += render_metric_card("Model Confidence", f"{ml['test_acc']*100:.0f}<small>%</small>", f"ML out-of-sample accuracy · edge {ml['edge']*100:+.0f}% · stable: {'yes' if ml['edge_min'] > 0 else 'no'}",
                                    "up" if ml["edge_min"] > 0 else "warn", small=True)
    html(f'<div class="{"mr-grid4" if ml is not None else "mr-grid3"}" style="margin-top:6px">{cards}</div>')
    render_section("Touch probability", "Chance the price touches a level at any point in the window (closing prices).")
    c1, c2 = st.columns([1, 2])
    with c1:
        tgt = st.number_input("Target price", value=float(round(spot * 1.1, 2)), step=1.0)
    touched = (mc.max(axis=1) >= tgt).mean() if tgt >= spot else (mc.min(axis=1) <= tgt).mean()
    with c2:
        html(render_metric_card(f"Chance of touching ${tgt:,.2f}", f"{touched*100:.0f}<small>%</small>", f"within {hname}", "blue", small=True))
    render_section("Machine-learning direction model", "Gradient boosting, purged walk-forward validation across yearly folds.")
    if st.button("Run the ML model (30–60 sec)"):
        with st.spinner("Training on 10 years of history…"):
            st.session_state["ml_" + ticker + str(horizon)] = train_ml(ticker, horizon)
        _rr = getattr(st, "rerun", None) or getattr(st, "experimental_rerun", None)
        if _rr: _rr()
    if ml is not None:
        html('<div class="mr-grid4">'
             + render_metric_card(f"P(up in {hname})", f"{ml['p_now']*100:.0f}<small>%</small>", "Today's model call", small=True)
             + render_metric_card("Out-of-sample accuracy", f"{ml['test_acc']*100:.0f}<small>%</small>", f"{ml['edge']*100:+.0f}% vs naive baseline", small=True)
             + render_metric_card("When confident", (f"{ml['conf_acc']*100:.0f}<small>%</small>" if not np.isnan(ml["conf_acc"]) else "—"), f"{ml['n_conf']} days >60% / <40%", small=True)
             + render_metric_card("Stable edge", "Yes" if ml["edge_min"] > 0 else "No", f"Edge range {ml['edge_min']*100:+.0f}% to {ml['edge_max']*100:+.0f}%", "up" if ml["edge_min"] > 0 else "down", small=True)
             + "</div>")
        with st.expander("Year-by-year results"):
            st.dataframe(ml["folds"], hide_index=True, use_container_width=True)
            st.caption(f"Purged gap {horizon} days · ~{ml['n_indep']} independent outcomes · Brier {ml['brier']:.3f} vs naive {ml['brier_naive']:.3f}")

def render_earnings_options(ctx):
    d, last, extras, info = ctx["d"], ctx["last"], ctx["extras"], ctx["info"]
    render_section("Earnings & Options", "Event-driven intelligence: what's scheduled, what's expected, and what options are pricing.")
    cards = []
    ed = extras.get("earnings_dates"); nxt = None; past = None; nxt_row = None
    try:
        e = ed.copy(); e.index = pd.to_datetime(e.index).tz_localize(None)
        fut = e[e.index >= datetime.now()]
        if len(fut):
            nxt = fut.index.min(); nxt_row = fut.loc[nxt]
        past = e[e.index < datetime.now()].head(8)
    except Exception:
        pass
    if nxt is not None:
        days = (nxt - datetime.now()).days
        cards.append(render_metric_card("Next earnings", nxt.strftime("%b %d"), f"{nxt.strftime('%A, %Y')} · in {days} days",
                                        "warn" if days <= 10 else None, small=True, badge=("Event risk", "warn") if days <= 10 else None))
    try:
        eps_est = nz(nxt_row["EPS Estimate"]) if nxt_row is not None else np.nan
        if not np.isnan(eps_est):
            cards.append(render_metric_card("Estimated EPS", f"${eps_est:.2f}", "Consensus for the next report", small=True))
    except Exception:
        pass
    try:
        rev = extras.get("revenue_estimate")
        if rev is not None and len(rev) and "avg" in rev.columns:
            rv = nz(rev["avg"].iloc[0])
            if not np.isnan(rv):
                cards.append(render_metric_card("Revenue expectation", f"${rv/1e9:,.1f}B", "Consensus, current quarter", small=True))
    except Exception:
        pass
    moves = []
    try:
        for dt in past.index:
            loc = d.index.searchsorted(pd.Timestamp(dt.date()))
            if loc < 1 or loc + 1 >= len(d):
                continue
            pre, post = (loc - 1, loc) if dt.hour < 12 else (loc, loc + 1)
            moves.append((dt.date(), (d["Close"].iloc[post] / d["Close"].iloc[pre] - 1) * 100, past.loc[dt, "Surprise(%)"]))
    except Exception:
        pass
    mv = pd.DataFrame(moves, columns=["Report date", "Reaction move %", "EPS surprise %"]) if moves else None
    if mv is not None:
        s = mv["EPS surprise %"].dropna()
        if len(s):
            beats = int((s > 0).sum())
            cards.append(render_metric_card("Beat history", f"{beats}/{len(s)}", "Quarters beating EPS estimates", "up" if beats >= len(s) * 0.75 else "down" if beats <= len(s) * 0.25 else None, small=True))
        cards.append(render_metric_card("Avg earnings move", f"±{mv['Reaction move %'].abs().mean():.1f}<small>%</small>", "Reaction on report (BMO/AMC-aware)", small=True))
    if cards:
        html(f'<div class="mr-grid4">{"".join(cards)}</div>')
    if mv is not None:
        render_section("Earnings reactions", "How the stock moved on each of its recent reports.")
        fig = go.Figure(go.Bar(x=[str(x) for x in mv["Report date"]], y=mv["Reaction move %"],
                               marker=dict(color=[P["up"] if v >= 0 else P["down"] for v in mv["Reaction move %"]], cornerradius=6),
                               text=[f"{v:+.1f}%" for v in mv["Reaction move %"]], textposition="outside", textfont=dict(color=P["text"], size=11),
                               customdata=mv["EPS surprise %"], hovertemplate="%{x}<br>Move %{y:+.1f}%<br>EPS surprise %{customdata:+.1f}%<extra></extra>"))
        lay = dict(PLOT); lay["yaxis"] = dict(PLOT["yaxis"], ticksuffix="%")
        fig.update_layout(height=260, showlegend=False, **lay)
        st.plotly_chart(fig, use_container_width=True, config=PCFG)
    chains = extras.get("chains") or []
    if chains:
        try:
            exp, calls, puts = chains[0]; spot = float(last["Close"])
            atm_c = calls.iloc[(calls["strike"] - spot).abs().argsort()[:1]].iloc[0]
            atm_p = puts.iloc[(puts["strike"] - spot).abs().argsort()[:1]].iloc[0]
            straddle = mid_price(atm_c) + mid_price(atm_p)
            pcr_v = nz(puts["volume"].sum()) / max(nz(calls["volume"].sum(), 1), 1)
            pcr_oi = nz(puts["openInterest"].sum()) / max(nz(calls["openInterest"].sum(), 1), 1)
            ivc = nz(atm_c["impliedVolatility"]); rv30 = nz(last["RealVol30"])
            strikes = sorted(set(calls["strike"]).union(puts["strike"]))
            pain = [(k, ((k - calls["strike"]).clip(lower=0) * calls["openInterest"]).sum() + ((puts["strike"] - k).clip(lower=0) * puts["openInterest"]).sum()) for k in strikes]
            max_pain = min(pain, key=lambda z: z[1])[0]
            sent = ("Bearish lean", "down") if pcr_v > 1.2 else ("Bullish lean", "up") if pcr_v < 0.6 else ("Balanced", "neu")
            render_section("Options market", f"Nearest expiry at least 7 days out: {exp}. Mid prices; junk IVs filtered.")
            ocards = [render_metric_card("Implied move", f"±{straddle/spot*100:.1f}<small>%</small>", f"±${straddle:.2f} by {exp} (ATM straddle)", "blue", small=True),
                      render_metric_card("Put / Call (volume)", f"{pcr_v:.2f}", f"Open interest {pcr_oi:.2f}", sent[1] if sent[1] != "neu" else None, small=True, badge=sent)]
            if clean_iv(ivc):
                rich = ivc * 100 - rv30
                ocards.append(render_metric_card("Implied vs realized vol", f"{ivc*100:.0f}<small>%</small>", f"Realized 30d {rv30:.0f}% · options {'expensive' if rich > 5 else 'cheap' if rich < -5 else 'fairly priced'}",
                                                 "warn" if rich > 15 else None, small=True))
            ocards.append(render_metric_card("Max pain", f"${max_pain:,.0f}", "Where option holders lose most at expiry · weak evidence, context only", small=True))
            html(f'<div class="mr-grid4">{"".join(ocards)}</div>')
        except Exception:
            st.info("Options chain incomplete for this ticker.")
    iv_tell = options_read(extras, nz(last["Close"]), nz(last["RealVol30"]))
    if iv_tell:
        html('<div class="mr-card" style="padding:4px 22px;margin-top:10px">' + render_signal_row(*iv_tell) + "</div>")
    if not cards and not chains:
        st.info("No earnings or options data available for this ticker.")

def render_backtest(ctx):
    render_section("Does the trend model work on this stock?", "In only when price > 200-day, 50-day > 200-day and 6-month momentum > 0. Includes 5 bps per switch. Not a test of the composite score.")
    bt = quick_backtest(ctx["d"], ctx["spy"])
    if not bt:
        st.info("Not enough history to backtest."); return
    html('<div class="mr-grid4">'
         + render_metric_card("Model return / yr", f"{bt['strat_cagr']:.1f}<small>%</small>", f"Buy & hold {bt['bh_cagr']:.1f}%", "up" if bt["strat_cagr"] >= bt["bh_cagr"] else None, small=True)
         + render_metric_card("Worst drawdown", f"{bt['strat_mdd']:.0f}<small>%</small>", f"Buy & hold {bt['bh_mdd']:.0f}%", "up" if bt["strat_mdd"] > bt["bh_mdd"] else "down", small=True)
         + render_metric_card("Sharpe", f"{bt['strat_sharpe']:.2f}", f"Buy & hold {bt['bh_sharpe']:.2f}", small=True)
         + render_metric_card("Time invested", f"{bt['time_in']:.0f}<small>%</small>", f"{bt['trades']} round trips", small=True) + "</div>")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=bt["strat"].index, y=bt["strat"], name="Trend model", line=dict(color=P["blue"], width=2)))
    fig.add_trace(go.Scatter(x=bt["bh"].index, y=bt["bh"], name="Buy & hold", line=dict(color=P["muted"], width=1.4)))
    fig.update_layout(height=340, **PLOT)
    st.plotly_chart(fig, use_container_width=True, config=PCFG)

def render_screener(ctx):
    render_section("Watchlist ranking", "Every ticker in your watchlist (Settings) scored by the same engine, ranked, plus peer percentile ranks.")
    weights, spy = ctx["weights"], ctx["spy"]
    if st.button("Run screener", type="primary"):
        rows, errors = [], []
        tickers = [t.strip().upper() for t in ctx["wl"].split(",") if t.strip()]
        prog = st.progress(0)
        for i, tk in enumerate(tickers):
            try:
                h = drop_partial_bar(get_hist(tk, "3y"))
                if h.empty: errors.append(f"{tk}: no price data"); continue
                dd = add_indicators(h); inf = get_info(tk); ex = get_extras(tk, with_options=False)
                sc, _ = factor_scores(dd, inf, ex, spy)
                comp = float(np.nansum([sc[k] * weights[k] for k in weights]) / np.nansum([weights[k] for k in weights if not np.isnan(sc[k])]))
                tl = between_the_lines(dd, inf, ex, spy)
                rows.append(dict(Ticker=tk, Score=round(comp), Momentum=round(nz(sc["Momentum"], 50)), Quality=round(nz(sc["Quality"], 50)),
                                 Value=round(nz(sc["Value"], 50)), Growth=round(nz(sc["Growth"], 50)), SmartMoney=round(nz(sc["SmartMoney"], 50)),
                                 Bull_tells=sum(1 for t in tl if t[0] == "🟢"), Bear_tells=sum(1 for t in tl if t[0] == "🔴"),
                                 Price=round(dd["Close"].iloc[-1], 2), From52wHigh=f"{nz(dd['DistHigh52'].iloc[-1]):.0f}%"))
            except Exception as e:
                errors.append(f"{tk}: {type(e).__name__} — {e}")
            prog.progress((i + 1) / len(tickers)); time.sleep(0.3)
        if rows:
            res = pd.DataFrame(rows)
            for col in ["Momentum", "Quality", "Value", "Growth", "SmartMoney"]:
                res[col + "_rank"] = (res[col].rank(pct=True) * 100).round()
            res["Peer_rank"] = res[[c + "_rank" for c in ["Momentum", "Quality", "Value", "Growth", "SmartMoney"]]].mean(axis=1).round()
            st.session_state["screen"] = res.sort_values("Score", ascending=False)
        st.session_state["screen_errors"] = errors
    res = st.session_state.get("screen")
    if res is not None:
        fig = go.Figure(go.Bar(x=res["Ticker"], y=res["Score"], marker=dict(color=[P["up"] if v >= 58 else P["down"] if v <= 35 else P["blue"] for v in res["Score"]], cornerradius=6),
                               text=res["Score"], textposition="outside", textfont=dict(color=P["text"], size=11)))
        fig.add_hline(y=58, line=dict(color=P["muted"], dash="dot", width=1), annotation_text="Setup threshold 58", annotation_font_color=P["text2"])
        fig.update_layout(height=280, showlegend=False, **PLOT)
        st.plotly_chart(fig, use_container_width=True, config=PCFG)
        st.dataframe(res, hide_index=True, use_container_width=True)
        st.download_button("Download CSV", res.to_csv(index=False), "screener.csv")
    errs = st.session_state.get("screen_errors") or []
    if errs:
        with st.expander(f"{len(errs)} ticker(s) skipped"):
            for e in errs: st.write("- " + e)

def render_market_insights(ctx):
    regime = ctx["regime"]
    render_section("Market regime", f"{regime['label']} · {regime['score']:.0f}/100 — this sets the factor weights for every score.")
    curve = nz(regime.get("curve"))
    html('<div class="mr-grid4">'
         + render_metric_card("VIX", f"{nz(regime['vix']):.1f}", f"20-day avg {nz(regime['vix_20d_avg']):.1f}", "down" if nz(regime["vix"]) >= 25 else "up" if nz(regime["vix"]) < 16 else None, small=True)
         + render_metric_card("Yield curve 10y−3m", f"{curve:+.2f}<small>%</small>", "Inverted — recession warning" if curve < 0 else "Normal", "down" if curve < 0 else None, small=True)
         + render_metric_card("S&P 500 · 1 month", f"{nz(regime.get('spy_1m')):+.1f}<small>%</small>", f"Drawdown {nz(regime.get('spy_dd')):.1f}%", "up" if nz(regime.get('spy_1m')) >= 0 else "down", small=True)
         + render_metric_card("S&P vs 200-day", "n/a" if "spy_above_200" not in regime else ("Above" if regime["spy_above_200"] else "Below"),
                              "50 > 200 day" if regime.get("sma50_above_200") else "50 < 200 day", "up" if regime.get("spy_above_200") else "down", small=True)
         + "</div>")
    try:
        snap, gdata = global_snapshot(); sectors = sector_table(gdata)
        alerts = build_alerts(regime, snap, sectors, gdata)
    except Exception:
        snap, gdata, sectors, alerts = pd.DataFrame(), {}, pd.DataFrame(), []
    left, right = st.columns([55, 45], gap="large")
    with left:
        if alerts:
            render_section("What changed that matters", "Live alerts from rates, volatility, credit, oil, the dollar and sector flows.")
            html('<div class="mr-card" style="padding:4px 22px">' + "".join(render_signal_row(c, t, f"{m} <b>Action:</b> {a}") for c, t, m, a in alerts[:6]) + "</div>")
        try:
            items = get_news(["SPY", "QQQ", "^VIX", "TLT", ctx["ticker"]])
            _, tcounts = classify_news(items)
            plays_syms = sorted({s for r in NEWS_RULES for s in r["winners"] + r["losers"]} - set(gdata.keys()))
            extra = batch_hist(plays_syms, "1mo") if tcounts else {}
            plays = plays_for_themes(tcounts, gdata, extra)
        except Exception:
            plays = []
        if plays:
            render_section("News → plays", "Themes in today's headlines, who usually wins or loses, and whether the market already agrees.")
            rows = []
            for p in plays[:5]:
                code = "🟢" if p.get("spread", 0) > 0.6 else "🔴" if p.get("spread", 0) < -0.6 else "🟡"
                w = ", ".join(f"{s} {m:+.1f}%" for s, m in p["winners"] if not np.isnan(m)); l = ", ".join(f"{s} {m:+.1f}%" for s, m in p["losers"] if not np.isnan(m))
                body = p["means"] + (f"<br><b>Usually wins:</b> {w}" if w else "") + (f"<br><b>Usually loses:</b> {l}" if l else "")
                rows.append(render_signal_row(code, f"{p['theme']} · {p['n']} headline{'s' if p['n'] > 1 else ''}", body, f"{p['confirm']} · {p['caveat']}" if p["confirm"] != "n/a" else p["caveat"]))
            html('<div class="mr-card" style="padding:4px 22px">' + "".join(rows) + "</div>")
    with right:
        if not sectors.empty:
            render_section("Sector flows", "Leading = strong and strengthening. Improving = early turn. Lagging = avoid.")
            qcol = {"Leading": P["up"], "Improving": P["blue"], "Weakening": P["warn"], "Lagging": P["down"]}
            fig = go.Figure()
            for _, r in sectors.iterrows():
                fig.add_trace(go.Scatter(x=r["RS_trail"], y=r["RSmom_trail"], mode="lines", line=dict(width=1.2, color=qcol[r["Quadrant"]], shape="spline"), opacity=.35, hoverinfo="skip", showlegend=False))
                fig.add_trace(go.Scatter(x=[r["RS"]], y=[r["RSmom"]], mode="markers+text", text=[r["Symbol"]], textposition="top center", textfont=dict(size=10, color=P["text2"]),
                                         marker=dict(size=11, color=qcol[r["Quadrant"]], line=dict(width=2, color="#fff")), showlegend=False,
                                         hovertemplate=f"<b>{r['Sector']}</b><br>{r['Quadrant']}<br>1w {r['W1']:+.1f}% · 1m {r['M1']:+.1f}%<extra></extra>"))
            fig.add_hline(y=100, line=dict(color="rgba(16,24,40,.15)", width=1)); fig.add_vline(x=100, line=dict(color="rgba(16,24,40,.15)", width=1))
            for x, y, t in [(.98, .98, "Leading"), (.02, .98, "Improving"), (.02, .02, "Lagging"), (.98, .02, "Weakening")]:
                fig.add_annotation(x=x, y=y, xref="paper", yref="paper", text=t, showarrow=False, font=dict(size=11, color=P["muted"]), xanchor="right" if x > .5 else "left")
            lay = dict(PLOT); lay["xaxis"] = dict(PLOT["xaxis"], title=dict(text="Relative strength vs S&P", font=dict(size=11))); lay["yaxis"] = dict(PLOT["yaxis"], title=dict(text="Momentum of RS", font=dict(size=11)))
            fig.update_layout(height=380, **lay)
            st.plotly_chart(fig, use_container_width=True, config=PCFG)
            show = sectors[["Sector", "W1", "M1", "M3", "Quadrant"]].rename(columns={"W1": "1 wk %", "M1": "1 mo %", "M3": "3 mo %"}).sort_values("1 mo %", ascending=False)
            st.dataframe(show.round(1), hide_index=True, use_container_width=True)

C = dict(navy="#0B1B3F", panel="#FFFFFF", panel2="#F7F9FC", border="#EAECF0", text="#101828", muted="#667085")
dark = False

def dragshot():
    """Drag Shot: bottom-right button (or Ctrl+drag) → box-select anything → composer to send it somewhere."""
    btn_bg = C["navy"]; btn_fg = "#0B0F1A" if dark else "#fff"
    st.components.v1.html(f"""
    <script>
    (function(){{
      const doc = window.parent.document;
      if (doc.getElementById('dragshot-btn')) return;
      const btn = doc.createElement('button'); btn.id='dragshot-btn'; btn.innerHTML='<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:-3px;margin-right:6px"><path d="M6 2v14a2 2 0 0 0 2 2h14"/><path d="M18 22V8a2 2 0 0 0-2-2H2"/></svg>Drag Shot';
      btn.style.cssText='position:fixed;right:22px;bottom:78px;z-index:9999;background:{btn_bg};color:{btn_fg};border:0;border-radius:999px;padding:12px 18px;font-weight:700;box-shadow:0 10px 30px rgba(33,37,91,.3);cursor:pointer;font-family:Inter,sans-serif';
      doc.body.appendChild(btn);
      let armed=false, box=null, sx=0, sy=0;
      const overlay = doc.createElement('div');
      overlay.style.cssText='position:fixed;inset:0;z-index:9998;cursor:crosshair;display:none;background:rgba(33,37,91,.12)';
      doc.body.appendChild(overlay);
      btn.onclick=()=>{{armed=true; overlay.style.display='block'; btn.innerHTML='<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:-3px;margin-right:6px"><path d="M6 2v14a2 2 0 0 0 2 2h14"/><path d="M18 22V8a2 2 0 0 0-2-2H2"/></svg>Drag a box…';}};
      doc.addEventListener('keydown',e=>{{ if(e.key==='Escape'){{armed=false;overlay.style.display='none';btn.innerHTML='<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:-3px;margin-right:6px"><path d="M6 2v14a2 2 0 0 0 2 2h14"/><path d="M18 22V8a2 2 0 0 0-2-2H2"/></svg>Drag Shot';}} }});
      overlay.addEventListener('mousedown',e=>{{ sx=e.clientX; sy=e.clientY; box=doc.createElement('div');
        box.style.cssText='position:fixed;border:2px solid {btn_bg};background:rgba(184,237,253,.35);border-radius:8px;z-index:9999;pointer-events:none';
        doc.body.appendChild(box); }});
      overlay.addEventListener('mousemove',e=>{{ if(!box) return; const x=Math.min(sx,e.clientX), y=Math.min(sy,e.clientY);
        box.style.left=x+'px'; box.style.top=y+'px'; box.style.width=Math.abs(e.clientX-sx)+'px'; box.style.height=Math.abs(e.clientY-sy)+'px'; }});
      overlay.addEventListener('mouseup',e=>{{ if(!box) return; const r=box.getBoundingClientRect(); box.remove(); box=null; overlay.style.display='none'; armed=false; btn.innerHTML='<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:-3px;margin-right:6px"><path d="M6 2v14a2 2 0 0 0 2 2h14"/><path d="M18 22V8a2 2 0 0 0-2-2H2"/></svg>Drag Shot';
        const els=[...doc.querySelectorAll('.stApp *')].filter(el=>{{ const b=el.getBoundingClientRect(); return el.children.length===0 && el.textContent.trim() && b.left>=r.left-2 && b.right<=r.right+2 && b.top>=r.top-2 && b.bottom<=r.bottom+2; }});
        const text=[...new Set(els.map(el=>el.textContent.trim()))].join('\\n');
        const pop=doc.createElement('div');
        pop.style.cssText='position:fixed;left:50%;top:50%;transform:translate(-50%,-50%);width:min(680px,92vw);background:{C["panel"]};border:1px solid {C["border"]};border-radius:18px;padding:18px;z-index:10000;box-shadow:0 30px 80px rgba(0,0,0,.35);font-family:Inter,sans-serif;color:{C["text"]}';
        pop.innerHTML='<div style="font-weight:800;font-size:16px;margin-bottom:8px"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:-3px;margin-right:6px"><path d="M6 2v14a2 2 0 0 0 2 2h14"/><path d="M18 22V8a2 2 0 0 0-2-2H2"/></svg>Drag Shot</div><textarea id="ds-text" style="width:100%;height:150px;background:{C["panel2"]};color:{C["text"]};border:1px solid {C["border"]};border-radius:10px;padding:10px;font-family:Inter"></textarea><input id="ds-ask" placeholder="What to do with this (explain simply, turn into a trade plan, summarize)…" style="width:100%;margin-top:8px;background:{C["panel2"]};color:{C["text"]};border:1px solid {C["border"]};border-radius:10px;padding:10px;font-family:Inter"><div style="display:flex;gap:8px;margin-top:10px;flex-wrap:wrap"><button id="ds-copy" class="dsb">Copy</button><button id="ds-claude" class="dsb">Open in Claude</button><button id="ds-note" class="dsb">Save to Notes</button><button id="ds-close" class="dsb" style="margin-left:auto;opacity:.7">Close</button></div><div id="ds-notes" style="margin-top:10px;font-size:12px;color:{C["muted"]}"></div>';
        doc.body.appendChild(pop);
        pop.querySelectorAll('.dsb').forEach(b=>b.style.cssText+=';background:{btn_bg};color:{btn_fg};border:0;border-radius:10px;padding:8px 14px;font-weight:700;cursor:pointer');
        pop.querySelector('#ds-text').value=text;
        pop.querySelector('#ds-close').onclick=()=>pop.remove();
        pop.querySelector('#ds-copy').onclick=()=>{{ navigator.clipboard.writeText(pop.querySelector('#ds-ask').value+'\\n\\n'+pop.querySelector('#ds-text').value); pop.querySelector('#ds-copy').textContent='Copied'; }};
        pop.querySelector('#ds-claude').onclick=()=>{{ const q=encodeURIComponent((pop.querySelector('#ds-ask').value||'Explain this Market Reader output in plain English and tell me what to do:')+'\\n\\n'+pop.querySelector('#ds-text').value); window.open('https://claude.ai/new?q='+q,'_blank'); }};
        pop.querySelector('#ds-note').onclick=()=>{{ const k='mr_notes'; const cur=JSON.parse(localStorage.getItem(k)||'[]'); cur.unshift({{t:new Date().toISOString(), ask:pop.querySelector('#ds-ask').value, text:pop.querySelector('#ds-text').value}}); localStorage.setItem(k,JSON.stringify(cur.slice(0,50))); pop.querySelector('#ds-notes').textContent='Saved. '+cur.length+' note(s) in this browser.'; }};
      }});
      doc.addEventListener('mousedown',e=>{{ if(e.ctrlKey && !armed){{ armed=true; overlay.style.display='block'; overlay.dispatchEvent(new MouseEvent('mousedown',e)); }} }});
    }})();
    </script>""", height=0)


# ======================================================================================================
# APP
# ======================================================================================================
inject_global_css()
now_et = pd.Timestamp.now(tz="America/New_York")
mkt_open = now_et.weekday() < 5 and (now_et.hour, now_et.minute) >= (9, 30) and now_et.hour < 16
render_navigation(mkt_open, now_et)
try:
    _snap, _gdata = global_snapshot()
except Exception:
    _snap, _gdata = pd.DataFrame(), {}
render_ticker_tape(_snap)

if "search" not in st.session_state:
    st.session_state["search"] = "AAPL"
hero_l, hero_r = st.columns([1.35, 1], gap="large")
with hero_r:
    html('<div style="height:22px"></div>')
    s1, s2 = st.columns([4, 1.25])
    with s1:
        raw = st.text_input("Search any stock or ticker", key="search", placeholder="Search any stock or ticker...")
    with s2:
        _pop = st.popover("Settings", use_container_width=True) if hasattr(st, "popover") else st.expander("Settings")
        with _pop:
            horizon = st.selectbox("Forecast horizon", [5, 10, 20, 63], index=2, format_func=lambda x: {5: "1 week", 10: "2 weeks", 20: "1 month", 63: "3 months"}[x])
            account = st.number_input("Account size ($)", 1000, 10_000_000, 10000, step=500)
            risk_pct = st.slider("Max loss per trade (%)", 0.25, 5.0, 1.0, 0.25)
            wl = st.text_area("Watchlist (Screener)", "AAPL, MSFT, NVDA, AMZN, GOOGL, META, TSLA, JPM, XOM, UNH, COST, AVGO, LLY, AMD, NFLX", height=90)
            st.caption("Yahoo Finance data, ~15-min delay. Research only — not financial advice.")
ticker = (raw or "AAPL").upper().strip().split()[0]

with st.spinner(f"Reading {ticker}…"):
    regime = market_regime()
    weights = factor_weights(regime["label"])
    hist = drop_partial_bar(get_hist(ticker, "5y"))
    spy = drop_partial_bar(get_hist("SPY", "5y"))

if hist.empty:
    with hero_l:
        html(f'<div class="mr-id"><div class="mr-co">?</div><div><div class="mr-tk">{ticker}</div><div class="mr-meta">No data found — try a US ticker like AAPL, MSFT or BRK-B.</div></div></div>')
    render_mobile_navigation(); dragshot(); st.stop()

d = add_indicators(hist)
info = get_info(ticker)
extras = get_extras(ticker)
scores, notes = factor_scores(d, info, extras, spy)
composite = float(np.nansum([scores[k] * weights[k] for k in weights]) / np.nansum([weights[k] for k in weights if not np.isnan(scores[k])]))
last = d.iloc[-1]
tells = between_the_lines(d, info, extras, spy)
green = sum(1 for t in tells if t[0] == "🟢"); red = sum(1 for t in tells if t[0] == "🔴")
verdict = ("STRONG SETUP — long bias" if composite >= 70 and green >= red else "LEAN LONG" if composite >= 58 else
           "AVOID / SHORT BIAS" if composite <= 35 else "NEUTRAL — wait for a better price or a catalyst")
if regime["label"].startswith("RISK-OFF") and composite >= 58:
    verdict += " (but the market is risk-off — size smaller, tighter stops)"
vkind = "up" if composite >= 58 else "down" if composite <= 35 else "warn"

with hero_l:
    render_stock_header(ticker, info, last, d["Close"].tail(90).values)

_mc = monte_carlo(d, days=horizon, demean=True)
p_up = float((_mc[:, -1] > last["Close"]).mean() * 100) if _mc is not None else 50.0
hname = {5: "1 week", 10: "2 weeks", 20: "1 month", 63: "3 months"}[horizon]
ctx = dict(p_up=p_up, hname=hname, ticker=ticker, d=d, info=info, extras=extras, scores=scores, notes=notes, weights=weights, composite=composite, last=last,
           tells=tells, green=green, red=red, verdict=verdict, vkind=vkind, regime=regime, spy=spy, horizon=horizon,
           account=account, risk_pct=risk_pct, wl=wl)

T = render_tabs()
with T[0]: render_verdict(ctx)
with T[1]: render_between_lines(ctx)
with T[2]: render_probability_engine(ctx)
with T[3]: render_earnings_options(ctx)
with T[4]: render_backtest(ctx)
with T[5]: render_screener(ctx)
with T[6]: render_market_insights(ctx)

html('<div class="mr-foot" style="margin-top:36px;color:#98A2B3">Data: Yahoo Finance (delayed). Research only — not financial advice. Every edge here is probabilistic; size positions so no single trade can hurt you.</div>')
render_mobile_navigation()
dragshot()
