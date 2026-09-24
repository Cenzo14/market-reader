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
from datetime import datetime, timedelta

st.set_page_config(page_title="Market Reader", page_icon="📈", layout="wide")

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

@st.cache_data(ttl=3600, show_spinner=False)
def get_info(ticker):
    try:
        return yf.Ticker(ticker).info or {}
    except Exception:
        return {}

@st.cache_data(ttl=3600, show_spinner=False)
def get_extras(ticker):
    t = yf.Ticker(ticker)
    out = {}
    for name in ["insider_transactions", "earnings_dates", "cashflow", "financials", "balance_sheet",
                 "upgrades_downgrades", "recommendations"]:
        try:
            out[name] = getattr(t, name)
        except Exception:
            out[name] = None
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

# ------------------------------------------------------------------ UI
st.title("📈 Market Reader")
st.caption("Reads between the lines: regime → factors → hidden tells → probabilities → honest backtest. Free data, no guarantees, stacked edges.")

with st.sidebar:
    st.header("Controls")
    ticker = st.text_input("Ticker", "AAPL").upper().strip()
    horizon = st.selectbox("Prediction horizon (trading days)", [5, 10, 20, 63], index=2)
    run_ml = st.checkbox("Run ML probability engine (slower)", True)
    st.divider()
    st.subheader("Screener watchlist")
    wl = st.text_area("Tickers (comma-separated)", "AAPL, MSFT, NVDA, AMZN, GOOGL, META, TSLA, JPM, XOM, UNH, COST, AVGO, LLY, AMD, NFLX")
    st.divider()
    st.subheader("Position sizing")
    account = st.number_input("Account size ($)", 1000, 10_000_000, 10000, step=500)
    risk_pct = st.slider("Risk per trade (%)", 0.25, 5.0, 1.0, 0.25)

regime = market_regime()
weights = factor_weights(regime["label"])

tab_over, tab_lines, tab_prob, tab_earn, tab_bt, tab_screen, tab_regime = st.tabs(
    ["Verdict", "Between the Lines", "Probability Engine", "Earnings & Options", "Backtest", "Screener", "Market Regime"])

with st.spinner(f"Reading {ticker}..."):
    hist = get_hist(ticker, "5y")
    spy = get_hist("SPY", "5y")

if hist.empty:
    st.error("No data for that ticker. Check the symbol (e.g. AAPL, MSFT, BRK-B).")
    st.stop()

d = add_indicators(hist)
info = get_info(ticker)
extras = get_extras(ticker)
scores, notes = factor_scores(d, info, extras, spy)
composite = float(np.nansum([scores[k] * weights[k] for k in weights]) / np.nansum([weights[k] for k in weights if not np.isnan(scores[k])]))
last = d.iloc[-1]
tells = between_the_lines(d, info, extras, spy)

# ---------------- Verdict
with tab_over:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"{ticker} price", f"${last['Close']:.2f}", f"{nz(last['Ret1'])*100:+.2f}% today")
    c2.metric("Composite score", f"{composite:.0f}/100")
    c3.metric("Market regime", regime["label"].split(" ")[0], f"{regime['score']:.0f}/100")
    green = sum(1 for t in tells if t[0] == "🟢"); red = sum(1 for t in tells if t[0] == "🔴")
    c4.metric("Hidden tells", f"{green} bullish / {red} bearish")

    verdict = "STRONG SETUP — long bias" if composite >= 70 and green >= red else \
              "LEAN LONG" if composite >= 58 else \
              "AVOID / SHORT BIAS" if composite <= 35 else "NEUTRAL — wait for a better price or a catalyst"
    if regime["label"].startswith("RISK-OFF") and composite >= 58:
        verdict += " (but the MARKET is risk-off — size smaller, tighter stops)"
    st.subheader(f"Verdict: {verdict}")
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
        with st.spinner("Training on 10 years of this ticker..."):
            ml = train_ml(ticker, horizon)
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
    st.subheader("Watchlist ranked by composite score (regime-weighted)")
    if st.button("Run screener", type="primary"):
        rows = []
        tickers = [t.strip().upper() for t in wl.split(",") if t.strip()]
        prog = st.progress(0)
        for i, tk in enumerate(tickers):
            try:
                h = get_hist(tk, "3y")
                if h.empty: continue
                dd = add_indicators(h)
                inf = get_info(tk); ex = get_extras(tk)
                sc, _ = factor_scores(dd, inf, ex, spy)
                comp = float(np.nansum([sc[k] * weights[k] for k in weights]) / np.nansum([weights[k] for k in weights if not np.isnan(sc[k])]))
                tl = between_the_lines(dd, inf, ex, spy)
                rows.append(dict(Ticker=tk, Score=round(comp), Momentum=round(nz(sc["Momentum"], 50)), Quality=round(nz(sc["Quality"], 50)),
                                 Value=round(nz(sc["Value"], 50)), Growth=round(nz(sc["Growth"], 50)), SmartMoney=round(nz(sc["SmartMoney"], 50)),
                                 Bull_tells=sum(1 for t in tl if t[0] == "🟢"), Bear_tells=sum(1 for t in tl if t[0] == "🔴"),
                                 Price=round(dd["Close"].iloc[-1], 2), From52wHigh=f"{nz(dd['DistHigh52'].iloc[-1]):.0f}%"))
            except Exception:
                pass
            prog.progress((i + 1) / len(tickers))
        if rows:
            res = pd.DataFrame(rows).sort_values("Score", ascending=False)
            st.dataframe(res, hide_index=True, use_container_width=True)
            st.download_button("Download CSV", res.to_csv(index=False), "screener.csv")
    else:
        st.write("Edit the watchlist in the sidebar, then press **Run screener**.")

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
