"""
Market Reader  -  v6
Single file. Free Yahoo data. No emoji: every icon and chart is drawn as vector SVG.
requirements.txt needs: streamlit, yfinance, pandas, numpy
"""
import math
import base64
import datetime as dt

import numpy as np
import pandas as pd
import streamlit as st

try:
    import yfinance as yf
except Exception:  # pragma: no cover
    yf = None

# ----------------------------------------------------------------------------
# Tokens
# ----------------------------------------------------------------------------
INK = "#0B1220"
MUTED = "#6B7488"
LINE = "#E3E7F0"
RED = "#E5484D"
GREEN = "#1F9D6B"
BLUE = "#2F4AA8"
NAVY = "#111D45"
SLATE = "#AAB3C7"
FONT = "Inter, Helvetica Neue, Arial, sans-serif"

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
:root{
  --ease:cubic-bezier(.22,1,.36,1);
  --glow-g:rgba(31,157,107,.35); --glow-r:rgba(229,72,77,.35); --glow-b:rgba(47,74,168,.35);
}
html, body, .stApp, [class*="css"] { font-family: 'Inter', -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif; color:#0B1220; }
.stApp { background:
  radial-gradient(1100px 560px at 88% -8%, #E1E7F8 0%, rgba(225,231,248,0) 62%),
  radial-gradient(900px 520px at -8% 6%, #ECEFFA 0%, rgba(236,239,250,0) 60%), #F6F8FC;
  position:relative; overflow-x:clip; }
footer, #MainMenu { display:none !important; }
header[data-testid="stHeader"] { background: transparent; }
.block-container { max-width: 1200px; padding: 1.2rem 1.25rem 4rem; position:relative; z-index:1; }

/* ambient aurora — one quiet, drifting background moment, dialed off for reduced motion */
.aurora { position:fixed; inset:0; z-index:0; pointer-events:none; overflow:hidden; }
.aurora i { position:absolute; border-radius:50%; filter:blur(70px); opacity:.5; }
.aurora i:nth-child(1){ width:520px; height:520px; top:-180px; right:-120px; background:radial-gradient(circle,#C9D4F7,transparent 70%); }
.aurora i:nth-child(2){ width:460px; height:460px; bottom:-200px; left:-140px; background:radial-gradient(circle,#D8E6DD,transparent 70%); }
.aurora i:nth-child(3){ width:360px; height:360px; top:30%; left:38%; background:radial-gradient(circle,#E4D9F5,transparent 70%); opacity:.35; }
@media (prefers-reduced-motion: no-preference){
  .aurora i:nth-child(1){ animation: drift1 26s ease-in-out infinite; }
  .aurora i:nth-child(2){ animation: drift2 32s ease-in-out infinite; }
  .aurora i:nth-child(3){ animation: drift3 22s ease-in-out infinite; }
}
@keyframes drift1{ 0%,100%{ transform:translate(0,0) scale(1);} 50%{ transform:translate(-40px,50px) scale(1.08);} }
@keyframes drift2{ 0%,100%{ transform:translate(0,0) scale(1);} 50%{ transform:translate(40px,-30px) scale(1.06);} }
@keyframes drift3{ 0%,100%{ transform:translate(0,0);} 50%{ transform:translate(30px,20px);} }

@keyframes riseIn { from { opacity:0; transform:translateY(14px); } to { opacity:1; transform:translateY(0); } }
@keyframes ringDraw { from { stroke-dashoffset: var(--circ); } to { stroke-dashoffset: var(--off); } }
@keyframes popIn { from { opacity:0; transform:scale(.85); } to { opacity:1; transform:scale(1); } }
.rise { animation: riseIn .7s var(--ease) both; }
@media (prefers-reduced-motion: reduce){ .rise, .aurora i { animation:none !important; } }

/* nav */
header[data-testid="stHeader"] + div .block-container > div:first-child { position:sticky; top:0; z-index:40; }
.nav { display:flex; align-items:center; justify-content:space-between; padding:12px 4px; margin:-1.2rem -1.25rem 14px; padding-left:1.25rem; padding-right:1.25rem;
  position:sticky; top:0; z-index:40; background:rgba(246,248,252,.68); backdrop-filter:blur(16px) saturate(160%); -webkit-backdrop-filter:blur(16px) saturate(160%);
  border-bottom:1px solid rgba(255,255,255,.7); }
.brand { display:flex; align-items:center; gap:10px; font-weight:700; font-size:19px; letter-spacing:-.01em; }
.links { display:flex; gap:30px; font-size:14px; color:#39425A; }
.links span { position:relative; cursor:default; transition:color .2s var(--ease); }
.links span:hover { color:#0B1220; }
.links span::after { content:""; position:absolute; left:0; right:100%; bottom:-4px; height:2px; background:#2F4AA8; transition:right .25s var(--ease); border-radius:2px; }
.links span:hover::after { right:0; }
.navr { display:flex; align-items:center; gap:14px; }
.cta { background:#0B1220; color:#fff; font-weight:600; font-size:14px; padding:11px 22px; border-radius:999px; position:relative; overflow:hidden;
  display:inline-block; transition:transform .25s var(--ease), box-shadow .25s var(--ease); box-shadow:0 6px 18px -8px rgba(11,18,32,.5); }
.cta::before { content:""; position:absolute; inset:0; background:linear-gradient(120deg,transparent 30%,rgba(255,255,255,.35) 50%,transparent 70%);
  transform:translateX(-120%); transition:transform .6s var(--ease); }
.cta:hover { transform:translateY(-2px); box-shadow:0 12px 26px -10px rgba(11,18,32,.6); }
.cta:hover::before { transform:translateX(120%); }
@media (max-width:760px){ .links{display:none;} .cta{padding:9px 16px;font-size:13px;} }

/* search + settings */
.stTextInput input { border-radius:999px !important; background:#fff !important; border:1px solid #DDE3EF !important;
  padding:.72rem 1.1rem !important; font-size:15px !important; box-shadow:0 8px 24px -14px rgba(30,50,110,.35);
  transition:border-color .2s var(--ease), box-shadow .25s var(--ease) !important; }
.stTextInput input:focus { border-color:#2F4AA8 !important; box-shadow:0 10px 28px -12px rgba(47,74,168,.45) !important; }
[data-testid="stPopover"] button, .stButton button { border-radius:999px; border:1px solid #DDE3EF; background:#fff; color:#0B1220; font-weight:600;
  transition:border-color .2s var(--ease), color .2s var(--ease), transform .2s var(--ease), box-shadow .25s var(--ease); }
.stButton button:hover, [data-testid="stPopover"] button:hover { border-color:#2F4AA8; color:#2F4AA8; transform:translateY(-1px); box-shadow:0 10px 22px -12px rgba(47,74,168,.4); }

/* hero */
.hero { display:grid; grid-template-columns: 1.05fr 1fr; gap:10px; align-items:center; margin:6px 0 0; }
.idrow { display:flex; align-items:center; gap:16px; animation: riseIn .6s var(--ease) both; }
.lg { position:relative; width:64px; height:64px; border-radius:18px; background:#fff; box-shadow:0 10px 26px -12px rgba(30,50,110,.4);
  display:flex; align-items:center; justify-content:center; overflow:hidden; font-weight:800; font-size:22px; color:#111D45;
  transition:transform .3s var(--ease); }
.lg:hover { transform:scale(1.05) rotate(-2deg); }
.lg img { position:absolute; inset:14px; width:36px; height:36px; object-fit:contain; background:#fff; }
.tk { font-size:32px; font-weight:800; letter-spacing:-.02em; line-height:1.05; }
.nm { font-size:14px; color:#6B7488; margin-top:2px; }
.prow { display:flex; align-items:baseline; gap:16px; margin-top:18px; flex-wrap:wrap; animation: riseIn .7s .08s var(--ease) both; }
.px { font-size:56px; font-weight:800; letter-spacing:-.035em; line-height:1; }
.chg { font-size:16px; font-weight:600; display:inline-flex; align-items:center; gap:4px; }
.chg.up { color:#1F9D6B; } .chg.dn { color:#E5484D; }
.chg svg, .chg img { animation: bob 2.2s ease-in-out infinite; }
@keyframes bob { 0%,100%{ transform:translateY(0);} 50%{ transform:translateY(-3px);} }
.art { position:relative; height:290px; perspective:1000px; }
.wave { position:absolute; left:-70%; right:-2%; top:34%; bottom:-6%; -webkit-mask-image:linear-gradient(90deg,transparent 0,#000 45%); mask-image:linear-gradient(90deg,transparent 0,#000 45%);
  opacity:0; animation: fadeSlow 1.4s .1s var(--ease) both; }
@keyframes fadeSlow { from{ opacity:0; } to { opacity:1; } }
.gp { position:absolute; top:22px; width:150px; height:196px; border-radius:22px; padding:18px 16px;
  background:linear-gradient(150deg, rgba(255,255,255,.92), rgba(255,255,255,.38));
  border:1px solid rgba(255,255,255,.95); box-shadow:0 34px 60px -26px rgba(40,60,130,.35);
  backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px);
  transform: rotateY(-24deg) rotateX(5deg); transform-origin:left center;
  opacity:0; animation: gpIn .65s var(--ease) both; transition:box-shadow .3s var(--ease), transform .3s var(--ease); }
.gp:hover { box-shadow:0 40px 70px -22px rgba(40,60,130,.5); transform: rotateY(-24deg) rotateX(5deg) translateY(-6px); }
@keyframes gpIn { from{ opacity:0; transform:rotateY(-24deg) rotateX(5deg) translateX(18px);} to { opacity:1; transform:rotateY(-24deg) rotateX(5deg) translateX(0);} }
.gp1 { left:3%; z-index:3; animation-delay:.12s; } .gp2 { left:31%; z-index:2; opacity:.96; animation-delay:.22s; } .gp3 { left:59%; z-index:1; opacity:.9; animation-delay:.32s; }
.gp .gi { width:46px; height:46px; border-radius:50%; display:flex; align-items:center; justify-content:center; margin-bottom:14px; position:relative; }
.gp .gi::after { content:""; position:absolute; inset:-6px; border-radius:50%; background:inherit; opacity:.5; filter:blur(6px); z-index:-1; }
.gp .gl { font-size:13px; color:#39425A; font-weight:600; }
.gp .gv { font-size:34px; font-weight:800; letter-spacing:-.03em; margin-top:2px; }
.gp1.bad { background:linear-gradient(150deg, rgba(255,232,232,.95), rgba(255,255,255,.4)); }
.tag { position:absolute; right:2%; bottom:0; font-size:12px; color:#8A93A8; opacity:0; animation: fadeSlow 1s .5s var(--ease) both; }
.ring-wrap { position:relative; width:52px; height:52px; margin-bottom:10px; }
.ring-wrap svg { display:block; }
@media (max-width:860px){
  .hero { grid-template-columns:1fr; }
  .px { font-size:44px; }
  .art { height:210px; }
  .gp { width:31%; height:150px; padding:12px 10px; top:14px; transform: rotateY(-14deg) rotateX(4deg); }
  .gp1 { left:1%; } .gp2 { left:34%; } .gp3 { left:67%; }
  .gp .gl { font-size:11px; }
  .gp .gi { width:34px; height:34px; margin-bottom:8px; }
  .gp .gv { font-size:26px; }
  .ring-wrap { width:40px; height:40px; }
}

/* tabs */
.stTabs [data-baseweb="tab-list"] { gap:30px; border-bottom:1px solid #E3E7F0; }
.stTabs [data-baseweb="tab"] { padding:12px 0; background:transparent; color:#6B7488; font-weight:600; font-size:15px; height:auto; transition:color .2s var(--ease); }
.stTabs [data-baseweb="tab"]:hover { color:#0B1220; }
.stTabs [aria-selected="true"] { color:#0B1220; }
.stTabs [data-baseweb="tab-highlight"] { background:#2F4AA8; height:2px; transition:left .3s var(--ease), width .3s var(--ease); }
.stTabs [data-baseweb="tab-border"] { display:none; }
.stTabs [data-baseweb="tab-panel"] { animation: fadeSlow .45s var(--ease) both; }

/* cards */
.card { background:rgba(255,255,255,.74); border:1px solid rgba(255,255,255,.95); border-radius:20px; padding:20px 22px;
  box-shadow:0 20px 44px -24px rgba(30,50,110,.28); backdrop-filter: blur(14px); -webkit-backdrop-filter: blur(14px);
  transition:transform .3s var(--ease), box-shadow .3s var(--ease), border-color .3s var(--ease); }
.card:hover { transform:translateY(-3px); box-shadow:0 28px 56px -22px rgba(30,50,110,.4); border-color:rgba(255,255,255,1); }
.card h4 { margin:0 0 12px; font-size:16px; font-weight:700; letter-spacing:-.01em; display:flex; align-items:center; gap:8px; }
.card h4 img { flex:0 0 auto; }
.k { font-size:13px; color:#6B7488; font-weight:500; }
.big { font-size:44px; font-weight:800; letter-spacing:-.03em; line-height:1.05; margin-top:6px; }
.big small { font-size:18px; color:#6B7488; font-weight:600; letter-spacing:0; margin-left:2px; }
.big.g { color:#1F9D6B; font-size:34px; } .big.r { color:#E5484D; font-size:34px; } .big.n { color:#39425A; font-size:34px; }
.chip { display:inline-flex; align-items:center; gap:4px; margin-top:12px; padding:8px 14px; border-radius:12px; font-weight:700; font-size:17px; transition:box-shadow .3s var(--ease); }
.chip.g { background:#E6F6EE; color:#1F9D6B; box-shadow:0 0 0 0 var(--glow-g); } .chip.r { background:#FDECEC; color:#E5484D; box-shadow:0 0 0 0 var(--glow-r); } .chip.n { background:#EEF1F7; color:#39425A; }
.card:hover .chip.g { box-shadow:0 6px 20px -4px var(--glow-g); } .card:hover .chip.r { box-shadow:0 6px 20px -4px var(--glow-r); }
.mob { display:none; } 
.g3 { display:grid; grid-template-columns: 1fr 1fr 2.3fr; gap:16px; margin-top:22px; }
.g2 { display:grid; grid-template-columns: 1fr 1fr; gap:16px; margin-top:16px; }
.g4 { display:grid; grid-template-columns: repeat(4,1fr); gap:14px; margin-top:16px; }
.g3s { display:grid; grid-template-columns: repeat(3,1fr); gap:16px; margin-top:20px; }
@media (max-width:860px){ .g3,.g2,.g3s { grid-template-columns:1fr; } .g4 { grid-template-columns:1fr 1fr; } .dsk { display:none; } .mob { display:block; } }
.stack { margin-top:16px; }

/* signals */
.sig { display:flex; gap:14px; padding:9px 0; }
.dot { flex:0 0 16px; height:16px; margin-top:3px; border-radius:50%; }
.dot.bad { background:radial-gradient(circle at 35% 30%, #FF8A8E, #E5484D); box-shadow:0 4px 10px -2px rgba(229,72,77,.55); }
.dot.good { background:radial-gradient(circle at 35% 30%, #5FD7A2, #1F9D6B); box-shadow:0 4px 10px -2px rgba(31,157,107,.5); }
.dot.flat { background:radial-gradient(circle at 35% 30%, #D7DCE8, #AAB3C7); }
.sig b { font-weight:700; font-size:15px; } .sig span { color:#4A546C; font-size:14.5px; line-height:1.4; }
.verdict { display:inline-flex; align-items:center; gap:8px; padding:8px 16px; border-radius:999px; font-weight:700; font-size:14px; }
.verdict.g { background:#E6F6EE; color:#1F9D6B; box-shadow:0 6px 18px -6px var(--glow-g); } .verdict.r { background:#FDECEC; color:#E5484D; box-shadow:0 6px 18px -6px var(--glow-r); } .verdict.n { background:#EEF1F7; color:#39425A; }

/* inline charts: one shared, restrained motion language */
.chart { display:block; width:100%; height:auto; overflow:visible; }
.chart .fadein { opacity:0; animation: fadeSlow .6s var(--ease) both; }
.chart .drawline { animation-name:drawPath; animation-timing-function:var(--ease); animation-fill-mode:both; }
@keyframes drawPath { from { stroke-dashoffset: var(--len, 600); } to { stroke-dashoffset: 0; } }
.chart .grow-y { transform-box:fill-box; transform-origin:center bottom; animation: growY .5s var(--ease) both; }
.chart .grow-up { transform-box:fill-box; transform-origin:center bottom; animation: growY .55s var(--ease) both; }
.chart .grow-down { transform-box:fill-box; transform-origin:center top; animation: growY .55s var(--ease) both; }
@keyframes growY { from { transform:scaleY(0); opacity:.5; } to { transform:scaleY(1); opacity:1; } }
@media (prefers-reduced-motion: reduce){ .chart .fadein,.chart .drawline,.chart .grow-y,.chart .grow-up,.chart .grow-down { animation:none !important; opacity:1 !important; transform:none !important; stroke-dashoffset:0 !important; } }

/* list rows: the same reveal extended down into their items, staggered, not a separate effect */
.sig, .row, .kv, .fb, .news div { animation: rowIn .5s var(--ease) both; }
@keyframes rowIn { from { opacity:0; transform:translateX(-6px); } to { opacity:1; transform:translateX(0); } }
.kv, .row { transition:background .2s var(--ease), padding-left .2s var(--ease); border-radius:8px; }
.kv:hover, .row:hover { background:rgba(47,74,168,.05); padding-left:6px; }
@media (prefers-reduced-motion: reduce){ .sig,.row,.kv,.fb,.news div { animation:none; } }
.card :nth-child(2){animation-delay:.03s} .card :nth-child(3){animation-delay:.06s} .card :nth-child(4){animation-delay:.09s}
.card :nth-child(5){animation-delay:.12s} .card :nth-child(6){animation-delay:.15s} .card :nth-child(7){animation-delay:.18s}
.card :nth-child(8){animation-delay:.21s} .card :nth-child(9){animation-delay:.24s} .card :nth-child(n+10){animation-delay:.27s}

/* Edge Finder */
.edge-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:16px; margin-top:4px; }
@media (max-width:900px){ .edge-grid { grid-template-columns:1fr 1fr; } }
@media (max-width:600px){ .edge-grid { grid-template-columns:1fr; } }
.card.edge { position:relative; padding-top:24px; }
.edge-top { display:flex; align-items:flex-start; justify-content:space-between; gap:8px; }
.edge-t { font-size:19px; font-weight:800; letter-spacing:-.01em; }
.edge-px { display:flex; align-items:baseline; justify-content:space-between; margin-top:14px; font-size:18px; font-weight:700; }
.edge-why { margin-top:10px; min-height:36px; }
.rank { position:absolute; top:-9px; left:16px; background:#0B1220; color:#fff; font-size:11px; font-weight:700; width:22px; height:22px;
  border-radius:50%; display:flex; align-items:center; justify-content:center; box-shadow:0 6px 14px -4px rgba(11,18,32,.5); }
.erow { grid-template-columns: 1.4fr 1fr 1.4fr; }
@media (max-width:760px){ .erow { grid-template-columns:1fr; } }

/* Learn tab — native <details> styled as glass cards, no JS needed */
.lesson { background:rgba(255,255,255,.74); border:1px solid rgba(255,255,255,.95); border-radius:16px; margin-bottom:12px;
  box-shadow:0 14px 30px -22px rgba(30,50,110,.25); backdrop-filter:blur(12px); -webkit-backdrop-filter:blur(12px); overflow:hidden;
  transition:box-shadow .3s var(--ease); }
.lesson:hover { box-shadow:0 22px 44px -20px rgba(30,50,110,.35); }
.lesson summary { list-style:none; cursor:pointer; padding:15px 20px; display:flex; align-items:center; gap:10px; font-weight:700; font-size:15px; }
.lesson summary::-webkit-details-marker { display:none; }
.lesson .lt { flex:1; }
.lesson .chev { transition:transform .25s var(--ease); opacity:.6; }
.lesson[open] .chev { transform:rotate(180deg); }
.lesson-body { padding:0 20px 20px 48px; font-size:14.5px; line-height:1.65; color:#39425A; }
.cl { margin:6px 0 0; padding-left:20px; color:#39425A; font-size:14.5px; line-height:1.7; }
.cl li { margin-bottom:6px; }

/* bars and rows */
.fb { display:grid; grid-template-columns:110px 1fr 34px; align-items:center; gap:12px; padding:7px 0; font-size:14px; }
.bar { height:8px; border-radius:99px; background:#E9EDF5; overflow:hidden; }
.bar i { display:block; height:100%; width:var(--w,0%); border-radius:99px; background:linear-gradient(90deg,#5570D6,#2F4AA8);
  animation: barIn .9s .1s var(--ease) both; transform-origin:left; }
.bar i.g { background:linear-gradient(90deg,#4CC79A,#1F9D6B); } .bar i.r { background:linear-gradient(90deg,#FF8A8E,#E5484D); }
@keyframes barIn { from { transform:scaleX(0); } to { transform:scaleX(1); } }
@media (prefers-reduced-motion: reduce){ .bar i { animation:none; } }
.fb b { text-align:right; font-weight:700; }
.row { display:grid; grid-template-columns: 1.3fr 1.2fr 1.6fr; gap:14px; align-items:center; padding:12px 0; border-top:1px solid #EAEEF6; font-size:14px; }
.row:first-of-type { border-top:0; }
.row .n { font-weight:600; } .row .m { color:#4A546C; }
@media (max-width:760px){ .row { grid-template-columns:1fr; gap:6px; } }
.kv { display:flex; justify-content:space-between; padding:9px 0; border-top:1px solid #EAEEF6; font-size:14.5px; }
.kv:first-of-type { border-top:0; } .kv span { color:#6B7488; } .kv b { font-weight:700; }
.help { font-size:12px; color:#8A93A8; margin-top:8px; line-height:1.4; }
.sc { border-radius:20px; padding:18px 20px; border:1px solid rgba(255,255,255,.95); background:rgba(255,255,255,.74);
  box-shadow:0 20px 44px -24px rgba(30,50,110,.28); }
.sc .pp { font-size:38px; font-weight:800; letter-spacing:-.03em; }
.sc .nm2 { font-weight:700; font-size:15px; display:flex; align-items:center; gap:8px; }
.sc ul { margin:10px 0 0; padding-left:18px; color:#4A546C; font-size:13.5px; line-height:1.5; }
.sc.bull { border-top:3px solid #1F9D6B; } .sc.bear { border-top:3px solid #E5484D; } .sc.base { border-top:3px solid #AAB3C7; }
.sc { transition:transform .3s var(--ease), box-shadow .3s var(--ease); }
.sc:hover { transform:translateY(-3px); }
.sc.bull:hover { box-shadow:0 24px 48px -20px var(--glow-g); } .sc.bear:hover { box-shadow:0 24px 48px -20px var(--glow-r); }
.news a { color:#0B1220; text-decoration:none; font-weight:600; font-size:14.5px; transition:color .2s var(--ease), padding-left .2s var(--ease); }
.news a:hover { color:#2F4AA8; padding-left:4px; }
.news div { padding:10px 0; border-top:1px solid #EAEEF6; } .news div:first-child { border-top:0; }
.news small { color:#8A93A8; display:block; margin-top:2px; }
.msg { background:rgba(255,255,255,.8); border:1px solid rgba(255,255,255,.95); border-radius:16px; padding:16px 20px; font-size:15px; margin-top:14px;
  display:flex; align-items:center; gap:12px; box-shadow:0 16px 34px -22px rgba(30,50,110,.3); backdrop-filter:blur(12px); -webkit-backdrop-filter:blur(12px);
  animation: riseIn .45s var(--ease) both; color:#39425A; }
.msg::before { content:""; flex:0 0 8px; width:8px; height:8px; border-radius:50%; background:#AAB3C7; box-shadow:0 0 0 4px rgba(170,179,199,.25); }
.msg.err::before { background:#E5484D; box-shadow:0 0 0 4px rgba(229,72,77,.2); }

/* live pulse next to the intraday change */
.livedot { display:inline-block; width:7px; height:7px; border-radius:50%; margin-right:2px; position:relative; top:-1px; }
.livedot.up { background:#1F9D6B; } .livedot.dn { background:#E5484D; }
.livedot::after { content:""; position:absolute; inset:-4px; border-radius:50%; border:1.5px solid currentColor; opacity:.6; animation: pulseRing 1.8s ease-out infinite; }
@media (prefers-reduced-motion: reduce){ .livedot::after { animation:none; } }
@keyframes pulseRing { 0%{ transform:scale(.4); opacity:.7; } 100%{ transform:scale(1.9); opacity:0; } }

/* settings popover, to match the rest of the glass system */
div[data-baseweb="popover"] [data-testid="stVerticalBlock"] { gap:.6rem; }
div[data-baseweb="popover"] > div { border-radius:18px !important; border:1px solid rgba(255,255,255,.9) !important;
  box-shadow:0 26px 50px -20px rgba(30,50,110,.35) !important; backdrop-filter:blur(16px); }
[data-testid="stSpinner"] { color:#39425A; }
[data-testid="stSpinner"] > div { border-top-color:#2F4AA8 !important; }
</style>
"""


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------
def H(s):
    """Collapse HTML to one block: no blank lines, no indentation (Markdown safe)."""
    return "\n".join(l.strip() for l in s.strip().splitlines() if l.strip())


def money(x, d=2):
    return "&#36;{:,.{d}f}".format(x, d=d)


def pct(x, d=1, sign=True):
    return ("{:+.%df}%%" % d if sign else "{:.%df}%%" % d).format(x)


def clip(x, lo=0, hi=100):
    return max(lo, min(hi, x))


def svg_img(svg, style="width:100%;display:block", alt=""):
    b = base64.b64encode(svg.encode("utf-8")).decode()
    return '<img alt="%s" style="%s" src="data:image/svg+xml;base64,%s"/>' % (alt, style, b)


_ICONS = {
    "down": '<path d="M12 5v14M6 13l6 6 6-6"/>',
    "up": '<path d="M12 19V5M6 11l6-6 6 6"/>',
    "trend": '<path d="M3 17l6-6 4 4 8-8M15 7h6v6"/>',
    "check": '<path d="M20 6L9 17l-5-5"/>',
    "search": '<circle cx="11" cy="11" r="7"/><path d="M21 21l-4.3-4.3"/>',
    "pulse": '<path d="M3 12h4l3-8 4 16 3-8h4"/>',
    "shield": '<path d="M12 3l8 3v6c0 5-3.5 8-8 9-4.5-1-8-4-8-9V6z"/>',
    "layers": '<path d="M12 3l9 5-9 5-9-5 9-5zM3 13l9 5 9-5"/>',
    "cal": '<rect x="3" y="5" width="18" height="16" rx="3"/><path d="M8 3v4M16 3v4M3 10h18"/>',
    "target": '<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="4.5"/><circle cx="12" cy="12" r="1"/>',
    "radar": '<circle cx="12" cy="12" r="9"/><path d="M12 12L19 7"/><path d="M12 3a9 9 0 0 1 9 9"/>',
    "book": '<path d="M4 4.5A2.5 2.5 0 0 1 6.5 2H20v17H6.5A2.5 2.5 0 0 0 4 21.5z"/><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/>',
    "compass": '<circle cx="12" cy="12" r="9"/><path d="M15 9l-2 6-6 2 2-6z"/>',
    "star": '<path d="M12 3l2.6 5.8 6.4.6-4.8 4.3 1.4 6.3L12 16.9 6.4 20l1.4-6.3-4.8-4.3 6.4-.6z"/>',
}



def icon(name, color=INK, size=20, sw=2):
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" '
           'stroke="%s" stroke-width="%s" stroke-linecap="round" stroke-linejoin="round">%s</svg>') % (color, sw, _ICONS[name])
    return svg_img(svg, "width:%dpx;height:%dpx;display:inline-block;vertical-align:-3px" % (size, size))


def hi(name):
    """Small muted header icon, used to open every card title with the same visual language."""
    return icon(name, BLUE, 17, 2.2)


def logo_mark(size=34):
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 40 40"><defs><linearGradient id="a" x1="0" y1="0" x2="1" y2="1">'
           '<stop offset="0" stop-color="#3A57C4"/><stop offset="1" stop-color="#111D45"/></linearGradient></defs>'
           '<rect width="40" height="40" rx="11" fill="url(#a)"/>'
           '<path d="M11 27V14l9 9 9-9v13" fill="none" stroke="#fff" stroke-width="3.2" stroke-linecap="round" stroke-linejoin="round"/>'
           '<circle cx="29" cy="11" r="2.4" fill="#7FD6B2"/></svg>')
    return svg_img(svg, "width:%dpx;height:%dpx;display:block" % (size, size))


def ring(value, color, size=64, label=None, track="#E4E9F3", stroke=7):
    """Inline (not base64) so the draw-in animation actually runs in the page."""
    r = (64 - stroke) / 2.0
    circ = 2 * math.pi * r
    off = circ * (1 - clip(value) / 100.0)
    txt = label if label is not None else "%d" % value
    return H('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64" style="width:%dpx;height:%dpx;display:block;overflow:visible">'
              '<circle cx="32" cy="32" r="%.2f" fill="none" stroke="%s" stroke-width="%s"/>'
              '<circle cx="32" cy="32" r="%.2f" fill="none" stroke="%s" stroke-width="%s" stroke-linecap="round" '
              'stroke-dasharray="%.2f" style="--circ:%.2fpx;--off:%.2fpx;stroke-dashoffset:%.2fpx;animation:ringDraw 1.1s .15s cubic-bezier(.22,1,.36,1) both" '
              'transform="rotate(-90 32 32)"/>'
              '<text x="32" y="37" text-anchor="middle" font-family="Inter,Arial" font-weight="700" font-size="15" fill="%s">%s</text></svg>'
              % (size, size, r, track, stroke, r, color, stroke, circ, circ, off, circ, INK, txt))
    # note: inline stroke-dashoffset starts at full circumference (hidden) so paint-before-JS/CSS never flashes a full ring


# ----------------------------------------------------------------------------
# SVG charts
# ----------------------------------------------------------------------------
def _path(arr, X, Y):
    d, pen = [], False
    for i, x in enumerate(arr):
        if x != x:
            pen = False
            continue
        d.append(("L" if pen else "M") + "%.1f,%.1f" % (X(i), Y(x)))
        pen = True
    return " ".join(d)


def _plen(arr, X, Y):
    """Approximate on-screen length of a line built from _path(), for a stroke draw-in animation."""
    pts = [(X(i), Y(x)) for i, x in enumerate(arr) if x == x]
    return sum(math.hypot(x1 - x0, y1 - y0) for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:])) or 1.0


def area_chart(series, dates, lines=(), color=BLUE, w=760, h=260, fmt="${:,.0f}"):
    v = np.asarray(series, float)
    n = len(v)
    pool = [v] + [np.asarray(l[0], float) for l in lines]
    lo = min(np.nanmin(a) for a in pool)
    hi = max(np.nanmax(a) for a in pool)
    pad = (hi - lo) * 0.08 or 1.0
    lo, hi = lo - pad, hi + pad
    L, R, T, B = 6, 62, 14, 26
    X = lambda i: L + (w - L - R) * i / max(n - 1, 1)
    Y = lambda x: T + (h - T - B) * (1 - (x - lo) / (hi - lo))
    s = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d" font-family="%s" class="chart">' % (w, h, FONT),
         '<defs><linearGradient id="f" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="%s" stop-opacity=".28"/>'
         '<stop offset="1" stop-color="%s" stop-opacity="0"/></linearGradient></defs>' % (color, color)]
    for k in range(5):
        gv = lo + (hi - lo) * k / 4
        y = Y(gv)
        s.append('<line x1="%d" x2="%d" y1="%.1f" y2="%.1f" stroke="#E7EBF3"/>' % (L, w - R, y, y))
        s.append('<text x="%d" y="%.1f" font-size="11" fill="#8A93A8">%s</text>' % (w - R + 8, y + 4, fmt.format(gv)))
    area = _path(v, X, Y) + " L%.1f,%.1f L%.1f,%.1f Z" % (X(n - 1), h - B, X(0), h - B)
    s.append('<path d="%s" fill="url(#f)" class="fadein" style="animation-delay:.5s"/>' % area)
    for arr, col, dash in lines:
        s.append('<path d="%s" fill="none" stroke="%s" stroke-width="1.6" %s class="fadein" style="animation-delay:.6s"/>' % (
            _path(np.asarray(arr, float), X, Y), col, ('stroke-dasharray="%s"' % dash) if dash else ""))
    ln = _plen(v, X, Y)
    s.append('<path d="%s" fill="none" stroke="%s" stroke-width="2.4" stroke-linejoin="round" stroke-linecap="round" '
              'class="drawline" style="--len:%.1f;stroke-dasharray:%.1f;stroke-dashoffset:%.1f;animation-duration:%.2fs"/>' % (_path(v, X, Y), color, ln, ln, ln, clip(ln / 420, 20, 220) / 10.0))
    s.append('<circle cx="%.1f" cy="%.1f" r="4.5" fill="#fff" stroke="%s" stroke-width="2.5" class="fadein" style="animation-delay:%.2fs"/>' % (
        X(n - 1), Y(v[-1]), color, clip(ln / 420, 20, 220) / 10.0))
    for i in (0, n // 2, n - 1):
        anchor = "start" if i == 0 else ("end" if i == n - 1 else "middle")
        s.append('<text x="%.1f" y="%d" font-size="11" fill="#8A93A8" text-anchor="%s">%s</text>' % (
            X(i) if i != n - 1 else w - R, h - 6, anchor, pd.Timestamp(dates[i]).strftime("%b %Y")))
    s.append("</svg>")
    return H("".join(s))


def hist_chart(term, w=760, h=250):
    mu, sd = term.mean(), term.std()
    cnt, edges = np.histogram(term * 100, bins=38)
    mx = cnt.max() or 1
    L, R, T, B = 8, 8, 14, 30
    bw = (w - L - R) / len(cnt)
    s = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d" font-family="%s" class="chart">' % (w, h, FONT)]
    for i, c in enumerate(cnt):
        mid = (edges[i] + edges[i + 1]) / 200.0
        col = RED if mid < mu - 0.5 * sd else (GREEN if mid > mu + 0.5 * sd else SLATE)
        bh = (h - T - B) * c / mx
        s.append('<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" rx="3" fill="%s" opacity=".88" '
                  'class="grow-y" style="animation-delay:%.2fs"/>' % (L + i * bw + 1, h - B - bh, bw - 2, bh, col, i * 0.008))
    xm = L + (w - L - R) * ((np.median(term) * 100 - edges[0]) / (edges[-1] - edges[0]))
    s.append('<line x1="%.1f" x2="%.1f" y1="%d" y2="%d" stroke="%s" stroke-width="2" stroke-dasharray="4 4" class="fadein" style="animation-delay:.5s"/>' % (xm, xm, T, h - B, INK))
    for k in range(5):
        val = edges[0] + (edges[-1] - edges[0]) * k / 4
        x = L + (w - L - R) * k / 4
        anchor = "start" if k == 0 else ("end" if k == 4 else "middle")
        s.append('<text x="%.1f" y="%d" font-size="11" fill="#8A93A8" text-anchor="%s">%+.0f%%</text>' % (x, h - 8, anchor, val))
    s.append("</svg>")
    return H("".join(s))


def reaction_chart(items, w=760, h=220):
    """items: list of (label, pct move)."""
    if not items:
        return ""
    m = max(abs(x[1]) for x in items) or 1
    L, R, T, B = 8, 8, 14, 28
    mid = T + (h - T - B) / 2
    bw = (w - L - R) / len(items)
    s = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d" font-family="%s" class="chart">' % (w, h, FONT),
         '<line x1="%d" x2="%d" y1="%.1f" y2="%.1f" stroke="#D8DEEA"/>' % (L, w - R, mid, mid)]
    for i, (lab, v) in enumerate(items):
        bh = (h - T - B) / 2 * abs(v) / m * 0.92
        y = mid - bh if v >= 0 else mid
        col = GREEN if v >= 0 else RED
        x = L + i * bw + bw * 0.2
        cls = "grow-up" if v >= 0 else "grow-down"
        s.append('<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" rx="4" fill="%s" class="%s" style="animation-delay:%.2fs"/>' % (
            x, y, bw * 0.6, max(bh, 2), col, cls, i * 0.05))
        ty = (y - 5) if v >= 0 else (y + bh + 13)
        s.append('<text x="%.1f" y="%.1f" font-size="11" font-weight="600" fill="%s" text-anchor="middle" class="fadein" style="animation-delay:%.2fs">%+.1f%%</text>' % (x + bw * 0.3, ty, col, i * 0.05 + .3, v))
        s.append('<text x="%.1f" y="%d" font-size="10.5" fill="#8A93A8" text-anchor="middle">%s</text>' % (x + bw * 0.3, h - 8, lab))
    s.append("</svg>")
    return H("".join(s))


def resp(fn, *a, **k):
    """Render a chart twice (wide + phone-sized) so labels stay readable on a phone."""
    wide = fn(*a, **k)
    kk = dict(k)
    kk["w"], kk["h"] = 380, 250
    return '<div class="dsk">%s</div><div class="mob">%s</div>' % (wide, fn(*a, **kk))


def wave_svg(series, w=900, h=300):
    v = np.asarray(series, float)
    v = v[~np.isnan(v)]
    if len(v) < 10:
        v = np.linspace(0, 1, 60)
    k = max(len(v) // 60, 1)
    v = pd.Series(v).rolling(k, min_periods=1).mean().values[::k]
    v = (v - v.min()) / ((v.max() - v.min()) or 1)
    xs = np.linspace(0, w, len(v))
    s = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d" preserveAspectRatio="none">' % (w, h),
         '<defs><linearGradient id="w" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="#9FB0EC" stop-opacity=".15"/>'
         '<stop offset=".5" stop-color="#7F95E4" stop-opacity=".8"/><stop offset="1" stop-color="#6E5FD0" stop-opacity=".9"/></linearGradient>'
         '<linearGradient id="v" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#8FA3E8" stop-opacity=".38"/>'
         '<stop offset="1" stop-color="#8FA3E8" stop-opacity="0"/></linearGradient></defs>']
    for j in range(9):
        ph = j * 0.5
        amp = 84 - j * 6
        base = h * (0.66 - j * 0.03)
        pts = [(x, base - v[i] * amp * 1.5 + math.sin(i / 5.0 + ph) * 14) for i, x in enumerate(xs)]
        d = "M%.1f,%.1f " % pts[0]
        for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
            cx = (x0 + x1) / 2
            d += "C%.1f,%.1f %.1f,%.1f %.1f,%.1f " % (cx, y0, cx, y1, x1, y1)
        if j == 0:
            s.append('<path d="%sL%d,%d L0,%d Z" fill="url(#v)"/>' % (d, w, h, h))
        s.append('<path d="%s" fill="none" stroke="url(#w)" stroke-width="%.1f" opacity="%.2f"/>' % (d, 2.6 - j * 0.22, 1 - j * 0.09))
    s.append("</svg>")
    return svg_img("".join(s), "width:100%;height:100%;display:block")


# ----------------------------------------------------------------------------
# Indicators
# ----------------------------------------------------------------------------
def rsi(c, n=14):
    d = c.diff()
    up, dn = d.clip(lower=0), (-d).clip(lower=0)
    ru = up.ewm(alpha=1.0 / n, adjust=False).mean()
    rd = dn.ewm(alpha=1.0 / n, adjust=False).mean()
    return 100 - 100 / (1 + ru / rd.replace(0, np.nan))


def atr(h, n=14):
    pc = h["Close"].shift(1)
    tr = pd.concat([h["High"] - h["Low"], (h["High"] - pc).abs(), (h["Low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / n, adjust=False).mean()


def ad_line(h):
    rng = (h["High"] - h["Low"]).replace(0, np.nan)
    mfm = ((h["Close"] - h["Low"]) - (h["High"] - h["Close"])) / rng
    return (mfm * h["Volume"]).fillna(0).cumsum()


# ----------------------------------------------------------------------------
# Data loading (only place that touches the network)
# ----------------------------------------------------------------------------
def _safe(fn, default=None):
    try:
        out = fn()
        return default if out is None else out
    except Exception:
        return default


@st.cache_data(ttl=900, show_spinner=False)
def load_market():
    spy = _safe(lambda: yf.Ticker("SPY").history(period="2y", auto_adjust=True))
    vix = _safe(lambda: yf.Ticker("^VIX").history(period="3mo", auto_adjust=True))
    secs = {"Technology": "XLK", "Financials": "XLF", "Energy": "XLE", "Health care": "XLV", "Consumer disc.": "XLY",
            "Staples": "XLP", "Industrials": "XLI", "Utilities": "XLU", "Materials": "XLB", "Real estate": "XLRE", "Comm. services": "XLC"}
    sec = {}
    px = _safe(lambda: yf.download(list(secs.values()), period="6mo", auto_adjust=True, progress=False)["Close"])
    if px is not None and len(px) > 25:
        for name, sym in secs.items():
            if sym in px.columns:
                s = px[sym].dropna()
                if len(s) > 25:
                    sec[name] = (float(s.iloc[-1] / s.iloc[-22] - 1) * 100, float(s.iloc[-1] / s.iloc[-64] - 1) * 100 if len(s) > 64 else np.nan)
    return {"spy": spy, "vix": vix, "sectors": sec}


@st.cache_data(ttl=900, show_spinner=False)
def load_raw(t):
    tk = yf.Ticker(t)
    h = tk.history(period="2y", interval="1d", auto_adjust=True)
    if h is None or h.empty or len(h) < 80:
        return None
    h = h.dropna(subset=["Close"])
    # drop a partial intraday bar for today
    if len(h) > 2 and h.index[-1].date() == dt.date.today() and dt.datetime.utcnow().hour < 21:
        if h["Volume"].iloc[-1] < 0.3 * h["Volume"].iloc[-20:-1].mean():
            h = h.iloc[:-1]
    raw = {"t": t, "hist": h}
    raw["info"] = _safe(lambda: tk.info, {}) or {}
    raw["ups"] = _safe(lambda: tk.upgrades_downgrades)
    raw["earn"] = _safe(lambda: tk.earnings_dates)
    raw["cal"] = _safe(lambda: tk.calendar, {})
    raw["shares"] = _safe(lambda: tk.get_shares_full(start=(dt.date.today() - dt.timedelta(days=430)).isoformat()))
    raw["ins"] = _safe(lambda: tk.insider_transactions)
    raw["news"] = _safe(lambda: tk.news, [])

    def opt():
        exps = list(tk.options)
        today = dt.date.today()
        pick = None
        for e in exps:
            d = (dt.date.fromisoformat(e) - today).days
            if d >= 7:
                pick = (e, d)
                break
        if not pick:
            return None
        ch = tk.option_chain(pick[0])
        return {"exp": pick[0], "dte": pick[1], "calls": ch.calls, "puts": ch.puts}

    raw["opt"] = _safe(opt)
    return raw


# ----------------------------------------------------------------------------
# Engine
# ----------------------------------------------------------------------------
def build_signals(raw):
    h, info = raw["hist"], raw["info"]
    c = h["Close"]
    r = rsi(c)
    sigs = []

    # 1 RSI divergence / stretch
    win = c.iloc[-60:]
    prev = c.iloc[-60:-8]
    if len(prev) > 10:
        if c.iloc[-1] >= win.max() * 0.995 and c.iloc[-1] > prev.max():
            pr = float(r.loc[prev.idxmax()])
            if r.iloc[-1] < pr - 3 and pr > 55:
                sigs.append(dict(tone="bad", w=9, head="Bearish RSI divergence",
                                 body="New price high with weaker momentum (RSI %.0f vs %.0f at the last high) - the move is losing fuel." % (r.iloc[-1], pr)))
        elif c.iloc[-1] <= win.min() * 1.005 and c.iloc[-1] < prev.min():
            pr = float(r.loc[prev.idxmin()])
            if r.iloc[-1] > pr + 3 and pr < 45:
                sigs.append(dict(tone="good", w=9, head="Bullish RSI divergence",
                                 body="New price low but momentum is rising (RSI %.0f vs %.0f) - sellers are tiring." % (r.iloc[-1], pr)))
    rv = float(r.iloc[-1])
    if rv > 72:
        sigs.append(dict(tone="bad", w=6, head="Overbought (RSI %.0f)" % rv, body="Stretched short term - pullbacks are common from here."))
    elif rv < 30:
        sigs.append(dict(tone="good", w=6, head="Oversold (RSI %.0f)" % rv, body="Stretched to the downside - bounces are common from here."))

    # 2 share count
    sh = raw.get("shares")
    if sh is not None and len(sh) > 20:
        sh = sh.dropna()
        try:
            last = float(sh.iloc[-1])
            old = float(sh[sh.index <= sh.index[-1] - pd.Timedelta(days=300)].iloc[-1])
            yrs = (sh.index[-1] - sh[sh.index <= sh.index[-1] - pd.Timedelta(days=300)].index[-1]).days / 365.0
            chg = (last / old - 1) * 100 / max(yrs, 0.5)
            if chg < -1:
                sigs.append(dict(tone="good", w=8, head="Share count shrinking (%.1f%%/yr)" % chg, body="Buybacks - management thinks it is cheap."))
            elif chg > 2:
                sigs.append(dict(tone="bad", w=8, head="Share count growing (+%.1f%%/yr)" % chg, body="Dilution - each share owns a smaller slice."))
        except Exception:
            pass

    # 3 analyst changes, 60 days
    u = raw.get("ups")
    if u is not None and len(u):
        try:
            u = u.copy()
            idx = pd.to_datetime(u.index)
            if getattr(idx, "tz", None) is not None:
                idx = idx.tz_localize(None)
            m = idx >= (pd.Timestamp.now() - pd.Timedelta(days=60))
            a = u["Action"].astype(str).str.lower()[m]
            dn_, up_ = int((a == "down").sum()), int((a == "up").sum())
            if dn_ + up_ > 0:
                tone = "bad" if dn_ > up_ else ("good" if up_ > dn_ else "flat")
                body = {"bad": "Sell-side is bailing.", "good": "Sell-side is warming up.", "flat": "Analysts are split."}[tone]
                sigs.append(dict(tone=tone, w=7, head="%d downgrade%s / %d upgrade%s (60d)" % (dn_, "" if dn_ == 1 else "s", up_, "" if up_ == 1 else "s"), body=body))
        except Exception:
            pass

    # 4 earnings beats
    e = raw.get("earn")
    if e is not None and len(e) and "Reported EPS" in e.columns:
        try:
            d = e.dropna(subset=["Reported EPS", "EPS Estimate"]).sort_index(ascending=False).head(6)
            if len(d) >= 4:
                beats = int((d["Reported EPS"] > d["EPS Estimate"]).sum())
                if beats >= len(d) - 1:
                    sigs.append(dict(tone="good", w=7, head="Beat estimates %d/%d straight quarters" % (beats, len(d)), body="Management sandbags guidance."))
                elif beats <= len(d) // 2 - 1:
                    sigs.append(dict(tone="bad", w=7, head="Missed estimates in %d of %d quarters" % (len(d) - beats, len(d)), body="Results keep undershooting expectations."))
        except Exception:
            pass

    # 5 money flow
    try:
        ad = ad_line(h)
        pchg = c.iloc[-1] / c.iloc[-63] - 1
        achg = ad.iloc[-1] - ad.iloc[-63]
        if pchg > 0.03 and achg < 0:
            sigs.append(dict(tone="bad", w=6, head="Quiet selling into strength", body="Price is up but money flow is falling - big holders are distributing."))
        elif pchg < -0.03 and achg > 0:
            sigs.append(dict(tone="good", w=6, head="Quiet buying into weakness", body="Price is down but money flow is rising - big holders are accumulating."))
    except Exception:
        pass

    # 6 trend
    s50, s200 = c.rolling(50).mean().iloc[-1], c.rolling(200).mean().iloc[-1]
    if s200 == s200:
        if c.iloc[-1] > s200 and s50 > s200:
            sigs.append(dict(tone="good", w=5, head="Uptrend intact", body="Price and the 50-day are both above the 200-day average."))
        elif c.iloc[-1] < s200 and s50 < s200:
            sigs.append(dict(tone="bad", w=5, head="Downtrend", body="Price and the 50-day are both below the 200-day average."))

    # 7 valuation
    fpe, tpe = info.get("forwardPE"), info.get("trailingPE")
    try:
        if fpe and tpe and 0 < fpe < tpe * 0.85:
            sigs.append(dict(tone="good", w=4, head="Earnings expected to grow", body="Forward P/E %.1f vs %.1f trailing." % (fpe, tpe)))
        elif tpe and tpe > 45:
            sigs.append(dict(tone="bad", w=4, head="Priced for perfection", body="P/E of %.0f leaves little room for a miss." % tpe))
    except Exception:
        pass

    sigs.sort(key=lambda s: -s["w"])
    return sigs


def build_factors(raw, sigs):
    h, info = raw["hist"], raw["info"]
    c = h["Close"]
    out = {}
    s50, s200 = c.rolling(50).mean().iloc[-1], c.rolling(200).mean().iloc[-1]
    t = 45
    t += 15 if c.iloc[-1] > s200 else -10
    t += 10 if c.iloc[-1] > s50 else -5
    t += 10 if s50 > s200 else -5
    t += 20 * float(np.tanh((c.iloc[-1] / c.iloc[-64] - 1) * 5))
    out["Trend"] = clip(t)
    rv = float(rsi(c).iloc[-1])
    r6 = c.iloc[-1] / c.iloc[-127] - 1 if len(c) > 130 else 0
    out["Momentum"] = clip((100 - abs(rv - 60) * 2.2 + 50 + r6 * 100) / 2)
    q = []
    if info.get("profitMargins") is not None:
        q.append(clip(30 + info["profitMargins"] * 240))
    if info.get("returnOnEquity") is not None:
        q.append(clip(30 + info["returnOnEquity"] * 120))
    if q:
        out["Quality"] = sum(q) / len(q)
    fpe = info.get("forwardPE") or info.get("trailingPE")
    if fpe and fpe > 0:
        out["Valuation"] = clip(100 - (fpe - 10) * 1.6)
    elif info.get("pegRatio"):
        out["Valuation"] = clip(100 - info["pegRatio"] * 30)
    sm = []
    if info.get("recommendationMean"):
        sm.append(clip((5 - info["recommendationMean"]) / 4 * 100))
    good = sum(1 for s in sigs if s["tone"] == "good")
    bad = sum(1 for s in sigs if s["tone"] == "bad")
    sm.append(clip(50 + (good - bad) * 12))
    out["Sentiment"] = sum(sm) / len(sm)
    vol = float(np.log(c / c.shift(1)).dropna().iloc[-252:].std() * math.sqrt(252))
    out["Risk"] = clip(100 - (vol - 0.12) * 190)
    return out, vol


WEIGHTS = {"Trend": .25, "Momentum": .20, "Quality": .15, "Valuation": .15, "Sentiment": .15, "Risk": .10}


def composite(f):
    tw = sum(WEIGHTS[k] for k in f)
    return int(round(sum(f[k] * WEIGHTS[k] for k in f) / tw))


def verdict(score):
    if score >= 70:
        return "Constructive", "g"
    if score >= 55:
        return "Leans positive", "g"
    if score >= 45:
        return "Neutral", "n"
    if score >= 30:
        return "Leans negative", "r"
    return "Cautious", "r"


def regime(mkt):
    spy, vix = mkt.get("spy"), mkt.get("vix")
    if spy is None or len(spy) < 210:
        return dict(score=None, label="N/A", tone="n", spy_trend="n/a", vix=None)
    c = spy["Close"]
    s = 40.0
    s += 20 if c.iloc[-1] > c.rolling(200).mean().iloc[-1] else -10
    s += 10 if c.iloc[-1] > c.rolling(50).mean().iloc[-1] else -5
    s += 10 if c.iloc[-1] / c.iloc[-64] - 1 > 0 else -5
    v = float(vix["Close"].iloc[-1]) if vix is not None and len(vix) else None
    if v is not None:
        s += 15 if v < 16 else (8 if v < 22 else (-5 if v < 30 else -15))
    s = int(clip(s))
    label, tone = ("RISK-ON", "g") if s >= 60 else (("NEUTRAL", "n") if s >= 40 else ("RISK-OFF", "r"))
    return dict(score=s, label=label, tone=tone, vix=v,
                spy_trend="above 200-day" if c.iloc[-1] > c.rolling(200).mean().iloc[-1] else "below 200-day",
                spy_1m=float(c.iloc[-1] / c.iloc[-22] - 1) * 100)


def monte_carlo(c, H, drift=True, sims=4000, seed=7):
    r = np.log(c / c.shift(1)).dropna().values[-756:]
    if not drift:
        r = r - r.mean()
    rng = np.random.default_rng(seed)
    b = 5
    nb = int(math.ceil(H / b))
    starts = rng.integers(0, len(r) - b, size=(sims, nb))
    idx = (starts[:, :, None] + np.arange(b)[None, None, :]).reshape(sims, nb * b)[:, :H]
    paths = np.cumsum(r[idx], axis=1)
    term = np.exp(paths[:, -1]) - 1
    mx = np.exp(paths.max(axis=1)) - 1
    mn = np.exp(paths.min(axis=1)) - 1
    return term, mx, mn


def scenarios(term, sigs):
    mu, sd = term.mean(), term.std()
    masks = {"bear": term < mu - 0.5 * sd, "base": (term >= mu - 0.5 * sd) & (term <= mu + 0.5 * sd), "bull": term > mu + 0.5 * sd}
    goods = [s["head"] for s in sigs if s["tone"] == "good"][:3]
    bads = [s["head"] for s in sigs if s["tone"] == "bad"][:3]
    drivers = {
        "bull": goods or ["Trend and momentum carry on"],
        "bear": bads or ["Broad market sells off"],
        "base": ["Recent pace of trading continues", "Signals stay mixed"],
    }
    out = {}
    for k, m in masks.items():
        x = term[m]
        out[k] = dict(p=float(m.mean()) * 100, lo=float(np.percentile(x, 10)) * 100, hi=float(np.percentile(x, 90)) * 100,
                      med=float(np.median(x)) * 100, drivers=drivers[k])
    return out


def hit_rates(c, fwd=20):
    r = rsi(c)
    s50, s200 = c.rolling(50).mean(), c.rolling(200).mean()
    f = c.shift(-fwd) / c - 1
    conds = [
        ("RSI above 70 (overbought)", r > 70),
        ("RSI below 30 (oversold)", r < 30),
        ("Price above 200-day average", c > s200),
        ("Price below 200-day average", c < s200),
        ("50-day crosses above 200-day", (s50 > s200) & (s50.shift(1) <= s200.shift(1))),
        ("Price at a 60-day high", c >= c.rolling(60).max()),
    ]
    rows = []
    for name, cond in conds:
        x = f[cond.fillna(False)].dropna()
        if len(x) >= 5:
            rows.append(dict(name=name, n=int(len(x)), hit=float((x > 0).mean()) * 100, avg=float(x.mean()) * 100))
    return rows


def earnings_block(raw):
    h = raw["hist"]
    c = h["Close"]
    idx = h.index.tz_localize(None) if getattr(h.index, "tz", None) is not None else h.index
    e = raw.get("earn")
    out = dict(next=None, eps_est=None, rev_est=None, beats=[], reactions=[])
    now = pd.Timestamp.now()
    if e is not None and len(e):
        e = e.copy()
        e.index = pd.to_datetime(e.index).tz_localize(None) if getattr(e.index, "tz", None) is not None else pd.to_datetime(e.index)
        fut = e[e.index > now].sort_index()
        if len(fut):
            out["next"] = fut.index[0]
            if "EPS Estimate" in fut.columns and fut["EPS Estimate"].iloc[0] == fut["EPS Estimate"].iloc[0]:
                out["eps_est"] = float(fut["EPS Estimate"].iloc[0])
        rep = e.dropna(subset=["Reported EPS"]).sort_index(ascending=False).head(8) if "Reported EPS" in e.columns else e.iloc[0:0]
        for ts, row in rep.iterrows():
            est = row.get("EPS Estimate")
            out["beats"].append((ts, float(row["Reported EPS"]), None if est != est else float(est)))
            pos = idx.searchsorted(ts.normalize())
            try:
                if ts.hour and ts.hour < 12:  # before open
                    v = c.iloc[pos] / c.iloc[pos - 1] - 1
                else:                         # after close
                    v = c.iloc[pos + 1] / c.iloc[pos] - 1
                if pos > 0 and pos + 1 < len(c) + 1:
                    out["reactions"].append((ts.strftime("%b %y"), float(v) * 100))
            except Exception:
                pass
        out["reactions"] = out["reactions"][::-1]
    cal = raw.get("cal") or {}
    try:
        if isinstance(cal, dict):
            ra = cal.get("Revenue Average")
            if ra:
                out["rev_est"] = float(ra)
            if out["next"] is None:
                ed = cal.get("Earnings Date")
                if ed:
                    out["next"] = pd.Timestamp(ed[0])
            if out["eps_est"] is None and cal.get("Earnings Average") is not None:
                out["eps_est"] = float(cal["Earnings Average"])
    except Exception:
        pass
    return out


def options_block(raw, price):
    o = raw.get("opt")
    if not o:
        return None
    calls, puts = o["calls"].copy(), o["puts"].copy()
    if calls is None or puts is None or not len(calls) or not len(puts):
        return None

    def mid(df):
        m = (df["bid"] + df["ask"]) / 2
        return m.where((df["bid"] > 0) & (df["ask"] > 0), df["lastPrice"])

    calls["mid"], puts["mid"] = mid(calls), mid(puts)
    k = calls["strike"].iloc[(calls["strike"] - price).abs().argsort().iloc[0]]
    cm = float(calls.loc[calls["strike"] == k, "mid"].iloc[0])
    pm = float(puts.loc[puts["strike"] == k, "mid"].iloc[0]) if (puts["strike"] == k).any() else cm
    ivs = []
    for df in (calls, puts):
        x = df.loc[(df["strike"] - price).abs().sort_values().index[:3], "impliedVolatility"]
        ivs += [float(v) for v in x if v and v > 0.02]
    iv = float(np.mean(ivs)) * 100 if ivs else None
    c = raw["hist"]["Close"]
    rv = float(np.log(c / c.shift(1)).dropna().iloc[-30:].std() * math.sqrt(252)) * 100
    coi, poi = calls["openInterest"].fillna(0).sum(), puts["openInterest"].fillna(0).sum()
    cv, pv = calls["volume"].fillna(0).sum(), puts["volume"].fillna(0).sum()
    strikes = sorted(set(calls["strike"]) | set(puts["strike"]))
    best, bp = None, None
    for K in strikes:
        pain = float((calls["openInterest"].fillna(0) * (K - calls["strike"]).clip(lower=0)).sum()
                     + (puts["openInterest"].fillna(0) * (puts["strike"] - K).clip(lower=0)).sum())
        if bp is None or pain < bp:
            best, bp = K, pain
    return dict(exp=o["exp"], dte=o["dte"], implied_move=(cm + pm) / price * 100 if price else None, iv=iv, rv=rv,
                pc_oi=float(poi / coi) if coi else None, pc_vol=float(pv / cv) if cv else None, max_pain=float(best) if best else None)


def trade_plan(price, atr_v, score, account, risk_pct):
    if score < 58:
        return None
    stop = price - 2 * atr_v
    target = price + 3 * atr_v
    rps = price - stop
    shares = int(min(account * risk_pct / 100.0 / rps, account / price))
    if shares < 1:
        return None
    return dict(entry=price, stop=stop, target=target, shares=shares, value=shares * price, risk=shares * rps, rr=(target - price) / rps)


def analyze(raw, mkt, horizon, account, risk_pct, drift):
    h = raw["hist"]
    c = h["Close"]
    info = raw["info"]
    price = float(c.iloc[-1])
    prev = float(c.iloc[-2])
    sigs = build_signals(raw)
    fac, vol = build_factors(raw, sigs)
    score = composite(fac)
    term, mx, mn = monte_carlo(c, horizon, drift)
    D = dict(raw=raw, t=raw["t"], hist=h, c=c, price=price, chg=(price / prev - 1) * 100, info=info, sigs=sigs, fac=fac, vol=vol,
             score=score, verdict=verdict(score), regime=regime(mkt), mkt=mkt, term=term, mx=mx, mn=mn, horizon=horizon,
             scen=scenarios(term, sigs), hits=hit_rates(c), earn=earnings_block(raw), opt=options_block(raw, price))
    D["atr"] = float(atr(h).iloc[-1])
    D["plan"] = trade_plan(price, D["atr"], score, account, risk_pct)
    D["nb"] = sum(1 for s in sigs if s["tone"] == "bad")
    D["ng"] = sum(1 for s in sigs if s["tone"] == "good")
    return D


def run_ml(c, H):
    """Walk-forward, purged logistic regression. Returns (out-of-fold accuracy, P(up over horizon))."""
    df = pd.DataFrame({"c": c})
    df["r5"], df["r20"] = c.pct_change(5), c.pct_change(20)
    df["rsi"] = rsi(c) / 100
    df["d200"] = c / c.rolling(200).mean() - 1
    df["v20"] = np.log(c / c.shift(1)).rolling(20).std() * math.sqrt(252)
    df["y"] = (c.shift(-H) / c - 1 > 0).astype(float)
    feats = ["r5", "r20", "rsi", "d200", "v20"]
    live = df[feats].iloc[[-1]]
    d = df.dropna(subset=feats).iloc[:-H].dropna(subset=["y"])
    if len(d) < 250:
        return None
    X = d[feats].values
    mu, sd = X.mean(0), X.std(0) + 1e-9
    X = (X - mu) / sd
    y = d["y"].values

    def fit(Xt, yt):
        w = np.zeros(Xt.shape[1] + 1)
        Xb = np.hstack([Xt, np.ones((len(Xt), 1))])
        for _ in range(400):
            p = 1 / (1 + np.exp(-Xb @ w))
            w -= 0.2 * (Xb.T @ (p - yt) / len(yt) + 0.02 * np.r_[w[:-1], 0])
        return w

    def pred(w, Xt):
        return 1 / (1 + np.exp(-np.hstack([Xt, np.ones((len(Xt), 1))]) @ w))

    n = len(X)
    folds, accs = 5, []
    edges = np.linspace(int(n * 0.4), n, folds + 1).astype(int)
    for i in range(folds):
        a, b = edges[i], edges[i + 1]
        tr = np.arange(0, max(a - H, 0))  # purge: drop the last H rows before the test window
        if len(tr) < 100:
            continue
        w = fit(X[tr], y[tr])
        accs.append(float(((pred(w, X[a:b]) > 0.5) == (y[a:b] > 0.5)).mean()))
    if not accs:
        return None
    w = fit(X, y)
    p_up = float(pred(w, (live.values - mu) / sd)[0])
    return dict(acc=float(np.mean(accs)) * 100, base=float(y.mean()) * 100, p_up=p_up * 100)


# ----------------------------------------------------------------------------
# Edge Finder — scans a universe of liquid stocks in ONE batched request
# (price + volume only, so it stays fast) and ranks them on a technical
# edge score. This is deliberately lighter than the single-ticker deep dive
# in Verdict, which also pulls fundamentals, analyst actions and earnings.
# ----------------------------------------------------------------------------
UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AVGO", "ORCL", "ADBE",
    "CRM", "AMD", "INTC", "CSCO", "QCOM", "TXN", "IBM", "NOW", "UBER", "SHOP",
    "JPM", "BAC", "WFC", "GS", "MS", "V", "MA", "AXP", "SCHW", "BLK",
    "UNH", "JNJ", "PFE", "LLY", "ABBV", "MRK", "TMO", "ABT", "DHR",
    "XOM", "CVX", "COP", "SLB",
    "HD", "MCD", "NKE", "SBUX", "TGT", "LOW", "BKNG",
    "PG", "KO", "PEP", "WMT", "COST",
    "DIS", "NFLX", "CMCSA",
    "BA", "CAT", "GE", "HON", "UPS", "LIN", "UNP",
]
SECTOR_OF = {  # rough grouping, for the "why this sector" line only
    "AAPL": "Technology", "MSFT": "Technology", "GOOGL": "Technology", "AMZN": "Consumer disc.", "NVDA": "Technology",
    "META": "Technology", "TSLA": "Consumer disc.", "AVGO": "Technology", "ORCL": "Technology", "ADBE": "Technology",
    "CRM": "Technology", "AMD": "Technology", "INTC": "Technology", "CSCO": "Technology", "QCOM": "Technology",
    "TXN": "Technology", "IBM": "Technology", "NOW": "Technology", "UBER": "Technology", "SHOP": "Technology",
    "JPM": "Financials", "BAC": "Financials", "WFC": "Financials", "GS": "Financials", "MS": "Financials",
    "V": "Financials", "MA": "Financials", "AXP": "Financials", "SCHW": "Financials", "BLK": "Financials",
    "UNH": "Health care", "JNJ": "Health care", "PFE": "Health care", "LLY": "Health care", "ABBV": "Health care",
    "MRK": "Health care", "TMO": "Health care", "ABT": "Health care", "DHR": "Health care",
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy", "SLB": "Energy",
    "HD": "Consumer disc.", "MCD": "Consumer disc.", "NKE": "Consumer disc.", "SBUX": "Consumer disc.",
    "TGT": "Consumer disc.", "LOW": "Consumer disc.", "BKNG": "Consumer disc.",
    "PG": "Staples", "KO": "Staples", "PEP": "Staples", "WMT": "Staples", "COST": "Staples",
    "DIS": "Comm. services", "NFLX": "Comm. services", "CMCSA": "Comm. services",
    "BA": "Industrials", "CAT": "Industrials", "GE": "Industrials", "HON": "Industrials",
    "UPS": "Industrials", "LIN": "Materials", "UNP": "Industrials",
}


@st.cache_data(ttl=1800, show_spinner=False)
def scan_universe(tickers=tuple(UNIVERSE)):
    px = _safe(lambda: yf.download(list(tickers), period="14mo", auto_adjust=True, progress=False, group_by="column"))
    if px is None or not len(px):
        return []
    out = []
    for t in tickers:
        try:
            c = px["Close"][t].dropna()
            v = px["Volume"][t].reindex(c.index).fillna(0)
        except Exception:
            continue
        if len(c) < 210:
            continue
        price, prev = float(c.iloc[-1]), float(c.iloc[-2])
        s50, s200 = c.rolling(50).mean(), c.rolling(200).mean()
        rv = float(rsi(c).iloc[-1])
        m21 = float(c.iloc[-1] / c.iloc[-22] - 1) * 100
        m63 = float(c.iloc[-1] / c.iloc[-64] - 1) * 100
        obv = (np.sign(c.diff()).fillna(0) * v).cumsum()
        obv_chg = float(obv.iloc[-1] - obv.iloc[-43]) if len(obv) > 43 else 0.0
        obv_scale = float(v.iloc[-60:].mean() * 40) or 1.0
        flow = clip(50 + 50 * math.tanh(obv_chg / obv_scale))
        vol60 = float(np.log(c / c.shift(1)).dropna().iloc[-60:].std() * math.sqrt(252))

        trend = 45
        trend += 20 if price > s200.iloc[-1] else -18
        trend += 10 if price > s50.iloc[-1] else -8
        trend += 10 if s50.iloc[-1] > s200.iloc[-1] else -8
        trend = clip(trend)
        momentum = clip(50 + m63 * 1.6 - abs(rv - 55) * 0.7)
        risk_s = clip(100 - (vol60 - 0.16) * 170)
        score = int(round(trend * .35 + momentum * .30 + flow * .20 + risk_s * .15))

        bits = []
        if price > s200.iloc[-1] and s50.iloc[-1] > s200.iloc[-1]:
            bits.append(("good", "Uptrend: price and the 50-day are above the 200-day"))
        elif price < s200.iloc[-1] and s50.iloc[-1] < s200.iloc[-1]:
            bits.append(("bad", "Downtrend: price and the 50-day are below the 200-day"))
        if rv > 70:
            bits.append(("bad", "Overbought (RSI %d)" % rv))
        elif rv < 30:
            bits.append(("good", "Oversold (RSI %d) — often a bounce zone" % rv))
        if flow > 62 and m21 < 1:
            bits.append(("good", "Quiet accumulation — buyers active while price is flat"))
        elif flow < 38 and m21 > -1:
            bits.append(("bad", "Quiet distribution — sellers active while price is flat"))
        if m63 > 12:
            bits.append(("good", "Strong 3-month momentum, +%.0f%%" % m63))
        elif m63 < -12:
            bits.append(("bad", "Heavy 3-month slide, %.0f%%" % m63))
        if not bits:
            bits.append(("flat", "No strong technical tilt right now"))

        out.append(dict(t=t, sector=SECTOR_OF.get(t, "Other"), price=price, chg=(price / prev - 1) * 100,
                         score=score, rsi=rv, m21=m21, m63=m63, flow=flow, vol=vol60 * 100,
                         reason=bits[0][1], tone=bits[0][1] and bits[0][0]))
    return out


# ----------------------------------------------------------------------------
# Rendering
# ----------------------------------------------------------------------------
def nav_html():
    return H('<div class="nav"><div class="brand">%s<span>Market Reader</span></div>'
             '<div class="links"><span>Home</span><span>Stocks</span><span>Watchlist</span><span>Insights</span></div>'
             '<div class="navr">%s<span class="cta">Get Started</span></div></div>' % (logo_mark(34), icon("search", INK, 20)))


def hero_html(D):
    info = D["info"]
    up = D["chg"] >= 0
    name = info.get("longName") or info.get("shortName") or D["t"]
    exch = info.get("fullExchangeName") or info.get("exchange") or ""
    site = (info.get("website") or "").replace("https://", "").replace("http://", "").replace("www.", "").split("/")[0]
    fav = ('<img alt="" src="https://www.google.com/s2/favicons?domain=%s&sz=128"/>' % site) if site else ""
    chg = '<span class="chg %s"><span class="livedot %s"></span>%s%.2f%% today</span>' % (
        "up" if up else "dn", "up" if up else "dn", icon("up" if up else "down", GREEN if up else RED, 16, 2.4), abs(D["chg"]))
    bad_lead = D["nb"] >= D["ng"]
    scol = GREEN if D["score"] >= 55 else (RED if D["score"] < 45 else SLATE)
    panels = (
        '<div class="gp gp1 %s"><div class="gi" style="background:%s">%s</div><div class="gl">Bearish signals</div><div class="gv">%d</div></div>'
        '<div class="gp gp2"><div class="gi" style="background:#E6F6EE">%s</div><div class="gl">Bullish signals</div><div class="gv">%d</div></div>'
        '<div class="gp gp3"><div class="ring-wrap">%s</div><div class="gl">Composite</div><div class="gv">%d<span style="font-size:16px;color:#6B7488">/100</span></div></div>'
    ) % ("bad" if bad_lead else "", "#FDECEC", icon("down", RED, 22, 2.4), D["nb"],
         icon("trend", GREEN, 22, 2.4), D["ng"], ring(D["score"], scol, 52, label="", stroke=6), D["score"])
    return H('<div class="hero"><div><div class="idrow"><div class="lg">%s%s</div><div><div class="tk">%s</div>'
             '<div class="nm">%s%s</div></div></div><div class="prow"><span class="px">%s</span>%s</div></div>'
             '<div class="art"><div class="wave">%s</div>%s<div class="tag">Turn data into confidence.</div></div></div>'
             % (D["t"][:1], fav, D["t"], name, (" &bull; " + exch) if exch else "", money(D["price"]), chg,
                wave_svg(D["c"].values[-160:]), panels))


def _sig_rows(sigs, n=4):
    show = [s for s in sigs if s["tone"] != "flat"][:n]
    if len(show) < n:
        show += [s for s in sigs if s["tone"] == "flat"][: n - len(show)]
    if not show:
        return '<div class="help">No unusual signals right now.</div>'
    return "".join('<div class="sig"><div class="dot %s"></div><div><b>%s</b> <span>&mdash; %s</span></div></div>' % (s["tone"], s["head"], s["body"]) for s in show)


def verdict_tab(D):
    reg = D["regime"]
    vtxt, vt = D["verdict"]
    rcol = {"g": GREEN, "r": RED, "n": "#39425A"}[reg["tone"]]
    reg_card = ('<div class="card"><div class="k">Market regime</div><div class="big %s">%s</div>%s</div>' % (
        reg["tone"], reg["label"],
        (('<div class="chip %s">%s%d/100</div>' % (reg["tone"], icon("up" if reg["tone"] != "r" else "down", rcol, 18, 2.6), reg["score"])) if reg["score"] is not None else "")
        + '<div class="help">Based on the S&amp;P 500 trend and the VIX fear gauge.</div>'))
    comp = ('<div class="card"><div class="k">Composite score</div><div class="big">%d<small>/100</small></div>'
            '<div style="margin-top:12px"><span class="verdict %s">%s</span></div><div class="help">Weighted from trend, momentum, quality, value, sentiment and risk.</div></div>') % (D["score"], vt, vtxt)
    sig = '<div class="card"><h4>%s What the headline numbers hide</h4>%s</div>' % (hi("pulse"), _sig_rows(D["sigs"]))
    top = '<div class="g3">%s%s%s</div>' % (comp, reg_card, sig)

    c = D["c"]
    n = min(252, len(c))
    pc = resp(area_chart, c.values[-n:], c.index[-n:], lines=[(c.rolling(50).mean().values[-n:], "#E0A030", None), (c.rolling(200).mean().values[-n:], "#8A93A8", "5 4")])
    chart = ('<div class="card stack"><h4>%s Price, last 12 months</h4>%s'
             '<div class="help">Amber line: 50-day average. Dashed grey line: 200-day average.</div></div>') % (hi("trend"), pc)

    fbars = ""
    for k, v in D["fac"].items():
        cls = "g" if v >= 65 else ("r" if v < 40 else "")
        fbars += '<div class="fb"><span>%s</span><div class="bar"><i class="%s" style="width:%d%%"></i></div><b>%d</b></div>' % (k, cls, v, v)
    factors = '<div class="card"><h4>%s What drives the score</h4>%s<div class="help">Factors with no data are left out, not counted as zero.</div></div>' % (hi("layers"), fbars)

    p = D["plan"]
    if p:
        plan = ('<div class="card"><h4>%s Trade plan</h4><div class="kv"><span>Entry</span><b>%s</b></div><div class="kv"><span>Stop</span><b>%s</b></div>'
                '<div class="kv"><span>Target</span><b>%s</b></div><div class="kv"><span>Size</span><b>%d shares (%s)</b></div>'
                '<div class="kv"><span>Money at risk</span><b>%s</b></div><div class="kv"><span>Reward to risk</span><b>%.1f to 1</b></div></div>') % (
            hi("target"), money(p["entry"]), money(p["stop"]), money(p["target"]), p["shares"], money(p["value"], 0), money(p["risk"], 0), p["rr"])
    else:
        plan = ('<div class="card"><h4>%s Trade plan</h4><div class="msg" style="margin:0">No plan. The composite score is below 58, so the model does not suggest a trade.</div></div>' % hi("target"))
    return H(top + chart + '<div class="g2">%s%s</div>' % (factors, plan))


def lines_tab(D):
    rows = "".join(
        '<div class="row"><div class="n">%s</div><div><div class="bar"><i class="%s" style="width:%d%%"></i></div></div>'
        '<div class="m">%.0f%% went up in the next 20 days, average %s, seen %d times</div></div>'
        % (r["name"], "g" if r["hit"] >= 58 else ("r" if r["hit"] < 45 else ""), r["hit"], r["hit"], pct(r["avg"]), r["n"]) for r in D["hits"])
    hits = '<div class="card"><h4>%s How each signal has played out on %s</h4>%s<div class="help">Two years of daily history. Past hit rates are context, not a promise.</div></div>' % (hi("pulse"), D["t"], rows or "No data")
    h = D["hist"]
    n = min(252, len(h))
    ad = ad_line(h)
    flow = ('<div class="card stack"><h4>%s Money flow</h4>%s<div class="help">Rising line: buyers are taking more shares than sellers. Falling line: the reverse.</div></div>'
            % (hi("trend"), resp(area_chart, ad.values[-n:] / 1e6, h.index[-n:], color="#5B6FD0", fmt="{:,.0f}M")))
    ins = D["raw"].get("ins")
    ins_html = ""
    try:
        if ins is not None and len(ins):
            d = ins.head(6)
            body = ""
            for _, r in d.iterrows():
                who = str(r.get("Insider", ""))
                tx = str(r.get("Text", "") or r.get("Transaction", ""))[:70]
                val = r.get("Value")
                body += '<div class="kv"><span>%s</span><b>%s%s</b></div>' % (who.title()[:30], tx.split(" at ")[0][:34] or "-", (" (" + money(abs(val), 0) + ")") if val == val and val else "")
            ins_html = '<div class="card stack"><h4>%s Insiders</h4>%s</div>' % (hi("layers"), body)
    except Exception:
        ins_html = ""
    return H(hits + flow + ins_html)


def prob_tab(D, ml):
    s = D["scen"]
    term = D["term"]
    card = lambda key, title, col: (
        '<div class="sc %s"><div class="nm2"><span class="dot %s" style="flex:0 0 12px;height:12px;margin:0"></span>%s</div><div class="pp">%.0f%%</div>'
        '<div class="k">Likely range %s to %s</div><ul>%s</ul></div>' % (
            key, col, title, s[key]["p"], pct(s[key]["lo"], 0), pct(s[key]["hi"], 0), "".join("<li>%s</li>" % d for d in s[key]["drivers"])))
    cards = '<div class="g3s">%s%s%s</div>' % (card("bear", "Bear", "bad"), card("base", "Base", "flat"), card("bull", "Bull", "good"))
    chart = ('<div class="card stack"><h4>%s Where the price could be in %d trading days</h4>%s'
             '<div class="help">4,000 simulated paths from this stock&rsquo;s own history. Red: bear, grey: base, green: bull. Dashed line: median.</div></div>') % (hi("pulse"), D["horizon"], resp(hist_chart, term))
    p5 = float((D["mx"] >= 0.05).mean()) * 100
    m5 = float((D["mn"] <= -0.05).mean()) * 100
    k = lambda a, b, c_="": '<div class="card"><div class="k">%s</div><div class="big" style="font-size:30px;color:%s">%s</div>%s</div>' % (a, c_ or INK, b, "")
    metrics = '<div class="g4">%s%s%s%s</div>' % (
        k("Expected return", pct(term.mean() * 100), GREEN if term.mean() >= 0 else RED),
        k("Downside (worst 10%)", pct(np.percentile(term, 10) * 100), RED),
        k("Upside (best 10%)", pct(np.percentile(term, 90) * 100), GREEN),
        k("Touches +5% / -5%", "%.0f%% / %.0f%%" % (p5, m5)))
    mlh = ""
    if ml:
        edge = ml["acc"] - ml["base"]
        mlh = ('<div class="card stack"><h4>%s ML model</h4><div class="kv"><span>Chance of finishing higher</span><b>%.0f%%</b></div>'
               '<div class="kv"><span>Out-of-sample accuracy</span><b>%.0f%%</b></div><div class="kv"><span>Always-guess-up baseline</span><b>%.0f%%</b></div>'
               '<div class="help">%s</div></div>') % (hi("target"), ml["p_up"], ml["acc"], ml["base"],
                                                     "Model beats the baseline by %.0f points." % edge if edge > 1.5 else "Model does not beat the baseline - treat its output as noise.")
    return H(cards + chart + metrics), mlh


def earnings_tab(D):
    e, o = D["earn"], D["opt"]
    kv = ""
    if e["next"] is not None:
        kv += '<div class="kv"><span>Next report</span><b>%s</b></div>' % pd.Timestamp(e["next"]).strftime("%b %d, %Y")
    if e["eps_est"] is not None:
        kv += '<div class="kv"><span>EPS estimate</span><b>%s</b></div>' % money(e["eps_est"])
    if e["rev_est"] is not None:
        kv += '<div class="kv"><span>Revenue estimate</span><b>%s</b></div>' % ("&#36;%.1fB" % (e["rev_est"] / 1e9))
    if e["beats"]:
        b = sum(1 for _, r, es in e["beats"] if es is not None and r > es)
        kv += '<div class="kv"><span>Beat history</span><b>%d of last %d</b></div>' % (b, len(e["beats"]))
    left = '<div class="card"><h4>%s Earnings</h4>%s</div>' % (hi("cal"), kv or '<div class="help">No earnings data from Yahoo for this ticker.</div>')
    okv = ""
    if o:
        okv += '<div class="kv"><span>Expiry used</span><b>%s (%d days)</b></div>' % (o["exp"], o["dte"])
        if o["implied_move"] is not None:
            okv += '<div class="kv"><span>Implied move to expiry</span><b>&plusmn;%.1f%%</b></div>' % o["implied_move"]
        if o["iv"] is not None:
            okv += '<div class="kv"><span>Implied vs realized volatility</span><b>%.0f%% vs %.0f%%</b></div>' % (o["iv"], o["rv"])
        if o["pc_oi"] is not None:
            okv += '<div class="kv"><span>Put/call (open interest)</span><b>%.2f</b></div>' % o["pc_oi"]
        if o["pc_vol"] is not None:
            okv += '<div class="kv"><span>Put/call (volume)</span><b>%.2f</b></div>' % o["pc_vol"]
        if o["max_pain"]:
            okv += '<div class="kv"><span>Max pain</span><b>%s</b></div>' % money(o["max_pain"])
    right = '<div class="card"><h4>%s Options</h4>%s</div>' % (hi("layers"), okv or '<div class="help">No usable options chain right now.</div>')
    react = ""
    if e["reactions"]:
        avg = np.mean([abs(v) for _, v in e["reactions"]])
        react = ('<div class="card stack"><h4>%s Price reaction after each report</h4>%s<div class="help">Average move %.1f%%. Before-open and after-close reports are measured from the right close.</div></div>'
                 % (hi("trend"), resp(reaction_chart, e["reactions"]), avg))
    return H('<div class="g2">%s%s</div>%s' % (left, right, react))


def insights_tab(D):
    reg, mkt = D["regime"], D["mkt"]
    kv = ""
    if reg["score"] is not None:
        kv += '<div class="kv"><span>Regime</span><b>%s (%d/100)</b></div><div class="kv"><span>S&amp;P 500</span><b>%s</b></div>' % (reg["label"], reg["score"], reg["spy_trend"])
        if reg.get("vix"):
            kv += '<div class="kv"><span>VIX (fear gauge)</span><b>%.1f</b></div>' % reg["vix"]
        kv += '<div class="kv"><span>S&amp;P 500, 1 month</span><b>%s</b></div>' % pct(reg["spy_1m"])
    alerts = []
    if reg["score"] is not None and reg.get("vix") and reg["vix"] > 25:
        alerts.append(("bad", "Fear is elevated", "VIX is above 25."))
    if reg["score"] is not None and reg["spy_trend"] == "below 200-day":
        alerts.append(("bad", "Market is in a downtrend", "S&P 500 is under its 200-day average."))
    if reg["score"] is not None and reg["label"] == "RISK-ON":
        alerts.append(("good", "Conditions favor risk", "Trend and volatility are supportive."))
    ah = "".join('<div class="sig"><div class="dot %s"></div><div><b>%s</b> <span>&mdash; %s</span></div></div>' % a for a in alerts) or '<div class="help">No alerts.</div>'
    top = '<div class="g2"><div class="card"><h4>%s Market regime</h4>%s</div><div class="card"><h4>%s Live alerts</h4>%s</div></div>' % (hi("shield"), kv or "No market data.", hi("pulse"), ah)
    sec = mkt.get("sectors") or {}
    sh = ""
    if sec:
        items = sorted(sec.items(), key=lambda kv_: -kv_[1][0])
        m = max(abs(v[0]) for _, v in items) or 1
        for name, (m1, _m3) in items:
            sh += '<div class="fb"><span>%s</span><div class="bar"><i class="%s" style="width:%d%%"></i></div><b style="width:60px;margin-left:-14px">%s</b></div>' % (
                name, "g" if m1 >= 0 else "r", max(4, abs(m1) / m * 100), pct(m1))
    sect = '<div class="card stack"><h4>%s Sector rotation, last month</h4>%s<div class="help">Which parts of the market money is moving into and out of.</div></div>' % (hi("layers"), sh or "No sector data.")
    news = ""
    for it in (D["raw"].get("news") or [])[:6]:
        try:
            cn = it.get("content") or it
            ttl = cn.get("title")
            link = (cn.get("canonicalUrl") or {}).get("url") or cn.get("link") or (cn.get("clickThroughUrl") or {}).get("url") or "#"
            pub = (cn.get("provider") or {}).get("displayName") or cn.get("publisher") or ""
            if ttl:
                news += '<div><a href="%s" target="_blank">%s</a><small>%s</small></div>' % (link, ttl, pub)
        except Exception:
            continue
    newsc = '<div class="card stack news"><h4>%s Headlines for %s</h4>%s</div>' % (hi("cal"), D["t"], news or "No headlines.")
    return H(top + sect + newsc)


def edge_card(row, rank=None):
    tone = "g" if row["score"] >= 60 else ("r" if row["score"] <= 40 else "n")
    col = {"g": GREEN, "r": RED, "n": SLATE}[tone]
    up = row["chg"] >= 0
    badge = '<div class="rank">%d</div>' % rank if rank else ""
    return H('<div class="card edge">%s<div class="edge-top"><div><div class="edge-t">%s</div><div class="k">%s</div></div>%s</div>'
              '<div class="edge-px"><span>%s</span><span class="chg %s">%s%.2f%%</span></div>'
              '<div class="help edge-why">%s</div></div>' % (
                  badge, row["t"], row["sector"], ring(row["score"], col, 44, stroke=5),
                  money(row["price"]), "up" if up else "dn", icon("up" if up else "down", GREEN if up else RED, 13, 3), abs(row["chg"]),
                  row["reason"]))


def edge_finder_tab(scan, D):
    if not scan:
        return H('<div class="msg">The scanner needs a live connection to Yahoo Finance to rank stocks &mdash; try again in a moment.</div>'), []
    ranked = sorted(scan, key=lambda r: -r["score"])
    bulls = [r for r in ranked if r["score"] >= 55][:6] or ranked[:6]
    bears = sorted(scan, key=lambda r: r["score"])[:6]
    intro = ('<div class="card"><h4>%s What this scan is doing</h4>'
             '<div class="help" style="font-size:14px;line-height:1.6">Every few minutes this scans %d well-known, heavily-traded stocks across every sector and scores each one 0&ndash;100 '
             'on trend, momentum, money flow, and risk &mdash; the same building blocks used in the Verdict tab, just without the per-stock fundamentals so it can scan this many at once. '
             'A high score means several technical tailwinds are lining up together; a low score means the opposite. This is a starting list to research further, not a buy or sell signal.</div></div>'
             ) % (hi("radar"), len(scan))
    bull_head = '<h3 style="margin:24px 0 2px;font-size:18px">%s Top setups to watch &mdash; bullish tilt</h3><div class="help" style="margin-bottom:14px">Ranked highest technical edge score first.</div>' % hi("star")
    bear_head = '<h3 style="margin:28px 0 2px;font-size:18px">%s Weakest setups &mdash; bearish tilt</h3><div class="help" style="margin-bottom:14px">Worth watching for a bounce, or avoiding new longs.</div>' % hi("compass")
    bull_grid = '<div class="edge-grid">%s</div>' % "".join(edge_card(r, i + 1) for i, r in enumerate(bulls))
    bear_grid = '<div class="edge-grid">%s</div>' % "".join(edge_card(r, i + 1) for i, r in enumerate(bears))
    rows = "".join(
        '<div class="row erow"><div class="n">%s <span class="k">%s</span></div><div class="m">%s &nbsp; <span class="chg %s">%.2f%%</span></div>'
        '<div class="m">RSI %d &middot; 3M %s &middot; score <b>%d</b></div></div>' % (
            r["t"], r["sector"], money(r["price"]), "up" if r["chg"] >= 0 else "dn", r["chg"], r["rsi"], pct(r["m63"], 0), r["score"])
        for r in ranked)
    table = '<div class="card stack"><h4>%s Full scan, every ticker</h4>%s</div>' % (hi("layers"), rows)
    return H(intro + bull_head + bull_grid + bear_head + bear_grid + table), [r["t"] for r in bulls + bears]


def lesson(icon_name, title, body):
    return H('<details class="lesson"><summary>%s<span class="lt">%s</span><span class="chev">%s</span></summary>'
              '<div class="lesson-body">%s</div></details>' % (hi(icon_name), title, icon("down", MUTED, 16, 2.4), body))


def learn_tab(D):
    rv = D["fac"]
    trend_now = "above" if D["c"].iloc[-1] > D["c"].rolling(200).mean().iloc[-1] else "below"
    rsi_now = float(rsi(D["c"]).iloc[-1])
    rsi_read = "overbought territory" if rsi_now > 70 else ("oversold territory" if rsi_now < 30 else "the neutral zone")
    intro = ('<div class="card"><h4>%s Learn to read the market</h4><div class="help" style="font-size:14px;line-height:1.6">'
             'Short, plain-English lessons behind every number in this app &mdash; each one uses %s&rsquo;s own live data as the example, '
             'so the ideas stick to a real stock instead of a textbook one. Open any topic below.</div></div>') % (hi("book"), D["t"])

    l1 = lesson("layers", "Trend: the 50-day and 200-day averages",
                'A moving average smooths out the daily noise so you can see the underlying direction. The 200-day average is the line most professionals watch for the '
                'big picture; the 50-day reacts faster to recent weeks. Right now %s is trading <b>%s</b> its 200-day average, and its own Trend factor scores <b>%d/100</b>. '
                '<br><br><b>Rule of thumb:</b> price above a rising 200-day average = uptrend. Price below a falling one = downtrend. When the 50-day crosses above the 200-day, '
                'traders call it a &ldquo;golden cross&rdquo; (bullish); the reverse is a &ldquo;death cross&rdquo; (bearish).' % (D["t"], trend_now, rv.get("Trend", 50)))

    l2 = lesson("pulse", "Momentum &amp; RSI (Relative Strength Index)",
                'RSI measures how fast and how far a stock has moved recently, on a scale of 0 to 100. %s&rsquo;s RSI is currently <b>%.0f</b>, which sits in %s. '
                '<br><br><b>Reading it:</b> above 70 usually means a stock has run hot and is due to cool off (overbought). Below 30 usually means it has been sold off hard and '
                'could be due for a bounce (oversold). The most useful signal isn&rsquo;t the level itself but a <b>divergence</b> &mdash; when price makes a new high but RSI '
                'makes a lower high, the rally is running out of fuel even though the price chart still looks strong. That is exactly what the &ldquo;What the headline numbers '
                'hide&rdquo; card on the Verdict tab is built to catch.' % (D["t"], rsi_now, rsi_read))

    l3 = lesson("trend", "Money flow: accumulation vs. distribution",
                'Price tells you what happened. Volume tells you how much conviction was behind it. The money-flow line on the Between the Lines tab adds up buying and selling '
                'pressure over time using where each day closes within its range. <br><br>When price drifts sideways but money flow keeps climbing, big holders may be quietly '
                '<b>accumulating</b> shares before a move. When price holds up but money flow rolls over, that can be quiet <b>distribution</b> &mdash; large holders selling into '
                'strength while the price hasn&rsquo;t reacted yet. Neither is a guarantee, but both are exactly the kind of thing that does not show up in the price chart alone.')

    l4 = lesson("shield", "Analyst actions &amp; earnings streaks",
                'Wall Street analysts publish buy/sell ratings and upgrade or downgrade them as their view changes. A cluster of downgrades in a short window is a signal that '
                'professional opinion is turning, even before the price fully reflects it &mdash; and the reverse for upgrades. <br><br>Earnings beat streaks matter for a different '
                'reason: many companies guide expectations low on purpose so they can comfortably beat them (&ldquo;sandbagging&rdquo;). A long beat streak isn&rsquo;t proof of a '
                'great business by itself, but a company that suddenly starts <i>missing</i> after years of beating is often the first sign something has changed.')

    l5 = lesson("target", "Valuation: P/E and PEG, in plain terms",
                'The P/E ratio is simply the stock price divided by its earnings per share &mdash; it tells you how many dollars of price you are paying for one dollar of yearly '
                'profit. A P/E of 30 means investors are paying $30 for every $1 of current profit. <br><br>On its own, a high P/E is not automatically &ldquo;expensive&rdquo; &mdash; '
                'fast-growing companies usually deserve one. That is what <b>forward</b> P/E (priced on next year&rsquo;s expected earnings) and the <b>PEG ratio</b> (P/E divided '
                'by expected growth rate) try to correct for: a PEG near 1 suggests the price roughly matches the growth on offer; well above 2 suggests the market has already '
                'priced in a lot of optimism.')

    l6 = lesson("radar", "The Probability Engine: what a Monte Carlo simulation actually is",
                'Nobody can predict one exact future price. Instead, the Probability Engine replays thousands of alternate paths built from this stock&rsquo;s own historical daily '
                'moves (a technique called <b>block-bootstrap simulation</b>), and looks at where all of those paths end up. <br><br>The Bear/Base/Bull split is just the worst third, '
                'middle third, and best third of those 4,000 simulated outcomes. It is a way of turning &ldquo;I don&rsquo;t know what will happen&rdquo; into an honest range of what '
                'has happened before, given this stock&rsquo;s own personality for calm or wild swings.')

    l7 = lesson("layers", "Options: implied move, IV vs. RV, put/call, and max pain",
                '<b>Implied move</b> is the size of price swing the options market is pricing in through the next expiration &mdash; back it out from what a call and a put at the '
                'nearest strike cost together. <b>Implied volatility (IV)</b> is the market&rsquo;s forecast of future swings; <b>realized volatility (RV)</b> is what actually '
                'happened recently. IV well above RV usually means options are pricing in extra uncertainty (often around earnings). <br><br>The <b>put/call ratio</b> compares '
                'bearish bets (puts) to bullish ones (calls) &mdash; a high ratio can mean traders are hedging or bearish. <b>Max pain</b> is the strike price at which the most '
                'options (both puts and calls) would expire worthless; some traders watch it as a rough magnet for where price drifts into an expiration.')

    l8 = lesson("shield", "Reading the Trade Plan: ATR, stops, and position size",
                'The Trade Plan only appears when the composite score is strong enough to suggest an edge. It sizes the trade using <b>ATR</b> (Average True Range) &mdash; a measure '
                'of how much a stock typically moves in a day &mdash; setting the stop 2 ATRs below entry and the target 3 ATRs above, so a winning trade is sized to be bigger than '
                'a losing one (a 1.5-to-1 reward-to-risk ratio or better). <br><br>Position size is then worked backward from how much money you are willing to risk (set in '
                'Settings) so that if the stop is hit, the loss matches that amount &mdash; not more. This is a risk-management framework, not a promise the trade will work.')

    l9 = lesson("compass", "Market regime &amp; the VIX",
                'A single stock does not trade in a vacuum &mdash; it trades inside a market climate. The Market Insights tab reads that climate using the S&amp;P 500&rsquo;s own '
                'trend and the <b>VIX</b>, often called Wall Street&rsquo;s &ldquo;fear gauge,&rdquo; which measures how much volatility options traders expect over the next 30 days. '
                '<br><br>VIX under 16 usually means calm, complacent markets (&ldquo;RISK-ON&rdquo;). Above 25&ndash;30 usually means fear is elevated (&ldquo;RISK-OFF&rdquo;) &mdash; '
                'the same bullish setup on an individual stock tends to work less reliably when the broader market is fighting a downtrend, which is why the regime badge sits right '
                'next to the composite score.')

    checklist = ('<div class="card stack"><h4>%s Five habits that actually find an edge</h4><ol class="cl">'
                 '<li>Never trade off one signal alone &mdash; look for two or three lining up (trend + momentum + money flow), the way the composite score does.</li>'
                 '<li>Check the market regime before the stock. A great setup in a RISK-OFF market carries more risk than the chart alone suggests.</li>'
                 '<li>Decide your stop and size <i>before</i> you enter, not after it starts moving against you.</li>'
                 '<li>Treat a Probability Engine bear case as a real possibility, not a footnote &mdash; it happens roughly a third of the time by design.</li>'
                 '<li>Beat streaks, upgrade clusters and low P/E can all reverse. Re-check the Verdict every time before acting on an old read.</li></ol></div>'
                 ) % hi("target")

    glossary_items = [
        ("RSI", "Relative Strength Index. 0&ndash;100 momentum gauge; above 70 = overbought, below 30 = oversold."),
        ("ATR", "Average True Range. The typical size of a stock's daily move, used to set stops and targets."),
        ("P/E", "Price-to-earnings ratio. Price divided by earnings per share."),
        ("PEG", "P/E divided by expected earnings growth. Near 1 suggests price and growth are in balance."),
        ("IV / RV", "Implied volatility (market's forecast) vs. realized volatility (what already happened)."),
        ("Put/Call ratio", "Bearish options bets divided by bullish ones. High = more hedging or bearish positioning."),
        ("Max pain", "The strike price where the most options would expire worthless."),
        ("VIX", "An index of expected S&amp;P 500 volatility over the next 30 days; the market's \"fear gauge\"."),
        ("Monte Carlo simulation", "Replaying thousands of randomized paths built from real historical moves to map a range of outcomes."),
        ("OBV / money flow", "A running total of volume, added on up days and subtracted on down days, used to gauge buying/selling pressure."),
    ]
    gl = "".join('<div class="kv"><span><b>%s</b></span><span class="m" style="text-align:right;max-width:70%%">%s</span></div>' % g for g in glossary_items)
    glossary = '<div class="card stack"><h4>%s Glossary</h4>%s</div>' % (hi("book"), gl)

    return H(intro + l1 + l2 + l3 + l4 + l5 + l6 + l7 + l8 + l9 + checklist + glossary)


# ----------------------------------------------------------------------------
# App
# ----------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Market Reader", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown('<div class="aurora"><i></i><i></i><i></i></div>', unsafe_allow_html=True)
    st.markdown(nav_html(), unsafe_allow_html=True)
    if "watch" not in st.session_state:
        st.session_state["watch"] = ["AAPL", "MSFT", "NVDA"]
    if "ticker" not in st.session_state:
        st.session_state["ticker"] = "AAPL"
    if st.session_state.get("_jump_to"):
        st.session_state["ticker"] = st.session_state.pop("_jump_to")

    c1, c2 = st.columns([5, 1.3])
    with c1:
        t = st.text_input("Ticker", key="ticker", placeholder="Search a ticker, e.g. AAPL", label_visibility="collapsed").strip().upper()
    with c2:
        pop = st.popover("Settings") if hasattr(st, "popover") else st.expander("Settings")
        with pop:
            horizon = st.selectbox("Forecast horizon (trading days)", [10, 21, 42, 63], index=1)
            account = st.number_input("Account size ($)", min_value=1000, value=10000, step=1000)
            risk = st.slider("Risk per trade (%)", 0.5, 3.0, 1.0, 0.5)
            drift = st.checkbox("Use the stock's historical drift", value=False, help="Off = neutral: simulated paths have no built-in up or down bias.")

    if not t or yf is None:
        st.markdown('<div class="msg">Type a ticker above to begin.</div>', unsafe_allow_html=True)
        return
    try:
        with st.spinner("Reading %s" % t):
            raw = load_raw(t)
            mkt = load_market()
    except Exception:
        st.markdown('<div class="msg err">Yahoo is busy right now. Wait a minute, then search again.</div>', unsafe_allow_html=True)
        return
    if raw is None:
        st.markdown('<div class="msg err">No data for &ldquo;%s&rdquo;. Check the ticker and try again, for example AAPL, MSFT or NVDA.</div>' % t.replace("<", ""), unsafe_allow_html=True)
        return

    D = analyze(raw, mkt, horizon, account, risk, drift)
    st.markdown(hero_html(D), unsafe_allow_html=True)

    tabs = st.tabs(["Verdict", "Between the Lines", "Probability Engine", "Earnings & Options", "Market Insights", "Edge Finder", "Learn", "Watchlist"])
    with tabs[0]:
        st.markdown(verdict_tab(D), unsafe_allow_html=True)
    with tabs[1]:
        st.markdown(lines_tab(D), unsafe_allow_html=True)
    with tabs[2]:
        key = "ml_%s_%d" % (t, horizon)
        if st.button("Run ML model", key="mlbtn"):
            with st.spinner("Training"):
                st.session_state[key] = run_ml(D["c"], horizon) or {}
        ml = st.session_state.get(key) or None
        body, mlh = prob_tab(D, ml)
        st.markdown(body, unsafe_allow_html=True)
        if mlh:
            st.markdown(mlh, unsafe_allow_html=True)
        elif st.session_state.get(key) == {}:
            st.markdown('<div class="msg err">Not enough history to train the ML model for this ticker.</div>', unsafe_allow_html=True)
    with tabs[3]:
        st.markdown(earnings_tab(D), unsafe_allow_html=True)
    with tabs[4]:
        st.markdown(insights_tab(D), unsafe_allow_html=True)
    with tabs[5]:
        with st.spinner("Scanning the market"):
            scan = scan_universe()
        body, shown = edge_finder_tab(scan, D)
        st.markdown(body, unsafe_allow_html=True)
        if shown:
            st.markdown('<div class="help" style="margin:16px 0 6px">Jump straight to the full Verdict for any pick above:</div>', unsafe_allow_html=True)
            cols = st.columns(6)
            for i, sym in enumerate(dict.fromkeys(shown)):
                with cols[i % 6]:
                    if st.button(sym, key="jump_%s" % sym, use_container_width=True):
                        st.session_state["_jump_to"] = sym
                        st.rerun()
    with tabs[6]:
        st.markdown(learn_tab(D), unsafe_allow_html=True)
    with tabs[7]:
        a, b = st.columns([4, 1])
        with a:
            add = st.text_input("Add ticker", placeholder="Add a ticker", label_visibility="collapsed", key="wadd").strip().upper()
        with b:
            if st.button("Add", key="wbtn") and add and add not in st.session_state["watch"]:
                st.session_state["watch"].append(add)
        rows = ""
        try:
            px = yf.download(st.session_state["watch"], period="5d", auto_adjust=True, progress=False)["Close"]
            if isinstance(px, pd.Series):
                px = px.to_frame(st.session_state["watch"][0])
            for s_ in st.session_state["watch"]:
                if s_ in px.columns:
                    x = px[s_].dropna()
                    if len(x) >= 2:
                        ch = (x.iloc[-1] / x.iloc[-2] - 1) * 100
                        up_ = ch >= 0
                        rows += ('<div class="kv"><span><span class="dot %s" style="width:9px;height:9px;margin-right:8px;display:inline-block;vertical-align:1px"></span>%s</span>'
                                 '<b style="color:%s">%s &nbsp; %s</b></div>') % ("good" if up_ else "bad", s_, GREEN if up_ else RED, money(x.iloc[-1]), pct(ch, 2))
        except Exception:
            pass
        st.markdown('<div class="card stack"><h4>%s Watchlist</h4>%s<div class="help">Saved for this visit only.</div></div>' % (hi("target"), rows or "No prices available."), unsafe_allow_html=True)


if __name__ == "__main__":
    main()
