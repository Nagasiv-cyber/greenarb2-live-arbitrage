"""app.py — GreenArb 2.0: Autonomous ADR Arbitrage Engine"""

import time
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime

from live_feed import fetch_baseline_prices, simulate_brownian_tick, compute_radar
from gate import evaluate_trade

EAA_THRESHOLD = 50.0
MAX_HIST      = 200

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="GreenArb 2.0", page_icon="⚡", layout="wide",
                   initial_sidebar_state="collapsed")

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;600;700&display=swap');
html,body,[data-testid="stAppViewContainer"],[data-testid="stApp"]{background:#0b0f12!important;color:#c8d6e5!important;font-family:'Rajdhani',sans-serif!important;}
#MainMenu,header,footer,[data-testid="stToolbar"],[data-testid="stDecoration"]{display:none!important;}
.block-container{padding:1rem 2rem 5rem 2rem!important;}
.header-bar{display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid #00ff9d33;padding-bottom:.5rem;margin-bottom:1rem;}
.header-title{font-family:'Share Tech Mono',monospace;font-size:1.4rem;color:#00ff9d;letter-spacing:2px;text-transform:uppercase;}
.header-status{font-family:'Share Tech Mono',monospace;font-size:.75rem;color:#00ff9d;animation:pulse 2s ease-in-out infinite;text-align:right;}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}
.metric-card{background:linear-gradient(145deg,#111820,#0d1520);border:1px solid #00ff9d22;border-radius:6px;padding:.9rem 1.1rem;position:relative;overflow:hidden;}
.metric-card::before{content:'';position:absolute;top:0;left:0;width:3px;height:100%;background:#00ff9d;}
.metric-label{font-family:'Share Tech Mono',monospace;font-size:.68rem;color:#5a7a6a;letter-spacing:2px;text-transform:uppercase;margin-bottom:.25rem;}
.metric-value{font-family:'Share Tech Mono',monospace;font-size:1.8rem;color:#00ff9d;font-weight:700;line-height:1.1;}
.metric-value.neg{color:#ff4b4b;}
.metric-sub{font-size:.68rem;color:#3d5a50;margin-top:.15rem;font-family:'Share Tech Mono',monospace;}
.gate-ok{background:linear-gradient(135deg,#001a0d,#002a14);border:1px solid #00ff9d;border-left:4px solid #00ff9d;border-radius:6px;padding:1rem 1.3rem;box-shadow:0 0 24px #00ff9d22;}
.gate-ko{background:linear-gradient(135deg,#1a0a00,#2a1200);border:1px solid #ff8c00;border-left:4px solid #ff4b4b;border-radius:6px;padding:1rem 1.3rem;box-shadow:0 0 24px #ff4b4b22;}
.gt{font-family:'Share Tech Mono',monospace;font-size:.95rem;letter-spacing:2px;font-weight:700;margin-bottom:.4rem;}
.gate-ok .gt{color:#00ff9d;}.gate-ko .gt{color:#ff4b4b;}
.gb{font-family:'Share Tech Mono',monospace;font-size:.75rem;color:#8aadaa;line-height:1.6;}
.gate-ko .gb{color:#c0856a;}
.hw{display:flex;gap:.8rem;margin-top:.6rem;flex-wrap:wrap;}
.hb{background:#0d1520;border:1px solid #1e3a2a;border-radius:4px;padding:.25rem .7rem;font-family:'Share Tech Mono',monospace;font-size:.65rem;color:#4a8a6a;letter-spacing:1px;}
.slbl{font-family:'Share Tech Mono',monospace;font-size:.65rem;color:#2a4a3a;letter-spacing:3px;text-transform:uppercase;border-bottom:1px solid #1a2a22;padding-bottom:.3rem;margin:1.2rem 0 .7rem;}
.tgt{background:linear-gradient(135deg,#001a0d,#00300f);border:2px solid #00ff9d;border-radius:8px;padding:1.2rem 2rem;text-align:center;box-shadow:0 0 50px #00ff9d33;margin-bottom:.8rem;animation:bglow 2.5s ease-in-out infinite;}
@keyframes bglow{0%,100%{box-shadow:0 0 30px #00ff9d33}50%{box-shadow:0 0 70px #00ff9d77}}
.tgt-lbl{font-family:'Share Tech Mono',monospace;font-size:.72rem;letter-spacing:4px;color:#4a8a6a;text-transform:uppercase;margin-bottom:.3rem;}
.tgt-name{font-family:'Share Tech Mono',monospace;font-size:2.6rem;font-weight:900;color:#00ff9d;letter-spacing:6px;text-shadow:0 0 25px #00ff9d88;line-height:1.1;}
.tgt-sub{font-family:'Share Tech Mono',monospace;font-size:.78rem;color:#4a8a6a;letter-spacing:2px;margin-top:.3rem;}
.fixed-footer{position:fixed;bottom:0;left:0;right:0;background:#07090b;border-top:1px solid #00ff9d22;padding:.4rem 2rem;font-family:'Share Tech Mono',monospace;font-size:.65rem;color:#2a4a3a;letter-spacing:2px;display:flex;justify-content:space-between;align-items:center;z-index:999;}
[data-testid="stExpander"]{background:#0d1318!important;border:1px solid #1a3328!important;border-radius:6px!important;}
</style>""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
for k, v in [("audit_log",[]),("total_ccu",0.0),("scan_count",0),
             ("last_data",None),("radar_df",None),
             ("scan_history",pd.DataFrame()),
             ("power_history",pd.DataFrame()),
             ("baseline_df",None),("usd_inr_rate",83.5)]:
    if k not in st.session_state:
        st.session_state[k] = v

# ── One-time baseline ─────────────────────────────────────────────────────────
if st.session_state.baseline_df is None:
    with st.spinner("⚡ Initializing Nifty 50 baseline prices… (once)"):
        _bl, _fx = fetch_baseline_prices()
        st.session_state.baseline_df  = _bl
        st.session_state.usd_inr_rate = _fx

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-bar">
  <div class="header-title">⚡ GreenArb 2.0 — Nifty 50 ADR Arbitrage</div>
  <div class="header-status">L1 FEED: Brownian Sim | KERNEL: GREENARB-V2.0.5-LIVE</div>
</div>""", unsafe_allow_html=True)

# ── Controls ──────────────────────────────────────────────────────────────────
ctrl, ts_col = st.columns([2, 5])
with ctrl:
    auto_pilot = st.toggle("🤖 ENGAGE AUTONOMOUS AGENT (Live Stream)", value=False)
    if not auto_pilot:
        manual_tick = st.button("🔄 MANUAL TICK", key="mtick")
    else:
        manual_tick = False

# ── Data execution ────────────────────────────────────────────────────────────
run_now = auto_pilot or manual_tick or (st.session_state.last_data is None)

if run_now:
    tick_df  = simulate_brownian_tick(st.session_state.baseline_df)
    radar_df = compute_radar(tick_df)

    infy = tick_df[tick_df["Ticker"] == "INFY"]
    if not infy.empty:
        r = infy.iloc[0]
        nse_price, nyse_inr, gross_spread = r["NSE ₹"], r["ADR ₹ (conv)"], round(r["ADR ₹ (conv)"] - r["NSE ₹"], 2)
    else:
        nse_price, nyse_inr, gross_spread = 1425.0, 1428.0, 3.0

    fx_rate     = st.session_state.usd_inr_rate
    gate_result = evaluate_trade(gross_spread=gross_spread, nse_price=nse_price, trade_qty=1000)
    ts          = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    top         = radar_df.iloc[0]
    wattage     = 1000 if top["Net EAA ₹"] > EAA_THRESHOLD else 150

    st.session_state.scan_count += 1
    st.session_state.total_ccu  += gate_result["ccu_minted"]
    st.session_state.radar_df    = radar_df
    st.session_state.last_data   = dict(
        nse_price=nse_price, nyse_price=nyse_inr, fx_rate=fx_rate,
        gross_spread=gross_spread, data_status="SIMULATED",
        gate=gate_result, timestamp=ts,
    )
    st.session_state.audit_log.append(dict(
        Timestamp=ts, Scan=st.session_state.scan_count,
        NSE=f"₹{nse_price:,.2f}", NYSE=f"₹{nyse_inr:,.2f}",
        Spread=f"₹{gross_spread:+,.2f}", EAA=f"₹{gate_result['eaa']:+,.2f}",
        Decision=gate_result["status"], Power=gate_result["power_mode"],
        CCU=f"{gate_result['ccu_minted']:.2f}kg",
    ))
    # power_history (per-scan)
    ph_row = pd.DataFrame([{"Timestamp": ts, "Net_EAA": top["Net EAA ₹"],
                             "Server_Wattage": wattage, "Top_Ticker": top["Ticker"]}])
    st.session_state.power_history = pd.concat(
        [st.session_state.power_history, ph_row], ignore_index=True).tail(MAX_HIST)
    # scan_history (per-ticker) — enriched with Wattage + CCU for leaderboard
    sh_rows = pd.DataFrame({
        "Timestamp":  ts,
        "Ticker":     radar_df["Ticker"].values,
        "Name":       radar_df["Name"].values,
        "Agent_ID":   radar_df["Agent_ID"].values if "Agent_ID" in radar_df.columns else "",
        "Net EAA ₹":  radar_df["Net EAA ₹"].values,
        "Wattage":    radar_df["Power Mode"].apply(lambda x: 1000 if "Peak" in x else 150).values,
        "CCU (kg)":   radar_df["CCU (kg)"].values,
    })
    st.session_state.scan_history = pd.concat(
        [st.session_state.scan_history, sh_rows], ignore_index=True).tail(MAX_HIST * 50)

# ── Timestamp display ─────────────────────────────────────────────────────────
with ts_col:
    if st.session_state.last_data:
        d = st.session_state.last_data
        st.markdown(
            f"<span style='font-family:Share Tech Mono,monospace;font-size:.68rem;color:#2a4a3a;letter-spacing:1px;'>"
            f"LAST TICK: {d['timestamp']} | SCANS: {st.session_state.scan_count} | "
            f"CCU: {st.session_state.total_ccu:.2f} kg CO₂</span>",
            unsafe_allow_html=True)

# ── Render ────────────────────────────────────────────────────────────────────
d  = st.session_state.last_data
df = st.session_state.radar_df

if d:
    gate   = d["gate"]
    spread = d["gross_spread"]
    eaa    = gate["eaa"]
    sc     = "neg" if spread < 0 else ""
    ec     = "neg" if eaa    < 0 else ""

    # Metric cards
    cards = st.empty()
    with cards.container():
        st.markdown('<div class="slbl">▸ INFY Snapshot — Lead Contract</div>', unsafe_allow_html=True)
        c1,c2,c3,c4 = st.columns(4)
        c1.markdown(f'<div class="metric-card"><div class="metric-label">INFY · NSE</div><div class="metric-value">₹{d["nse_price"]:,.2f}</div><div class="metric-sub">NSE Mumbai · INR</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="metric-card"><div class="metric-label">INFY · NYSE ADR</div><div class="metric-value">₹{d["nyse_price"]:,.2f}</div><div class="metric-sub">FX: {d["fx_rate"]:.2f} USD/INR</div></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="metric-card"><div class="metric-label">Gross Spread</div><div class="metric-value {sc}">₹{spread:+,.2f}</div><div class="metric-sub">NYSE − NSE</div></div>', unsafe_allow_html=True)
        c4.markdown(f'<div class="metric-card"><div class="metric-label">Net EAA</div><div class="metric-value {ec}">₹{eaa:+,.2f}</div><div class="metric-sub">After fees · 1 000 sh</div></div>', unsafe_allow_html=True)

    # Gate — Decision Attribution Matrix
    gate_slot = st.empty()
    with gate_slot.container():
        top_r = df.iloc[0] if df is not None and not df.empty else None
        agt   = top_r["Agent_ID"] if top_r is not None and "Agent_ID" in top_r else "AGT-??"

        st.markdown(f'<div class="slbl">▸ Agent Decision Matrix — {agt} | AMD MI325X Hardware Gate</div>', unsafe_allow_html=True)

        # ── 4-column math decomposition ──────────────────────────────────────
        dm1, dm2, dm3, dm4 = st.columns(4)
        gross_p = gate["gross_profit"]
        fees    = gate["total_fees"]
        compute = 0.50   # ENERGY_COST_INR constant

        dm1.markdown(f"""
        <div class="metric-card" style="border-left-color:#c8d6e5">
          <div class="metric-label">① Gross Spread</div>
          <div class="metric-value" style="font-size:1.4rem;color:#c8d6e5">₹{gross_p:+,.2f}</div>
          <div class="metric-sub">NYSE ADR − NSE Spot × 1000 sh</div>
        </div>""", unsafe_allow_html=True)

        dm2.markdown(f"""
        <div class="metric-card" style="border-left-color:#ff8c00">
          <div class="metric-label">② − Notional Fees</div>
          <div class="metric-value" style="font-size:1.4rem;color:#ff8c00">−₹{fees:,.2f}</div>
          <div class="metric-sub">Broker 0.05% × NSE price × qty</div>
        </div>""", unsafe_allow_html=True)

        dm3.markdown(f"""
        <div class="metric-card" style="border-left-color:#ff4b4b">
          <div class="metric-label">③ − Compute Cost</div>
          <div class="metric-value" style="font-size:1.4rem;color:#ff4b4b">−₹{compute:.2f}</div>
          <div class="metric-sub">MI325X energy overhead / tick</div>
        </div>""", unsafe_allow_html=True)

        eaa_col = "#00ff9d" if eaa > EAA_THRESHOLD else "#ff4b4b"
        dm4.markdown(f"""
        <div class="metric-card" style="border-left-color:{eaa_col}">
          <div class="metric-label">④ = Net EAA</div>
          <div class="metric-value" style="font-size:1.4rem;color:{eaa_col}">₹{eaa:+,.2f}</div>
          <div class="metric-sub">Efficiency-Adjusted Alpha</div>
        </div>""", unsafe_allow_html=True)

        # ── Power state justification ─────────────────────────────────────────
        if gate["status"] == "EXECUTE":
            st.markdown(f"""
            <div class="gate-ok" style="margin-top:.6rem">
              <div class="gt">[EXECUTE] — Net EAA ₹{eaa:,.2f} &gt; Threshold ₹{EAA_THRESHOLD:.0f}</div>
              <div class="gb">
                {agt} has authorized <b>AMD MI325X → 1000W Peak Power</b>.<br>
                Route: DMA pipeline → NSE Colocation → Order submitted at &lt;5µs latency.<br>
                Gross Profit: ₹{gross_p:,.2f} | Fees: ₹{fees:,.2f} | Compute: ₹{compute:.2f}
              </div>
              <div class="hw"><span class="hb">▶ FPGA ARMED</span><span class="hb">▶ &lt;5µs</span><span class="hb">▶ 1000W PEAK</span><span class="hb">▶ DMA→NSE</span></div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="gate-ko" style="margin-top:.6rem">
              <div class="gt">[UNDERCLOCK] — Net EAA ₹{eaa:,.2f} &lt; Threshold ₹{EAA_THRESHOLD:.0f}</div>
              <div class="gb">
                {agt} denied execution. Throttling hardware to <b>150W Eco-Mode</b>.<br>
                Spread does not cover fees + compute. Minting <b style="color:#ffaa44">{gate['ccu_minted']:.2f} kg CO₂ CCU</b> from idle capacity.
              </div>
              <div class="hw"><span class="hb">⬇ 150W ECO</span><span class="hb">♻ CCU MINT</span><span class="hb">⏸ EXECUTION DENIED</span></div>
            </div>""", unsafe_allow_html=True)

    # Agent radar section
    st.markdown('<div class="slbl">▸ Agent Radar — Nifty 50 ADR Sweep</div>', unsafe_allow_html=True)

    if df is not None and not df.empty:
        top = df.iloc[0]

        # Target banner — with Agent ID
        banner = st.empty()
        with banner.container():
            agt_id  = top["Agent_ID"] if "Agent_ID" in top.index else "AGT-??"
            st.markdown(f"""
            <div class="tgt">
              <div class="tgt-lbl">▸ ACTIVE THREAD: {agt_id} tracking {top["Ticker"]}</div>
              <div class="tgt-name">🎯 {top["Ticker"]} — {top["Name"]}</div>
              <div class="tgt-sub">
                {agt_id} | EAA: ₹{top["Net EAA ₹"]:+,.2f} | SPREAD: ₹{top["Spread ₹"]:+,.2f} | {top["Decision"]}
              </div>
            </div>""", unsafe_allow_html=True)

        tab1, tab2, tab3, tab4 = st.tabs([
            "🎯  Agent Radar Matrix",
            "📈  Live Telemetry & Power",
            "🏆  Swarm Analytics Leaderboard",
            "🌐  Geospatial Arc Radar",
        ])

        # ── TAB 1: Treemap ────────────────────────────────────────────────────
        with tab1:
            mx = max(df["Net EAA ₹"].abs().max(), 1.0)
            ft = px.treemap(df,
                path=[px.Constant("🟢 Nifty 50"), "Sector", "Ticker"],
                values="NSE ₹", color="Net EAA ₹",
                hover_data={"Name":True,"Spread ₹":":.2f","Decision":True,"Power Mode":True,"Status":True,"NSE ₹":False},
                color_continuous_scale=[[0,"#ff4b4b"],[.44,"#2a0a0a"],[.5,"#0d1318"],[.56,"#001a0d"],[1,"#00ff9d"]],
                range_color=[-mx, mx], color_continuous_midpoint=0,
                title=f"Nifty 50 ADR Heatmap — Scan #{st.session_state.scan_count}",
            )
            ft.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Share Tech Mono",color="#c8d6e5",size=11),
                title_font=dict(color="#00ff9d",size=12),
                coloraxis_colorbar=dict(title="Net EAA",tickprefix="₹",tickfont=dict(color="#c8d6e5",size=9),title_font=dict(color="#00ff9d"),bgcolor="rgba(13,19,24,0.9)",bordercolor="#1e3a2a",borderwidth=1),
                margin=dict(l=0,r=0,t=36,b=0))
            ft.update_traces(texttemplate="<b>%{label}</b><br>₹%{color:+,.0f}", textfont_size=11)
            st.plotly_chart(ft, use_container_width=True, config={"displayModeBar":False})
            ex = (df["Decision"].str.contains("EXECUTE")).sum()
            st.markdown(f"<div style='font-family:Share Tech Mono,monospace;font-size:.65rem;color:#4a6a5a;letter-spacing:2px;'>▸ EXECUTE: <span style='color:#00ff9d'>{ex}</span> | UNDERCLOCKING: <span style='color:#ff8c00'>{len(df)-ex}</span> | TARGET: <span style='color:#00ff9d'>{df.iloc[0]['Ticker']}</span> (₹{df.iloc[0]['Net EAA ₹']:+,.2f})</div>", unsafe_allow_html=True)

        # ── TAB 2: Dual-Axis Telemetry ────────────────────────────────────────
        with tab2:
            ph = st.session_state.power_history
            if ph.empty:
                st.markdown("<div style='font-family:Share Tech Mono,monospace;font-size:.8rem;color:#2a4a3a;padding:2rem;text-align:center;letter-spacing:2px;'>⌛ AWAITING DATA — engage auto-pilot or click MANUAL TICK</div>", unsafe_allow_html=True)
            else:
                fig = make_subplots(
                    rows=2, cols=1, shared_xaxes=True,
                    subplot_titles=("Net EAA (₹) — Top Opportunity Per Tick",
                                    "Server Wattage (W) — Power Consumption"),
                    vertical_spacing=0.10, row_heights=[0.6, 0.4],
                )

                # Top: EAA line
                fig.add_trace(go.Scatter(
                    x=ph["Timestamp"], y=ph["Net_EAA"],
                    mode="lines+markers", name="Net EAA",
                    line=dict(color="#00ff9d", width=2),
                    marker=dict(size=5, color="#00ff9d"),
                    fill="tozeroy", fillcolor="rgba(0,255,157,0.06)",
                    hovertemplate="<b>EAA: ₹%{y:,.2f}</b><extra></extra>",
                ), row=1, col=1)

                # Threshold line (manual, avoids add_hline row/col issues)
                if len(ph) >= 2:
                    fig.add_trace(go.Scatter(
                        x=[ph["Timestamp"].iloc[0], ph["Timestamp"].iloc[-1]],
                        y=[EAA_THRESHOLD, EAA_THRESHOLD],
                        mode="lines", name="Execute Threshold",
                        line=dict(color="#ff4b4b", width=1, dash="dash"),
                        showlegend=True,
                        hoverinfo="skip",
                    ), row=1, col=1)

                # Bottom: eco-zone fill (0 → 150W, always green)
                fig.add_trace(go.Scatter(
                    x=ph["Timestamp"], y=[150]*len(ph),
                    mode="lines", line=dict(color="rgba(0,0,0,0)", width=0),
                    fill="tozeroy", fillcolor="rgba(0,255,157,0.09)",
                    showlegend=False, hoverinfo="skip", name="Eco Zone",
                ), row=2, col=1)

                # Wattage line + red fill above eco zone
                fig.add_trace(go.Scatter(
                    x=ph["Timestamp"], y=ph["Server_Wattage"],
                    mode="lines+markers", name="Server Wattage",
                    line=dict(color="#ff4b4b", width=2),
                    marker=dict(size=5, color=["#00ff9d" if w==150 else "#ff4b4b" for w in ph["Server_Wattage"]]),
                    fill="tonexty", fillcolor="rgba(255,75,75,0.12)",
                    hovertemplate="<b>%{y}W</b><extra></extra>",
                ), row=2, col=1)

                _layout = dict(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(13,19,24,0.8)",
                    font=dict(family="Share Tech Mono, monospace", color="#c8d6e5", size=10),
                    legend=dict(bgcolor="rgba(13,19,24,0.9)", bordercolor="#1e3a2a",
                                borderwidth=1, font=dict(color="#c8d6e5", size=9)),
                    hovermode="x unified",
                    hoverlabel=dict(bgcolor="#0d1318", bordercolor="#00ff9d",
                                    font=dict(color="#c8d6e5", size=10, family="Share Tech Mono")),
                    margin=dict(l=60, r=20, t=40, b=40),
                )
                fig.update_layout(**_layout)
                fig.update_xaxes(gridcolor="#1a2a22", tickfont=dict(color="#4a6a5a", size=9), showgrid=True)
                fig.update_yaxes(gridcolor="#1a2a22", tickfont=dict(color="#4a6a5a", size=9),
                                 zerolinecolor="#1a2a22")
                fig.update_yaxes(tickprefix="₹", title_text="Net EAA (₹)",
                                 title_font=dict(color="#4a6a5a"), row=1, col=1)
                fig.update_yaxes(title_text="Wattage (W)", title_font=dict(color="#4a6a5a"),
                                 range=[0, 1100], row=2, col=1)
                # Subplot title styling
                for ann in fig.layout.annotations:
                    ann.update(font=dict(color="#00ff9d", size=11, family="Share Tech Mono"))

                st.plotly_chart(fig, use_container_width=True)

        # ── TAB 3: Swarm Analytics Leaderboard ───────────────────────────────
        with tab3:
            sh = st.session_state.scan_history
            if sh.empty or "Wattage" not in sh.columns:
                st.markdown("<div style='font-family:Share Tech Mono,monospace;font-size:.8rem;color:#2a4a3a;padding:2rem;text-align:center;letter-spacing:2px;'>⌛ AWAITING DATA — run at least one scan</div>", unsafe_allow_html=True)
            else:
                # ── Aggregate per ticker ────────────────────────────────────────────
                exec_sh   = sh[sh["Wattage"] == 1000]
                undcl_sh  = sh[sh["Wattage"] == 150]

                exec_agg  = exec_sh.groupby("Ticker").agg(
                    Trades_Executed=("Net EAA ₹", "count"),
                    Cumulative_EAA=("Net EAA ₹", "sum"),
                ).reset_index()
                ccu_agg   = undcl_sh.groupby("Ticker").agg(
                    Total_CCU_kg=("CCU (kg)", "sum"),
                ).reset_index()
                # Merge + fill tickers with no executes
                all_tickers = sh[["Ticker","Name","Agent_ID"]].drop_duplicates("Ticker")
                lb = all_tickers.merge(exec_agg,  on="Ticker", how="left") \
                                .merge(ccu_agg,   on="Ticker", how="left")
                lb["Trades_Executed"]  = lb["Trades_Executed"].fillna(0).astype(int)
                lb["Cumulative_EAA"]   = lb["Cumulative_EAA"].fillna(0).round(2)
                lb["Total_CCU_kg"]     = lb["Total_CCU_kg"].fillna(0).round(3)
                lb = lb.sort_values("Cumulative_EAA", ascending=False).reset_index(drop=True)
                lb.insert(0, "Rank", ["🥇" if i==0 else f"#{i+1}" for i in range(len(lb))])

                # ── TOP PERFORMER card ───────────────────────────────────────────
                best = lb.iloc[0]
                st.markdown(f"""
                <div style="background:linear-gradient(135deg,#001a0d,#003010);border:2px solid #00ff9d;
                  border-radius:8px;padding:1.2rem 2rem;text-align:center;
                  box-shadow:0 0 50px #00ff9d44;margin-bottom:1rem;">
                  <div style="font-family:Share Tech Mono,monospace;font-size:.7rem;color:#4a8a6a;letter-spacing:4px;margin-bottom:.3rem;">
                    🏆 TOP PERFORMER THIS SESSION</div>
                  <div style="font-family:Share Tech Mono,monospace;font-size:2.4rem;font-weight:900;
                    color:#00ff9d;letter-spacing:5px;text-shadow:0 0 20px #00ff9d88;">
                    {best['Agent_ID']} — {best['Ticker']}</div>
                  <div style="font-family:Share Tech Mono,monospace;font-size:.8rem;color:#4a8a6a;margin-top:.3rem;letter-spacing:2px;">
                    CUMULATIVE EAA: ₹{best['Cumulative_EAA']:+,.2f} &nbsp;|
                    TRADES: {best['Trades_Executed']} &nbsp;|
                    CCU MINTED: {best['Total_CCU_kg']:.3f} kg</div>
                </div>""", unsafe_allow_html=True)

                # ── Styled dataframe ───────────────────────────────────────────────
                display_lb = lb.rename(columns={
                    "Agent_ID":       "Agent",
                    "Name":           "Company",
                    "Trades_Executed":"Executions",
                    "Cumulative_EAA": "Cum. EAA (₹)",
                    "Total_CCU_kg":   "CCU Minted (kg)",
                })[[ "Rank","Agent","Ticker","Company","Executions","Cum. EAA (₹)","CCU Minted (kg)"]]

                def _eaa_color(val):
                    """Green if positive, red if negative — no matplotlib needed."""
                    if isinstance(val, str):
                        return ""
                    color = "#00ff9d" if val >= 0 else "#ff4b4b"
                    intensity = min(abs(val) / max(lb["Cumulative_EAA"].abs().max(), 1) * 0.4, 0.4)
                    return f"background-color:rgba({0 if val>=0 else 255},{255 if val>=0 else 75},{0 if val>=0 else 75},{intensity:.2f});color:{color};font-weight:bold;"

                def _ccu_color(val):
                    """Amber tint scaled by CCU amount — no matplotlib needed."""
                    if isinstance(val, str) or val == 0:
                        return ""
                    intensity = min(val / max(lb["Total_CCU_kg"].max(), 0.001) * 0.35, 0.35)
                    return f"background-color:rgba(255,170,68,{intensity:.2f});color:#ffaa44;"

                styled = display_lb.style \
                    .format({"Cum. EAA (₹)": "₹{:+,.2f}", "CCU Minted (kg)": "{:.3f}"}) \
                    .map(_eaa_color, subset=["Cum. EAA (₹)"]) \
                    .map(_ccu_color, subset=["CCU Minted (kg)"]) \
                    .set_properties(**{"font-family": "Share Tech Mono, monospace",
                                       "font-size": "0.75rem", "color": "#c8d6e5",
                                       "background-color": "#0d1318"})
                st.dataframe(styled, use_container_width=True, height=420)

                # ── Grouped bar chart ─────────────────────────────────────────────
                # Build long-form for grouped bars
                bar_df = lb[["Ticker","Agent_ID","Cumulative_EAA","Total_CCU_kg"]].copy()
                bar_long = pd.concat([
                    bar_df.assign(Metric="Cumulative EAA (₹)",  Value=bar_df["Cumulative_EAA"]),
                    bar_df.assign(Metric="CCU Minted (kg)",      Value=bar_df["Total_CCU_kg"] * 100),  # scaled for visibility
                ], ignore_index=True)

                fb = px.bar(
                    bar_long, x="Ticker", y="Value", color="Metric",
                    barmode="group",
                    text_auto=".0f",
                    color_discrete_map={
                        "Cumulative EAA (₹)":  "#00ff9d",
                        "CCU Minted (kg)":      "#ffaa44",
                    },
                    title="Agent Swarm: Cumulative EAA vs CCU Minted (CCU ×100 for scale)",
                    labels={"Value":"", "Ticker":"Agent Ticker"},
                )
                fb.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(13,19,24,0.8)",
                    font=dict(family="Share Tech Mono, monospace", color="#c8d6e5", size=10),
                    title_font=dict(color="#00ff9d", size=12),
                    legend=dict(bgcolor="rgba(13,19,24,0.9)", bordercolor="#1e3a2a",
                                borderwidth=1, font=dict(color="#c8d6e5", size=9)),
                    xaxis=dict(gridcolor="#1a2a22", tickfont=dict(color="#c8d6e5", size=9)),
                    yaxis=dict(gridcolor="#1a2a22", tickfont=dict(color="#4a6a5a", size=9),
                               zerolinecolor="#1a2a22"),
                    bargap=0.2, bargroupgap=0.05,
                    hoverlabel=dict(bgcolor="#0d1318", bordercolor="#00ff9d",
                                    font=dict(color="#c8d6e5", size=10, family="Share Tech Mono")),
                    margin=dict(l=40, r=20, t=40, b=40),
                )
                fb.update_traces(textfont=dict(color="#c8d6e5", size=9))
                st.plotly_chart(fb, use_container_width=True)

        # ── TAB 4: Geospatial Arc Radar ───────────────────────────────────────
        with tab4:
            ph = st.session_state.power_history

            # ── Helper: great-circle interpolation ──────────────────────────
            import numpy as np

            def _arc_lats_lons(lat1, lon1, lat2, lon2, n=60):
                """Return (lats, lons) for a great-circle arc between two points."""
                lat1r, lon1r = np.radians(lat1), np.radians(lon1)
                lat2r, lon2r = np.radians(lat2), np.radians(lon2)
                d = 2 * np.arcsin(np.sqrt(np.sin((lat2r-lat1r)/2)**2 +
                                           np.cos(lat1r)*np.cos(lat2r)*np.sin((lon2r-lon1r)/2)**2))
                t = np.linspace(0, 1, n)
                A = np.sin((1-t)*d) / np.sin(d)
                B = np.sin(t*d)    / np.sin(d)
                x = A*np.cos(lat1r)*np.cos(lon1r) + B*np.cos(lat2r)*np.cos(lon2r)
                y = A*np.cos(lat1r)*np.sin(lon1r) + B*np.cos(lat2r)*np.sin(lon2r)
                z = A*np.sin(lat1r)               + B*np.sin(lat2r)
                lats = np.degrees(np.arctan2(z, np.sqrt(x**2+y**2)))
                lons = np.degrees(np.arctan2(y, x))
                return lats.tolist(), lons.tolist()

            NYSE_LAT, NYSE_LON = 40.7128, -74.0060
            NSE_LAT,  NSE_LON  = 19.0760,  72.8777
            arc_lats, arc_lons = _arc_lats_lons(NYSE_LAT, NYSE_LON, NSE_LAT, NSE_LON)

            fig_globe = go.Figure()

            # ── Ocean / land base ────────────────────────────────────────────
            fig_globe.update_geos(
                projection_type    = "orthographic",
                projection_rotation= dict(lon=-10, lat=20, roll=0),
                showcoastlines     = True,  coastlinecolor  = "#1a3328",
                showland           = True,  landcolor       = "#0d1318",
                showocean          = True,  oceancolor      = "#060a0f",
                showlakes          = False,
                showcountries      = True,  countrycolor    = "#0f1f18",
                bgcolor            = "rgba(0,0,0,0)",
                showframe          = False,
            )

            # ── Execute arcs from history (most recent 12) ───────────────────
            ARC_COLORS = ["#00ff9d", "#00ccff", "#ffaa44", "#cc44ff"]
            execute_scans = ph[ph["Server_Wattage"] == 1000].tail(12) if not ph.empty else pd.DataFrame()
            n_arcs = len(execute_scans)

            for i, (_, row) in enumerate(execute_scans.iterrows()):
                age       = n_arcs - i          # newest = n_arcs, oldest = 1
                opacity   = max(0.15, age / n_arcs * 0.9)
                color     = ARC_COLORS[i % len(ARC_COLORS)]
                # slight lat jitter for fiber-optic bundle spread
                jitter    = np.random.uniform(-0.5, 0.5, len(arc_lats))
                j_lats    = [l + jitter[k] for k, l in enumerate(arc_lats)]

                fig_globe.add_trace(go.Scattergeo(
                    lat        = j_lats,
                    lon        = arc_lons,
                    mode       = "lines",
                    line       = dict(width=1.5, color=color),
                    opacity    = opacity,
                    showlegend = False,
                    hoverinfo  = "skip",
                ))

            # ── NYSE marker ──────────────────────────────────────────────────
            fig_globe.add_trace(go.Scattergeo(
                lat        = [NYSE_LAT],
                lon        = [NYSE_LON],
                mode       = "markers+text",
                marker     = dict(size=12, color="#00ff9d", symbol="circle",
                                  line=dict(width=2, color="#00ff9d")),
                text       = ["NYSE · NY"],
                textposition = "top right",
                textfont   = dict(family="Share Tech Mono", size=10, color="#00ff9d"),
                name       = "NYSE · New York",
                hovertemplate = "<b>NYSE</b><br>40.71°N 74.01°W<extra></extra>",
            ))

            # ── NSE marker ───────────────────────────────────────────────────
            fig_globe.add_trace(go.Scattergeo(
                lat        = [NSE_LAT],
                lon        = [NSE_LON],
                mode       = "markers+text",
                marker     = dict(size=12, color="#00ccff", symbol="circle",
                                  line=dict(width=2, color="#00ccff")),
                text       = ["NSE · Mumbai"],
                textposition = "bottom right",
                textfont   = dict(family="Share Tech Mono", size=10, color="#00ccff"),
                name       = "NSE · Mumbai",
                hovertemplate = "<b>NSE</b><br>19.08°N 72.88°E<extra></extra>",
            ))

            fig_globe.update_layout(
                paper_bgcolor = "rgba(0,0,0,0)",
                plot_bgcolor  = "rgba(0,0,0,0)",
                height        = 540,
                margin        = dict(l=0, r=0, t=36, b=0),
                title         = dict(
                    text      = f"⚡ GEOSPATIAL ARC RADAR — {n_arcs} SWARM ROUTES IN FLIGHT",
                    font      = dict(family="Share Tech Mono", color="#00ff9d", size=12),
                    x         = 0.01, y=0.97,
                ),
                legend        = dict(
                    bgcolor     = "rgba(11,15,18,0.85)",
                    bordercolor = "#1e3a2a", borderwidth=1,
                    font        = dict(family="Share Tech Mono", color="#c8d6e5", size=9),
                    x=0.01, y=0.08,
                ),
                hoverlabel    = dict(
                    bgcolor   = "#0d1318", bordercolor="#00ff9d",
                    font      = dict(family="Share Tech Mono", color="#c8d6e5", size=10),
                ),
            )

            st.plotly_chart(fig_globe, use_container_width=True, config={"displayModeBar": False})

            # ── Telemetry row below globe ─────────────────────────────────────
            tl1, tl2, tl3, tl4 = st.columns(4)
            status_text  = "ARBITRAGE LOCKED ✅" if n_arcs > 0 else "Scanning Order Books…"
            status_color = "#00ff9d" if n_arcs > 0 else "#2a4a3a"
            tl1.markdown(f"<div class='metric-card' style='border-left-color:#00ff9d'><div class='metric-label'>Swarm Routes In Flight</div><div class='metric-value' style='font-size:1.6rem'>{n_arcs}</div><div class='metric-sub'>NYSE ⟶ NSE arcs</div></div>", unsafe_allow_html=True)
            tl2.markdown(f"<div class='metric-card' style='border-left-color:#00ccff'><div class='metric-label'>Route</div><div class='metric-value' style='font-size:1rem;color:#00ccff'>NYSE ⟶ NSE</div><div class='metric-sub'>New York → Mumbai</div></div>", unsafe_allow_html=True)
            tl3.markdown(f"<div class='metric-card' style='border-left-color:#ffaa44'><div class='metric-label'>Sim Latency</div><div class='metric-value' style='font-size:1.6rem;color:#ffaa44'>42ms</div><div class='metric-sub'>Fiber optic route</div></div>", unsafe_allow_html=True)
            tl4.markdown(f"<div class='metric-card' style='border-left-color:{status_color}'><div class='metric-label'>Status</div><div class='metric-value' style='font-size:.9rem;color:{status_color}'>{status_text}</div><div class='metric-sub'>Real-time gate decision</div></div>", unsafe_allow_html=True)

    sc_col = "#00ff9d" if d["data_status"] == "SIMULATED" else "#ff8c00"
    st.markdown(f"<div style='font-family:Share Tech Mono,monospace;font-size:.63rem;color:{sc_col};letter-spacing:2px;margin-top:.4rem;'>▸ {d['data_status']} | BROWNIAN VOL: ±0.2% | LAST: {d['timestamp']}</div>", unsafe_allow_html=True)

    # Audit log
    st.markdown('<div class="slbl">▸ Immutable Power Ledger</div>', unsafe_allow_html=True)
    with st.expander("📒  Audit Log", expanded=False):
        if st.session_state.audit_log:
            rows = ""
            for r in reversed(st.session_state.audit_log[-50:]):
                c = "#00ff9d" if r["Decision"]=="EXECUTE" else "#ff8c00"
                rows += f"<tr><td style='color:#5a7a6a'>{r['Timestamp']}</td><td>{r['NSE']}</td><td>{r['NYSE']}</td><td>{r['Spread']}</td><td>{r['EAA']}</td><td style='color:{c}'>{r['Decision']}</td><td>{r['Power']}</td><td style='color:#ffaa44'>{r['CCU']}</td></tr>"
            st.markdown(f"<table style='width:100%;border-collapse:collapse;font-family:Share Tech Mono,monospace;font-size:.7rem;color:#7aaa9a;'><thead><tr style='color:#00ff9d;border-bottom:1px solid #1a3328;'><th>Time</th><th>NSE</th><th>NYSE</th><th>Spread</th><th>EAA</th><th>Decision</th><th>Power</th><th>CCU</th></tr></thead><tbody>{rows}</tbody></table>", unsafe_allow_html=True)

# ── Advanced Modules ──────────────────────────────────────────────────────────
st.markdown('<div class="slbl">▸ Advanced Intelligence Modules</div>', unsafe_allow_html=True)
adv1, adv2 = st.columns(2)

# ── NLP SENTINEL ──────────────────────────────────────────────────────────────
with adv1:
    NLP_HEADLINES = [
        ("Fed signals two additional rate cuts in H2 2025 — dovish pivot confirmed",    0.72),
        ("India NSE halts trading circuit breaker triggered by FII sell-off",           -0.85),
        ("Infosys Q4 beats estimates: EPS ₹22.4 vs ₹20.1 expected",                    0.68),
        ("SEC launches probe into cross-border ADR arbitrage practices",                -0.61),
        ("SEBI proposes T+0 settlement for Nifty 50 constituents by Q3",                0.45),
        ("Global credit tightening fears resurface as ECB holds rates",                 -0.38),
        ("USD/INR approaches 84.20 — RBI likely to intervene via OMO",                 -0.29),
        ("HDFC Bank ADR premium hits 6-month high on strong NIM guidance",              0.81),
        ("Crude oil spikes 4.2% on OPEC+ surprise output cut announcement",             -0.55),
        ("Wipro wins $1.2B AI infrastructure deal with US defense contractor",          0.77),
        ("China PMI contracts for third consecutive month — EM contagion risk",         -0.73),
        ("Tata Motors EV division IPO files S-1 with NYSE — $8B valuation",             0.88),
        ("Bank of Japan unexpectedly widens YCC band — yen carry unwind risk",          -0.44),
        ("CPI data hotter than expected: 3.8% vs 3.5% forecast — risk-off mode",       -0.79),
        ("Reliance Industries to spin off Jio Financial — ADR implications positive",   0.62),
    ]

    if "nlp_ptr" not in st.session_state:
        st.session_state.nlp_ptr  = 0
        st.session_state.nlp_feed = []

    # Advance pointer on each scan
    if run_now:
        ptr = st.session_state.nlp_ptr % len(NLP_HEADLINES)
        h, base = NLP_HEADLINES[ptr]
        s = round(base + np.random.uniform(-0.1, 0.1), 3)
        ts = datetime.now().strftime("%H:%M:%S")
        st.session_state.nlp_feed.append({"text": h, "s": s, "ts": ts})
        st.session_state.nlp_feed = st.session_state.nlp_feed[-10:]
        st.session_state.nlp_ptr += 1

    feed = st.session_state.nlp_feed
    rolling = round(sum(e["s"] for e in feed[-5:]) / max(len(feed[-5:]), 1), 3) if feed else 0
    gauge_pct = int(((rolling + 1) / 2) * 100)
    vol_color  = "#ff4b4b" if rolling < -0.3 else "#00ff9d" if rolling > 0 else "#ffaa44"
    vol_text   = "⚠ VOLATILITY DETECTED — Raising Execution Threshold" if rolling < -0.3 else \
                 "✅ MACRO STABLE — Normal EAA Gate Active" if rolling > 0 else "⏳ SENTIMENT NEUTRAL"

    st.markdown(f"""
    <div style="background:rgba(11,15,18,0.72);backdrop-filter:blur(14px);
      border:1px solid {vol_color}44;border-left:3px solid {vol_color};
      border-radius:8px;padding:12px 14px;font-family:Share Tech Mono,monospace;">
      <div style="font-size:10px;letter-spacing:3px;color:#00ff9d;margin-bottom:8px;">
        ⚡ NLP SENTINEL — Macro Sentiment Engine</div>
      <div style="background:{vol_color}18;border:1px solid {vol_color}55;border-radius:4px;
        padding:5px 10px;font-size:10px;font-weight:bold;color:{vol_color};
        letter-spacing:1px;margin-bottom:10px;">{vol_text}</div>
      <div style="display:flex;justify-content:space-between;font-size:9px;color:#4a6a5a;margin-bottom:4px;">
        <span>BEAR -1.0</span>
        <span style="color:{vol_color};font-weight:bold">ROLLING AVG: {'+' if rolling>=0 else ''}{rolling}</span>
        <span>BULL +1.0</span></div>
      <div style="background:#0d1318;border-radius:3px;height:8px;border:1px solid #1e3a2a;overflow:hidden;margin-bottom:10px;">
        <div style="width:{gauge_pct}%;height:100%;
          background:linear-gradient(90deg,#ff4b4b,{vol_color});
          border-radius:3px;transition:width .6s;"></div></div>
    </div>""", unsafe_allow_html=True)

    if feed:
        rows_html = ""
        for e in reversed(feed[-8:]):
            sc = "#00ff9d" if e["s"] > 0.2 else "#ff4b4b" if e["s"] < -0.2 else "#ffaa44"
            rows_html += f"<tr><td style='color:#2a4a3a;white-space:nowrap'>{e['ts']}</td><td style='color:#8aadaa;padding:2px 8px'>{e['text'][:68]}…</td><td style='color:{sc};font-weight:bold;white-space:nowrap'>{'+' if e['s']>=0 else ''}{e['s']}</td></tr>"
        st.markdown(f"<table style='width:100%;border-collapse:collapse;font-family:Share Tech Mono,monospace;font-size:.68rem;'><tbody>{rows_html}</tbody></table>", unsafe_allow_html=True)
    else:
        st.markdown("<div style='font-family:Share Tech Mono,monospace;font-size:.7rem;color:#2a4a3a;padding:1rem;text-align:center;'>⌛ AWAITING FIRST SCAN…</div>", unsafe_allow_html=True)

# ── DARK POOL CCU LEDGER ───────────────────────────────────────────────────────
with adv2:
    if "dp_revenue" not in st.session_state:
        st.session_state.dp_revenue  = 0.0
        st.session_state.dp_notif    = ""
        st.session_state.dp_prev_ccu = 0.0

    prev_ccu = st.session_state.dp_prev_ccu
    curr_ccu = st.session_state.total_ccu
    if curr_ccu > prev_ccu and run_now:
        diff = curr_ccu - prev_ccu
        price = round(22 + np.random.uniform(0, 4), 2)
        st.session_state.dp_revenue  += round(diff * price, 2)
        st.session_state.dp_notif     = f"EXECUTING DARK POOL LIQUIDATION: {diff:.4f} CCU → ${diff*price:.2f}"
    elif run_now:
        st.session_state.dp_notif = ""
    st.session_state.dp_prev_ccu = curr_ccu

    BUYERS  = ["BlackRock ESG", "Vanguard Carbon", "PIMCO Climate", "Citadel Green", "BofA Climate"]
    SELLERS = ["JPM Carbon Desk", "Goldman ESG", "Morgan Stanley", "Barclays CCU", "Bridgewater"]
    import random
    bids = sorted([(round(19+random.random()*5,2), random.randint(1,50), random.choice(BUYERS)) for _ in range(5)], reverse=True)
    asks = sorted([(round(22+random.random()*5,2), random.randint(1,50), random.choice(SELLERS)) for _ in range(5)])

    notif_html = f"""<div style="background:rgba(0,255,157,0.1);border:1px solid #00ff9d;border-radius:4px;
      padding:5px 10px;font-size:9px;color:#00ff9d;letter-spacing:1px;font-weight:bold;margin-bottom:8px;">
      ▶ {st.session_state.dp_notif}</div>""" if st.session_state.dp_notif else ""

    bid_rows = "".join(f"<tr><td style='color:#00ff9d;font-weight:bold'>${p}</td><td style='color:#4a8a6a'>{q}</td><td style='color:#2a4a3a'>{e[:16]}</td></tr>" for p,q,e in bids)
    ask_rows = "".join(f"<tr><td style='color:#ff4b4b;font-weight:bold'>${p}</td><td style='color:#8a4a4a'>{q}</td><td style='color:#4a2a2a'>{e[:16]}</td></tr>" for p,q,e in asks)

    st.markdown(f"""
    <div style="background:rgba(11,15,18,0.72);backdrop-filter:blur(14px);
      border:1px solid rgba(0,255,157,0.15);border-left:3px solid #00ff9d;
      border-radius:8px;padding:12px 14px;font-family:Share Tech Mono,monospace;">
      <div style="font-size:10px;letter-spacing:3px;color:#00ff9d;margin-bottom:8px;">
        ♻ DARK POOL LEDGER — CCU Secondary Market</div>
      <div style="background:linear-gradient(135deg,#001a0d,#002a14);border:1px solid #00ff9d33;
        border-radius:6px;padding:8px 12px;text-align:center;margin-bottom:8px;">
        <div style="font-size:8px;letter-spacing:3px;color:#4a8a6a;margin-bottom:2px;">SECONDARY ESG REVENUE YIELD</div>
        <div style="font-size:26px;font-weight:bold;color:#00ff9d;text-shadow:0 0 12px #00ff9d66;">
          ${st.session_state.dp_revenue:,.2f}</div>
        <div style="font-size:8px;color:#4a6a5a;margin-top:2px;">CCUs MINTED: {curr_ccu:.4f} kg · POOL: ACTIVE</div></div>
      {notif_html}
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;">
        <div>
          <div style="font-size:8px;color:#00ff9d;letter-spacing:2px;margin-bottom:4px;">▼ BIDS (BUY)</div>
          <table style="width:100%;border-collapse:collapse;font-size:8px;">
            <tr style="color:#4a6a5a;font-size:7px;"><th>PRICE</th><th>QTY</th><th>ENTITY</th></tr>
            {bid_rows}</table></div>
        <div>
          <div style="font-size:8px;color:#ff4b4b;letter-spacing:2px;margin-bottom:4px;">▲ ASKS (SELL)</div>
          <table style="width:100%;border-collapse:collapse;font-size:8px;">
            <tr style="color:#4a6a5a;font-size:7px;"><th>PRICE</th><th>QTY</th><th>ENTITY</th></tr>
            {ask_rows}</table></div></div></div>
    """, unsafe_allow_html=True)

# ── THERMAL MATRIX ─────────────────────────────────────────────────────────────
st.markdown('<div class="slbl">▸ Thermal Matrix — AMD MI325X Bare-Metal Array</div>', unsafe_allow_html=True)

if "node_temps" not in st.session_state:
    st.session_state.node_temps = [45.0] * 8

active = st.session_state.power_history.get("Server_Wattage", pd.Series()).eq(1000).sum() \
    if not st.session_state.power_history.empty and "Server_Wattage" in st.session_state.power_history.columns else 0

if run_now:
    new_temps = []
    for t in st.session_state.node_temps:
        if active > 0:
            new_temps.append(min(85.0, round(t + np.random.uniform(1.0, 3.0), 1)))
        else:
            new_temps.append(max(45.0, round(t - np.random.uniform(0.5, 1.5), 1)))
    st.session_state.node_temps = new_temps

temps = st.session_state.node_temps
max_temp = max(temps)
avg_temp = round(sum(temps)/8, 1)
is_critical = max_temp >= 80
master_color = "#ff4b4b" if is_critical else "#4488ff"
master_text  = "🔴 THERMAL DEGRADATION WARNING" if is_critical else "🔵 THERMAL STABLE"

NODE_LABELS = ["MI325X-01","MI325X-02","MI325X-03","MI325X-04","MI325X-05","MI325X-06","MI325X-07","MI325X-08"]
NODE_ROLES  = ["EAA GATE","BROWNIAN","SPREAD","FX ENGINE","CCU MINT","LEDGER","PREDICTOR","WATCHDOG"]

def _temp_color(t):
    if t <= 50:  return "#003080"
    if t <= 60:  return "#5500bb"
    if t <= 70:  return "#8b0050"
    if t <= 78:  return "#c82800"
    return "#ff4b4b"

def _temp_glow(t):
    if t <= 50:  return "0 0 12px rgba(0,80,255,0.5)"
    if t <= 60:  return "0 0 14px rgba(120,0,255,0.5)"
    if t <= 70:  return "0 0 16px rgba(255,80,0,0.5)"
    return "0 0 22px rgba(255,75,75,0.7)"

node_cells = ""
for i, t in enumerate(temps):
    bg   = _temp_color(t)
    glow = _temp_glow(t)
    pct  = int(((t-45)/(85-45))*100)
    tc   = "#4488ff" if t<=50 else "#aa44ff" if t<=60 else "#ff8800" if t<=70 else "#ff4b4b"
    node_cells += f"""
    <div style="background:{bg};border:1px solid {tc}55;border-radius:6px;padding:10px 8px;
      text-align:center;box-shadow:{glow};transition:background .4s;">
      <div style="font-size:8px;color:#c8d6e5;letter-spacing:1px;margin-bottom:4px;">{NODE_LABELS[i]}</div>
      <div style="font-size:18px;font-weight:bold;color:{tc};line-height:1.1">{t}°</div>
      <div style="font-size:7px;color:{tc}aa;margin-top:2px">{NODE_ROLES[i]}</div>
      <div style="margin-top:5px;background:rgba(0,0,0,0.4);border-radius:2px;height:3px;overflow:hidden">
        <div style="width:{pct}%;height:100%;background:{tc};transition:width .4s"></div></div>
      <div style="font-size:7px;color:{tc}88;margin-top:2px">{pct}%</div></div>"""

st.markdown(f"""
<div style="background:rgba(11,15,18,0.72);backdrop-filter:blur(14px);
  border:1px solid {master_color}33;border-left:3px solid {master_color};
  border-radius:8px;padding:14px 16px;font-family:Share Tech Mono,monospace;">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
    <div style="font-size:10px;letter-spacing:2px;color:{master_color};font-weight:bold;">{master_text}</div>
    <div style="font-size:9px;color:#4a6a5a">AVG: {avg_temp}°C · PEAK: {max_temp}°C · ACTIVE SCANS: {active}</div></div>
  <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:8px">{node_cells}</div>
  <div style="margin-top:8px;display:flex;justify-content:space-between;font-size:8px;color:#4a6a5a;
    border-top:1px solid #1a2a22;padding-top:6px;letter-spacing:1px">
    <span>ACTIVE SCANS: <span style="color:#00ff9d">{active}</span></span>
    <span>ESTIMATED DRAW: <span style="color:{master_color}">{'~8×1000W' if is_critical else '~8×150W'}</span></span>
    <span>MODE: <span style="color:{master_color}">{'PEAK' if active>0 else 'ECO'}</span></span></div></div>
""", unsafe_allow_html=True)

# Footer
st.markdown(f"""<div class="fixed-footer"><span>NODE: REC-AIML-SEC-D | ID: 251501249 | OPERATOR: K. VISAGAN | KERNEL: GREENARB-V2.0.5</span><span style='color:#1a3a2a'>SCANS: {st.session_state.scan_count} | CCU: {st.session_state.total_ccu:.2f}kg | AUTO: {"ON ●" if "auto_pilot" in dir() and auto_pilot else "OFF ○"}</span></div>""", unsafe_allow_html=True)

# ── AUTO-PILOT LOOP ───────────────────────────────────────────────────────────
if auto_pilot:
    time.sleep(1.5)
    st.rerun()
