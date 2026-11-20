import math
import pandas as pd
import streamlit as st
from datetime import timedelta
from data import PRODUCTS, LOCATIONS
from orchestrator import Sim, Orchestrator, Params

st.set_page_config(page_title="Autonomous Supply Chain Agent", page_icon="📦", layout="wide")
orch = Orchestrator()
ICON = {"HIGH": "🔴", "MEDIUM": "🟠", "LOW": "🟢", "OVERSTOCK": "🔵"}


def new_sim(seed, params):
    st.session_state.sim = Sim(int(seed))
    st.session_state.result = orch.run_cycle(st.session_state.sim, params, execute=False)


# ---------------- sidebar
with st.sidebar:
    st.title("📦 Supply Chain Agent")
    st.caption("Detect → Decide → Order → Track → Adjust")
    seed = st.number_input("Random seed", 0, 9999, 7)
    params = Params(
        service_level=st.select_slider("Target service level", [0.90, 0.95, 0.98, 0.99], 0.95),
        review_days=st.slider("Review / cover period (days)", 3, 21, 7),
        auto=st.toggle("Auto-procurement (agent places POs)", True),
        profit_aware=st.toggle("Profit-aware ordering (bulk discounts)", True))
    if "sim" not in st.session_state:
        new_sim(seed, params)
    sim = st.session_state.sim

    def advance(n):
        for _ in range(n):
            sim.advance_day()
            st.session_state.result = orch.run_cycle(sim, params)

    c1, c2 = st.columns(2)
    if c1.button("▶ Run cycle"):
        st.session_state.result = orch.run_cycle(sim, params)
    if c2.button("⏭ +1 day"):
        advance(1)
    if st.button("⏩ +7 days"):
        advance(7)
    st.divider()
    st.caption("Chaos injectors")
    sp = st.selectbox("Product", list(PRODUCTS))
    if st.button("💥 Demand spike ×3 (next day)"):
        sim.inject_spike(sp, 3.0)
        st.toast(f"{sp}: spike queued - press +1 day")
    if st.button("❄️ Cold chain disruption (spoilage)"):
        sim.force_spoilage(sp)
        st.toast(f"{sp}: batches expired - press +1 day")
    if st.button("🚧 Delay in-transit orders +2d"):
        st.toast(f"{sim.delay_in_transit(2)} orders delayed")
    if st.button("🔄 Reset simulation"):
        new_sim(seed, params)
        st.rerun()
    st.divider()
    with st.expander("🤖 AI Co-Pilot Settings", expanded=False):
        gemini_key = st.text_input("Gemini API Key (Optional)", type="password", help="Leave empty to use built-in offline reasoning engine")
        if gemini_key:
            orch.copilot.api_key = gemini_key

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "👋 Hello! I am your **Autonomous AI Supply Chain Co-Pilot**. Ask me any question about today's inventory, stockout risks, vendor selections, perishability warnings, or dynamic pricing strategies!"}
    ]

res = st.session_state.result
dec, k = res["decisions"], res["kpis"]
st.header("Autonomous Supply Chain Intelligence & Execution System")
m = st.columns(6)
m[0].metric("Sim day", sim.day, sim.today.strftime("%d %b %Y"))
m[1].metric("Fill rate", f"{k['fill_rate']:.1%}")
m[2].metric("Revenue", f"₹{k['revenue']:,.0f}")
m[3].metric("Net Profit", f"₹{k['profit']:,.0f}")
m[4].metric("Procurement spend", f"₹{k['spend']:,.0f}")
m[5].metric("Spoilage waste", f"₹{k['spoilage_cost']:,.0f}", f"{k['spoilage_units']} units", delta_color="inverse")

tab_dash, tab_copilot, tab_cock, tab_scn, tab_ord, tab_log = st.tabs(
    ["📊 Dashboard", "💬 AI Co-Pilot", "🧠 Agent cockpit", "🧪 Scenario simulator", "🧾 Orders & vendors", "📜 Activity log"])

# ---------------- dashboard
with tab_dash:
    briefing = orch.copilot.generate_briefing(sim, res)
    st.info(f"📋 **Daily Executive Briefing (AI Co-Pilot)**\n\n{briefing}")
    rows = []
    for p, d in dec.items():
        r, ro, pr, per = d["risk"], d["reorder"], d["pricing"], d["perish"]
        price_tag = f"₹{pr['current_price']}"
        if pr["discount_pct"] > 0:
            price_tag += f" (-{pr['discount_pct']:.0%})"
        elif pr["surge_pct"] > 0:
            price_tag += f" (+{pr['surge_pct']:.0%})"
        
        per_icon = "🔴" if per["status"] == "HIGH" else "🟠" if per["status"] == "MEDIUM" else "🟢"

        rows.append({
            "Product": p, "Stock": d["inv"]["total"], "On order": d["inv"]["on_order"],
            "Price": price_tag,
            "Shelf Life": f"{per['shelf_life']}d",
            "Expiry Risk": f"{per_icon} {per['status']}",
            "Forecast/day": round(d["fc"]["mean7"]), "Days cover": round(min(r["cover_days"], 99), 1),
            "Stockout in": "∞" if math.isinf(r["days_to_stockout"]) else f"{r['days_to_stockout']:.1f}d",
            "Stock Risk": f"{ICON[r['level']]} {r['level']}", "Reorder?": "YES" if ro["trigger"] else "no",
            "Order qty": d["po"]["qty"] if d["po"] else 0, "Vendor": d["po"]["vendor"] if d["po"] else "-",
            "Status": f"PO #{d['placed']['id']} placed" if d.get("placed") else ("Draft - awaiting approval" if d["po"] else "-")
        })
    st.subheader("Risk, Perishability & Recommended Actions")
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    a, b = st.columns([3, 2])
    with a:
        st.subheader("Unified inventory graph")
        st.bar_chart(pd.DataFrame(sim.recorded).T[list(LOCATIONS)])
    with b:
        st.subheader("Anomaly alerts")
        if res["alerts"]:
            for al in res["alerts"]:
                st.warning(f"**{al['type']}** · {al['product']} - {al['message']}" + (" ✅ reconciled" if al["done"] else ""))
        else:
            st.success("No anomalies detected")
    st.subheader("Warehouse balancing")
    wrows = [{"Product": p, "Location": l, **s} for p, d in dec.items() for l, s in d["loc_status"].items() if s["status"] != "Balanced"]
    st.dataframe(pd.DataFrame(wrows), hide_index=True) if wrows else st.info("All locations balanced")

# ---------------- AI Co-Pilot
with tab_copilot:
    st.subheader("💬 Autonomous AI Supply Chain Co-Pilot")
    mode_tag = "🟢 Connected to Google Gemini 1.5 Flash" if orch.copilot.api_key else "⚡ Running Built-in Offline Reasoning Engine"
    st.caption(f"Status: **{mode_tag}** · Natural language interface explaining multi-agent decisions in real-time.")

    st.markdown("##### Quick Inquiries (Click to Ask)")
    qcol1, qcol2, qcol3, qcol4 = st.columns(4)
    suggested = None
    if qcol1.button("🏭 Why this vendor?"):
        suggested = "Why did we choose our recent vendor for purchase orders?"
    if qcol2.button("🥦 Spoilage risks?"):
        suggested = "What are our critical perishability and spoilage risks today?"
    if qcol3.button("🏷️ Dynamic discounts?"):
        suggested = "Explain our active dynamic pricing and markdown strategies."
    if qcol4.button("💰 Financial health?"):
        suggested = "Give me an executive summary of our revenue, profit, and spend."

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_query = st.chat_input("Ask the AI Co-Pilot anything about your supply chain...")
    final_query = suggested or user_query

    if final_query:
        st.session_state.messages.append({"role": "user", "content": final_query})
        with st.chat_message("user"):
            st.markdown(final_query)
        with st.spinner("AI Co-Pilot analyzing agent consensus..."):
            ans = orch.copilot.answer_question(final_query, sim, res)
        st.session_state.messages.append({"role": "assistant", "content": ans})
        with st.chat_message("assistant"):
            st.markdown(ans)
        if suggested:
            st.rerun()

# ---------------- cockpit
with tab_cock:
    p = st.selectbox("Inspect product", list(PRODUCTS), key="cock")
    d = dec[p]
    c = st.columns(6)
    c[0].metric("Stock Risk", f"{ICON[d['risk']['level']]} {d['risk']['level']}")
    c[1].metric("Stockout in", "∞" if math.isinf(d["risk"]["days_to_stockout"]) else f"{d['risk']['days_to_stockout']:.1f} d")
    c[2].metric("Dynamic Price", f"₹{d['pricing']['current_price']}", f"Base ₹{d['pricing']['base_price']}", delta_color="off")
    c[3].metric("Shelf Life", f"{d['perish']['shelf_life']} d")
    c[4].metric("Expiring ≤3d", f"{d['perish']['expiring_soon']} u")
    c[5].metric("Spoilage Risk", f"₹{d['perish']['exp_loss_est']:,.0f}", f"{d['perish']['at_risk_qty']} at-risk", delta_color="inverse")
    
    st.markdown("##### Agent reasoning trace (Multi-Agent Consensus)")
    for ag, msg in d["trace"]:
        st.markdown(f"**{ag}** - {msg}")

    # Perishability and Dynamic Pricing Detailed Cards
    col_perish, col_price = st.columns(2)
    with col_perish:
        st.markdown("##### 🥦 Perishability & FEFO Batch Rotation")
        st.info(d["perish"]["note"])
        if d["perish"]["batches"]:
            b_rows = []
            for b in d["perish"]["batches"]:
                b_status = "🔴 Expiring Soon (≤3d)" if b["days_left"] <= 3 else "🟠 Mid-life" if b["days_left"] <= 7 else "🟢 Fresh"
                b_rows.append({"Location": b["location"], "Qty": b["qty"], "Days Left": f"{b['days_left']}d", "Status": b_status})
            st.dataframe(pd.DataFrame(b_rows), hide_index=True)
        else:
            st.caption("No batch tracking required for non-perishable goods.")

    with col_price:
        st.markdown("##### 🏷️ Dynamic Pricing & Revenue Strategy")
        st.info(f"**Strategy**: {d['pricing']['reason']}")
        dp_metric1, dp_metric2 = st.columns(2)
        dp_metric1.metric("Discount / Surge", f"{'-' + str(int(d['pricing']['discount_pct']*100)) + '%' if d['pricing']['discount_pct'] > 0 else '+' + str(int(d['pricing']['surge_pct']*100)) + '%' if d['pricing']['surge_pct'] > 0 else '0%'}")
        dp_metric2.metric("Demand Multiplier", f"×{d['pricing']['demand_mult']}")
    h = sim.hist_df(p)["demand"].tail(30)
    fdates = [sim.today + timedelta(days=i + 1) for i in range(14)]
    chart = pd.DataFrame({"history": h, "forecast": pd.Series(d["fc"]["daily"][:14], index=fdates)})
    chart.index = pd.to_datetime(chart.index)
    st.markdown("##### Demand: history vs forecast")
    st.line_chart(chart)
    if d["ranking"]:
        st.markdown("##### Vendor intelligence (multi-criteria)")
        vr = pd.DataFrame(d["ranking"]["rows"])[["vendor", "price", "lead", "reliability", "bulk", "score", "misses_stockout_window"]]
        st.dataframe(vr, hide_index=True)
        st.markdown("##### Cost optimisation (expected total cost)")
        co = pd.DataFrame(d["cost"]["options"])[["vendor", "qty", "unit_price", "lead", "exp_short_units", "purchase", "holding", "exp_total"]].round(1)
        st.dataframe(co, hide_index=True)
        st.info(d["cost"]["note"])
        st.markdown("##### Purchase order")
        st.json(d["po"])
    if d["transfers"]:
        st.markdown("##### Warehouse transfers")
        st.dataframe(pd.DataFrame(d["transfers"]), hide_index=True)

# ---------------- scenario simulator
with tab_scn:
    st.markdown("**What if a vendor is late?** The agents re-run vendor selection and cost optimisation with the delay applied.")
    s1, s2, s3 = st.columns(3)
    sp_ = s1.selectbox("Product", list(PRODUCTS), key="scn_p")
    sv = s2.selectbox("Delayed vendor", ["Vendor A", "Vendor B", "Vendor C"], index=1)
    sd = s3.slider("Extra delay (days)", 0, 7, 2)
    base = orch.analyze_product(sim, sp_, params)
    alt = orch.analyze_product(sim, sp_, params, delays={sv: sd})
    if not base["po"]:
        st.info(f"No reorder is triggered for {sp_} right now - try a product flagged 🔴/🟠 (e.g. Milk).")
    else:
        def summ(x):
            ch = x["cost"]["chosen"]
            return {"Vendor": ch["vendor"], "Qty": ch["qty"], "Lead (d)": ch["lead"], "Unit price": ch["unit_price"], "Expected total ₹": round(ch["exp_total"])}
        st.dataframe(pd.DataFrame({"Baseline": summ(base), f"{sv} +{sd}d": summ(alt)}).T)
        if base["po"]["vendor"] != alt["po"]["vendor"]:
            st.warning(f"Decision changed: {base['po']['vendor']} → {alt['po']['vendor']}")
        else:
            st.success("Decision unchanged - the plan is robust to this delay.")
        st.dataframe(pd.DataFrame(alt["cost"]["options"])[["vendor", "qty", "unit_price", "lead", "exp_short_units", "exp_total"]].round(1), hide_index=True)

# ---------------- orders & vendors
with tab_ord:
    drafts = [d for d in dec.values() if d["po"] and not d.get("placed")]
    if drafts:
        st.warning(f"{len(drafts)} draft PO(s) awaiting approval (auto-procurement is off or not run yet).")
        if st.button("✅ Approve & place draft POs"):
            orch.execute(sim, res)
            st.rerun()
    st.subheader("Purchase orders")
    if sim.pos:
        st.dataframe(pd.DataFrame(sim.pos[::-1]).drop(columns=["actual_eta_day"]), hide_index=True)
    else:
        st.info("No POs yet - press ▶ Run cycle (auto-procurement on) or approve the drafts.")
    st.subheader("Vendor performance (learned from deliveries)")
    vrows = [{"Product": p, "Vendor": n, "Price": v["price"], "Lead (d)": v["lead"], "Advertised reliability": v["rel"],
              "Delivered / on-time": f"{v['total']} / {v['ontime']}", "Estimated reliability": round(v["est_rel"], 2)}
             for p in PRODUCTS for n, v in sim.vendor_view(p).items()]
    st.dataframe(pd.DataFrame(vrows), hide_index=True)

# ---------------- log
with tab_log:
    st.subheader("Event log")
    st.dataframe(pd.DataFrame(sim.events[::-1][:100], columns=["Day", "Event"]), hide_index=True)
