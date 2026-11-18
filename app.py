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
    if st.button("🚧 Delay in-transit orders +2d"):
        st.toast(f"{sim.delay_in_transit(2)} orders delayed")
    if st.button("🔄 Reset simulation"):
        new_sim(seed, params)
        st.rerun()

res = st.session_state.result
dec, k = res["decisions"], res["kpis"]
st.header("Autonomous Supply Chain Intelligence & Execution System")
m = st.columns(5)
m[0].metric("Sim day", sim.day, sim.today.strftime("%d %b %Y"))
m[1].metric("Fill rate", f"{k['fill_rate']:.1%}")
m[2].metric("Lost sales (units)", k["lost_units"])
m[3].metric("Procurement spend", f"₹{k['spend']:,.0f}")
m[4].metric("Open POs", k["open_pos"])

tab_dash, tab_cock, tab_scn, tab_ord, tab_log = st.tabs(
    ["📊 Dashboard", "🧠 Agent cockpit", "🧪 Scenario simulator", "🧾 Orders & vendors", "📜 Activity log"])

# ---------------- dashboard
with tab_dash:
    rows = []
    for p, d in dec.items():
        r, ro = d["risk"], d["reorder"]
        rows.append({"Product": p, "Stock": d["inv"]["total"], "On order": d["inv"]["on_order"],
                     "Forecast/day": round(d["fc"]["mean7"]), "Days of cover": round(min(r["cover_days"], 99), 1),
                     "Stockout in (days)": "∞" if math.isinf(r["days_to_stockout"]) else f"{r['days_to_stockout']:.1f}",
                     "Risk": f"{ICON[r['level']]} {r['level']}", "Reorder?": "YES" if ro["trigger"] else "no",
                     "Order qty": d["po"]["qty"] if d["po"] else 0, "Vendor": d["po"]["vendor"] if d["po"] else "-",
                     "Status": f"PO #{d['placed']['id']} placed" if d.get("placed") else ("Draft - awaiting approval" if d["po"] else "-")})
    st.subheader("Risk & recommended actions")
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

# ---------------- cockpit
with tab_cock:
    p = st.selectbox("Inspect product", list(PRODUCTS), key="cock")
    d = dec[p]
    c = st.columns(4)
    c[0].metric("Risk", f"{ICON[d['risk']['level']]} {d['risk']['level']}")
    c[1].metric("Stockout in", "∞" if math.isinf(d["risk"]["days_to_stockout"]) else f"{d['risk']['days_to_stockout']:.1f} d")
    c[2].metric("Reorder point", f"{d['reorder']['rop']:.0f}")
    c[3].metric("Dynamic safety stock", f"{d['reorder']['safety_stock']:.0f}", f"vs fixed {d['reorder']['fixed_safety']:.0f}", delta_color="off")
    st.markdown("##### Agent reasoning trace")
    for ag, msg in d["trace"]:
        st.markdown(f"**{ag}** - {msg}")
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
