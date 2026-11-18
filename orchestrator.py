"""Simulation world + Master Orchestrator (Forecast -> Risk -> Decision -> Vendor -> PO -> Execution loop)."""
import math
from dataclasses import dataclass
from datetime import date, timedelta
import numpy as np
import pandas as pd
from data import PRODUCTS, LOCATIONS, HISTORY, demand_value, make_vendors
from agents import (InventoryTrackingAgent, DemandForecastAgent, StockoutRiskAgent, ReorderDecisionAgent,
                    VendorIntelligenceAgent, CostOptimizationAgent, PurchaseOrderAgent,
                    WarehouseOptimizationAgent, AnomalyDetectionAgent, ReportingDashboardAgent)


@dataclass
class Params:
    service_level: float = 0.95
    review_days: int = 7
    auto: bool = True
    profit_aware: bool = True


class Sim:
    """A tiny supply-chain world: sales, deliveries, supplier delays, shrinkage, promo spikes."""

    def __init__(self, seed=7):
        self.rng = np.random.default_rng(seed)
        self.today = date.today()
        self.start = self.today - timedelta(days=HISTORY - 1)
        self.day, self.next_po = 0, 1024
        self.vendors = make_vendors(self.rng)
        self.history, self.pos, self.events, self.alert_log, self.spike = {}, [], [], [], {}
        self.kpi = dict(demand=0, lost=0, spend=0.0)
        for p in PRODUCTS:
            rows = []
            for i in range(HISTORY):
                d = self.start + timedelta(days=i)
                promo = bool(self.rng.random() < 0.06)
                rows.append((d, demand_value(p, d, i, self.rng, self.rng.uniform(1.4, 1.8) if promo else 1.0), promo))
            self.history[p] = rows
        self.physical, self.recorded = {}, {}
        for p, m in PRODUCTS.items():
            total = int(np.mean([h[1] for h in self.history[p][-7:]]) * m["cover"])
            w = np.array([s * self.rng.uniform(0.4, 1.8) for s in LOCATIONS.values()])
            alloc = np.floor(total * w / w.sum()).astype(int)
            alloc[0] += total - alloc.sum()
            self.physical[p] = dict(zip(LOCATIONS, map(int, alloc)))
            self.recorded[p] = dict(self.physical[p])

    # ---- helpers
    def pending(self, p=None):
        return [po for po in self.pos if po["status"] == "In Transit" and (p is None or po["product"] == p)]

    def vendor_view(self, p):
        out = {}
        for n, v in self.vendors[p].items():
            est = (v["rel"] * 5 + v["ontime"]) / (5 + v["total"])  # prior blended with observed on-time deliveries
            out[n] = dict(v, est_rel=est)
        return out

    def log(self, msg):
        self.events.append((self.day, msg))

    def hist_df(self, p):
        return pd.DataFrame(self.history[p], columns=["date", "demand", "promo"]).set_index("date")

    # ---- actions
    def place_po(self, draft):
        v = self.vendors[draft["product"]][draft["vendor"]]
        delay = 0 if self.rng.random() < v["true_rel"] else int(self.rng.integers(1, 4))  # hidden from the agents
        po = dict(id=self.next_po, product=draft["product"], vendor=draft["vendor"], qty=draft["qty"],
                  unit_price=draft["unit_price"], total=draft["total"], priority=draft["priority"],
                  created_day=self.day, promised_eta_day=self.day + draft["lead"],
                  actual_eta_day=self.day + draft["lead"] + delay, status="In Transit", delivered_day=None)
        self.next_po += 1
        self.pos.append(po)
        self.kpi["spend"] += po["total"]
        self.log(f"📝 PO #{po['id']} placed: {po['qty']} x {po['product']} from {po['vendor']} (₹{po['total']:,.0f}, ETA day {po['promised_eta_day']})")
        return po

    def apply_transfer(self, p, frm, to, qty):
        qty = min(qty, self.physical[p][frm], self.recorded[p][frm])
        for d in (self.physical, self.recorded):
            d[p][frm] -= qty
            d[p][to] += qty
        self.log(f"🔁 Transfer {qty} x {p}: {frm} -> {to}")

    def inject_spike(self, p, mult=3.0):
        self.spike[p] = mult

    def delay_in_transit(self, days=2):
        n = 0
        for po in self.pending():
            po["actual_eta_day"] += days
            n += 1
        self.log(f"🚧 Supplier disruption: {n} in-transit orders delayed by {days}d")
        return n

    def advance_day(self):
        self.day += 1
        self.today += timedelta(days=1)
        for po in self.pending():  # 1. deliveries
            if po["actual_eta_day"] <= self.day:
                p = po["product"]
                po["status"], po["delivered_day"] = "Delivered", self.day
                self.physical[p]["Warehouse A"] += po["qty"]
                self.recorded[p]["Warehouse A"] += po["qty"]
                v = self.vendors[p][po["vendor"]]
                v["total"] += 1
                v["ontime"] += int(self.day <= po["promised_eta_day"])
                self.log(f"📦 PO #{po['id']} delivered: {po['qty']} x {p} ({'on time' if self.day <= po['promised_eta_day'] else 'LATE'})")
        for p in PRODUCTS:  # 2. demand
            mult = self.spike.pop(p, None)
            promo = mult is not None or self.rng.random() < 0.04
            if promo and mult is None:
                mult = self.rng.uniform(1.3, 1.7)
            total = demand_value(p, self.today, (self.today - self.start).days, self.rng, mult or 1.0)
            self.history[p].append((self.today, total, bool(promo)))
            self.kpi["demand"] += total
            for loc, q in zip(LOCATIONS, self.rng.multinomial(total, list(LOCATIONS.values()))):
                sold = min(int(q), self.physical[p][loc])
                self.kpi["lost"] += int(q) - sold
                self.physical[p][loc] -= sold
                self.recorded[p][loc] = max(0, self.recorded[p][loc] - sold)
                if sold < q:
                    self.log(f"❌ Stockout: {p} at {loc} ({int(q) - sold} units of demand lost)")
        if self.rng.random() < 0.05:  # 3. silent shrinkage (theft / loss)
            p = list(PRODUCTS)[self.rng.integers(len(PRODUCTS))]
            loc = list(LOCATIONS)[self.rng.integers(len(LOCATIONS))]
            loss = int(self.physical[p][loc] * self.rng.uniform(0.08, 0.2))
            if loss > 0:
                self.physical[p][loc] -= loss


class Orchestrator:
    name = "Master Orchestrator"

    def __init__(self):
        self.inventory, self.forecaster, self.risk = InventoryTrackingAgent(), DemandForecastAgent(), StockoutRiskAgent()
        self.reorder, self.vendor, self.cost = ReorderDecisionAgent(), VendorIntelligenceAgent(), CostOptimizationAgent()
        self.po_agent, self.warehouse = PurchaseOrderAgent(), WarehouseOptimizationAgent()
        self.anomaly, self.reporter = AnomalyDetectionAgent(), ReportingDashboardAgent()

    def analyze_product(self, sim, p, params, delays=None):
        meta, T = PRODUCTS[p], []
        inv = self.inventory.snapshot(sim, p)
        T.append((self.inventory.name, f"{inv['total']} units across {len(LOCATIONS)} locations ({inv['on_order']} on order)"))
        fc = self.forecaster.forecast(p, sim.history[p], sim.today)
        T.append((self.forecaster.name, f"~{fc['mean7']:.0f} units/day next week (trend {fc['trend']:+.1f}/day, σ={fc['sigma']:.1f}, promo uplift ×{fc['uplift']:.2f})"))
        vendors = sim.vendor_view(p)
        lead = max(1, round(np.mean([v["lead"] for v in vendors.values()])))
        arrivals = [(max(1, po["promised_eta_day"] - sim.day), po["qty"]) for po in sim.pending(p)]
        risk = self.risk.assess(inv, fc, lead, arrivals)
        dts = "∞" if math.isinf(risk["days_to_stockout"]) else f"{risk['days_to_stockout']:.1f}"
        T.append((self.risk.name, f"{risk['level']} - runs out in {dts} days vs {lead}-day lead time; deficit {risk['deficit']:.0f} units; P(stockout)={risk['p_stockout']:.0%}"))
        ro = self.reorder.decide(inv, fc, lead, params)
        T.append((self.reorder.name, f"position {inv['position']} vs reorder point {ro['rop']:.0f} -> {'REORDER ' + str(ro['qty']) + ' units' if ro['trigger'] else 'no reorder'}; dynamic safety stock {ro['safety_stock']:.0f} (fixed rule would be {ro['fixed_safety']:.0f})"))
        ranking = cost = po = None
        if ro["trigger"]:
            ranking = self.vendor.rank(vendors, ro["qty"], risk["days_to_stockout"], delays)
            top = ranking["rows"][0]
            T.append((self.vendor.name, f"top score: {top['vendor']} - {ranking['reason']}"))
            cost = self.cost.optimize(ranking["rows"], ro["qty"], risk["avail"], fc, meta, params.profit_aware)
            ch = cost["chosen"]
            override = " (overrides vendor ranking)" if ch["vendor"] != top["vendor"] else ""
            T.append((self.cost.name, f"{ch['vendor']} x {ch['qty']}{override}: expected total ₹{ch['exp_total']:,.0f}. {cost['note']}"))
            po = self.po_agent.draft(p, ch, risk["level"])
            T.append((self.po_agent.name, f"Draft PO: {po['qty']} x {p} from {po['vendor']}, ₹{po['total']:,.0f}, deliver in {po['lead']}d, priority {po['priority']}"))
        wh = self.warehouse.plan(sim.recorded[p], fc["mean7"])
        if wh["transfers"]:
            T.append((self.warehouse.name, "; ".join(f"move {t['qty']} from {t['frm']} to {t['to']}" for t in wh["transfers"])))
        return dict(product=p, inv=inv, fc=fc, risk=risk, reorder=ro, ranking=ranking, cost=cost, po=po,
                    transfers=wh["transfers"], loc_status=wh["status"], trace=T, lead=lead)

    def run_cycle(self, sim, params, execute=True):
        decisions, alerts = {}, []
        for p in PRODUCTS:
            decisions[p] = self.analyze_product(sim, p, params)
            alerts += self.anomaly.scan(sim, p)
        res = dict(day=sim.day, decisions=decisions, alerts=alerts)
        if execute and params.auto:
            self.execute(sim, res)
        sim.alert_log += [a for a in alerts]
        res["kpis"] = self.reporter.kpis(sim)
        return res

    def execute(self, sim, res):
        """Act on drafts: place POs, rebalance warehouses, reconcile inventory mismatches."""
        for d in res["decisions"].values():
            if d["po"] and not d.get("placed"):
                d["placed"] = sim.place_po(d["po"])
            if d["transfers"] and not d.get("moved"):
                for t in d["transfers"]:
                    sim.apply_transfer(d["product"], t["frm"], t["to"], t["qty"])
                d["moved"] = True
        for a in res["alerts"]:
            if a["fix"] and not a["done"]:
                p, loc = a["fix"]
                sim.recorded[p][loc] = sim.physical[p][loc]
                a["done"] = True
                sim.log(f"🧮 Reconciled {p} @ {loc} to physical count {sim.physical[p][loc]}")
        res["kpis"] = self.reporter.kpis(sim)
