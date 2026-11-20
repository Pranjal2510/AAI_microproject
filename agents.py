"""All agents. Each agent is a small class with one clear responsibility."""
import math
from datetime import timedelta
from statistics import NormalDist
import numpy as np
from data import PRODUCTS, LOCATIONS, season_index

ND = NormalDist()


def cum_demand(daily, L):
    """Forecast demand over the next L days (L may be fractional)."""
    n, frac = int(L), L - int(L)
    tot = float(sum(daily[:n]))
    if frac > 0 and n < len(daily):
        tot += daily[n] * frac
    return tot


def ceil10(x):
    return int(math.ceil(max(x, 0) / 10.0) * 10)


class InventoryTrackingAgent:
    name = "Inventory Tracking Agent"

    def snapshot(self, sim, p):
        by_loc = dict(sim.recorded[p])
        total = sum(by_loc.values())
        on_order = sum(po["qty"] for po in sim.pending(p))
        return dict(total=total, by_loc=by_loc, on_order=on_order, position=total + on_order)


class DemandForecastAgent:
    """Deseasonalise -> day-of-week factors -> promo uplift -> damped Holt trend -> re-seasonalise."""
    name = "Demand Forecasting Agent"

    def forecast(self, product, hist, today, horizon=30):
        hist = hist[-56:]
        dates = [h[0] for h in hist]
        y = np.array([h[1] for h in hist], float)
        promo = np.array([h[2] for h in hist], bool)
        s = np.array([season_index(product, d) for d in dates])
        wd = np.array([d.weekday() for d in dates])
        y1 = y / s
        base = y1[~promo].mean() if (~promo).any() else y1.mean()
        dowf = np.ones(7)
        for k in range(7):
            m = (wd == k) & ~promo
            if m.sum() >= 2:
                dowf[k] = y1[m].mean() / base
        dowf = dowf / dowf.mean()
        y2 = y1 / dowf[wd]
        uplift = 1.0
        if promo.sum() >= 2 and (~promo).sum() >= 5:
            uplift = float(np.clip(y2[promo].mean() / y2[~promo].mean(), 1.0, 3.0))
        y3 = np.where(promo, y2 / uplift, y2)
        level, trend, res = float(y3[:7].mean()), 0.0, []
        for i in range(len(y3)):
            pred = level + trend
            if not promo[i]:
                res.append((y3[i] - pred) * dowf[wd[i]] * s[i])
            new = 0.3 * y3[i] + 0.7 * pred
            trend = 0.1 * (new - level) + 0.9 * trend
            level = new
        daily = []
        for h in range(1, horizon + 1):
            d = today + timedelta(days=h)
            lv = level + trend * sum(0.9 ** k for k in range(1, h + 1))
            daily.append(max(1.0, lv * dowf[d.weekday()] * season_index(product, d)))
        sigma = float(np.std(res[-28:])) if len(res) >= 7 else 0.1 * level
        return dict(daily=np.array(daily), mean7=float(np.mean(daily[:7])), sigma=max(sigma, 1.0),
                    trend=float(trend), uplift=uplift)


class StockoutRiskAgent:
    name = "Stockout Risk Agent"

    def assess(self, inv, fc, lead, arrivals):
        daily = fc["daily"]
        arr = {}
        for rem, q in arrivals:  # delivery lands before that day's demand
            k = min(len(daily) - 1, max(1, int(math.ceil(rem))) - 1)
            arr[k] = arr.get(k, 0) + q
        s, days_to = float(inv["total"]), math.inf
        for k in range(len(daily)):
            s += arr.get(k, 0)
            if s < daily[k]:
                days_to = k + s / daily[k]
                break
            s -= daily[k]
        required = cum_demand(daily, lead)
        avail = inv["total"] + sum(q for rem, q in arrivals if rem <= lead)
        deficit = max(0.0, required - avail)
        sig = fc["sigma"] * math.sqrt(lead)
        p_out = 1 - ND.cdf((avail - required) / sig)
        cover = inv["total"] / fc["mean7"]
        if days_to < lead or p_out > 0.5:
            level = "HIGH"
        elif days_to < lead + 3 or p_out > 0.2:
            level = "MEDIUM"
        elif inv["position"] / fc["mean7"] > 35:
            level = "OVERSTOCK"
        else:
            level = "LOW"
        return dict(level=level, days_to_stockout=days_to, required=required, avail=avail, deficit=deficit,
                    p_stockout=p_out, cover_days=cover, lead=lead)


class ReorderDecisionAgent:
    """(s, S) policy with *dynamic* safety stock: SS = z * sigma_daily * sqrt(lead)."""
    name = "Reorder Decision Agent"

    def decide(self, inv, fc, lead, params):
        daily, mu = fc["daily"], fc["mean7"]
        z = ND.inv_cdf(params.service_level)
        ss = z * fc["sigma"] * math.sqrt(lead)
        rop = cum_demand(daily, lead) + ss
        up_to = cum_demand(daily, lead + params.review_days) + ss
        qty = ceil10(up_to - inv["position"])
        return dict(trigger=bool(inv["position"] <= rop and qty > 0), rop=rop, safety_stock=ss,
                    fixed_safety=2 * mu, order_up_to=up_to, qty=qty, volatility=fc["sigma"] / mu, z=z)


def quote(list_price, disc, minq, q):
    return round(list_price * (1 - disc) if q >= minq else list_price, 2)


class VendorIntelligenceAgent:
    """Multi-criteria score: price, speed, reliability (observed + prior), bulk discount, urgency."""
    name = "Vendor Intelligence Agent"

    def rank(self, vendors, qty, days_to_stockout, delays=None):
        delays = delays or {}
        rows = []
        for n, v in vendors.items():
            rows.append(dict(vendor=n, list_price=v["price"], disc=v["disc"], minq=v["minq"],
                             price=quote(v["price"], v["disc"], v["minq"], qty), lead=v["lead"] + delays.get(n, 0),
                             reliability=v["est_rel"], bulk=qty >= v["minq"]))
        mean_lead = np.mean([r["lead"] for r in rows])
        urgent = days_to_stockout < mean_lead + 1
        w = dict(cost=0.2, speed=0.5, rel=0.3) if urgent else dict(cost=0.5, speed=0.2, rel=0.3)

        def norm(key, inv=False):
            vals = [r[key] for r in rows]
            lo, hi = min(vals), max(vals)
            return [0.5 if hi == lo else ((v - lo) / (hi - lo) if not inv else (hi - v) / (hi - lo)) for v in vals]
        nc, ns, nr = norm("price"), norm("lead"), norm("reliability", inv=True)
        for i, r in enumerate(rows):
            r["score"] = round(w["cost"] * (1 - nc[i]) + w["speed"] * (1 - ns[i]) + w["rel"] * (1 - nr[i])
                               - (0.3 if r["lead"] > days_to_stockout else 0), 3)
            r["misses_stockout_window"] = bool(r["lead"] > days_to_stockout)
        rows.sort(key=lambda r: -r["score"])
        top, why = rows[0], []
        if top["lead"] == min(r["lead"] for r in rows): why.append(f"fastest delivery ({top['lead']}d)")
        if top["price"] == min(r["price"] for r in rows): why.append("lowest price")
        if top["reliability"] == max(r["reliability"] for r in rows): why.append(f"best reliability ({top['reliability']:.0%})")
        if urgent: why.append("stockout urgency weights speed higher")
        if top["bulk"]: why.append("bulk discount applied")
        return dict(rows=rows, urgent=urgent, weights=w, reason=", ".join(why) or "best weighted score")


class CostOptimizationAgent:
    """Final arbiter: expected total cost = purchase + holding + P(delay)-weighted stockout penalty."""
    name = "Cost Optimization Agent"

    def optimize(self, rows, qty, avail, fc, meta, profit_aware):
        daily, mu = fc["daily"], fc["mean7"]
        qtys = sorted({ceil10(qty * f) for f in (1, 1.25, 1.5)}) if profit_aware else [qty]
        options = []
        for q in qtys:
            for r in rows:
                price = quote(r["list_price"], r["disc"], r["minq"], q)
                rel, L = r["reliability"], r["lead"]
                short = rel * max(0, cum_demand(daily, L) - avail) + (1 - rel) * max(0, cum_demand(daily, L + 2) - avail)
                purchase = price * q
                holding = meta["hold"] * (q / 2) * (q / max(mu, 1))
                total = purchase + holding + meta["penalty"] * short
                options.append(dict(vendor=r["vendor"], qty=q, unit_price=price, lead=L, purchase=purchase,
                                    holding=holding, exp_short_units=short, exp_total=total, cost_per_unit=total / q))
        best = min(options, key=lambda o: o["cost_per_unit"])
        at_q = [o for o in options if o["qty"] == best["qty"]]
        cheapest = min(at_q, key=lambda o: o["unit_price"])
        if cheapest["vendor"] == best["vendor"]:
            note = "Chosen option is also the cheapest per unit."
        else:
            note = (f"{cheapest['vendor']} is cheaper per unit (₹{cheapest['unit_price']}) but risks ~{cheapest['exp_short_units']:.0f} "
                    f"units of stockout; expected total is ₹{cheapest['exp_total'] - best['exp_total']:,.0f} higher -> safer option wins.")
        return dict(chosen=best, options=at_q, cheapest=cheapest, note=note, qty_candidates=qtys)


class PurchaseOrderAgent:
    name = "Purchase Order Agent"

    def draft(self, p, chosen, risk_level):
        pr = {"HIGH": "High", "MEDIUM": "Medium"}.get(risk_level, "Normal")
        return dict(product=p, qty=chosen["qty"], vendor=chosen["vendor"], unit_price=chosen["unit_price"],
                    total=round(chosen["purchase"], 2), lead=int(chosen["lead"]), priority=pr)


class WarehouseOptimizationAgent:
    name = "Warehouse Optimization Agent"

    def plan(self, by_loc, mu, tol=0.2, min_move=10):
        total = sum(by_loc.values())
        target = {l: total * s for l, s in LOCATIONS.items()}
        surplus = {l: by_loc[l] - target[l] for l in LOCATIONS}
        status = {}
        for l, s in LOCATIONS.items():
            cover = by_loc[l] / max(mu * s, 1e-9)
            status[l] = dict(stock=by_loc[l], cover_days=round(cover, 1),
                             status="Overstock" if surplus[l] > tol * target[l] and target[l] > 0 else
                                    "Stockout risk" if -surplus[l] > tol * target[l] and target[l] > 0 else "Balanced")
        donors = {l: s for l, s in surplus.items() if status[l]["status"] == "Overstock"}
        needy = sorted([(l, -s) for l, s in surplus.items() if status[l]["status"] == "Stockout risk"], key=lambda x: -x[1])
        moves = []
        for r, need in needy:
            for d in sorted(donors, key=lambda k: -donors[k]):
                mv = int(min(need, donors[d]))
                if mv >= min_move:
                    moves.append(dict(frm=d, to=r, qty=mv))
                    donors[d] -= mv
                    need -= mv
        return dict(transfers=moves, status=status)


class PerishabilityAgent:
    name = "Perishability & Expiry Agent"

    def assess(self, sim, p, fc):
        meta = PRODUCTS[p]
        shelf_life = meta.get("shelf_life", 365)
        if shelf_life >= 180:
            return dict(status="SAFE", expiring_soon=0, at_risk_qty=0, exp_loss_est=0.0,
                        batches=[], shelf_life=shelf_life,
                        note=f"Shelf life is {shelf_life} days (non-perishable). Spoilage risk negligible.")

        all_batches = []
        expiring_soon = 0
        if hasattr(sim, "batches") and p in sim.batches:
            for loc, blist in sim.batches[p].items():
                for b in blist:
                    rem_days = max(0, b["expiry_day"] - sim.day)
                    all_batches.append(dict(location=loc, qty=b["qty"], days_left=rem_days, expiry_day=b["expiry_day"]))
                    if rem_days <= 3:
                        expiring_soon += b["qty"]

        all_batches.sort(key=lambda x: x["days_left"])
        exp_3d_demand = float(sum(fc["daily"][:3])) if len(fc["daily"]) >= 3 else float(fc["mean7"] * 3)
        at_risk_qty = max(0, expiring_soon - int(exp_3d_demand))
        loss_est = at_risk_qty * meta["cost"]

        if at_risk_qty > 0.3 * exp_3d_demand and at_risk_qty >= 10:
            status = "HIGH"
        elif at_risk_qty > 0:
            status = "MEDIUM"
        else:
            status = "SAFE"

        if status == "HIGH":
            note = f"CRITICAL: {expiring_soon} units expire in ≤3d vs 3d forecast demand of {exp_3d_demand:.0f}. Projected spoilage: {at_risk_qty} units (₹{loss_est:,.0f}). Requires clearance markdown!"
        elif status == "MEDIUM":
            note = f"WARNING: {expiring_soon} units near expiry (≤3d). At-risk surplus: {at_risk_qty} units. Recommending promotional discount."
        else:
            note = f"Healthy rotation (FEFO). {expiring_soon} units expiring in ≤3d within projected demand capacity ({exp_3d_demand:.0f} units)."

        return dict(status=status, expiring_soon=expiring_soon, at_risk_qty=at_risk_qty,
                    exp_loss_est=loss_est, batches=all_batches, shelf_life=shelf_life, note=note)


class DynamicPricingAgent:
    name = "Dynamic Pricing & Markdown Agent"

    def evaluate(self, p, base_price, risk, perish, inv, fc):
        meta = PRODUCTS[p]
        disc = 0.0
        surge = 0.0
        reason = "Standard baseline price"

        if perish["status"] == "HIGH" and perish["at_risk_qty"] > 0:
            disc = 0.25
            reason = f"Clearance Markdown (-25%): Liquidate {perish['at_risk_qty']} expiring units before spoilage"
        elif perish["status"] == "MEDIUM" and perish["at_risk_qty"] > 0:
            disc = 0.15
            reason = f"Expiry Promotion (-15%): Accelerate sell-through of {perish['at_risk_qty']} near-expiry units"
        elif risk["level"] == "OVERSTOCK":
            disc = 0.10
            reason = "Overstock Markdown (-10%): Stimulate demand velocity to relieve warehouse capacity"
        elif risk["level"] == "HIGH" and risk["days_to_stockout"] < risk["lead"]:
            surge = 0.08
            reason = f"Surge Protection (+8%): Dampen stockout risk and preserve margin during supplier lead time"

        current_price = round(base_price * (1.0 - disc + surge), 2)
        elasticity = meta.get("elasticity", 1.0)
        price_ratio = current_price / base_price
        demand_mult = round((1.0 / price_ratio) ** elasticity, 2)

        return dict(product=p, base_price=base_price, current_price=current_price,
                    discount_pct=disc, surge_pct=surge, reason=reason,
                    demand_mult=demand_mult, is_adjusted=bool(disc > 0 or surge > 0))


class AnomalyDetectionAgent:
    name = "Anomaly Detection Agent"

    def scan(self, sim, p):
        out = []
        y = np.array([h[1] for h in sim.history[p]], float)
        promo_flag = sim.history[p][-1][2]
        if len(y) >= 15:
            prev, last = y[-29:-1], y[-1]
            z = (last - prev.mean()) / max(prev.std(), 1)
            ratio = last / max(prev.mean(), 1)
            if z > 3 and ratio > 1.8:
                out.append(self._a(sim, p, "Demand spike", "HIGH",
                                   f"Demand spike +{(ratio - 1) * 100:.0f}% detected" + (" - possible promotion effect" if promo_flag else " - unexplained")))
            elif ratio < 0.4 and z < -3:
                out.append(self._a(sim, p, "Demand drop", "MEDIUM", f"Demand fell {(1 - ratio) * 100:.0f}% below normal"))
        for po in sim.pending(p):
            if sim.day > po["promised_eta_day"]:
                out.append(self._a(sim, p, "Supplier delay", "HIGH",
                                   f"PO #{po['id']} from {po['vendor']} is {sim.day - po['promised_eta_day']}d late"))
        for loc in LOCATIONS:
            diff = sim.recorded[p][loc] - sim.physical[p][loc]
            if diff >= max(5, 0.05 * sim.recorded[p][loc]):
                out.append(self._a(sim, p, "Inventory mismatch", "MEDIUM",
                                   f"{loc}: system says {sim.recorded[p][loc]}, physical count {sim.physical[p][loc]} (-{diff}) - possible theft/loss",
                                   fix=(p, loc)))
        if hasattr(sim, "daily_spoilage") and sim.daily_spoilage.get(p, 0) > 0:
            sp_units = sim.daily_spoilage[p]
            cost = sp_units * PRODUCTS[p]["cost"]
            out.append(self._a(sim, p, "Spoilage loss", "HIGH",
                               f"{sp_units} units spoiled due to shelf-life expiry (Loss: ₹{cost:,.0f})"))
        return out

    @staticmethod
    def _a(sim, p, typ, sev, msg, fix=None):
        return dict(day=sim.day, product=p, type=typ, severity=sev, message=msg, fix=fix, done=False)


class ReportingDashboardAgent:
    name = "Reporting Dashboard Agent"

    def kpis(self, sim):
        k = sim.kpi
        inv_val = sum(sum(sim.recorded[p].values()) * PRODUCTS[p]["cost"] for p in PRODUCTS)
        rev = k.get("revenue", 0.0)
        spoil_units = k.get("spoilage_units", 0)
        spoil_cost = k.get("spoilage_cost", 0.0)
        spend = k.get("spend", 0.0)
        profit = rev - spend - spoil_cost
        return dict(fill_rate=1 - k["lost"] / max(k["demand"], 1), lost_units=k["lost"], spend=spend,
                    revenue=rev, profit=profit, spoilage_units=spoil_units, spoilage_cost=spoil_cost,
                    open_pos=len(sim.pending()), inventory_value=inv_val)
