"""Synthetic data + demand model for the supply-chain simulation."""
import math
import numpy as np

HISTORY = 60  # days of sales history
LOCATIONS = {"Warehouse A": 0.35, "Warehouse B": 0.25, "Store C": 0.15, "Online FC": 0.25}  # demand share
WEEKLY = [0.95, 0.90, 0.92, 0.97, 1.10, 1.25, 1.00]  # Mon..Sun

# base = units/day, cost/price in INR, hold = holding cost/unit/day, penalty = stockout cost/unit
# cover = initial days of stock, peak/amp = seasonality (day-of-year of peak, amplitude)
PRODUCTS = {
    "Milk":        dict(base=50,  cost=50,  price=62,  hold=0.9,  penalty=45, cover=2.4, trend=0.0,   peak=150, amp=0.25),
    "Rice":        dict(base=100, cost=40,  price=52,  hold=0.2,  penalty=25, cover=7.0, trend=0.005, peak=0,   amp=0.0),
    "Bread":       dict(base=80,  cost=30,  price=40,  hold=0.8,  penalty=30, cover=3.5, trend=0.0,   peak=0,   amp=0.0),
    "Eggs":        dict(base=200, cost=6,   price=8,   hold=0.1,  penalty=6,  cover=12,  trend=0.0,   peak=15,  amp=0.15),
    "Sugar":       dict(base=70,  cost=42,  price=50,  hold=0.15, penalty=20, cover=14,  trend=0.0,   peak=305, amp=0.20),
    "Cooking Oil": dict(base=60,  cost=120, price=145, hold=0.3,  penalty=60, cover=45,  trend=0.0,   peak=0,   amp=0.0),
}


def season_index(product, d):
    m = PRODUCTS[product]
    if m["amp"] == 0:
        return 1.0
    doy = d.timetuple().tm_yday
    return 1 + m["amp"] * math.cos(2 * math.pi * (doy - m["peak"]) / 365)


def demand_value(product, d, t, rng, promo_mult=1.0):
    m = PRODUCTS[product]
    mu = m["base"] * (1 + m["trend"] * t) * WEEKLY[d.weekday()] * season_index(product, d) * promo_mult
    return int(max(1, round(mu * rng.normal(1, 0.07))))


def make_vendors(rng):
    # name, price multiplier, lead days, advertised reliability, bulk threshold (days of demand), bulk discount
    spec = [("Vendor A", 0.92, 5, 0.80, 8, 0.05),
            ("Vendor B", 1.00, 2, 0.93, 5, 0.03),
            ("Vendor C", 1.12, 1, 0.97, 3, 0.02)]
    out = {}
    for p, m in PRODUCTS.items():
        out[p] = {}
        for name, pm, lead, rel, minq_days, disc in spec:
            out[p][name] = dict(
                price=round(m["cost"] * pm * rng.uniform(0.98, 1.02), 2), lead=lead, rel=rel,
                true_rel=float(np.clip(rel + rng.normal(0, 0.04), 0.5, 0.99)),  # hidden ground truth
                minq=int(m["base"] * minq_days), disc=disc, ontime=0, total=0)
    return out
