# 📦 Autonomous Supply Chain & Inventory Management Agent (PS-08)

Multi-agent system + Streamlit demo. Runs fully offline (no API keys, no external data).

## Run
```bash
python -m venv venv && source venv/bin/activate     # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Agents (agents.py) → orchestrated by `Orchestrator` (orchestrator.py)
| # | Agent | What it does |
|---|-------|--------------|
| 1 | Inventory Tracking | Unified stock view across Warehouse A/B, Store C, Online FC + on-order qty |
| 2 | Demand Forecasting | Deseasonalise → day-of-week factors → promo uplift → damped Holt trend → seasonal re-application |
| 3 | Stockout Risk | Simulates stock burn-down incl. incoming POs → days-to-stockout, deficit, P(stockout), HIGH/MEDIUM/LOW/OVERSTOCK |
| 4 | Reorder Decision | (s,S) policy with **dynamic safety stock** `z·σ·√L` → WHEN and HOW MUCH |
| 5 | Vendor Intelligence | Price, speed, reliability (prior + observed on-time rate), bulk discount, urgency-aware weights |
| 6 | Purchase Order | Structured PO (ID, product, qty, vendor, cost, deadline, priority) |
| 7 | Warehouse Optimization | Detects overstock/stockout-risk locations, proposes & executes transfers |
| 8 | Anomaly Detection | Demand spikes, supplier delays, inventory mismatch (theft/loss) |
| 9 | Cost Optimization | Expected total cost (purchase + holding + stockout penalty × P(delay)); final vendor/qty arbiter |
| 10 | Reporting Dashboard | KPIs: fill rate, lost sales, spend, open POs |
| 11 | Master Orchestrator | Forecast → Risk → Decision → Vendor → PO → Execution loop |

Advanced features included: dynamic safety stock, multi-vendor delay simulation (Scenario tab), profit-aware ordering with bulk discounts, seasonal intelligence, live dashboard, auto-procurement loop (Detect → Decide → Order → Track → Adjust).

## Demo script (3 minutes)
1. Open the app: **Milk** is 🔴 HIGH (~2 days of stock vs 3-day lead). Open **Agent cockpit → Milk** to show each agent's reasoning and the Cost agent overriding the vendor ranking.
2. Toggle **Auto-procurement** off → show draft POs → **Orders & vendors → Approve**.
3. Press **⏩ +7 days**: watch deliveries, vendor reliability learning, event log.
4. Sidebar → **💥 Demand spike ×3** on Rice → **+1 day** → anomaly alert + forecast adjusts.
5. Sidebar → **🚧 Delay in-transit orders** → **+1 day** → supplier-delay alert.
6. **Scenario simulator**: "What if Vendor B delays 2 days?" → decision recomputed.

## Plug in real data / LLMs
- Replace `Sim` in `orchestrator.py` with your DB/ERP/CSV loaders (`history`, `recorded`, `vendors`).
- Add an LLM (Gemini/Claude) in `ReportingDashboardAgent` to narrate decisions from the `trace` list.
