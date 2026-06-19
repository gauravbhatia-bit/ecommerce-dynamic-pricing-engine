# 🛒 E-Commerce Dynamic Pricing Engine

> **Algorithmusbasierte Dynamic Pricing Engine für E-Commerce** —  
> Preiselastizität, Kundensegmentierung und margenoptimierte Preisempfehlungen  
> mit XGBoost und Generative AI.

---

## 💡 What This Project Does

In e-commerce, millions of SKUs need continuous price adjustments based on competitor moves, stock levels, customer segments, and demand elasticity. Manual pricing is slow and leaves margin on the table.

This project builds an **end-to-end algorithmic pricing engine** that:
- Estimates **price elasticity** per product category to identify price-sensitive vs. inelastic segments
- Scores customers into **Champions / Loyal / Price-Sensitive** using RFM methodology
- Applies **6 rule-based pricing strategies** that balance margin optimisation with customer retention
- Trains an **XGBoost ML model** that learns optimal price recommendations (MAPE: 2.14%)
- Serves everything in a **Streamlit KPI dashboard** with a one-click AI management report generator

This directly mirrors the workflow of a **Pricing & Analytics team** in e-commerce: *"die Brücke zwischen Marge und Kundenzufriedenheit schlagen."*

---

## 🚀 Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/gauravbhatia-bit/ecommerce-dynamic-pricing-engine
cd ecommerce-dynamic-pricing-engine

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate          # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the full pipeline (generates data + models)
python run_pipeline.py

# 5. Launch the dashboard
streamlit run dashboard/app.py
```

Expected terminal output after `run_pipeline.py`:
```
Step 1 — Generating dataset...
Step 2 — Computing price elasticity...
Step 3 — RFM segmentation...
Step 4 — Applying rule-based pricing engine...
  8,352 / 15,000 SKUs repriced (55.7%)
Step 5 — Training XGBoost ML pricing model...
  MAE  : €2.31
  MAPE : 2.14%
  R²   : 0.999
Pipeline complete → data/processed/final_pricing_output.csv
```

---

## 📁 Project Structure

```
ecommerce-dynamic-pricing-engine/
│
├── run_pipeline.py              ← Entry point: runs all 5 steps
├── requirements.txt
├── .gitignore
│
├── src/
│   ├── data_prep.py             ← Step 1: Generate dataset (15K SKUs, 8 categories)
│   ├── elasticity_model.py      ← Step 2: Log-log OLS price elasticity per category
│   ├── rfm_segmentation.py      ← Step 3: RFM customer scoring & segmentation
│   └── pricing_engine.py        ← Step 4+5: Rule-based engine + XGBoost ML layer
│
├── dashboard/
│   └── app.py                   ← Streamlit KPI dashboard + AI report generator
│
├── data/
│   ├── raw/                     ← Auto-generated on first run (gitignored)
│   └── processed/               ← Elasticity, RFM, pricing outputs
│
└── charts/                      ← Saved visualisations
```

---

## 🏗️ Pipeline Architecture

```
Raw Data (15K SKUs, 8 categories)
         │
         ▼
┌─────────────────────────────────────────┐
│ Step 1 · Data Prep                      │
│ price, cost, stock, competitor_price,   │
│ reviews, margin, day_of_week            │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│ Step 2 · Price Elasticity (OLS log-log) │
│ → elastic vs inelastic per category     │
│ → feeds pricing rule thresholds         │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│ Step 3 · RFM Customer Segmentation      │
│ → Champions / Loyal / Price-Sensitive   │
│ → segment-aware pricing decisions       │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│ Step 4 · Rule-Based Pricing Engine      │
│ → 6 business rules                      │
│ → 55.7% of SKUs repriced                │
│ → every decision has a business reason  │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│ Step 5 · XGBoost ML Pricing Layer       │
│ → learns from rule-based targets        │
│ → MAE: €2.31 | MAPE: 2.14% | R²: 0.999 │
│ → top feature: competitor_price (62%)   │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│ Step 6 · Streamlit Dashboard            │
│ → Pricing KPIs + margin monitoring      │
│ → SKU-level recommendations table       │
│ → AI-generated management report        │
└─────────────────────────────────────────┘
```

---

## 📊 Key Results

| Metric | Value |
|--------|-------|
| SKUs analysed | 15,000 |
| SKUs repriced | 8,352 (55.7%) |
| XGBoost MAE | €2.31 |
| XGBoost MAPE | 2.14% |
| XGBoost R² | 0.999 |
| Top ML driver | competitor_price (62.2% importance) |
| Most elastic category | Toys (-1.9) & Fashion (-2.1) |
| Safest margin uplift | Food & Beverages (-0.5) & Beauty (-0.8) |

---

## 📈 Business Impact

- **55.7% of the catalogue** has a data-driven pricing action vs. holding a static price
- **Elastic categories** (Toys, Fashion, Electronics) flagged for volume-first strategy — avoid blind margin increases
- **Inelastic categories** (Food & Beverages, Beauty, Books) identified as safe margin uplift opportunities of +4–6%
- **Champions segment** (highest RFM) receives scarcity premium pricing when stock is low
- **Loyal customers** are protected from churn via automatic competitor price matching
- Management gets a **one-click AI pricing report** instead of manually summarising data — saves analyst time

---

## ⚙️ Pricing Rules Implemented

| # | Trigger | Action | Business Logic |
|---|---------|--------|----------------|
| 1 | Overstock >300 units + elasticity >1.5 | −8% price | Drive volume to clear excess inventory |
| 2 | Inelastic demand + margin >45% | +6% price | Capture margin safely without losing customers |
| 3 | Competitor undercuts >7% + Loyal/Champions segment | Match competitor price | Retain high-value customers |
| 4 | Stock <50 units + Champions segment | +4% price | Scarcity premium for top customers |
| 5 | Competitor priced >5% higher + elastic market | +3% price | Opportunistic uptick without risk |
| 6 | No trigger matches | Hold price | No action needed |

---

## 🤖 Generative AI Integration

The Streamlit dashboard includes a **one-click AI report generator** that creates natural-language pricing summaries for management:

> *"Category Electronics shows high price elasticity (-1.8). Current overstock combined with elastic demand suggests an 8% price reduction could increase sales volume significantly. Categories Food & Beverages and Beauty are inelastic — safe to apply +4–6% margin uplift. Champion segment (537 SKUs): apply scarcity premium where stock < 50 units."*

This demonstrates: **Generative AI und AI-Coding-Tools konstruktiv einsetzen, um die Effizienz der Analysen zu maximieren.**

---

## 🛠️ Tech Stack

| Layer | Tools |
|-------|-------|
| Data Processing | Python, Pandas, NumPy |
| Elasticity Modelling | Statsmodels (OLS log-log regression) |
| Customer Segmentation | RFM scoring, Scikit-learn |
| ML Pricing Model | XGBoost, Scikit-learn |
| Dashboard | Streamlit, Plotly |
| Generative AI | Pattern: LLM-generated management summaries |
| Version Control | Git, GitHub |

---

## 🗂️ Output Files (auto-generated)

| File | Description |
|------|-------------|
| `data/processed/price_elasticity.csv` | Elasticity coefficient per category |
| `data/processed/rfm_segments.csv` | RFM scores and segment per product |
| `data/processed/final_pricing_output.csv` | Full pricing recommendations (rule + ML) |
| `data/processed/feature_importance.csv` | XGBoost feature importance ranking |

---

*Built by [Gaurav Bhatia](https://github.com/gauravbhatia-bit) | MSc Data Science & AI, GISMA University Berlin*  
*Portfolio: [github.com/gauravbhatia-bit](https://github.com/gauravbhatia-bit)*
