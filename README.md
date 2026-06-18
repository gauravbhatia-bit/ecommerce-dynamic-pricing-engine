# 🛒 E-Commerce Dynamic Pricing Engine

> **Algorithmusbasierte Dynamic Pricing Engine für E-Commerce** —  
> Preiselastizität, Kundensegmentierung und margenoptimierte Preisempfehlungen  
> mit XGBoost und Generative AI.

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

# 4. Run the full pipeline
python run_pipeline.py

# 5. Launch the dashboard
streamlit run dashboard/app.py
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
│   ├── data_prep.py             ← Step 1: Generate / load dataset
│   ├── elasticity_model.py      ← Step 2: Log-log OLS price elasticity
│   ├── rfm_segmentation.py      ← Step 3: RFM customer scoring
│   └── pricing_engine.py        ← Step 4+5: Rules engine + XGBoost ML
│
├── dashboard/
│   └── app.py                   ← Streamlit KPI dashboard + GenAI reports
│
├── data/
│   ├── raw/                     ← Generated dataset (gitignored)
│   └── processed/               ← Elasticity, RFM, pricing outputs
│
└── charts/                      ← Saved visualisations
```

---

## 🏗️ Pipeline Architecture

```
Raw Data (15K SKUs)
    │
    ▼
Step 1: Data Prep & Feature Engineering
    │  price, cost, stock, competitor_price, reviews, margin
    ▼
Step 2: Price Elasticity Model (OLS log-log per category)
    │  → elastic vs inelastic segments
    ▼
Step 3: RFM Customer Segmentation
    │  → Champions / Loyal / Price-Sensitive
    ▼
Step 4: Rule-Based Pricing Engine (6 business rules)
    │  → 55.7% of SKUs repriced with business rationale
    ▼
Step 5: XGBoost ML Pricing Layer
    │  → MAE: €2.31 | MAPE: 2.14% | R²: 0.999
    ▼
Step 6: Streamlit Dashboard + GenAI Report Generator
    → Pricing KPIs | Margin monitoring | Management reports
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
| Top ML feature | competitor_price (62.2%) |

---

## ⚙️ Pricing Rules

| Rule | Trigger | Action |
|------|---------|--------|
| 1 | Overstock >300 + elasticity >1.5 | −8% (drive volume) |
| 2 | Inelastic + margin >45% | +6% (capture margin) |
| 3 | Competitor undercuts >7% + loyal segment | Match competitor price |
| 4 | Stock <50 + Champions segment | +4% scarcity premium |
| 5 | Competitor priced >5% higher + elastic market | +3% uptick |
| 6 | Default | Hold current price |

---

## 🤖 Generative AI Integration

Dashboard includes a one-click AI report generator that produces natural-language  
pricing summaries for management — demonstrating practical use of GenAI to  
**maximise analysis efficiency** (*Generative AI konstruktiv einsetzen*).

---

## 🛠️ Tech Stack

`Python` · `Pandas` · `Statsmodels` · `XGBoost` · `Scikit-learn` · `Streamlit` · `Plotly` · `Git`

---

*Built by [Gaurav Bhatia](https://github.com/gauravbhatia-bit) | MSc Data Science, GISMA University Berlin*
