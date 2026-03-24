# 💳 Credit Card Fraud Detection — EDA Project

Structured exploratory data analysis for the [Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

---

## 📁 Project Structure
```
credit_card_fraud_eda_project/
├── data/
│   ├── raw/                # Place creditcard.csv here
│   └── processed/          # Cleaned dataset output
├── notebooks/
│   ├── 01_data_loading.ipynb
│   └── 02_eda_visualization.ipynb
├── src/
│   ├── data_preprocessing.py
│   └── eda_utils.py
├── reports/
│   └── figures/            # All saved plots
├── models/                 # Saved ML models (later)
├── configs/
│   └── config.yaml
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚙️ Setup

```bash
# 1. Clone / unzip the project
cd credit_card_fraud_eda_project

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Usage

1. Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
2. Place it in `data/raw/`.
3. Run notebooks in order:
   - `01_data_loading.ipynb` → inspect raw data
   - `02_eda_visualization.ipynb` → generate all plots

---

## 📊 Dataset Info
| Property | Detail |
|---|---|
| Rows | 284,807 transactions |
| Features | Time, V1–V28 (PCA), Amount, Class |
| Fraud rate | ~0.17% (highly imbalanced) |
| Missing values | None |

---

## 📌 Next Steps
- Handle class imbalance with SMOTE or undersampling
- Train models: Logistic Regression, Random Forest, XGBoost
- Evaluate using Precision-Recall AUC
