# Employee Attrition Analysis & HR Dashboard

![Python](https://img.shields.io/badge/Python-3.x-blue?style=flat-square)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?style=flat-square)
![Status](https://img.shields.io/badge/Status-Deployed-brightgreen?style=flat-square)

> Predicts which employees are at risk of leaving, and surfaces actionable HR insights through a live Streamlit dashboard.

---

## Problem Statement

Employee attrition costs organizations significant time and money in rehiring and retraining. This project uses historical HR data to build a machine learning model that predicts individual attrition risk — enabling HR teams to intervene proactively rather than reactively.

---

## Live Demo

> Run locally using the steps below, or see screenshots in `/assets`.
## 🚀 Live Demo
👉 [thejal-attrition-dashboard.streamlit.app](https://thejal-attrition-dashboard.streamlit.app)
---

## Results

| Model | Accuracy | Recall | ROC-AUC |
|---|---|---|---|
| Logistic Regression | — | — | — |
| **Random Forest (Best)** | — | — | **0.90** |

> Random Forest was selected as the final model based on ROC-AUC score. Saved as `best_attrition_model.pkl`.

**Dataset KPIs:**
- Total Employees: 1,470
- Attrition Count: 237
- Attrition Rate: 16.12%
- Avg Monthly Income: $6,500

---

## Features

- Exploratory Data Analysis (EDA) with attrition breakdown by department, age, income
- Two ML models compared: Logistic Regression and Random Forest
- Class imbalance handled via `class_weight='balanced'`
- Hyperparameter tuning with `GridSearchCV`
- Feature importance analysis (Random Forest + Logistic Regression coefficients)
- Interactive Streamlit dashboard:
  - Live KPI metrics
  - Attrition proportion by department
  - Top 10 employees by predicted attrition risk
  - Single employee attrition prediction with custom inputs

---

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.x |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn (Logistic Regression, Random Forest, GridSearchCV) |
| Visualization | Matplotlib, Seaborn, Streamlit |
| Deployment | Streamlit + joblib |
| Version Control | Git, GitHub |

---

## Project Structure

```
employee-attrition/
│
├── employee_attrition_analysis_and_dashboard.py   # Main script (analysis + dashboard)
├── best_attrition_model.pkl                       # Saved best model
├── WA_Fn-UseC_-HR-Employee-Attrition.csv          # Dataset
├── assets/
│   └── attrition_by_department.png                # EDA chart
└── README.md
```

---

## How to Run

```bash
# 1. Clone the repo
git clone https://github.com/cthejal/employee-attrition.git
cd employee-attrition

# 2. Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn joblib streamlit

# 3. Run analysis (trains and evaluates models)
python employee_attrition_analysis_and_dashboard.py --run-analysis --csv "WA_Fn-UseC_-HR-Employee-Attrition.csv"

# 4. Launch the Streamlit dashboard
streamlit run employee_attrition_analysis_and_dashboard.py
```

---

## Key Learnings

- Handling mixed feature types with `ColumnTransformer` (OneHotEncoder + StandardScaler)
- Model evaluation beyond accuracy — ROC-AUC and recall on imbalanced datasets
- Building end-to-end ML pipelines with `sklearn.pipeline.Pipeline`
- Debugging sparse matrix compatibility issues across scikit-learn versions
- Deploying interactive ML apps with Streamlit

---

## Dataset

- **Source:** IBM HR Analytics (via internship material)
- **Size:** 1,470 rows × 35 columns
- **Target:** `Attrition` (Yes/No)

---

## Author

**Thejal C Kotian**
[LinkedIn](https://linkedin.com/in/thejalckotian2003) · [GitHub](https://github.com/cthejal) · thejalck@gmail.com
