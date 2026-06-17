# Loan Default Predictor

Predicts the probability that a loan applicant will default, served as a
**Dockerized FastAPI** service with a **Streamlit** demo UI. End-to-end ML:
data analysis → feature engineering → model training → evaluation → deployment.

## Problem
Binary classification on imbalanced financial data (~22% of loans default). A
missed default (false negative) is the costly error for a lender, so models are
compared on **ROC-AUC, precision, and recall** — not accuracy. (A model that
predicts "repaid" for everyone would be ~78% accurate yet catch zero defaulters.)

## Dataset
Kaggle **Credit Risk Dataset** (~32,500 loan records) — features include income,
loan amount, interest rate, loan grade, home ownership, employment length, and
prior default history. Target: `loan_status` (1 = default, 0 = repaid).

## Results (held-out test set)
| Model               | ROC-AUC |
|---------------------|---------|
| Logistic Regression | 0.843   |
| Random Forest       | 0.928   |
| **XGBoost**         | **0.948** |

**Best model: XGBoost (ROC-AUC 0.948).** Logistic Regression looked fine on
accuracy (~0.84) but had only ~0.43 recall on the default class — i.e. it missed
most actual defaulters. XGBoost ranks risky vs. safe applicants far better, which
is why it was selected.

## Tech
Python · pandas · NumPy · scikit-learn · XGBoost · Matplotlib · Seaborn ·
FastAPI · Streamlit · Docker · pytest

## Project structure
```
Project-1/
├── data/credit_risk_dataset.csv   # dataset
├── notebooks/eda.ipynb            # EDA, feature engineering, model training
├── src/app.py                     # FastAPI /predict service
├── model/                         # saved model.pkl + columns.json
├── images/                        # EDA charts
├── tests/test_model.py            # automated tests
├── streamlit_app.py               # interactive demo UI
├── requirements.txt
├── Dockerfile
└── README.md
```

## Run locally
```bash
pip install -r requirements.txt
uvicorn src.app:app --reload        # API at http://localhost:8000/docs
streamlit run streamlit_app.py      # demo UI at http://localhost:8501
```

## Run with Docker
```bash
docker build -t loan-default-predictor .
docker run -p 8000:8000 loan-default-predictor
```

## Test
```bash
pytest -q
```

## Example request
```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{
  "person_age": 23, "person_income": 18000, "person_home_ownership": "RENT",
  "person_emp_length": 1, "loan_intent": "DEBTCONSOLIDATION", "loan_grade": "E",
  "loan_amnt": 12000, "loan_int_rate": 18.5, "loan_percent_income": 0.66,
  "cb_person_default_on_file": "Y", "cb_person_cred_hist_length": 3 }'
# -> {"default_probability": 0.9x, "prediction": "high_risk"}
```
