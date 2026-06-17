import json, joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Loan Default Predictor")

# Load the trained model + the exact column order it was trained on
model = joblib.load("model/model.pkl")
TRAIN_COLUMNS = json.load(open("model/columns.json"))

# Defines the shape of the incoming request (the raw applicant fields)
class Applicant(BaseModel):
    person_age: int = 35
    person_income: int = 55000
    person_home_ownership: str = "RENT"
    person_emp_length: float = 5
    loan_intent: str = "PERSONAL"
    loan_grade: str = "B"
    loan_amnt: int = 10000
    loan_int_rate: float = 11.0
    loan_percent_income: float = 0.18
    cb_person_default_on_file: str = "N"
    cb_person_cred_hist_length: int = 5

@app.get("/")
def health():
    return {"status": "ok", "message": "Loan Default Predictor running. See /docs"}

@app.post("/predict")
def predict(a: Applicant):
    row = pd.DataFrame([a.dict()])                      # 1) one-row table from the request
    row = pd.get_dummies(row, drop_first=True)          # 2) encode like training
    row = row.reindex(columns=TRAIN_COLUMNS, fill_value=0)  # 3) line columns up with training
    proba = float(model.predict_proba(row)[0][1])       # 4) probability of default
    return {"default_probability": round(proba, 3),
            "prediction": "high_risk" if proba > 0.5 else "low_risk"}