"""
test_model.py — automated checks that the model behaves sensibly.
Run from the project root:  pytest -q
"""
import json, joblib
import pandas as pd

model = joblib.load("model/model.pkl")
COLS = json.load(open("model/columns.json"))


def prob_of_default(applicant: dict) -> float:
    """Same logic as the API: encode -> align columns -> predict probability."""
    row = pd.DataFrame([applicant])
    row = pd.get_dummies(row, drop_first=True).reindex(columns=COLS, fill_value=0)
    return float(model.predict_proba(row)[0][1])


SAFE = {
    "person_age": 45, "person_income": 120000, "person_home_ownership": "MORTGAGE",
    "person_emp_length": 15.0, "loan_intent": "PERSONAL", "loan_grade": "A",
    "loan_amnt": 6000, "loan_int_rate": 7.2, "loan_percent_income": 0.05,
    "cb_person_default_on_file": "N", "cb_person_cred_hist_length": 18,
}
RISKY = {
    "person_age": 23, "person_income": 18000, "person_home_ownership": "RENT",
    "person_emp_length": 1.0, "loan_intent": "DEBTCONSOLIDATION", "loan_grade": "E",
    "loan_amnt": 12000, "loan_int_rate": 18.5, "loan_percent_income": 0.66,
    "cb_person_default_on_file": "Y", "cb_person_cred_hist_length": 2,
}


def test_probabilities_are_valid():
    # a probability must always be between 0 and 1
    for applicant in (SAFE, RISKY):
        p = prob_of_default(applicant)
        assert 0.0 <= p <= 1.0


def test_risky_scores_higher_than_safe():
    # the model should rate the risky applicant as more likely to default
    assert prob_of_default(RISKY) > prob_of_default(SAFE)