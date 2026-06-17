import json, joblib
import pandas as pd
import streamlit as st

model = joblib.load("model/model.pkl")
TRAIN_COLUMNS = json.load(open("model/columns.json"))

st.set_page_config(page_title="Loan Default Predictor", page_icon="💰")
st.title("💰 Loan Default Predictor")
st.write("Pick a sample applicant below, or tweak any field yourself.")

# --- default values (the "Custom" starting point) -------------------------
DEFAULTS = {
    "person_age": 30, "person_income": 55000, "person_home_ownership": "RENT",
    "person_emp_length": 5.0, "loan_intent": "PERSONAL", "loan_grade": "B",
    "loan_amnt": 10000, "loan_int_rate": 11.0, "cb_person_default_on_file": "N",
    "cb_person_cred_hist_length": 5,
}

# --- ready-made example applicants ----------------------------------------
PRESETS = {
    "🟢 Likely to repay": {
        "person_age": 45, "person_income": 120000, "person_home_ownership": "MORTGAGE",
        "person_emp_length": 15.0, "loan_intent": "PERSONAL", "loan_grade": "A",
        "loan_amnt": 6000, "loan_int_rate": 7.2, "cb_person_default_on_file": "N",
        "cb_person_cred_hist_length": 18,
    },
    "🟡 Borderline": {
        "person_age": 30, "person_income": 45000, "person_home_ownership": "RENT",
        "person_emp_length": 4.0, "loan_intent": "MEDICAL", "loan_grade": "C",
        "loan_amnt": 9000, "loan_int_rate": 13.5, "cb_person_default_on_file": "N",
        "cb_person_cred_hist_length": 6,
    },
    "🔴 Likely to default": {
        "person_age": 23, "person_income": 18000, "person_home_ownership": "RENT",
        "person_emp_length": 1.0, "loan_intent": "DEBTCONSOLIDATION", "loan_grade": "E",
        "loan_amnt": 12000, "loan_int_rate": 18.5, "cb_person_default_on_file": "Y",
        "cb_person_cred_hist_length": 2,
    },
}

# seed the form with defaults the first time the app loads
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)

# clicking a preset button copies its values into the form (then the app reruns)
def apply_preset(name):
    for k, v in PRESETS[name].items():
        st.session_state[k] = v

st.write("**Quick fill — try a sample applicant:**")
cols = st.columns(len(PRESETS))
for i, name in enumerate(PRESETS):
    cols[i].button(name, on_click=apply_preset, args=(name,), use_container_width=True)

st.divider()

# --- the form (widgets read/write st.session_state via key=) --------------
c1, c2 = st.columns(2)
with c1:
    st.number_input("Age", 18, 100, key="person_age")
    st.number_input("Annual income ($)", 4000, 500000, step=1000, key="person_income")
    st.number_input("Employment length (years)", 0.0, 60.0, key="person_emp_length")
    st.number_input("Loan amount ($)", 500, 40000, step=500, key="loan_amnt")
    st.number_input("Interest rate (%)", 5.0, 25.0, key="loan_int_rate")
with c2:
    st.selectbox("Home ownership", ["RENT", "MORTGAGE", "OWN", "OTHER"], key="person_home_ownership")
    st.selectbox("Loan intent",
        ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"],
        key="loan_intent")
    st.selectbox("Loan grade", ["A", "B", "C", "D", "E", "F", "G"], key="loan_grade")
    st.selectbox("Prior default on file?", ["N", "Y"], key="cb_person_default_on_file")
    st.number_input("Credit history length (years)", 1, 30, key="cb_person_cred_hist_length")

loan_percent_income = round(st.session_state.loan_amnt / st.session_state.person_income, 3) \
    if st.session_state.person_income else 0.0
st.caption(f"Loan as % of income (auto-calculated): {loan_percent_income}")

# --- predict --------------------------------------------------------------
if st.button("Predict default risk", type="primary"):
    row = pd.DataFrame([{
        "person_age": st.session_state.person_age,
        "person_income": st.session_state.person_income,
        "person_home_ownership": st.session_state.person_home_ownership,
        "person_emp_length": st.session_state.person_emp_length,
        "loan_intent": st.session_state.loan_intent,
        "loan_grade": st.session_state.loan_grade,
        "loan_amnt": st.session_state.loan_amnt,
        "loan_int_rate": st.session_state.loan_int_rate,
        "loan_percent_income": loan_percent_income,
        "cb_person_default_on_file": st.session_state.cb_person_default_on_file,
        "cb_person_cred_hist_length": st.session_state.cb_person_cred_hist_length,
    }])
    row = pd.get_dummies(row, drop_first=True).reindex(columns=TRAIN_COLUMNS, fill_value=0)
    proba = float(model.predict_proba(row)[0][1])

    st.metric("Default probability", f"{proba:.1%}")
    if proba > 0.5:
        st.error("⚠️ High risk")
    else:
        st.success("✅ Low risk")