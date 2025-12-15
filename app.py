import streamlit as st
import pandas as pd
import joblib

# ==================================================
# PAGE CONFIG (HARUS PALING ATAS)
# ==================================================
st.set_page_config(
    page_title="Telemarketing Lead Scoring",
    page_icon="📞",
    layout="centered"
)

# ==================================================
# LOAD MODEL
# ==================================================
@st.cache_resource
def load_model():
    return joblib.load("model_telemarketing_campaign.sav")

model = load_model()

# ==================================================
# TITLE
# ==================================================
st.title("📞 Telemarketing Lead Scoring")
st.write("Prediksi kemungkinan nasabah **subscribe Term Deposit**")

st.divider()

# ==================================================
# INPUT NASABAH (MINIMAL & PENTING)
# ==================================================
st.subheader("🧾 Data Nasabah")

# ---- Numeric ----
age = st.number_input("Usia", min_value=18, max_value=100, value=35)
balance = st.number_input(
    "Tabungan / Balance (€)",
    value=1000,
)

# ---- Job (label → value model) ----
job_label_to_value = {
    "Admin": "admin.",
    "Blue Collar": "blue-collar",
    "Technician": "technician",
    "Services": "services",
    "Management": "management",
    "Retired": "retired",
    "Entrepreneur": "entrepreneur",
    "Self Employed": "self-employed",
    "Student": "student",
    "Unemployed": "unemployed",
    "Unknown": "unknown"
}

job_label = st.selectbox("Pekerjaan", list(job_label_to_value.keys()))
job = job_label_to_value[job_label]

# Marital Status (Kapital di UI)
marital_map = {
    "Married": "married",
    "Single": "single",
    "Divorced": "divorced"
}
marital_label = st.selectbox("Marital Status", list(marital_map.keys()))
marital = marital_map[marital_label]

# Education (tanpa titik di UI)
education_map = {
    "Basic 4y": "basic.4y",
    "Basic 6y": "basic.6y",
    "Basic 9y": "basic.9y",
    "High School": "high.school",
    "Professional Course": "professional.course",
    "University Degree": "university.degree",
    "Unknown": "unknown"
}
education_label = st.selectbox("Education", list(education_map.keys()))
education = education_map[education_label]

housing = st.selectbox("Housing Loan", ["No", "Yes"])
housing = housing.lower()

loan = st.selectbox("Personal Loan", ["No", "Yes"])
loan = loan.lower()

# ===============================
# DATAFRAME INPUT (MODEL FORMAT)
# ===============================
input_df = pd.DataFrame([{
    # Input utama
    "age": age,
    "job": job,
    "marital": marital,
    "education": education,
    "balance": balance,
    "housing": housing,
    "loan": loan,

    # ===============================
    # DEFAULT VALUE (TIDAK DITAMPILKAN)
    # ===============================
    "duration": 120,
    "campaign": 1,
    "default": "no",
    "contact": "cellular",
    "day_of_week": "mon",
    "month": "may",
    "poutcome": "nonexistent",
    "pdays": 999,
    "previous": 0,
    "emp.var.rate": 1.1,
    "cons.price.idx": 93.994,
    "cons.conf.idx": -36.4,
    "euribor3m": 4.857,
    "nr.employed": 5191
}])

# ===============================
# PREDICTION
# ===============================
st.divider()

if st.button("🔍 Predict"):
    pred_label = model.predict(input_df)[0]
    pred_prob = model.predict_proba(input_df)[0][1]

    st.subheader("📌 Prediction Result")

    if pred_label == 1:
        st.success(f"✅ **Subscribe**\n\nProbability: **{pred_prob:.2%}**")
    else:
        st.error(f"❌ **Not Subscribe**\n\nProbability: **{pred_prob:.2%}**")
