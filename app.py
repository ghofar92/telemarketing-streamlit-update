import streamlit as st
import pandas as pd
import joblib

# ===============================
# PAGE CONFIG (WAJIB PALING ATAS)
# ===============================
st.set_page_config(
    page_title="Telemarketing Lead Scoring",
    page_icon="📞",
    layout="centered"
)

# ===============================
# LOAD MODEL & PREPROCESSOR
# ===============================
@st.cache_resource
def load_assets():
    model = joblib.load("model_telemarketing_campaign.sav")
    preprocess = joblib.load("preprocess.pkl")
    return model, preprocess

model, preprocess = load_assets()

# ===============================
# UI - HEADER
# ===============================
st.title("📞 Telemarketing Lead Scoring")
st.write("Prediksi kemungkinan nasabah **subscribe Term Deposit**")

st.divider()

# ===============================
# INPUT NASABAH (MINIMAL & CLEAN)
# ===============================
st.subheader("🧾 Data Nasabah")

age = st.number_input("Age", min_value=18, max_value=100, value=30)

job_label_map = {
    "Admin": "admin.",
    "Blue Collar": "blue-collar",
    "Technician": "technician",
    "Services": "services",
    "Management": "management",
    "Retired": "retired",
    "Student": "student",
    "Entrepreneur": "entrepreneur",
    "Unemployed": "unemployed"
}
job_label = st.selectbox("Job", list(job_label_map.keys()))
job = job_label_map[job_label]

marital = st.selectbox(
    "Marital Status",
    ["Single", "Married", "Divorced"]
).lower()

education_label_map = {
    "Basic 4 Years": "basic.4y",
    "Basic 6 Years": "basic.6y",
    "Basic 9 Years": "basic.9y",
    "High School": "high.school",
    "Professional Course": "professional.course",
    "University Degree": "university.degree",
    "Unknown": "unknown"
}
education_label = st.selectbox("Education", list(education_label_map.keys()))
education = education_label_map[education_label]

balance = st.number_input("Balance (€)", min_value=0, value=1000)

housing = st.selectbox("Housing Loan", ["No", "Yes"]).lower()
loan = st.selectbox("Personal Loan", ["No", "Yes"]).lower()

duration = st.number_input(
    "Call Duration (seconds)",
    min_value=0,
    value=120,
    help="Durasi panggilan. Biasanya sangat berpengaruh pada hasil prediksi."
)

# ===============================
# BUILD INPUT DATAFRAME
# ===============================
input_df = pd.DataFrame([{
    "age": age,
    "job": job,
    "marital": marital,
    "education": education,
    "balance": balance,
    "housing": housing,
    "loan": loan,
    "duration": duration,

    # ===== DEFAULT VALUE (TIDAK DITAMPILKAN DI UI) =====
    "campaign": 1,
    "default": "no",
    "contact": "cellular",
    "day_of_week": "mon",
    "month": "may",
    "poutcome": "unknown",
    "pdays": -1,
    "previous": 0,
    "emp.var.rate": 1.1,
    "cons.price.idx": 93.994,
    "cons.conf.idx": -36.4,
    "euribor3m": 4.857,
    "nr.employed": 5191
}])

# ===============================
# PREDICT
# ===============================
st.divider()

if st.button("🔍 Predict"):
    # 🔑 WAJIB TRANSFORM DULU
    X_encoded = preprocess.transform(input_df)

    pred_label = model.predict(X_encoded)[0]
    pred_prob = model.predict_proba(X_encoded)[0, 1]

    st.subheader("📊 Prediction Result")

    if pred_label == 1:
        st.success(f"✅ **Subscribe**\n\nProbability: **{pred_prob:.2%}**")
    else:
        st.error(f"❌ **Not Subscribe**\n\nProbability: **{pred_prob:.2%}**")
