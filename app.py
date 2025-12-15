import streamlit as st
import pandas as pd
import joblib

# ===============================
# PAGE CONFIG (HARUS PALING ATAS)
# ===============================
st.set_page_config(
    page_title="📞 Telemarketing Prediction",
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
# UI
# ===============================
st.title("📞 Telemarketing Campaign Prediction")
st.write("Prediksi probabilitas nasabah subscribe **Term Deposit**")

# ===============================
# USER INPUT (SIMPLE)
# ===============================
age = st.number_input("Age", 18, 100, 35)

job = st.selectbox("Job", [
    "admin", "blue-collar", "technician", "services",
    "management", "retired", "self-employed",
    "student", "unemployed", "entrepreneur"
])

marital = st.selectbox("Marital Status", ["Married", "Single", "Divorced"])
education = st.selectbox("Education", [
    "Basic 4y", "Basic 6y", "Basic 9y",
    "High School", "Professional Course",
    "University Degree", "Unknown"
])

duration = st.number_input("Call Duration (seconds)", 0, 5000, 120)

# ===============================
# MAPPING (UI → MODEL)
# ===============================
education_map = {
    "Basic 4y": "basic.4y",
    "Basic 6y": "basic.6y",
    "Basic 9y": "basic.9y",
    "High School": "high.school",
    "Professional Course": "professional.course",
    "University Degree": "university.degree",
    "Unknown": "unknown"
}

marital = marital.lower()

# ===============================
# INPUT DATAFRAME (RAW)
# ===============================
input_df = pd.DataFrame([{
    "age": age,
    "job": job,
    "marital": marital,
    "education": education_map[education],
    "duration": duration,

    # DEFAULT (TIDAK DITAMPILKAN)
    "campaign": 1,
    "default": "no",
    "housing": "no",
    "loan": "no",
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
# PREDICTION
# ===============================
if st.button("Predict"):
    X_encoded = preprocess.transform(input_df)

    prob = model.predict_proba(X_encoded)[0][1]

    st.subheader("📊 Prediction Result")

    if prob >= 0.5:
        st.success(f"✅ Subscribe Probability: **{prob:.2%}**")
    else:
        st.warning(f"❌ Not Subscribe Probability: **{prob:.2%}**")
