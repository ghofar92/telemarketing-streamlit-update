import streamlit as st
import pandas as pd
import joblib

# ===============================
# PAGE CONFIG (HARUS PALING ATAS)
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
# UI HEADER
# ===============================
st.title("📞 Telemarketing Campaign Prediction")
st.write("Prediksi probabilitas nasabah **subscribe Term Deposit**")

st.divider()

# ===============================
# INPUT USER (YANG PENTING SAJA)
# ===============================
age = st.number_input("Age", min_value=18, max_value=100, value=35)

job = st.selectbox("Job", [
    "admin.", "blue-collar", "technician", "services",
    "management", "retired", "self-employed", "student",
    "unemployed", "entrepreneur", "housemaid", "unknown"
])

marital = st.selectbox("Marital Status", [
    "married", "single", "divorced"
])

education_label_to_value = {
    "Basic 4y": "basic.4y",
    "Basic 6y": "basic.6y",
    "Basic 9y": "basic.9y",
    "High School": "high.school",
    "Professional Course": "professional.course",
    "University Degree": "university.degree",
    "Unknown": "unknown"
}

education_label = st.selectbox(
    "Education",
    list(education_label_to_value.keys())
)
education = education_label_to_value[education_label]

housing = st.selectbox("Housing Loan", ["yes", "no"])
loan = st.selectbox("Personal Loan", ["yes", "no"])

# OPTIONAL (untuk eksperimen pengaruh durasi)
duration = st.number_input(
    "Call Duration (seconds)",
    min_value=0,
    value=120,
    help="Durasi panggilan (fitur penting)"
)

# ===============================
# BUILD INPUT DATAFRAME
# ===============================
input_df = pd.DataFrame([{
    # INPUT USER
    "age": age,
    "job": job,
    "marital": marital,
    "education": education,
    "housing": housing,
    "loan": loan,
    "duration": duration,

    # DEFAULT VALUE (TIDAK DITAMPILKAN)
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
    "nr.employed": 5191,
    "generation": "Other"
}])

# ===============================
# PREDICT
# ===============================
if st.button("🔍 Predict"):
    try:
        # WAJIB: preprocessing dulu
        X_processed = preprocess.transform(input_df)

        pred = model.predict(X_processed)[0]
        prob = model.predict_proba(X_processed)[0][1]

        st.subheader("📊 Prediction Result")

        if pred == 1:
            st.success(f"✅ **SUBSCRIBE**\n\nProbabilitas: **{prob:.2%}**")
        else:
            st.error(f"❌ **NOT SUBSCRIBE**\n\nProbabilitas: **{prob:.2%}**")

    except Exception as e:
        st.error("❌ Terjadi error saat prediksi")
        st.exception(e)
