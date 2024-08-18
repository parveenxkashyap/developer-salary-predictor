import streamlit as st

from utils import canonicalize_employment, predict_salary

st.set_page_config(page_title="Developer Salary Predictor", layout="centered")

st.title("Developer Salary Prediction App")
st.write("Estimate your expected annual salary (USD) based on your profile.")

EDUCATION_LEVELS = [
    "Bachelor’s degree (B.A., B.S., B.Eng., etc.)",
    "Master’s degree (M.A., M.S., M.Eng., MBA, etc.)",
    "Professional degree (JD, MD, etc.)",
    "Associate degree (A.A., A.S., etc.)",
    "Some college/university study without earning a degree",
    "Secondary school (e.g. American high school, German Realschule or Gymnasium, etc.)",
    "Primary/elementary school",
    "I never completed any formal education",
    "Other",
]

COUNTRIES = [
    "Brazil",
    "Canada",
    "France",
    "Germany",
    "India",
    "Italy",
    "Netherlands",
    "Poland",
    "Spain",
    "Sweden",
    "Switzerland",
    "Ukraine",
    "United Kingdom of Great Britain and Northern Ireland",
    "United States of America",
    "Other",
]

EMPLOYMENT_OPTIONS = [
    "Employed, full-time",
    "Employed, part-time",
    "Independent contractor, freelancer, or self-employed",
    "Student, full-time",
    "Student, part-time",
    "Not employed, but looking for work",
    "Not employed, and not looking for work",
    "Retired",
]

REMOTE_OPTIONS = ["Remote", "In-person", "Other"]

ORG_SIZES = [
    "Just me - I am a freelancer, sole proprietor, etc.",
    "2 to 9 employees",
    "10 to 19 employees",
    "20 to 99 employees",
    "100 to 499 employees",
    "500 to 999 employees",
    "5,000 to 9,999 employees",
    "10,000 or more employees",
    "I don’t know",
    "Other",
]

with st.form("prediction_form"):
    st.subheader("Enter your details")

    years_exp = st.number_input(
        "Years of professional coding experience",
        min_value=0,
        max_value=50,
        value=1,
        step=1,
    )

    country = st.selectbox("Country", COUNTRIES, index=COUNTRIES.index("India"))

    education = st.selectbox("Education level", EDUCATION_LEVELS, index=0)

    employment = st.multiselect(
        "Employment status (you can select multiple)",
        EMPLOYMENT_OPTIONS,
        default=["Employed, full-time"],
    )

    remote = st.selectbox("Work arrangement", REMOTE_OPTIONS, index=0)

    org_size = st.selectbox("Organization size", ORG_SIZES, index=0)

    submitted = st.form_submit_button("Predict salary")

if submitted:
    employment_str = canonicalize_employment(employment)

    user_input = {
        "YearsCodePro": years_exp,
        "Country": country,
        "EdLevel": education,
        "Employment": employment_str,
        "RemoteWork": remote,
        "OrgSize": org_size,
    }

    try:
        salary = predict_salary(user_input)
        st.success(f"Estimated Annual Salary: ${salary:,.2f} USD")
    except FileNotFoundError as e:
        st.error(str(e))
        st.info("Tip: put the trained model at Models/reg_model.pkl or set MODEL_PATH.")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
