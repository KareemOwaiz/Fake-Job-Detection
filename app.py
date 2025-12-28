import streamlit as st
import joblib

st.set_page_config(page_title="Fake Job Detection", layout="centered")

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.title("🕵️ Fake Job Posting Detection")
st.write("Enter a job description to check if it is fake or genuine.")

text = st.text_area("Job Description")

if st.button("Predict"):
    if text.strip() == "":
        st.warning("Please enter a job description")
    else:
        X = vectorizer.transform([text])
        prediction = model.predict(X)[0]

        if prediction == 1:
            st.error("🚨 This job posting is FAKE")
        else:
            st.success("✅ This job posting is REAL")
