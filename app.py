import streamlit as st
import pickle

# Page setting
st.set_page_config(page_title="Smart Review Detector", page_icon="🛍️")

st.title("🛍️ Amazon & Flipkart Review Checker")
st.write("Flipkart ya Amazon se review copy karein aur niche check karein.")

# Model load karna
try:
    model = pickle.load(open('model.pkl', 'rb'))
    vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

    user_input = st.text_area("Paste Review Here:", placeholder="Example: The product is very good...")

    if st.button("Check Authenticity"):
        if user_input:
            # Prediction logic
            data = vectorizer.transform([user_input])
            prediction = model.predict(data)
            
            if prediction[0] == 1:
                st.success("✅ REAL: This review looks genuine.")
            else:
                st.error("⚠️ FAKE: This review looks suspicious or paid.")
        else:
            st.warning("Please paste a review first!")
except:
    st.error("Model files not found! Please run 'python train.py' in terminal first.")