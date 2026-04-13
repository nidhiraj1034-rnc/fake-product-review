import streamlit as st
import pickle

st.title("🛡️ Fake Product Review Detector")

try:
    model = pickle.load(open('model.pkl', 'rb'))
    vector = pickle.load(open('vectorizer.pkl', 'rb'))

    user_input = st.text_area("Enter Review:")
    if st.button("Check"):
        data = vector.transform([user_input])
        prediction = model.predict(data)
        if prediction[0] == 1:
            st.success("Real Review")
        else:
            st.error("Fake Review")
except:
    st.warning("Please run train.py first to generate model files.")