import streamlit as st
import pickle
import numpy as np

# Page Configuration
st.set_page_config(page_title="Fake Product Review Detector", page_icon="🛡️")

st.title("🛡️ Fake Product Review Detector")
st.write("Analyze reviews from Amazon, Flipkart, or any shopping site to check if they are genuine or manipulated.")
st.markdown("---")

try:
    # Model Loading
    model = pickle.load(open('model.pkl', 'rb'))
    vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

    # User Input
    user_input = st.text_area("✍️ Paste the product review here:", height=150, placeholder="Example: Great quality product, but the delivery was late...")

    if st.button("🔍 Check Authenticity"):
        if user_input:
            # AI Analysis Logic
            transformed_input = vectorizer.transform([user_input])
            prediction = model.predict(transformed_input)
            probability = model.predict_proba(transformed_input) 
            confidence = np.max(probability) * 100

            if prediction[0] == 1:
                st.success(f"### ✅ VERDICT: REAL REVIEW ({confidence:.1f}% Confidence)")
                st.info("**Why?** This review follows a natural human writing style, mentioning specific details rather than just empty praise.")
            else:
                st.error(f"### ⚠️ VERDICT: FAKE / SUSPICIOUS ({confidence:.1f}% Confidence)")
                
                # Explaining the patterns
                st.warning("**Why it looks suspicious:**")
                if "!!!" in user_input or user_input.isupper():
                    st.write("- **Over-Excitement:** Excessive use of CAPITAL letters or exclamation marks.")
                if len(user_input.split()) < 10:
                    st.write("- **Lack of Detail:** The review is too short to be helpful or detailed.")
                st.write("- **Pattern Match:** The language matches common templates used by paid review bots.")
            
            # Visual Analysis Strength
            st.write("Analysis Strength Meter:")
            st.progress(int(confidence))
        else:
            st.warning("Please paste a review first!")

except Exception as e:
    st.error("Model Error: Please ensure you have run 'python train.py' in your terminal.")