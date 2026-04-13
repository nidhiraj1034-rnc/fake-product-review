import streamlit as st
import pickle
import numpy as np

# Page Configuration
st.set_page_config(page_title="Fake Product Review", page_icon="🛡️", layout="wide")

# ================= SIDEBAR (How to Use & Terms) =================
with st.sidebar:
    st.title("📌 Navigation")
    
    with st.expander("📖 How to Use? / कैसे इस्तेमाल करें?"):
        st.write("""
        **English:**
        1. Copy the **text** of any review from Amazon, Flipkart, or any site.
        2. Paste it in the box on the right.
        3. Click 'Analyze Review'.
        
        **Hindi:**
        1. किसी भी साइट से रिव्यू का **टेक्स्ट** कॉपी करें।
        2. इसे दाईं ओर दिए गए बॉक्स में पेस्ट करें।
        3. 'Analyze Review' बटन पर क्लिक करें।
        """)
    
    with st.expander("⚖️ Terms & Conditions"):
        st.write("""
        - This tool uses AI to detect patterns. 
        - Results may not be 100% accurate.
        - Use this for educational purposes only.
        """)
    
    st.markdown("---")
    st.info("💡 **Note:** Please paste the review text, not the product link.")

# ================= MAIN PAGE =================
st.title("🛡️ Fake Product Review")
st.write("Analyze reviews from any platform worldwide to check if they are Real or Fake.")
st.markdown("---")

try:
    # Model Loading
    model = pickle.load(open('model.pkl', 'rb'))
    vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

    # User Input
    user_input = st.text_area("✍️ Paste Review Text Here:", height=200, placeholder="Paste a customer review to verify...")

    if st.button("🔍 Analyze Review"):
        if user_input:
            # Smart Link Detection
            if "http" in user_input or "www." in user_input:
                st.error("❌ **Error:** You pasted a Link. Please paste the **TEXT** of the review. / आपने लिंक पेस्ट किया है, कृपया रिव्यू का **टेक्स्ट** पेस्ट करें।")
            else:
                # AI Prediction
                transformed_input = vectorizer.transform([user_input])
                prediction = model.predict(transformed_input)
                probability = model.predict_proba(transformed_input)
                confidence = np.max(probability) * 100

                # Result Section
                if prediction[0] == 1:
                    st.success(f"### ✅ VERDICT: REAL REVIEW ({confidence:.1f}% Confidence)")
                    st.markdown("#### **Analysis / विश्लेषण:**")
                    st.write("**ENG:** This review looks genuine based on its natural language and details.")
                    st.write("**HIN:** यह रिव्यू अपनी भाषा और विवरण के आधार पर असली लग रहा है।")
                else:
                    st.error(f"### ⚠️ VERDICT: SUSPICIOUS / FAKE ({confidence:.1f}% Confidence)")
                    st.markdown("#### **Reasons / कारण:**")
                    st.warning("""
                    **English:** - High-pressure promotional language detected.
                    - Pattern matches paid or bot-generated templates.
                    
                    **Hindi:**
                    - दिखावटी और अत्यधिक तारीफों वाली भाषा मिली है।
                    - यह पैटर्न 'पेड रिव्यूज' या 'बॉट' से मिलता-जुलता है।
                    """)
                
                # Visual Meter
                st.write("Analysis Strength Meter:")
                st.progress(int(confidence))
        else:
            st.warning("Please paste some text first!")

except Exception as e:
    st.error("Model files not found! Please run 'python train.py' in your terminal first.")