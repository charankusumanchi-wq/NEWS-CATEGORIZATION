import streamlit as st
import joblib
import pandas as pd

st.set_page_config(page_title="News Article Categorizer", page_icon="📰")
st.title("📰 News Article Categorization System")
st.write("Paste any news headline or article below to classify its topic.")

@st.cache_resource
def load_model():
    return joblib.load("news_model .pkl")

model = load_model()

article = st.text_area("Enter News Text:", height=150, placeholder="e.g., NASA launches new space telescope to study distant exoplanets...")

if st.button("Classify Article"):
    if article.strip():
        prediction = model.predict([article])[0]
        probs = model.predict_proba([article])[0]
        
        st.success(f"**Predicted Category:** {prediction}")
        
        prob_df = pd.DataFrame({
            "Category": model.classes_,
            "Confidence (%)": [round(p * 100, 2) for p in probs]
        }).set_index("Category")
        st.bar_chart(prob_df)
    else:
        st.warning("Please enter some text first.")