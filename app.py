import streamlit as st
import joblib
import numpy as np
import pandas as pd

st.set_page_config(
    page_title="News Article Categorizer",
    page_icon="📰",
    layout="centered"
)

@st.cache_resource
def load_model():
    return joblib.load("news_model.pkl")

try:
    model = load_model()
except Exception as e:
    st.error(f"Could not load 'news_model.pkl'. Make sure the filename has no extra spaces. Details: {e}")
    st.stop()

CATEGORY_META = {
    "World": {"icon": "🌍", "desc": "International affairs, diplomacy, and global news"},
    "Sports": {"icon": "🏆", "desc": "Athletics, tournaments, teams, and player updates"},
    "Business": {"icon": "💼", "desc": "Markets, finance, corporate news, and economy"},
    "Sci/Tech": {"icon": "💻", "desc": "Technology, software, scientific research, and space"},
    0: {"icon": "🌍", "desc": "International affairs, diplomacy, and global news"},
    1: {"icon": "🏆", "desc": "Athletics, tournaments, teams, and player updates"},
    2: {"icon": "💼", "desc": "Markets, finance, corporate news, and economy"},
    3: {"icon": "💻", "desc": "Technology, software, scientific research, and space"},
}

LABEL_NAMES = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}

SAMPLE_ARTICLES = {
    "Select a sample headline...": "",
    "🌍 World News": "United Nations diplomats gathered in Geneva today to negotiate a new international climate accord and humanitarian aid framework.",
    "🏆 Sports News": "The championship final went into overtime as the star striker scored a dramatic late goal to secure the league trophy.",
    "💼 Business News": "Global stock markets rallied after the central bank signaled steady interest rates and major tech firms reported strong quarterly earnings.",
    "💻 Sci/Tech News": "Researchers unveiled a new quantum computing processor capable of reducing error rates while running complex machine learning algorithms."
}

def get_top_keywords(pipeline, text, class_idx, top_n=6):
    try:
        vectorizer = pipeline.steps[0][1]
        classifier = pipeline.steps[-1][1]
        if not hasattr(classifier, "coef_"):
            return []
        feature_names = np.array(vectorizer.get_feature_names_out())
        tfidf_vec = vectorizer.transform([text]).toarray()[0]
        weights = classifier.coef_[class_idx] if classifier.coef_.shape[0] > 1 else classifier.coef_[0]
        contributions = tfidf_vec * weights
        top_indices = np.argsort(contributions)[::-1]
        return [
            feature_names[i]
            for i in top_indices
            if contributions[i] > 0
        ][:top_n]
    except Exception:
        return []

st.title("📰 News Article Categorizer")
st.markdown("Classify news headlines and articles into **World**, **Sports**, **Business**, or **Sci/Tech** with confidence scores and keyword insights.")

selected_sample = st.selectbox("Try a quick sample or paste your own text below:", list(SAMPLE_ARTICLES.keys()))

if "article_input" not in st.session_state:
    st.session_state.article_input = ""

if selected_sample != "Select a sample headline...":
    st.session_state.article_input = SAMPLE_ARTICLES[selected_sample]

user_text = st.text_area(
    "News Headline or Article Text:",
    value=st.session_state.article_input,
    height=150,
    placeholder="Paste any news headline or paragraph here..."
)

col1, col2 = st.columns([1, 4])
with col1:
    predict_btn = st.button("Classify", type="primary", use_container_width=True)
with col2:
    if st.button("Clear"):
        st.session_state.article_input = ""
        st.rerun()

if predict_btn:
    if not user_text.strip():
        st.warning("Please enter a news headline or article text first.")
    else:
        raw_pred = model.predict([user_text])[0]
        pred_label = LABEL_NAMES.get(raw_pred, str(raw_pred))
        meta = CATEGORY_META.get(raw_pred, {"icon": "📄", "desc": ""})

        st.divider()
        st.subheader(f"{meta['icon']} Predicted Category: **{pred_label}**")
        if meta["desc"]:
            st.caption(meta["desc"])

        classes = list(model.classes_)
        class_idx = classes.index(raw_pred)

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba([user_text])[0]
            readable_classes = [LABEL_NAMES.get(c, str(c)) for c in classes]
            prob_df = pd.DataFrame({
                "Category": readable_classes,
                "Confidence (%)": np.round(probs * 100, 2)
            }).sort_values("Confidence (%)", ascending=False)

            st.metric("Model Confidence", f"{probs[class_idx] * 100:.1f}%")
            st.bar_chart(prob_df.set_index("Category"))

        keywords = get_top_keywords(model, user_text, class_idx)
        if keywords:
            st.markdown("**Top Influential Keywords Detected:**")
            st.write(" • ".join([f"`{word}`" for word in keywords]))
