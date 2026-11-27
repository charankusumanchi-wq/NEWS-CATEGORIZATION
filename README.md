# NEWS-CATEGORIZATION
# 📰 News Article Categorization Using Machine Learning

A supervised multi-class NLP text classification system that automatically categorizes news headlines and articles into topics using **TF-IDF Vectorization** and **Logistic Regression**, deployed with an interactive **Streamlit** web interface.

## Tech Stack
- **Language:** Python 3
- **Machine Learning & NLP:** Scikit-Learn (`TfidfVectorizer`, `LogisticRegression`, `Pipeline`)
- **Web Interface:** Streamlit
- **Visualization:** Matplotlib, Seaborn

## Project Structure
- `News_Categorization.ipynb` — Dataset loading, TF-IDF feature extraction, model training, and confusion matrix evaluation.
- `app.py` — Streamlit web application for live news categorization with confidence scores.
- `news_model.pkl` — Serialized Scikit-Learn pipeline (`TF-IDF` + `LogisticRegression`).
- `confusion_matrix.png` — Model evaluation heatmap on unseen test data.
- `requirements.txt` — Python dependencies.

## How to Run Locally
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
