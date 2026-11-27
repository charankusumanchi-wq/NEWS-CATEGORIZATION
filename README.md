# 📰 News Article Categorization Using Machine Learning

An end-to-end Natural Language Processing (NLP) web application that classifies news headlines and articles into four categories—**World**, **Sports**, **Business**, and **Sci/Tech**—using a Scikit-Learn `TF-IDF` + `Logistic Regression` pipeline deployed with **Streamlit**.

---

## ✨ Key Features
- **Instant Multi-Class Classification:** Categorizes news text across 4 major domains (`World`, `Sports`, `Business`, `Sci/Tech`).
- **Confidence Breakdown:** Displays probability distribution across all four categories via an interactive bar chart.
- **Keyword Explainability:** Extracts and highlights the highest-weighted TF-IDF terms driving each prediction.
- **One-Click Sample Testing:** Built-in sample headlines allow instant testing without manual typing.
- **Leakage-Free Pipeline:** Combines text vectorization and classification inside a single serialized Scikit-Learn `Pipeline`.

---

## 🛠️ Tech Stack
- **Language:** Python 3
- **Machine Learning & NLP:** Scikit-Learn (`TfidfVectorizer`, `LogisticRegression`, `Pipeline`), NumPy, Pandas
- **Web Framework:** Streamlit
- **Model Serialization:** Joblib

---

## 📂 Repository Structure
```text
├── app.py                        # Interactive Streamlit web application
├── News_Categorization.ipynb     # Data preprocessing, model training & evaluation notebook
├── news_model.pkl                # Serialized TF-IDF + Logistic Regression pipeline
├── confusion_matrix.png          # Model evaluation confusion matrix
├── requirements.txt              # Project dependencies
└── README.md                     # Project documentation
