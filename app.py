# ============================================
# 📌 Streamlit NLP Phase-wise with All Models (Enhanced UI)
# ============================================

import streamlit as st
import pandas as pd
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from textblob import TextBlob

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
import plotly.express as px

# ============================
# Load SpaCy & Globals
# ============================
nlp = spacy.load("en_core_web_sm")
stop_words = STOP_WORDS

# ============================
# Phase Feature Extractors
# ============================
def lexical_preprocess(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if token.text not in stop_words and token.is_alpha]
    return " ".join(tokens)

def syntactic_features(text):
    doc = nlp(text)
    return " ".join([token.pos_ for token in doc])

def semantic_features(text):
    blob = TextBlob(text)
    return [blob.sentiment.polarity, blob.sentiment.subjectivity]

def discourse_features(text):
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]
    return f"{len(sentences)} {' '.join([s.split()[0] for s in sentences if len(s.split()) > 0])}"

pragmatic_words = ["must", "should", "might", "could", "will", "?", "!"]
def pragmatic_features(text):
    text = text.lower()
    return [text.count(w) for w in pragmatic_words]

# ============================
# Train & Evaluate All Models
# ============================
def evaluate_models(X_features, y):
    results = {}
    models = {
        "Naive Bayes": MultinomialNB(),
        "Decision Tree": DecisionTreeClassifier(),
        "Logistic Regression": LogisticRegression(max_iter=200),
        "SVM": SVC()
    }

    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y, test_size=0.2, random_state=42
    )

    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred) * 100
            results[name] = round(acc, 2)
        except Exception as e:
            results[name] = None

    return results

# ============================
# Streamlit Enhanced UI
# ============================
st.set_page_config(page_title="🧠 NLP Phase-wise Analysis", layout="wide")

st.title("🧠 Phase-wise NLP Analysis with Model Comparison")
st.markdown(
    """
    <div style="background-color:#f0f2f6;padding:10px;border-radius:10px;">
    Upload your dataset and evaluate ML models across different **linguistic phases**  
    *(Lexical, Syntactic, Semantic, Discourse, Pragmatic)*.  
    </div>
    """,
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader("📂 Upload a CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    st.subheader("📊 Data Preview")
    st.dataframe(df.head(), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        text_col = st.selectbox("📝 Select Text Column:", df.columns)
    with col2:
        target_col = st.selectbox("🎯 Select Target Column:", df.columns)

    phase = st.radio(
        "🔎 Select NLP Phase:",
        ["Lexical & Morphological", "Syntactic", "Semantic", "Discourse", "Pragmatic"],
        horizontal=True
    )

    if st.button("🚀 Run Comparison", use_container_width=True, type="primary"):
        X = df[text_col].astype(str)
        y = df[target_col]

        if phase == "Lexical & Morphological":
            X_processed = X.apply(lexical_preprocess)
            X_features = CountVectorizer().fit_transform(X_processed)

        elif phase == "Syntactic":
            X_processed = X.apply(syntactic_features)
            X_features = CountVectorizer().fit_transform(X_processed)

        elif phase == "Semantic":
            X_features = pd.DataFrame(
                X.apply(semantic_features).tolist(),
                columns=["polarity", "subjectivity"]
            )

        elif phase == "Discourse":
            X_processed = X.apply(discourse_features)
            X_features = CountVectorizer().fit_transform(X_processed)

        elif phase == "Pragmatic":
            X_features = pd.DataFrame(
                X.apply(pragmatic_features).tolist(),
                columns=pragmatic_words
            )

        # Run models
        results = evaluate_models(X_features, y)

        # Convert results to DataFrame
        results_df = pd.DataFrame(
            [{"Model": m, "Accuracy": acc} for m, acc in results.items()]
        )
        results_df = results_df.dropna().sort_values(by="Accuracy", ascending=False)

        st.subheader("📈 Model Comparison Results")
        st.dataframe(results_df, use_container_width=True)

        # Interactive bar chart with Plotly
        fig = px.bar(
            results_df,
            x="Model",
            y="Accuracy",
            color="Accuracy",
            text=results_df["Accuracy"].astype(str) + "%",
            title=f"Model Performance on {phase}",
            height=500
        )
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)
