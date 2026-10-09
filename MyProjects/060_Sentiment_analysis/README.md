# Customer Feedback & Review Triage System

An end-to-end machine learning project designed to process raw customer reviews, classify sentiment, and isolate specific operational issues from negative feedback.

---

## Overview

E-commerce businesses handle thousands of customer reviews every week. Reading every single comment manually is inefficient, and critical complaints regarding battery defects, late deliveries, or damaged goods often get buried under positive feedback.

This project solves that problem through a simple two-tier pipeline:
1. It screens incoming reviews and classifies them as Positive or Negative using an optimized text classification model.
2. If a review is negative, it immediately tags the core issue into categories like Hardware and Battery, Shipping and Logistics, Build Quality, or Customer Support.
3. The results are served through a clean Streamlit dashboard where teams can test and review individual customer comments in real time.

---

## Performance Summary

During experiments, we compared a standard baseline model against a tuned support vector classifier.

- Baseline Model (Multinomial Naive Bayes): 82.19% accuracy
- Final Model (LinearSVC with tuned hyperparameters): 86.69% accuracy
- Representation: TF-IDF vectorization with unigrams, bigrams, and trigrams alongside sublinear term frequency scaling.

Why LinearSVC performed better:
High-dimensional sparse text data works particularly well with linear decision boundaries. By including phrase patterns (n-grams up to 3 words) and dampening repeated common words with sublinear scaling, LinearSVC separated subtle negative phrases much more reliably than Naive Bayes.

---

## Tech Stack

- Programming: Python
- Data Processing: Pandas, NumPy, Regular Expressions
- Machine Learning & NLP: Scikit-learn (TfidfVectorizer, LinearSVC, GridSearchCV)
- Model Serialization: Joblib
- Web Framework: Streamlit
- Deployment: Streamlit Community Cloud

---

## Project Structure

```text
060_Sentiment_analysis/
│
├── app.py                      # Streamlit application file
├── requirements.txt            # Python dependencies
├── best_sentiment_model.pkl    # Serialized LinearSVC model
├── best_tfidf_vectorizer.pkl   # Serialized TF-IDF vectorizer
└── README.md                   # Documentation