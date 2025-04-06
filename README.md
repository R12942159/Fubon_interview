# Fubon Interview – NLP Disaster Tweet Classification

This project was prepared for an interview with Fubon Financial. It demonstrates the application of Natural Language Processing (NLP) to a real-world problem: building a machine learning model that determines whether a tweet is related to an actual disaster.

## 📌 Project Objective

The goal is to apply NLP and machine learning techniques to analyze the semantics of tweets and classify whether they refer to real disaster events. This kind of model could serve as a foundation for disaster monitoring systems to improve response speed and accuracy.

## 📊 Dataset

This project uses a public dataset from [Kaggle - Natural Language Processing with Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started/overview):

- 10,000 English tweets
- Each tweet is manually labeled as:
  - `1`: related to a real disaster
  - `0`: not related to any real disaster

## 🔍 Problem Definition

The task is to train a classification model that can learn from the semantic content of tweets and predict whether each tweet is disaster-related.

---
## Usage
- **To run TF-IDF + Logistic Regression, use the following command:** <br>
    python3 LogisticRegression.py

- **To run TF-IDF + Decision Tree, use the following command:** <br>
    python3 DecisionTree.py

- **To run Llama embedding + Fully Connected, use the following command:** <br>
    python3 Llama_emb.py

- **To run BERT + Fine Tune, use the following command:** <br>
    python3 BERT_emb.py

---
## Submission
- Deadline: 2025/04/07 (Mon.) 10:00