import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


df = pd.read_csv("./data/train.csv", encoding='utf-8')
test_df = pd.read_csv("./data/test.csv", encoding='utf-8')

x = df['text']
y = df['target']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_df=0.9)),
    ('clf', LogisticRegression(solver='liblinear'))
])

model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print("LR-Accuracy:", accuracy_score(y_test, y_pred))

# Predicting on the test set
test_pred = model.predict(test_df['text'])
kaggle_LR = pd.DataFrame({
    'id': test_df['id'],
    'target': test_pred,
})
kaggle_LR.to_csv("./results/LR.csv", index=False)