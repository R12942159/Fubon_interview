import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


df = pd.read_csv("./data/train.csv", encoding='utf-8')
test_df = pd.read_csv("/home/r12942159/kaggle/data/test.csv", encoding='utf-8')

x = df['text']
y = df['target']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_df=0.9)),
    ('clf', DecisionTreeClassifier(max_depth=128, random_state=42))  # 控制深度防止 overfit
])

model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print("DT-Accuracy:", accuracy_score(y_test, y_pred))

test_pred = model.predict(test_df['text'])
kaggle_DT = pd.DataFrame({
    'id': test_df['id'],
    'target': test_pred,
})
kaggle_DT.to_csv("private-test/DT.csv", index=False)