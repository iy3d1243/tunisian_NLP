# Part 1: Train and Save Models
# ==============================

import pandas as pd
import re
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Load and clean data
df = pd.read_csv("C:\\Users\\iyed\\Downloads\\finals.csv")
df.dropna(inplace=True)

def clean_arabic(text):
    text = re.sub(r'[\u064B-\u065F]', '', text)
    text = re.sub(r'[\W_]+', ' ', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[إأآا]', 'ا', text)
    text = re.sub(r'ة', 'ه', text)
    text = re.sub(r'ى', 'ي', text)
    return text.strip()

df['clean_text'] = df['Comment_Text_Arabic'].apply(clean_arabic)

# Vectorization
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X = vectorizer.fit_transform(df['clean_text'])
y = df['Problem_Source']

# Save vectorizer
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Models to train
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "NaiveBayes": MultinomialNB(),
    "SVM": LinearSVC(),
    "RandomForest": RandomForestClassifier(n_estimators=100),
    }

# Train, evaluate, and save each model
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print("Classification Report:\n", classification_report(y_test, y_pred))

    with open(f"model_{name}.pkl", "wb") as f:
        pickle.dump(model, f)

print("\nAll models and vectorizer saved.")