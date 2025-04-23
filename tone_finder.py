import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load dataset
train_data = pd.read_csv("empathetic-dialogues-contexts/train.csv")
test_data = pd.read_csv("empathetic-dialogues-contexts/test.csv")
valid_data = pd.read_csv("empathetic-dialogues-contexts/valid.csv")

print(train_data.head())


# Combine train and valid datasets for training
data = pd.concat([train_data, valid_data])

# Split dataset
X_train = data["situation"]
y_train = data["emotion"]
X_test = test_data["situation"]
y_test = test_data["emotion"]

# Convert text to numerical features using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train a classifier
classifier = RandomForestClassifier(random_state=42)
classifier.fit(X_train_tfidf, y_train)

# Evaluate the model
y_pred = classifier.predict(X_test_tfidf)
print(classification_report(y_test, y_pred))