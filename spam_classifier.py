
    import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import io

# Load dataset (SMS Spam Collection)
# uploaded["SMSSpamCollection"] contains the file content as bytes.
# Use io.BytesIO to create a file-like object from the bytes.
df = pd.read_csv(io.BytesIO(uploaded["SMSSpamCollection"]), encoding="latin-1", sep='	') 

# Rename columns
df.columns = ["label", "message"]
df["label"] = df["label"].map({"ham": 0, "spam": 1})

# Split data
X_train, X_test, y_train, y_test = train_test_split(df["message"], df["label"], test_size=0.2, random_state=42)

# Build a Pipeline (TF-IDF + Naïve Bayes)
model = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")),
    ("classifier", MultinomialNB())
])

# Train the model
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, y_pred))


