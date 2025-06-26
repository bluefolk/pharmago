import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# 1. Load data CSV
df = pd.read_csv("Medicine_Details.csv")
df["Uses"] = df["Uses"].astype(str).str.replace("Treatment of ", "", regex=False).str.strip()

# 2. Buat TF-IDF
vectorizer = TfidfVectorizer(max_features=500)
vectorizer.fit(df["Uses"])

# 3. Simpan ke file .pkl
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
print("✅ tfidf_vectorizer.pkl berhasil dibuat.")
