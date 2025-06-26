import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# 1. Load data
df = pd.read_csv("Medicine_Details.csv")

# 2. Ekstrak teks input & label penyakit (hapus 'Treatment of ')
df["Uses"] = df["Uses"].astype(str).str.replace("Treatment of ", "", regex=False).str.strip()
texts = df["Uses"].values  # input teks
labels = df["Uses"].values  # label juga diambil dari kolom yang sama

# 3. TF-IDF Vectorizer
vectorizer = TfidfVectorizer(max_features=500)
X = vectorizer.fit_transform(texts).toarray()

# 4. Encode label
le = LabelEncoder()
y = le.fit_transform(labels)
label_names = le.classes_.tolist()

# 5. Simpan labels.txt
with open("labels.txt", "w", encoding="utf-8") as f:
    for label in label_names:
        f.write(label + "\n")

# 6. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. Model
model = tf.keras.Sequential([
    tf.keras.Input(shape=(X.shape[1],)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(len(label_names), activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 8. Train
model.fit(X_train, y_train, epochs=10, batch_size=8, verbose=1)

# 9. Simpan SavedModel format
model.save("model_diagnosis")

# 10. Konversi ke TFLite
converter = tf.lite.TFLiteConverter.from_saved_model("model_diagnosis")
tflite_model = converter.convert()
with open("text_model.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Model dan labels berhasil disimpan!")
