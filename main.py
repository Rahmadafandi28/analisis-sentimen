import os
import re
import emoji
import pickle
import numpy as np
import pandas as pd
import nltk
import matplotlib.pyplot as plt
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, BatchNormalization, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint


DATA_PATH = "/Users/gertrudisrms/Documents/Kuliah/DL/data/gojek 2025.csv"
TEXT_COL = "content"
SCORE_COL = "score"

DROP_NEUTRAL = True   
SAMPLE_SIZE = 10000  
MAX_WORDS = 10000
MAX_LEN = 120
EMBEDDING_DIM = 128
EPOCHS = 5
BATCH_SIZE = 64

os.makedirs("models", exist_ok=True)
BEST_MODEL_PATH = "models/best_model2.keras"


nltk.download("punkt_tab")
nltk.download("stopwords")

factory = StemmerFactory()
stemmer = factory.create_stemmer()
stopwords = set(nltk.corpus.stopwords.words("indonesian"))

def clean_text(text):
    """Membersihkan teks ulasan"""
    text = str(text).lower()
    text = emoji.replace_emoji(text, replace='')
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = word_tokenize(text)
    tokens = [stemmer.stem(t) for t in tokens if t not in stopwords and len(t) > 2]
    return " ".join(tokens)

#Load dataset
df = pd.read_csv(DATA_PATH)
print("Dataset shape:", df.shape)

if SAMPLE_SIZE and len(df) > SAMPLE_SIZE:
    df = df.sample(SAMPLE_SIZE, random_state=42).reset_index(drop=True)

# Labeling score
def map_sentiment(score):
    if score <= 2:
        return 0   
    elif score == 3:
        return 2   
    else:
        return 1   

df["label_raw"] = df[SCORE_COL].apply(map_sentiment)

if DROP_NEUTRAL:
    before = len(df)
    df = df[df["label_raw"] != 2].reset_index(drop=True)
    print(f"Dropped {before - len(df)} neutral reviews (score=3).")

df["label"] = df["label_raw"].astype(int)

print("\nDistribusi Label:")
print(df["label"].value_counts())
print("\nPersentase (%):")
print((df["label"].value_counts(normalize=True) * 100).round(2))

# Prepocessing
print("\nMembersihkan teks... (bisa agak lama)")
df["cleaned"] = df[TEXT_COL].astype(str).apply(clean_text)

# Tokenisasi
tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
tokenizer.fit_on_texts(df["cleaned"])
seqs = tokenizer.texts_to_sequences(df["cleaned"])
X = pad_sequences(seqs, maxlen=MAX_LEN, padding="post", truncating="post")
y = df["label"].values

print("X shape:", X.shape, "y shape:", y.shape)

# Split data
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

cw = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
cw = dict(enumerate(cw))
print("\nClass weights:", cw)

# Modeling
model = Sequential([
    Embedding(MAX_WORDS, EMBEDDING_DIM, input_length=MAX_LEN),
    Bidirectional(LSTM(64, return_sequences=True, dropout=0.3, recurrent_dropout=0.3)),
    BatchNormalization(),
    Bidirectional(LSTM(32, dropout=0.3, recurrent_dropout=0.3)),
    Dense(64, activation="relu"),
    Dropout(0.5),
    Dense(1, activation="sigmoid")
])

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()
callbacks = [
    EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1, min_lr=1e-6),
    ModelCheckpoint(BEST_MODEL_PATH, monitor="val_loss", save_best_only=True, verbose=1)
]


# Training
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=cw,
    callbacks=callbacks,
    verbose=1
)


print("\n📦 Menyimpan model dan tokenizer...")
model.save("models/final_bilstm.keras")

with open("models/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

pd.DataFrame(history.history).to_csv("models/training_history.csv", index=False)

# Visualisasi hasil training
plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Akurasi')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.legend()

plt.tight_layout()
plt.savefig("models/training_history.png", dpi=300)
plt.show()

# Evaluasi
best_model = tf.keras.models.load_model(BEST_MODEL_PATH)
print("\nEvaluasi di test set:")
loss, acc = best_model.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.4f}, Test Accuracy: {acc:.4f}")

y_pred = (best_model.predict(X_test) > 0.5).astype(int).flatten()
print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=4))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negatif','Positif'], yticklabels=['Negatif','Positif'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("models/confusion_matrix.png", dpi=300)
plt.show()

# Simpan metrik evaluasi ke CSV
metrics = {
    "accuracy": [acc],
    "precision": [precision_score(y_test, y_pred)],
    "recall": [recall_score(y_test, y_pred)],
    "f1": [f1_score(y_test, y_pred)]
}
pd.DataFrame(metrics).to_csv("models/test_metrics.csv", index=False)
print("\n📊 Metrik evaluasi disimpan di models/test_metrics.csv")
