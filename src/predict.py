import pickle
import tensorflow as tf
import numpy as np
import pandas as pd
import re
import emoji
import nltk
from tqdm import tqdm
tqdm.pandas()
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory


# ======== Load Model & Tokenizer ========
MODEL_PATH = "models/final_bilstm.keras"
TOKENIZER_PATH = "models/tokenizer.pkl"

model = tf.keras.models.load_model(MODEL_PATH)
with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

print("Model & tokenizer berhasil dimuat!")

# ======== Preprocessing Function (sama kayak training) ========
factory = StemmerFactory()
stemmer = factory.create_stemmer()
stopwords = set(nltk.corpus.stopwords.words("indonesian"))

def clean_text(text):
    text = str(text).lower()
    text = emoji.replace_emoji(text, replace='')
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = word_tokenize(text)
    tokens = [stemmer.stem(t) for t in tokens if t not in stopwords and len(t) > 2]
    return " ".join(tokens)

MAX_LEN = 120

# ======== 1. Load dataset ulasan untuk analisis akhir ========
# DATA_PATH = "/Users/gertrudisrms/Documents/Kuliah/DL/data/gojek 2025.csv"
# DATA_PATH = "C:\\Users\\Afandi\\Downloads\\DL GACOR\\Project-DL\\data\\gojek 2025.csv"
DATA_PATH = "gojek 2025.csv"
df = pd.read_csv(DATA_PATH)

SAMPLE_SIZE = 10000  # Batas jumlah data yang diprediksi

df = pd.read_csv(DATA_PATH)
print("Dataset loaded:", df.shape)

if len(df) > SAMPLE_SIZE:
    df = df.sample(SAMPLE_SIZE, random_state=42).reset_index(drop=True)
    print("Dataset dibatasi menjadi:", df.shape)


# ======== 2. Preprocess teks ========
df["cleaned"] = df["content"].astype(str).progress_apply(clean_text)

# ======== 3. Tokenisasi + padding ========
sequences = tokenizer.texts_to_sequences(df["cleaned"])
X = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=MAX_LEN, padding="post")

# ======== 4. Prediksi ========
pred = (model.predict(X) > 0.5).astype(int).flatten()
df["sentiment_pred"] = pred   # 1 = positif, 0 = negatif

# ======== 5. Hitung distribusi sentimen ========
print("\nDistribusi Sentimen:")
print(df["sentiment_pred"].value_counts())
print("\nPersentase (%):")
print((df["sentiment_pred"].value_counts(normalize=True)*100).round(2))

# ======== 6. Simpan hasil ========
df.to_csv("hasil_prediksi_sentimen.csv", index=False)
print("\nHasil prediksi disimpan sebagai hasil_prediksi_sentimen.csv")
