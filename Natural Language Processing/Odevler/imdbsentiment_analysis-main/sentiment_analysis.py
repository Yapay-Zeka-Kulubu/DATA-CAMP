import nltk
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

from datasets import load_dataset
dataset = load_dataset("imdb")
from nltk.corpus import stopwords
stop_words = set(stopwords.words("english"))
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer
import re
from tqdm import tqdm #preprocessing sırasında ilerleme çubuğunu göstermek için
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
   accuracy_score, precision_score, recall_score, f1_score, confusion_matrix )
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# 1) Preprocessing Fonksiyonları
# -----------------------------

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]","",text) #noktalama isaretlerini kaldirir
    text = re.sub(r"\d+","",text) #sayilari siler
    words = text.split()
    new_words = []
    for w in words:
        if w not in stop_words and len(w) > 1:
           lemmatized_word = lemmatizer.lemmatize(w)
           new_words.append(lemmatized_word)
    return " ".join(new_words)

def preprocess_dataset(dataset):
    texts = dataset["text"]
    labels = dataset["label"]

    cleaned_texts = []
    for t in tqdm(texts, desc="Preprocessing"):
        cleaned = preprocess_text(t)
        cleaned_texts.append(cleaned)
    return cleaned_texts, labels



# -----------------------------
# 2) Dataset Yükleme
# -----------------------------   
dataset = load_dataset("imdb")
train_texts, train_labels = preprocess_dataset(dataset["train"])
test_texts, test_labels = preprocess_dataset(dataset["test"])


# -----------------------------
# 3) TF-IDF Vektörleştirme
# ----------------------------- 
tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1,2),
    stop_words="english"
)


X_train = tfidf.fit_transform(train_texts) #train_texts e bakarak kelime haznesi olusutur ve sayisal matrise dönüstürür
X_test = tfidf.transform(test_texts) #sadece dönüstürme yapilir


# -----------------------------
# 4) Model Eğitimi
# ----------------------------- 
model = LogisticRegression(max_iter=200, random_state=42)
model.fit(X_train, train_labels)
print("Model eğitildi!")
y_pred = model.predict(X_test) #test verisi üzerinde tahmin yapilmasi


# -----------------------------
# 5) Metrik Hesaplama
# -----------------------------
accuracy = accuracy_score(test_labels, y_pred)
precision =precision_score(test_labels,y_pred)
recall = recall_score(test_labels, y_pred)
f1 = f1_score(test_labels, y_pred)

print("Accuracy :", accuracy) #doğruluk
print("Precision:", precision) #kesinlik
print("Recall   :", recall) #hassasiyet
print("F1-score :", f1)


# -----------------------------
# 6) Confusion Matrix Kaydetme
# -----------------------------
cm = confusion_matrix(test_labels,y_pred)
os.makedirs("results", exist_ok=True)

# Görselleştirme için figür boyutunu ayarla (6 inç genişlik, 5 inç yükseklik).
plt.figure(figsize=(6,5)) 
# Confusion Matrix'i Isı Haritası (Heatmap) olarak çiz:
# annot=True: Kutucuklara sayısal değerleri yaz.
# fmt="d": Sayıları tam sayı olarak formatla.
# cmap="Blues": Mavi renk skalasını kullan.
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues") 
plt.title("Confusion Matrix") # Grafiğe başlık ekle.
plt.xlabel("Predicted")      # X ekseni etiketini (Tahmin Edilen) ayarla.
plt.ylabel("Actual")         # Y ekseni etiketini (Gerçek) ayarla.
save_path = "results/confusion_matrix.png"
plt.savefig(save_path)
plt.close()
print("Confusion Matrix kaydedildi:", save_path)


# -----------------------------
# 7) Custom Sentence Prediction
# -----------------------------
def predict_sentence(sentence):
    cleaned = preprocess_text(sentence)
    vector = tfidf.transform([cleaned])
    prediction = model.predict(vector)[0]
    return "Positive" if prediction == 1 else "Negative"

test_sentences = [
    "They said the movie was absolutely legendary!",
    "I laughed much,this movie was comical.",
    "Amazing visuals and great soundtrack.",
    "The acting was perfect but the story was disappointing.",
    "Terrible movie, waste of time."
] 

print("\n--- Custom Sentence Predictions ---")
for sentence in test_sentences:
    print(sentence, "→", predict_sentence(sentence))

metrics_path = "results/metrics.txt"

with open(metrics_path, "w", encoding="utf-8") as f:
    f.write("Model Evaluation Metrics\n")
    f.write("========================\n")
    f.write(f"Accuracy : {accuracy}\n")
    f.write(f"Precision: {precision}\n")
    f.write(f"Recall   : {recall}\n")
    f.write(f"F1-score : {f1}\n")

print("Metrikler kaydedildi:", metrics_path)
