from datasets import load_dataset
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix,precision_score, recall_score,f1_score,ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os

dataset = load_dataset("imdb")

#Duygu analizinin performansı için ham metnin temizlenmesi ve işlenmesi gerekir.
#noktalama işaretleri genelde gürültülü veri olur
#yorumlarda yapılan sayısal içerikler duyguyla çok ilgili olmaz.
#stopword kelimeler(the,was,is...) çok anlam taşımaz çıkartırsak model daha iyi odaklanabilir.
#lemminization(kök haline getirmek) kelimeleri kök durumuna indirger.
#temizleme işlemi sonunda çok fazla boşluk kalabilir, bunlar da temizlenmeli.

nltk.download("stopwords")
nltk.download("wordnet")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocessing(text):
    text = text.lower()
    text = re.sub(r"\d+", "", text) #d rakamdır, + bir veya daha fazla (regular exp'ten substitute degistirme)
    text = text.translate(str.maketrans("","",string.punctuation)) #punct. tüm nokt. işaretleri
    tokens = text.split() #kelimelere ayırma işlemi yapıyorz
    tokens = [word for word in tokens if word not in stop_words] #stopword'de olmayan kelimeleri al
    tokens = [lemmatizer.lemmatize(word) for word in tokens] #kelimeler kök haline gelir
    text = " ".join(tokens) #kelimeleri birlestir.
    text = re.sub(r"\s+", " ",text).strip() #fazla boşlukları sil.
    return text

#TF-IDF metin verilerini sayısal forma dönüştürür,kullanım sıklığını dikkate alır
tfidf_vectorizer = TfidfVectorizer(
    max_features = 5000, #en fazla 5000 kelime
    ngram_range= (1,2), #tekli + ikili kelime grubu
    stop_words="english" 
)

X_train = [preprocessing(sample["text"]) for sample in dataset["train"]]
X_test = [preprocessing(sample["text"]) for sample in dataset["test"]]
y_train = [sample["label"] for sample in dataset["train"]]
y_test = [sample["label"] for sample in dataset["test"]]

X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

model = LogisticRegression(max_iter=1000,random_state=13)
model.fit(X_train_tfidf,y_train)
y_pred = model.predict(X_test_tfidf)


#metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

os.makedirs("project/results", exist_ok=True) #otomatik olustursun

with open("project/results/metrics.txt","w") as f:
    f.write(f"Accuracy: {accuracy:.2f}\n")
    f.write(f"Precision: {precision:.2f}\n")
    f.write(f"Recall: {recall:.2f}\n")
    f.write(f"F1 Score: {f1:.2f}\n")

#confusion matrx
cm = confusion_matrix(y_test, y_pred)
display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Negative","Positive"])
display.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.savefig("project/results/confusion_matrix.png")
plt.close()

#classification report
print("Classification Report:", classification_report(y_test, y_pred, target_names=["Negative","Positive"]))

#test edelim
custom_sentences = [
    "This movie was boring and slow.",
    "I absolutely loved this film!",
    "The acting was terrible and the story was weak.",
    "Amazing visuals and great storyline.",
    "It was not good, I would not recommend it."
]

process_sentences = [preprocessing(sentence) for sentence in custom_sentences]
vectors = tfidf_vectorizer.transform(process_sentences)
predictions = model.predict(vectors)

print("Custom Settence Predictions:")

for sentence, pred in zip(custom_sentences, predictions):
    label = "Positive" if pred==1 else "Negative"    
    print(f"Sentence:{sentence} => Prediction: {label}")