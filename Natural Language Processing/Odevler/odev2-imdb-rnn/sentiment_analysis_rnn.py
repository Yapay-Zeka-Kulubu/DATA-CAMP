"""
İlk ödevde klasik ML kullanarak metni TF-IDF ile vektörleştirmiştik,model de log. reg.'du
şimdi ise metne tokenization+padding uygulayıp, embedding+RNN/LSTM kullanacagız
Klasik ML'de kelime sırası önemli degildi ama şimdi DL'de önemli olacak.

RNN her adımda önceki adımını çıktısına dikkat eder ama unutma problemi vardır.(vanishing gradient)
LSTM ise unutma problemini çözer, neyin tutulup neyin unutulacagının kararını gate'ler verir.(forget,input,output)
Embedding ise kelimeleri one-hot degil, anlamlı vektörlere dönüştürür.
"""

import time
import numpy as np
from datasets import load_dataset
import nltk
from nltk.corpus import stopwords
import string
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.metrics import (accuracy_score, precision_score, recall_score,f1_score, confusion_matrix, ConfusionMatrixDisplay)
import os
import matplotlib.pyplot as plt


os.makedirs("project2/results/confusion_matrices",exist_ok=True)

dataset = load_dataset("imdb")

X_train_texts = dataset["train"]["text"]
y_train = np.array(dataset["train"]["label"])

X_test_texts = dataset["test"]["text"]
y_test = np.array(dataset["test"]["label"])

#Derin ogrenmede preprocessing adımları biraz daha hafif olur, anlamı -> Embedding+LSTM ögrenir.

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

def preprocessing(text):
    text = text.lower() #küçük harf
    text = text.translate(str.maketrans("","", string.punctuation)) #nokt. işaretleri
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words] #stopwords temizleme

    text = " ".join(tokens)

    return text

X_train_cleaned = [preprocessing(text) for text in X_train_texts]
X_test_cleaned = [preprocessing(text) for text in X_test_texts]

#Tokenization yapalım.
#Tokenization, metni modelin anlayabilecegi en küçük anlamlı parçalara bölme işlemidir.
#model cümleyi anlamaz, token dizisini anlar. Metni sayısal verilere dönüştürür.

vocab_size = 10000 #en sık geçen 10k kelimeyi alıyoruz

tokenizer = Tokenizer(num_words = vocab_size, oov_token="<OOV>") # vocab dışı kelimeler için <OOV>
tokenizer.fit_on_texts(X_train_cleaned) #egitim

X_train_sequences = tokenizer.texts_to_sequences(X_train_cleaned)
X_test_sequences = tokenizer.texts_to_sequences(X_test_cleaned)

#Padding ve Truncating işlemi yapalım
max_length = 200
X_train_padded = pad_sequences(
    X_train_sequences,
    maxlen = max_length,
    padding = "post",
    truncating= "post"
)
X_test_padded = pad_sequences(
    X_test_sequences,
    maxlen = max_length,
    padding = "post",
    truncating= "post"
)

#Embedding+LSTM Modeli
embedding_dim = 128 #Her kelime 128 boyutlu vektörle temsil edilir.
model = Sequential([
    Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        input_length=max_length
    ),
    LSTM(128, return_sequences=False),
    Dropout(0.5),
    Dense(1, activation="sigmoid")
])
model.compile(
    optimizer = "adam",
    loss = "binary_crossentropy",
    metrics = ["accuracy"]
)
model.summary()

#Model Egitimi
start_time = time.time()

history = model.fit(
    X_train_padded,
    y_train,
    epochs=5,
    batch_size = 64,
    validation_split=0.2,
    verbose=1
)

training_time = time.time() - start_time

#Model Degerlendirme
y_pred_probs = model.predict(X_test_padded)
y_pred = (y_pred_probs > 0.5).astype(int).flatten()

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)


#Kaydedelim.
with open("project2/results/confusion_matrices/rnn_metrics.txt", "w") as f:
    f.write(f"Accuracy : {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall   : {recall:.4f}\n")
    f.write(f"F1-score : {f1:.4f}\n")
    f.write(f"Training Time (sec): {training_time:.2f}\n")


# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Negative", "Positive"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")

plt.savefig("project2/results/confusion_matrices/rnn_confusion_matrix.png")
plt.close()