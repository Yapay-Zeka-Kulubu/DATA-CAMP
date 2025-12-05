import pandas as pd 
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

# CSV verisini yükleme
df = pd.read_csv('twitter_training.csv', names=['id', 'topic', 'sentiment', 'text'])
df = df.dropna()

# Temel veri keşfi
print("Veri boyutu:", df.shape)
print("Sütunlar:", df.columns.tolist())
print("\nDuygu dağılımı:")
print(df['sentiment'].value_counts())
print("\nİlk 5 satır:")
print(df.head())

# Metin temizleme fonksiyonu
def clean_text(text):
    if isinstance(text, float):
        text = str(text)
    text = re.sub(r'http\S+', '', text)       # URL'leri kaldır
    text = re.sub(r'@\w+', '', text)          # Kullanıcı adlarını kaldır
    text = re.sub(r'#\w+', '', text)          # Hashtag'leri kaldır
    text = re.sub(r'\d+', '', text)           # Rakamları kaldır
    text = re.sub(r'[^\w\s]', ' ', text)      # Noktalama işaretlerini kaldır
    text = re.sub(r'\s+', ' ', text)          # Çoklu boşlukları temizle
    text = text.lower().strip()                # Küçük harf ve baş/son boşluk
    return text

df['cleaned_text'] = df['text'].apply(clean_text)

# Stopword'leri kaldırma
stop_words = set(stopwords.words('english'))
def remove_stopwords(text):
    words = text.split()
    words = [word for word in words if word not in stop_words and len(word) > 2]
    return ' '.join(words)

df['cleaned_text'] = df['cleaned_text'].apply(remove_stopwords)

# Etiket kodlama
label_encoder = LabelEncoder()
df['sentiment_encoded'] = label_encoder.fit_transform(df['sentiment'])
print("\nEtiket eşlemesi:")
for i, label in enumerate(label_encoder.classes_):
    print(f"{label}: {i}")

# Kelime haznesi oluşturma
def build_vocab(texts, min_freq=2):
    word_counts = Counter()
    for text in texts:
        words = text.split()
        word_counts.update(words)
    
    vocab = {word: idx+2 for idx, (word, count) in enumerate(word_counts.items()) if count >= min_freq}
    vocab['<PAD>'] = 0
    vocab['<UNK>'] = 1
    return vocab

vocab = build_vocab(df['cleaned_text'])
vocab_size = max(vocab.values()) + 1  # Embedding hatası düzeltmesi
print(f"\nKelime haznesi boyutu: {len(vocab)}, Embedding için vocab_size: {vocab_size}")

# Metinleri sayısal indekslere dönüştürme
def text_to_indices(text, vocab, max_length=50):
    words = text.split()[:max_length]
    indices = [vocab.get(word, vocab['<UNK>']) for word in words]
    if len(indices) < max_length:
        indices.extend([vocab['<PAD>']] * (max_length - len(indices)))
    return indices

max_length = 50
df['text_indices'] = df['cleaned_text'].apply(lambda x: text_to_indices(x, vocab, max_length))

# Özel Dataset sınıfı
class TweetDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = torch.tensor(self.texts[idx], dtype=torch.long)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return text, label

# Train-test split
X = list(df['text_indices'])
y = list(df['sentiment_encoded'])
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nEğitim verisi: {len(X_train)}")
print(f"Test verisi: {len(X_test)}")

# DataLoader'ları oluşturma
batch_size = 32
train_loader = DataLoader(TweetDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(TweetDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

# RNN Modeli
class SentimentRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, n_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        embedded = self.dropout(self.embedding(text))           # [batch, seq_len, emb_dim]
        output, hidden = self.rnn(embedded)                     # output: [batch, seq_len, hidden]
        hidden = hidden[-1]                                     # Son katman hidden
        return self.fc(hidden)                                   # [batch, output_dim]

# Model parametreleri
embedding_dim = 100
hidden_dim = 128
output_dim = len(label_encoder.classes_)
n_layers = 2
dropout = 0.3

model = SentimentRNN(vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout)
print(f"\nModel: {model}")

# Loss ve optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Eğitim fonksiyonu
def train_model(model, train_loader, criterion, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct_predictions = 0
        total_samples = 0
        
        for texts, labels in train_loader:
            optimizer.zero_grad()
            predictions = model(texts)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(predictions, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)
        
        accuracy = correct_predictions / total_samples
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}')

# Test fonksiyonu
def evaluate_model(model, test_loader):
    model.eval()
    all_predictions, all_labels = [], []
    with torch.no_grad():
        for texts, labels in test_loader:
            predictions = model(texts)
            _, predicted = torch.max(predictions, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    accuracy = accuracy_score(all_labels, all_predictions)
    print(f"\nTest Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_predictions, target_names=label_encoder.classes_))
    return all_predictions, all_labels

# Modeli eğitme
print("\nModel eğitimi başlıyor...")
train_model(model, train_loader, criterion, optimizer, epochs=10)

# Modeli değerlendirme
print("\nModel değerlendirme...")
predictions, true_labels = evaluate_model(model, test_loader)

# Tahmin fonksiyonu
def predict_sentiment(text, model, vocab, max_length=50):
    model.eval()
    cleaned_text = remove_stopwords(clean_text(text))
    indices = text_to_indices(cleaned_text, vocab, max_length)
    text_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        prediction = model(text_tensor)
        _, predicted_class = torch.max(prediction, 1)
    sentiment = label_encoder.inverse_transform([predicted_class.item()])[0]
    probabilities = torch.softmax(prediction, dim=1)
    return sentiment, probabilities.numpy()[0]

# Tahmin örnekleri
print("\nTahmin örnekleri:")
test_texts = [
    "I love this game, it's amazing!",
    "This is the worst product ever.",
    "The game is okay, nothing special.",
    "Borderlands is fantastic and fun to play!"
]

for text in test_texts:
    sentiment, probs = predict_sentiment(text, model, vocab)
    print(f"Text: '{text}'")
    print(f"Predicted sentiment: {sentiment}")
    print(f"Probabilities: {dict(zip(label_encoder.classes_, probs))}\n")

# Modeli kaydetme
torch.save({
    'model_state_dict': model.state_dict(),
    'vocab': vocab,
    'label_encoder': label_encoder,
    'model_params': {
        'vocab_size': vocab_size,
        'embedding_dim': embedding_dim,
        'hidden_dim': hidden_dim,
        'output_dim': output_dim,
        'n_layers': n_layers,
        'dropout': dropout
    }
}, 'sentiment_rnn_model.pth')

print("Model 'sentiment_rnn_model.pth' olarak kaydedildi.")
