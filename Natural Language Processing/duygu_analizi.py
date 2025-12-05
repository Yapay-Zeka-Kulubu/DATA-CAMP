import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords

# NLTK verilerini indirme
nltk.download('stopwords')

# CSV verisini yükleme
df = pd.read_csv('twitter_training.csv', names=['id', 'topic', 'sentiment', 'text'])
df = df.dropna()

# Temel veri keşfi
print("Veri boyutu:", df.shape)
print("Sütunlar:", df.columns.tolist())
print("\nDuygu dağılımı:")
print(df['sentiment'].value_counts())

# Metin temizleme fonksiyonu
def clean_text(text):
    if isinstance(text, float):
        text = str(text)
    
    # URL'leri kaldır
    text = re.sub(r'http\S+', '', text)
    # Kullanıcı adlarını kaldır
    text = re.sub(r'@\w+', '', text)
    # Hashtag'leri kaldır
    text = re.sub(r'#\w+', '', text)
    # Rakamları kaldır
    text = re.sub(r'\d+', '', text)
    # Noktalama işaretlerini kaldır
    text = re.sub(r'[^\w\s]', ' ', text)
    # Birden fazla boşlukları tek boşluk yap
    text = re.sub(r'\s+', ' ', text)
    # Küçük harfe çevir
    text = text.lower()
    # Baştaki ve sondaki boşlukları kaldır
    text = text.strip()
    
    return text

# Metinleri temizle
df['cleaned_text'] = df['text'].apply(clean_text)

# Stopword'leri kaldırma
stop_words = set(stopwords.words('english'))
def remove_stopwords(text):
    words = text.split()
    words = [word for word in words if word not in stop_words and len(word) > 1]
    return ' '.join(words)

df['cleaned_text'] = df['cleaned_text'].apply(remove_stopwords)

# Boş metinleri temizle
df = df[df['cleaned_text'].str.len() > 0]

# Etiket kodlama
label_encoder = LabelEncoder()
df['sentiment_encoded'] = label_encoder.fit_transform(df['sentiment'])
print("\nEtiket eşlemesi:")
for i, label in enumerate(label_encoder.classes_):
    print(f"{label}: {i}")

# Kelime haznesi oluşturma - DÜZELTİLMİŞ
def build_vocab(texts, min_freq=1):  # min_freq=1 yapıyoruz
    word_counts = Counter()
    for text in texts:
        words = text.split()
        word_counts.update(words)
    
    # Tüm kelimeleri al, minimum frekans kontrolünü kaldır
    vocab = {word: idx+2 for idx, word in enumerate(word_counts.keys())}
    vocab['<PAD>'] = 0
    vocab['<UNK>'] = 1
    
    return vocab, word_counts

vocab, word_counts = build_vocab(df['cleaned_text'])
print(f"\nKelime haznesi boyutu: {len(vocab)}")
print(f"En sık 10 kelime: {word_counts.most_common(10)}")

# Metinleri sayısal indekslere dönüştürme - DÜZELTİLMİŞ
def text_to_indices(text, vocab, max_length=50):
    words = text.split()[:max_length]
    indices = []
    for word in words:
        if word in vocab:
            indices.append(vocab[word])
        else:
            indices.append(vocab['<UNK>'])
    
    # Padding
    if len(indices) < max_length:
        indices.extend([vocab['<PAD>']] * (max_length - len(indices)))
    
    return indices

max_length = 50
df['text_indices'] = df['cleaned_text'].apply(lambda x: text_to_indices(x, vocab, max_length))

# İndeksleri kontrol et
all_indices = [idx for sublist in df['text_indices'] for idx in sublist]
max_index = max(all_indices)
min_index = min(all_indices)
print(f"\nİndeks aralığı: {min_index} - {max_index}")
print(f"Vocab boyutu: {len(vocab)}")
print(f"Maksimum indeks vocab içinde mi: {max_index < len(vocab)}")

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
train_dataset = TweetDataset(X_train, y_train)
test_dataset = TweetDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# RNN Modeli - DÜZELTİLMİŞ
class SentimentRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, n_layers, 
                         batch_first=True, dropout=dropout if n_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        # text shape: [batch_size, seq_len]
        embedded = self.embedding(text)  # Dropout'u embedding'den sonra uygula
        embedded = self.dropout(embedded)
        # embedded shape: [batch_size, seq_len, embedding_dim]
        
        output, hidden = self.rnn(embedded)
        # output shape: [batch_size, seq_len, hidden_dim]
        # hidden shape: [n_layers, batch_size, hidden_dim]
        
        # Son hidden state'i al
        hidden = hidden[-1]
        # hidden shape: [batch_size, hidden_dim]
        
        return self.fc(hidden)

# Model parametreleri - DÜZELTİLMİŞ
vocab_size = len(vocab)
embedding_dim = 100
hidden_dim = 128
output_dim = len(label_encoder.classes_)
n_layers = 1  # Başlangıç için 1 layer
dropout = 0.3

print(f"\nModel parametreleri:")
print(f"Vocab size: {vocab_size}")
print(f"Embedding dim: {embedding_dim}")
print(f"Hidden dim: {hidden_dim}")
print(f"Output dim: {output_dim}")
print(f"Number of layers: {n_layers}")

model = SentimentRNN(vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout)
print(f"\nModel: {model}")

# Model parametrelerini kontrol et
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Eğitilebilir parametre sayısı: {count_parameters(model):,}")

# Loss fonksiyonu ve optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Eğitim fonksiyonu - DÜZELTİLMİŞ
def train_model(model, train_loader, criterion, optimizer, epochs=10):
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        correct_predictions = 0
        total_samples = 0
        
        for batch_idx, (texts, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            
            # Gradyanları sıfırla
            if texts.nelement() == 0:  # Boş batch kontrolü
                continue
                
            predictions = model(texts)
            loss = criterion(predictions, labels)
            
            loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(predictions, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)
            
            if batch_idx % 50 == 0:
                print(f'Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        accuracy = correct_predictions / total_samples
        avg_loss = total_loss / len(train_loader)
        
        print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}')

# Test fonksiyonu
def evaluate_model(model, test_loader):
    model.eval()
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for texts, labels in test_loader:
            if texts.nelement() == 0:  # Boş batch kontrolü
                continue
                
            predictions = model(texts)
            _, predicted = torch.max(predictions, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_predictions)
    print(f"\nTest Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_predictions, 
                              target_names=label_encoder.classes_))
    
    return all_predictions, all_labels

# Modeli eğitme
print("\nModel eğitimi başlıyor...")
train_model(model, train_loader, criterion, optimizer, epochs=5)  # Önce 5 epoch deneyelim

# Modeli değerlendirme
print("\nModel değerlendirme...")
predictions, true_labels = evaluate_model(model, test_loader)

# Tahmin fonksiyonu
def predict_sentiment(text, model, vocab, max_length=50):
    model.eval()
    
    # Metni temizle ve hazırla
    cleaned_text = clean_text(text)
    cleaned_text = remove_stopwords(cleaned_text)
    indices = text_to_indices(cleaned_text, vocab, max_length)
    
    # Tensor'a çevir
    text_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0)
    
    # Tahmin yap
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
    print(f"Probabilities: {dict(zip(label_encoder.classes_, probs))}")
    print()

print("Eğitim tamamlandı!")