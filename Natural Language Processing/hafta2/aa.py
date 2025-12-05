# =============================================================================
# CUDA ve veri kontrolü
# =============================================================================
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Kullanılan cihaz: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"GPU Bellek: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
# =============================================================================
# Veri yükleme ve temizleme (print ile adım adım)
# =============================================================================
import pandas as pd
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

nltk.download('stopwords')

df = pd.read_csv(r'C:\Users\w\Desktop\Kodlama\VsCode\HelloWorld\DataKamp\DATA-CAMP\Natural Language Processing\twitter_training.csv', names=['id','topic','sentiment','text']).dropna()
print("Orijinal veri örneği:\n", df.head(), "\n")

# Temizleme fonksiyonu
def clean_text(text):
    text = str(text)
    text = re.sub(r'http\S+|@\w+|#\w+|\d+', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.lower().strip()

df['cleaned_text'] = df['text'].apply(clean_text)
print("Temizlenmiş veri örneği:\n", df[['text','cleaned_text']].head(), "\n")

# Stopwords çıkarma
stop_words = set(stopwords.words('english'))
df['cleaned_text'] = df['cleaned_text'].apply(lambda x: ' '.join([w for w in x.split() if w not in stop_words and len(w)>1]))
df = df[df['cleaned_text'].str.len()>0]
print("Stopwords çıkarıldıktan sonra örnek:\n", df[['cleaned_text']].head(), "\n")

# Label encoding
label_encoder = LabelEncoder()
df['sentiment_encoded'] = label_encoder.fit_transform(df['sentiment'])
print("Label encoding örneği:\n", df[['sentiment','sentiment_encoded']].head(), "\n")

# Kelime haznesi ve indeksleme
def build_vocab(texts):
    word_counts = Counter()
    for text in texts:
        word_counts.update(text.split())
    vocab = {w:i+2 for i,w in enumerate(word_counts)}
    vocab['<PAD>'] = 0
    vocab['<UNK>'] = 1
    return vocab

vocab = build_vocab(df['cleaned_text'])
max_length = 50
def text_to_indices(text):
    words = text.split()[:max_length]
    idx = [vocab.get(w, vocab['<UNK>']) for w in words]
    idx += [vocab['<PAD>']] * (max_length - len(idx))
    return idx

df['text_indices'] = df['cleaned_text'].apply(text_to_indices)
print("İndekslenmiş veri örneği (ilk 2 tweet):\n", df['text_indices'].head(2).tolist(), "\n")

# =============================================================================
# Veri küçültme
# =============================================================================
X = list(df['text_indices'])
y = list(df['sentiment_encoded'])

X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size=20000, stratify=y, random_state=42)
X_test, _, y_test, _ = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

# =============================================================================
# Dataset ve DataLoader
# =============================================================================
from torch.utils.data import Dataset, DataLoader

class TweetDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        return torch.tensor(self.texts[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)

batch_size = 64
train_loader = DataLoader(TweetDataset(X_train, y_train), batch_size=batch_size, shuffle=True, pin_memory=True if device.type=='cuda' else False)
test_loader = DataLoader(TweetDataset(X_test, y_test), batch_size=batch_size, shuffle=False, pin_memory=True if device.type=='cuda' else False)

# =============================================================================
# Model Tanımları
# =============================================================================
import torch.nn as nn

class SentimentRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, n_layers, batch_first=True, dropout=dropout if n_layers>1 else 0)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        output, hidden = self.rnn(embedded)
        hidden = hidden[-1]
        return self.fc(hidden)

class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout, bidirectional=True):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers,
                            batch_first=True, dropout=dropout if n_layers>1 else 0,
                            bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_dim*2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.bidirectional = bidirectional
    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        output, (hidden, cell) = self.lstm(embedded)
        if self.bidirectional:
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            hidden = hidden[-1]
        return self.fc(hidden)

# =============================================================================
# Model Parametreleri
# =============================================================================
vocab_size = len(vocab)
embedding_dim = 256
hidden_dim = 384
output_dim = len(label_encoder.classes_)
n_layers = 2
dropout = 0.2
bidirectional = True

rnn_model = SentimentRNN(vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout).to(device)
lstm_model = SentimentLSTM(vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout, bidirectional).to(device)

# =============================================================================
# Eğitim ve Değerlendirme Fonksiyonları
# =============================================================================
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report
import time

criterion = nn.CrossEntropyLoss()
rnn_optimizer = optim.Adam(rnn_model.parameters(), lr=0.001, weight_decay=1e-4)
lstm_optimizer = optim.Adam(lstm_model.parameters(), lr=0.001, weight_decay=1e-4)

def train_model(model, loader, optimizer, epochs=10, model_name="Model"):
    model.train()
    for epoch in range(epochs):
        total_loss, total_correct, total_samples = 0,0,0
        start_time = time.time()
        for texts, labels in loader:
            texts, labels = texts.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            total_correct += (outputs.argmax(1)==labels).sum().item()
            total_samples += labels.size(0)
        print(f"{model_name} - Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.4f}, "
              f"Accuracy: {total_correct/total_samples:.4f}, Time: {time.time()-start_time:.1f}s")

def evaluate_model(model, loader, model_name="Model"):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for texts, labels in loader:
            texts, labels = texts.to(device), labels.to(device)
            outputs = model(texts)
            preds = outputs.argmax(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    print(f"{model_name} Test Accuracy: {acc:.4f}")
    print(classification_report(all_labels, all_preds, target_names=label_encoder.classes_))

# =============================================================================
# Eğitim ve Test
# =============================================================================
print("\n==== RNN Eğitim ====")
train_model(rnn_model, train_loader, rnn_optimizer, epochs=10, model_name="RNN")
evaluate_model(rnn_model, test_loader, model_name="RNN")

print("\n==== LSTM Eğitim ====")
train_model(lstm_model, train_loader, lstm_optimizer, epochs=10, model_name="LSTM")
evaluate_model(lstm_model, test_loader, model_name="LSTM")



# =============================================================================
# Manuel Deneme
# =============================================================================
def predict_text(model, text, vocab, max_length=50):
    model.eval()
    # Temizleme ve stopword çıkarma
    text_clean = clean_text(text)
    text_clean = ' '.join([w for w in text_clean.split() if w not in stop_words and len(w)>1])
    # İndekslere çevirme ve padding
    idx = text_to_indices(text_clean)
    tensor = torch.tensor([idx], dtype=torch.long).to(device)
    with torch.no_grad():
        output = model(tensor)
        pred = torch.argmax(output, dim=1).item()
    return label_encoder.inverse_transform([pred])[0]

# Örnek kullanım
sample_texts = [
    "I love this product! It's amazing.",
    "Worst experience ever, I hate it.",
    "I don't know what to think about this."
]

for text in sample_texts:
    pred = predict_text(lstm_model, text, vocab)
    print(f"Text: {text}\nPredicted Sentiment: {pred}\n")
