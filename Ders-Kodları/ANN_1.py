from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler


CSV_PATH = Path(r"merc_data.csv")
TARGET_COL = "price"
FEATURE_COLS = ["year", "mileage", "tax", "mpg", "engineSize"]
BATCH_SIZE = 64
EPOCHS = 500
LR = 1e-5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("kullanılan cihaz:", device)



class CarPriceDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.from_numpy(features.astype(np.float32))
        self.targets = torch.from_numpy(targets.astype(np.float32))
    def __len__(self):
        return len(self.targets)
    def __getitem__(self, index):
        x = self.features[index]
        y = self.targets[index]
        return x, y



class PriceRegressionMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.act2 = nn.ReLU()
        self.fc3 = nn.Linear(128, 64)
        self.act3 = nn.ReLU()
        self.fc4 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.fc3(x)
        x = self.act3(x)
        x = self.fc4(x)
        return x


df = pd.read_csv(CSV_PATH)

X = df[FEATURE_COLS].to_numpy(copy=True).astype(np.float32)
y = df[TARGET_COL].to_numpy(copy=True).astype(np.float32)

if np.isnan(X).any():
    col_means = np.nanmean(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_means, inds[1])

idx = np.arange(len(X))
np.random.shuffle(idx)
X = X[idx]
y = y[idx]

split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

train_dataset = CarPriceDataset(X_train, y_train)
test_dataset = CarPriceDataset(X_test, y_test)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)



model = PriceRegressionMLP(input_dim=len(FEATURE_COLS)).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)



for epoch in range(1, EPOCHS + 1):
    model.train()
    epoch_loss = 0.0
    n_samples = 0

    for features, targets in train_loader:
        features = features.to(device)
        targets = targets.to(device).unsqueeze(1)

        preds = model(features)
        loss = criterion(preds, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = features.size(0)
        epoch_loss += loss.item() * batch_size
        n_samples += batch_size

    avg_loss = epoch_loss / n_samples

    if epoch % 20 == 0 or epoch == 1:
        print(f"epoch {epoch:03d}  train_mse {avg_loss:.2f}")



model.eval()
all_true = []
all_pred = []

with torch.no_grad():
    for features, targets in test_loader:
        features = features.to(device)
        targets = targets.to(device)
        preds = model(features).squeeze(1)
        all_true.append(targets.cpu().numpy())
        all_pred.append(preds.cpu().numpy())

all_true = np.concatenate(all_true, axis=0)
all_pred = np.concatenate(all_pred, axis=0)

mse = np.mean((all_true - all_pred))
ss_res = np.sum((all_true - all_pred) ** 2)
ss_tot = np.sum((all_true - all_true.mean()) ** 2)
r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0.0

print("Test MSE:", float(mse))
print("Test R^2:", float(r2))
