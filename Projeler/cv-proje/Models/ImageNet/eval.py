import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import config

def test_model(model, test_loader):
    model.to(config.DEVICE)
    model.eval()

    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(config.DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            y_true.extend(labels.numpy())
            y_pred.extend(preds.cpu().numpy())

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro")
    rec = recall_score(y_true, y_pred, average="macro")
    f1 = f1_score(y_true, y_pred, average="macro")

    print("TEST SONUÇLARI")
    print(f"Accuracy       : {acc:.4f}")
    print(f"Precision(Macro): {prec:.4f}")
    print(f"Recall(Macro)   : {rec:.4f}")
    print(f"F1-Score(Macro): {f1:.4f}")
