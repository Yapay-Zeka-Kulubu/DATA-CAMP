import torch

def load_weights(model, weight_path):
    state_dict = torch.load(weight_path, map_location="cpu")
    model.load_state_dict(state_dict)
    print("Model ağırlıkları yüklendi.")
