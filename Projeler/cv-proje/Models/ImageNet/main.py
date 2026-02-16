import argparse
import config
from dataset import get_dataloaders
from model import build_model
from train import train_model
from eval import test_model
from utils import load_weights

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "test"], required=True)
    parser.add_argument("--weights", type=str, help="Model ağırlık yolu")
    args = parser.parse_args()

    train_loader, val_loader, test_loader = get_dataloaders()
    model = build_model()

    if args.mode == "train":
        train_model(model, train_loader, val_loader)
        test_model(model, test_loader)

    elif args.mode == "test":
        load_weights(model, args.weights)
        test_model(model, test_loader)

if __name__ == "__main__":
    main()
