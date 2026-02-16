from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import config

def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    test_transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return train_transform, test_transform


def get_dataloaders():
    train_tf, test_tf = get_transforms()

    full_dataset = datasets.ImageFolder(
        root=config.DATA_DIR,
        transform=train_tf
    )

    total_size = len(full_dataset)
    train_size = int(total_size * config.TRAIN_SPLIT)
    val_size = int(total_size * config.VAL_SPLIT)
    test_size = total_size - train_size - val_size

    train_ds, val_ds, test_ds = random_split(
        full_dataset,
        [train_size, val_size, test_size]
    )

    val_ds.dataset.transform = test_tf
    test_ds.dataset.transform = test_tf

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, test_loader

