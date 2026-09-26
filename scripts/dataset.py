import os
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch


def get_transforms(config, is_train=True):
    """Аугментация и препроцессинг."""
    if is_train:
        return A.Compose([
            A.Resize(config.IMG_SIZE, config.IMG_SIZE),
            A.HorizontalFlip(p=0.5),
            A.RandomResizedCrop(size=(config.IMG_SIZE, config.IMG_SIZE), scale=(0.7, 1.0), p=0.5),
            A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0, hue=0, p=0.3),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    return A.Compose([
        A.Resize(config.IMG_SIZE, config.IMG_SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


class DishDataset(Dataset):
    """Мультимодальный датасет: изображение, текст ингредиентов, нормализованная масса, таргет."""

    def __init__(self, df, tokenizer, mass_mean, mass_std, config, transform=None):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.mass_mean = mass_mean
        self.mass_std = mass_std
        self.config = config
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = os.path.join(self.config.IMAGES_DIR, str(row['dish_id']), "rgb.png")
        image = np.array(Image.open(img_path).convert("RGB"))
        if self.transform:
            image = self.transform(image=image)['image']

        encoded = self.tokenizer(
            row['ingredients'],
            padding='max_length',
            truncation=True,
            max_length=self.config.MAX_LEN,
            return_tensors='pt',
        )

        mass = np.log1p(row['total_mass'])
        mass = (mass - self.mass_mean) / self.mass_std

        return {
            'image': image,
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'mass': torch.tensor([mass], dtype=torch.float32),
            'target': torch.tensor([row['total_calories']], dtype=torch.float32),
        }


def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_dataloaders(config, train_df, val_df, tokenizer, mass_mean, mass_std):
    """DataLoader для train и val с детерминированной загрузкой."""
    train_ds = DishDataset(train_df, tokenizer, mass_mean, mass_std, config, get_transforms(config, True))
    val_ds = DishDataset(val_df, tokenizer, mass_mean, mass_std, config, get_transforms(config, False))

    g = torch.Generator()
    g.manual_seed(config.SEED)

    train_loader = DataLoader(
        train_ds, batch_size=config.BATCH_SIZE, shuffle=True,
        num_workers=config.NUM_WORKERS, pin_memory=True,
        worker_init_fn=worker_init_fn,
        generator=g,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=True,
        worker_init_fn=worker_init_fn,
    )
    return train_loader, val_loader


def get_test_loader(config, test_df, tokenizer, mass_mean, mass_std):
    """DataLoader для тестовой выборки."""
    test_ds = DishDataset(
        test_df, tokenizer, mass_mean, mass_std,
        config, get_transforms(config, False),
    )
    return DataLoader(
        test_ds, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )
