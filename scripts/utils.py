import random
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
import torchmetrics
import timm
from transformers import AutoModel, AutoTokenizer
from sklearn.model_selection import train_test_split

from dataset import get_dataloaders


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MultimodalModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.text_encoder = AutoModel.from_pretrained(config.TEXT_MODEL_NAME)
        self.image_encoder = timm.create_model(
            config.IMAGE_MODEL_NAME, pretrained=True, num_classes=0,
        )
        text_dim = self.text_encoder.config.hidden_size
        image_dim = self.image_encoder.num_features

        self.text_proj = nn.Linear(text_dim, config.EMB_DIM)
        self.image_proj = nn.Linear(image_dim, config.EMB_DIM)
        self.mass_proj = nn.Linear(1, config.EMB_DIM)

        self.head = nn.Sequential(
            nn.Linear(config.EMB_DIM * 3, config.EMB_DIM),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(config.EMB_DIM, 1),
        )

    def forward(self, images, input_ids, attention_mask, mass):
        text_features = self.text_encoder(
            input_ids=input_ids, attention_mask=attention_mask,
        ).last_hidden_state[:, 0, :]
        image_features = self.image_encoder(images)

        text_emb = self.text_proj(text_features)
        image_emb = self.image_proj(image_features)
        mass_emb = self.mass_proj(mass)

        fused = torch.cat([text_emb, image_emb, mass_emb], dim=1)
        return self.head(fused)


def evaluate(model, dataloader, config):
    """MAE на даталоадере + массивы предсказаний и таргетов."""
    model.eval()
    mae_metric = torchmetrics.MeanAbsoluteError().to(config.DEVICE)
    preds_all, targets_all = [], []

    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(config.DEVICE)
            input_ids = batch['input_ids'].to(config.DEVICE)
            attention_mask = batch['attention_mask'].to(config.DEVICE)
            mass = batch['mass'].to(config.DEVICE)
            targets = batch['target'].to(config.DEVICE)

            preds = model(images, input_ids, attention_mask, mass)
            mae_metric.update(preds, targets)
            preds_all.append(preds.cpu())
            targets_all.append(targets.cpu())

    mae = mae_metric.compute().item()
    preds_all = torch.cat(preds_all).numpy().flatten()
    targets_all = torch.cat(targets_all).numpy().flatten()
    return mae, preds_all, targets_all


def train(config, dish):
    """Обучение на обработанном датафрейме из ноутбука."""
    set_seed(config.SEED)
    print(f"Device: {config.DEVICE}")

    train_df = dish[dish['split'] == 'train'].reset_index(drop=True)
    test_df = dish[dish['split'] == 'test'].reset_index(drop=True)

    train_df, val_df = train_test_split(
        train_df, test_size=config.VAL_SIZE, random_state=config.SEED,
    )

    log_mass = np.log1p(train_df['total_mass'])
    mass_mean, mass_std = float(log_mass.mean()), float(max(log_mass.std(), 1e-6))
    print(f"mass_mean={mass_mean:.3f}, mass_std={mass_std:.3f}")

    tokenizer = AutoTokenizer.from_pretrained(config.TEXT_MODEL_NAME)

    train_loader, val_loader = get_dataloaders(
        config, train_df, val_df, tokenizer, mass_mean, mass_std,
    )

    model = MultimodalModel(config).to(config.DEVICE)

    optimizer = AdamW([
        {"params": model.text_encoder.parameters(), "lr": config.TEXT_LR},
        {"params": model.image_encoder.parameters(), "lr": config.IMAGE_LR},
        {"params": model.text_proj.parameters(), "lr": config.CLASSIFIER_LR},
        {"params": model.image_proj.parameters(), "lr": config.CLASSIFIER_LR},
        {"params": model.mass_proj.parameters(), "lr": config.CLASSIFIER_LR},
        {"params": model.head.parameters(), "lr": config.CLASSIFIER_LR},
    ])

    criterion = nn.L1Loss()
    best_val_mae = float('inf')

    open(config.LOG_PATH, "w").close()

    for epoch in range(config.EPOCHS):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            images = batch['image'].to(config.DEVICE)
            input_ids = batch['input_ids'].to(config.DEVICE)
            attention_mask = batch['attention_mask'].to(config.DEVICE)
            mass = batch['mass'].to(config.DEVICE)
            targets = batch['target'].to(config.DEVICE)

            optimizer.zero_grad()
            preds = model(images, input_ids, attention_mask, mass)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        val_mae, _, _ = evaluate(model, val_loader, config)

        msg = f"epoch {epoch+1}/{config.EPOCHS} | train_loss {train_loss:.2f} | val_mae {val_mae:.2f}"
        print(msg)
        with open(config.LOG_PATH, "a") as f:
            f.write(msg + "\n")

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save({
                'model_state': model.state_dict(),
                'mass_mean': mass_mean,
                'mass_std': mass_std,
                'val_mae': val_mae,
            }, config.MODEL_PATH)
            print(f"  saved: {config.MODEL_PATH}")

    print(f"Best val MAE: {best_val_mae:.2f}")
    return model, mass_mean, mass_std, test_df


def load_model(config):
    """Загрузка лучшей модели для инференса."""
    model = MultimodalModel(config).to(config.DEVICE)
    ckpt = torch.load(config.MODEL_PATH, map_location=config.DEVICE)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    return model, ckpt['mass_mean'], ckpt['mass_std']
