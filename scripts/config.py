import torch


class Config:
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    DISH_CSV = "data/dish.csv"
    INGREDIENTS_CSV = "data/ingredients.csv"
    IMAGES_DIR = "data/images"
    MODEL_PATH = "checkpoints/best.pt"
    LOG_PATH = "checkpoints/train_log.txt"

    TEXT_MODEL_NAME = "distilbert-base-uncased"
    IMAGE_MODEL_NAME = "resnet18"
    EMB_DIM = 256

    MAX_LEN = 64
    IMG_SIZE = 224
    VAL_SIZE = 0.15

    BATCH_SIZE = 4
    EPOCHS = 12
    NUM_WORKERS = 2

    TEXT_LR = 2e-6
    IMAGE_LR = 2e-6
    CLASSIFIER_LR = 5e-4
