import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import recall_score, accuracy_score, f1_score
import numpy as np
from tqdm import tqdm
import os
import random

# ==== CONFIG ====
DATA_DIR = r"D:\Protein_Cancer\Protein_Project\dataset_images" 
SAVE_DIR = r"D:\Protein_Cancer\Protein_Project\models_3"
os.makedirs(SAVE_DIR, exist_ok=True)

MODEL_LIST = ["inception_v3", "efficientnet_b4", "densenet201"]

BATCH_SIZE = 16
EPOCHS = 30
IMG_SIZE = 299
ACCUM_STEPS = 2
LR = 2e-4
THRESHOLD = 0.30  
EARLY_STOP_PATIENCE = 7

USE_FOCAL_LOSS = True
LABEL_SMOOTHING = 0.05


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n🔥 Using: {device}")
if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0), "\n")


# ========= Focal Loss ========= #
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.9, gamma=3):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss()  # Removed label_smoothing

    def forward(self, preds, targets):
        bce_loss = self.bce(preds, targets)
        pt = torch.exp(-bce_loss)
        return self.alpha * (1-pt)**self.gamma * bce_loss


criterion = FocalLoss() if USE_FOCAL_LOSS else nn.BCEWithLogitsLoss()


# ========= Cancer-focused Augmentation ========= #
train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomApply([
        transforms.RandomRotation(40),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.GaussianBlur(kernel_size=3),
        transforms.ColorJitter(0.4,0.4)
    ], p=0.7),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])


# ========= Dataset ========= #
full_ds = datasets.ImageFolder(DATA_DIR)
train_size = int(0.8 * len(full_ds))
val_size = len(full_ds) - train_size
train_ds, val_ds = torch.utils.data.random_split(full_ds, [train_size, val_size])

train_ds.dataset.transform = train_tf
val_ds.dataset.transform = val_tf

train_targets = np.array(full_ds.targets)[train_ds.indices]
class_counts = np.bincount(train_targets)
weights = 1.0 / class_counts
sample_weights = torch.tensor([weights[t] for t in train_targets], dtype=torch.double)
sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                          num_workers=0, pin_memory=True, drop_last=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=0, pin_memory=True)


# ========= Model Factory ========= #
def create_model(name):
    print(f"🏗️ Building {name}...")

    if name == "inception_v3":
        model = models.inception_v3(weights="IMAGENET1K_V1")
        model.aux_logits = False
        model.fc = nn.Sequential(nn.Dropout(0.4),
                                 nn.Linear(2048, 1))

    elif name == "efficientnet_b4":
        model = models.efficientnet_b4(weights="IMAGENET1K_V1")
        model.classifier[1] = nn.Linear(1792, 1)

    elif name == "densenet201":
        model = models.densenet201(weights="IMAGENET1K_V1")
        model.classifier = nn.Linear(1920, 1)

    return model.to(device)


# ========= Training ========= #
def train_model(name):
    model = create_model(name)

    # Model-Specific Learning Rate Tuning
    lr = LR
    if name == "efficientnet_b4": lr = 1.5e-4

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = torch.cuda.amp.GradScaler()

    best_recall = 0.0
    patience_counter = 0

    print(f"\n🚀 Training Started: {name}\n")

    for epoch in range(EPOCHS):
        model.train()
        loop = tqdm(train_loader, desc=f"{name} | Epoch {epoch+1}/{EPOCHS}")
        optimizer.zero_grad()

        for i, (imgs, lbls) in enumerate(loop):
            imgs, lbls = imgs.to(device), lbls.float().unsqueeze(1).to(device)

            with torch.cuda.amp.autocast():
                out = model(imgs)
                loss = criterion(out, lbls) / ACCUM_STEPS

            scaler.scale(loss).backward()

            if (i + 1) % ACCUM_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            loop.set_postfix(loss=loss.item()*ACCUM_STEPS)

        scheduler.step()

        # === Validation === #
        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs = imgs.to(device)
                out = torch.sigmoid(model(imgs)).cpu().numpy()
                preds.extend(out > THRESHOLD)
                labels.extend(lbls.numpy())

        acc = accuracy_score(labels, preds)
        rec = recall_score(labels, preds, zero_division=0)
        f1 = f1_score(labels, preds, zero_division=0)

        print(f"📌 {name} | Acc: {acc:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")

        # Save highest cancer recall
        if rec > best_recall:
            best_recall = rec
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"{name}_best.pth"))
            print(f"💾 Saved Best {name} | Recall: {best_recall:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= EARLY_STOP_PATIENCE:
            print("🛑 Early Stopping (Recall plateau)")
            break

    print(f"🎯 Final Best Recall for {name}: {best_recall:.4f}\n")


# ========= Train All ========= #
if __name__ == "__main__":
    for name in MODEL_LIST:
        train_model(name)
