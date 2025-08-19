import argparse
import random
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as T
from torchvision.models import resnet18, ResNet18_Weights

from dataset import ImageDataset
from utils import set_seed, train_one_epoch, evaluate, unfreeze_layer4_and_fc, freeze_all, unfreeze_fc, unfreeze_layer4_and_fc
import time

DATASETS = {
    "1": {
        "train": "../../Dataset/dataset/1-by-species/train",
        "test":  "../../Dataset/dataset/1-by-species/test",
        "val":   "../../Dataset/dataset/1-by-species/val",
    },
    "2": {
        "train": "../../Dataset/dataset/2-by-species-margaretha/train",
        "test":  "../../Dataset/dataset/2-by-species-margaretha/test",
        "val":   "../../Dataset/dataset/2-by-species-margaretha/val",
    },
    "3": {
        "train": "../../Dataset/dataset/3-by-ripeness/train",
        "test":  "../../Dataset/dataset/3-by-ripeness/test",
        "val":   "../../Dataset/dataset/3-by-ripeness/val",
    },
}

# ---------------- Config (keep it simple) ----------------
BATCH_SIZE = 32
VAL_SPLIT = 0.2
EPOCHS_STAGE1 = 5        # train only fc (feature-extraction phase)
EPOCHS_STAGE2 = 10       # unfreeze layer4 + fc (fine-tune for classification)
LR_HEAD_S1 = 1e-3        # lr for fc in stage 1
LR_HEAD_S2 = 1e-3        # lr for fc in stage 2
LR_BACKBONE_S2 = 1e-4    # lower lr for backbone in stage 2
WEIGHT_DECAY = 1e-4
OUT_PATH = "models/resnet18_best.pth"

def main():
    start_time = time.time()
    parser = argparse.ArgumentParser(description="ResNet18 two-stage trainer with explicit train/val/test")
    parser.add_argument("dataset", choices=["1","2","3"], help="Choose dataset mapping")
    args = parser.parse_args()

    paths = DATASETS[args.dataset]
    train_dir, val_dir, test_dir = paths["train"], paths["val"], paths["test"]
    for p in (train_dir, val_dir, test_dir):
        if not os.path.isdir(p): raise FileNotFoundError(f"Missing dir: {p}")

    set_seed()
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    mean = [0.485, 0.456, 0.406]; std = [0.229, 0.224, 0.225]
    tf_train = T.Compose([T.Resize((224,224)), T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize(mean, std)])
    tf_eval  = T.Compose([T.Resize((224,224)), T.ToTensor(), T.Normalize(mean, std)])

    # Ensure consistent class->idx across splits (derive from TRAIN)
    temp_train = ImageDataset(train_dir)  # infer class_to_idx from train
    class_to_idx = temp_train.class_to_idx
    classes = temp_train.classes
    num_classes = len(classes)
    print(f"Classes ({num_classes}): {classes}")

    ds_train = ImageDataset(train_dir, transform=tf_train, class_to_idx=class_to_idx)
    ds_val   = ImageDataset(val_dir,   transform=tf_eval,  class_to_idx=class_to_idx)
    ds_test  = ImageDataset(test_dir,  transform=tf_eval,  class_to_idx=class_to_idx)

    train_loader = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(ds_val,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(ds_test,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    # Model
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.to(device)

    criterion1 = nn.CrossEntropyLoss(label_smoothing=0.1)
    criterion2 = nn.CrossEntropyLoss(label_smoothing=0.05)

    # ----- Stage 1: Feature extraction (train only fc) -----
    freeze_all(model); unfreeze_fc(model)
    optimizer = torch.optim.AdamW(model.fc.parameters(), lr=LR_HEAD_S1, weight_decay=WEIGHT_DECAY)

    best_val_acc = 0.0
    for epoch in range(1, EPOCHS_STAGE1 + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion1, optimizer, device)
        va_loss, va_acc = evaluate(model, val_loader, criterion1, device)
        print(f"[Stage 1] Epoch {epoch:02d}/{EPOCHS_STAGE1} | train {tr_loss:.4f}/{tr_acc:.4f} | val {va_loss:.4f}/{va_acc:.4f}")
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save({"model_state": model.state_dict(),
                        "classes": classes,
                        "val_acc": best_val_acc}, OUT_PATH)
            print(f"  ✓ Saved best to {OUT_PATH} (val_acc={best_val_acc:.4f})")

    # ----- Stage 2: Fine-tune (unfreeze layer4 + fc) -----
    unfreeze_layer4_and_fc(model)
    backbone_params, head_params = [], []
    for name, p in model.named_parameters():
        if p.requires_grad:
            (head_params if name.startswith("fc") else backbone_params).append(p)

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": LR_BACKBONE_S2, "weight_decay": WEIGHT_DECAY},
        {"params": head_params,     "lr": LR_HEAD_S2,     "weight_decay": WEIGHT_DECAY},
    ])

    for epoch in range(1, EPOCHS_STAGE2 + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion2, optimizer, device)
        va_loss, va_acc = evaluate(model, val_loader, criterion2, device)
        print(f"[Stage 2] Epoch {epoch:02d}/{EPOCHS_STAGE2} | train {tr_loss:.4f}/{tr_acc:.4f} | val {va_loss:.4f}/{va_acc:.4f}")
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save({"model_state": model.state_dict(),
                        "classes": classes,
                        "val_acc": best_val_acc}, OUT_PATH)
            print(f"  ✓ Saved best to {OUT_PATH} (val_acc={best_val_acc:.4f})")
    train_time = time.time() - start_time
    print(f"Training time: {train_time:.2f} seconds")
    writer.add_scalar("TrainingTime", train_time, global_step)

    # ----- Final test evaluation -----
    test_start_time = time.time()
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"[Test] loss {test_loss:.4f} | acc {test_acc:.4f}")
    test_time = time.time() - test_start_time
    print(f"Test time: {test_time:.2f} seconds")
    writer.add_scalar("TestTime", test_time, global_step)
    print("Done.")

if __name__ == "__main__":
    main()