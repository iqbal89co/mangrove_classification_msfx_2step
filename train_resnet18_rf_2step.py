import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.ensemble import RandomForestClassifier

from dataset import ImageDataset
from utils import set_seed, train_one_epoch, evaluate, extract_features, compute_prf1_cm, plot_confusion_matrix
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import pickle

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
SEED = 42
BATCH_SIZE = 32
VAL_SPLIT = 0.2
EPOCHS_STAGE1 = 5        # train only fc (feature-extraction phase)
EPOCHS_STAGE2 = 10       # unfreeze layer4 + fc (fine-tune for classification)
LR_HEAD_S1 = 1e-3        # lr for fc in stage 1
LR_HEAD_S2 = 1e-3        # lr for fc in stage 2
LR_BACKBONE_S2 = 1e-4    # lower lr for backbone in stage 2
WEIGHT_DECAY = 1e-4

def main():
    parser = argparse.ArgumentParser(description="ResNet18 two-stage trainer with explicit train/val/test")
    parser.add_argument("dataset", choices=["1","2","3"], help="Choose dataset mapping")
    args = parser.parse_args()

    paths = DATASETS[args.dataset]
    train_dir, val_dir, test_dir = paths["train"], paths["val"], paths["test"]
    for p in (train_dir, val_dir, test_dir):
        if not os.path.isdir(p): raise FileNotFoundError(f"Missing dir: {p}")

    set_seed()
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    
    OUT_RF = f"models/resnet18_rf_ds{args.dataset}_best.pkl"
    
    # TensorBoard
    run_name = f"resnet18_rf_ds{args.dataset}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    writer = SummaryWriter(log_dir=f"runs/{run_name}")

    mean = [0.485, 0.456, 0.406]; std = [0.229, 0.224, 0.225]
    tf_train = T.Compose([T.Resize((224,224)), T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize(mean, std)])
    tf_eval  = T.Compose([T.Resize((224,224)), T.ToTensor(), T.Normalize(mean, std)])

    # Ensure consistent class->idx across splits (derive from TRAIN)
    temp_train = ImageDataset(train_dir)  # infer class_to_idx from train
    class_to_idx = temp_train.class_to_idx
    classes = temp_train.classes
    num_classes = len(classes)
    print(f"Classes ({num_classes}): {classes}")
    writer.add_text("classes", ", ".join(f"{i}:{c}" for i, c in enumerate(classes)))

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

    criterion_s1 = nn.CrossEntropyLoss(label_smoothing=0.1)

    global_step = 0
    best_val_acc = 0.0

    # ----- Stage 1: Feature extraction (train only fc) -----
    for p in model.parameters(): p.requires_grad = False
    for p in model.fc.parameters(): p.requires_grad = True
    optimizer = torch.optim.AdamW(model.fc.parameters(), lr=LR_HEAD_S1, weight_decay=WEIGHT_DECAY)

    for epoch in range(1, EPOCHS_STAGE1 + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion_s1, optimizer, device)
        va_loss, va_acc, y_true, y_pred = evaluate(model, val_loader, criterion_s1, device, return_preds=True)

        # Metrics
        p, r, f1, support, pm, rm, f1m, cm = compute_prf1_cm(y_true, y_pred, num_classes)

        # Logs
        writer.add_scalar("Train/Loss", tr_loss, global_step)
        writer.add_scalar("Train/Acc",  tr_acc,  global_step)
        writer.add_scalar("Val/Loss",   va_loss, global_step)
        writer.add_scalar("Val/Acc",    va_acc,  global_step)
        writer.add_scalar("Val/Precision_macro", pm,  global_step)
        writer.add_scalar("Val/Recall_macro",    rm,  global_step)
        writer.add_scalar("Val/F1_macro",        f1m, global_step)

        # Per-class (kept light: only F1; uncomment if you want P/R too)
        for i, cls in enumerate(classes):
            writer.add_scalar(f"Val/F1_per_class/{cls}", f1[i], global_step)

        # Confusion matrix (raw + normalized)
        fig_cm = plot_confusion_matrix(cm, classes, normalize=False)
        writer.add_figure("Val/ConfusionMatrix", fig_cm, global_step); plt.close(fig_cm)
        fig_cmn = plot_confusion_matrix(cm, classes, normalize=True)
        writer.add_figure("Val/ConfusionMatrix_Normalized", fig_cmn, global_step); plt.close(fig_cmn)

        print(f"[Stage 1] Epoch {epoch:02d}/{EPOCHS_STAGE1} | "
              f"train {tr_loss:.4f}/{tr_acc:.4f} | val {va_loss:.4f}/{va_acc:.4f} | "
              f"macro P/R/F1 {pm:.4f}/{rm:.4f}/{f1m:.4f}")

        if va_acc > best_val_acc:
            best_val_acc = va_acc

        global_step += 1

    # ----- Stage 2: RF on ResNet18 features -----
    # Use non-augmented images for feature extraction
    ds_train_noaug = ImageDataset(train_dir, transform=tf_eval, class_to_idx=class_to_idx)
    feat_train_loader = DataLoader(ds_train_noaug, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    # Extract features
    X_train, y_train = extract_features(model, feat_train_loader, device)
    X_val,   y_val   = extract_features(model, val_loader,        device)
    X_test,  y_test  = extract_features(model, test_loader,       device)

    # RF config (CPU by default; switch tree_method to 'gpu_hist' if you want GPU)
    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        bootstrap=True,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )

    print("[Stage 2] Training RF on extracted features...")
    rf.fit(X_train, y_train)

    # Validation metrics
    y_pred_val = rf.predict(X_val)
    _, _, f1v, _, pmv, rmv, f1mv, cmv = compute_prf1_cm(y_val, y_pred_val, num_classes)
    acc_val = (y_pred_val == y_val).mean()
    writer.add_scalar("RF/Val/Acc", acc_val, global_step)
    writer.add_scalar("RF/Val/Precision_macro", pmv, global_step)
    writer.add_scalar("RF/Val/Recall_macro",    rmv, global_step)
    writer.add_scalar("RF/Val/F1_macro",        f1mv, global_step)
    fig_cmv = plot_confusion_matrix(cmv, classes, normalize=False)
    writer.add_figure("RF/Val/ConfusionMatrix", fig_cmv, global_step); plt.close(fig_cmv)
    fig_cmvn = plot_confusion_matrix(cmv, classes, normalize=True)
    writer.add_figure("RF/Val/ConfusionMatrix_Normalized", fig_cmvn, global_step); plt.close(fig_cmvn)

    # Test metrics
    y_pred_test = rf.predict(X_test)
    _, _, f1t, _, pmt, rmt, f1mt, cmt = compute_prf1_cm(y_test, y_pred_test, num_classes)
    acc_test = (y_pred_test == y_test).mean()
    print(f"[Test | RF] acc {acc_test:.4f} | macro P/R/F1 {pmt:.4f}/{rmt:.4f}/{f1mt:.4f}")
    writer.add_scalar("RF/Test/Acc",  acc_test, global_step)
    writer.add_scalar("RF/Test/Precision_macro", pmt, global_step)
    writer.add_scalar("RF/Test/Recall_macro",    rmt, global_step)
    writer.add_scalar("RF/Test/F1_macro",        f1mt, global_step)
    fig_cmt = plot_confusion_matrix(cmt, classes, normalize=False)
    writer.add_figure("RF/Test/ConfusionMatrix", fig_cmt, global_step); plt.close(fig_cmt)
    fig_cmtn = plot_confusion_matrix(cmt, classes, normalize=True)
    writer.add_figure("RF/Test/ConfusionMatrix_Normalized", fig_cmtn, global_step); plt.close(fig_cmtn)

    # Save artifacts
    with open(OUT_RF,'wb') as f:
        pickle.dump(rf,f)
        
    print(f"  ✓ Saved RF model to {OUT_RF}")
    writer.close()
    print("Done.")

if __name__ == "__main__":
    main()