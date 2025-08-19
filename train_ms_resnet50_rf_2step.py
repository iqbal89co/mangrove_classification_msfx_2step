import argparse
import random
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as T
from torchvision.models import resnet50, ResNet50_Weights

from dataset import ImageDataset
from utils import set_seed, train_one_epoch, evaluate, compute_prf1_cm, plot_confusion_matrix, extract_features_multiscale_resnet50
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
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
BATCH_SIZE = 32
EPOCHS_STAGE1 = 5        # train only head (feature-extraction phase)
EPOCHS_STAGE2 = 10       # unfreeze layer4 + head (fine-tune for classification)
LR_HEAD_S1 = 1e-3        # lr for head in stage 1
LR_HEAD_S2 = 1e-3        # lr for head in stage 2
LR_BACKBONE_S2 = 1e-4    # lower lr for backbone in stage 2
WEIGHT_DECAY = 1e-4
MS_LAYERS = ("layer1", "layer2", "layer3", "layer4")  # multi-scale fusion set

# -------------- NEW: Multi-scale ResNet50 wrapper --------------
class MultiScaleResNet50(nn.Module):
    """
    ResNet50 backbone + multi-scale fusion (GAP on layer1..layer4, concat) + classification head.
    'head' replaces the usual 'fc'.
    """
    def __init__(self, num_classes: int, layers=MS_LAYERS, head_hidden: int = 0):
        super().__init__()
        self.layers = set(layers)

        base = resnet50(weights=ResNet50_Weights.DEFAULT)
        # reuse backbone blocks
        self.conv1 = base.conv1; self.bn1 = base.bn1; self.relu = base.relu; self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4

        def _out_ch(seq):
            last = seq[-1]
            # Bottleneck has conv3, BasicBlock has conv2
            return getattr(last, "conv3", getattr(last, "conv2")).out_channels

        # compute fused dim
        dims = []
        if "layer1" in self.layers: dims.append(_out_ch(self.layer1))   # 256 on ResNet50, 64 on ResNet18
        if "layer2" in self.layers: dims.append(_out_ch(self.layer2))   # 512 or 128
        if "layer3" in self.layers: dims.append(_out_ch(self.layer3))   # 1024 or 256
        if "layer4" in self.layers: dims.append(_out_ch(self.layer4))   # 2048 or 512

        self.feat_dim = sum(dims)

        # classifier head (same as you had)
        if head_hidden and head_hidden > 0:
            self.head = nn.Sequential(
                nn.Linear(self.feat_dim, head_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.2),
                nn.Linear(head_hidden, num_classes),
            )
        else:
            self.head = nn.Linear(self.feat_dim, num_classes)

    def forward_features(self, x):
        z = self.conv1(x); z = self.bn1(z); z = self.relu(z); z = self.maxpool(z)
        f1 = self.layer1(z)        # 64, 56x56
        f2 = self.layer2(f1)       # 128, 28x28
        f3 = self.layer3(f2)       # 256, 14x14
        f4 = self.layer4(f3)       # 512, 7x7

        gap = lambda t: t.mean(dim=(2,3))
        parts = []
        if "layer1" in self.layers: parts.append(gap(f1))
        if "layer2" in self.layers: parts.append(gap(f2))
        if "layer3" in self.layers: parts.append(gap(f3))
        if "layer4" in self.layers: parts.append(gap(f4))
        fused = torch.cat(parts, dim=1)  # (B, feat_dim)
        return fused

    def forward(self, x):
        fused = self.forward_features(x)
        logits = self.head(fused)
        return logits

def main():
    start_time = time.time()
    parser = argparse.ArgumentParser(description="Multi-scale ResNet50 two-stage trainer with explicit train/val/test")
    parser.add_argument("dataset", choices=["1","2","3"], help="Choose dataset mapping")
    parser.add_argument("--head-hidden", type=int, default=0, help="MLP head hidden size (0 = linear)")
    args = parser.parse_args()

    paths = DATASETS[args.dataset]
    train_dir, val_dir, test_dir = paths["train"], paths["val"], paths["test"]
    for p in (train_dir, val_dir, test_dir):
        if not os.path.isdir(p): raise FileNotFoundError(f"Missing dir: {p}")

    set_seed()
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    os.makedirs("models", exist_ok=True)
    OUT_RF = f"models/ms_resnet50_rf_ds{args.dataset}_best.pkl"

    # TensorBoard
    run_name = f"ms_resnet50_rf_ds{args.dataset}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    writer = SummaryWriter(log_dir=f"runs/{run_name}")

    mean = [0.485, 0.456, 0.406]; std = [0.229, 0.224, 0.225]
    # Multi-scale aware train aug via scale jitter
    tf_train = T.Compose([T.RandomResizedCrop(224, scale=(0.5, 1.0)), T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize(mean, std)])
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

    # -------------------- MODEL (multi-scale) --------------------
    model = MultiScaleResNet50(num_classes=num_classes, layers=MS_LAYERS, head_hidden=args.head_hidden).to(device)

    criterion_s1 = nn.CrossEntropyLoss(label_smoothing=0.10)
    criterion_s2 = nn.CrossEntropyLoss(label_smoothing=0.05)

    global_step = 0
    best_val_acc = 0.0

    # ----- Stage 1: Feature extraction (freeze backbone, train head only) -----
    for m in [model.conv1, model.bn1, model.layer1, model.layer2, model.layer3, model.layer4]:
        for p in m.parameters(): p.requires_grad = False
    for p in model.head.parameters(): p.requires_grad = True

    optimizer = torch.optim.AdamW(model.head.parameters(), lr=LR_HEAD_S1, weight_decay=WEIGHT_DECAY)

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

        # Per-class F1
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

    # ----- Stage 2: Fine-tune (unfreeze layer4 + head) -----
    for p in model.parameters():
        p.requires_grad = False
    model.eval()

    print("Extracting features for Random Forest ...")
    X_tr, y_tr = extract_features_multiscale_resnet50(model, train_loader, device, layers=MS_LAYERS)
    X_va, y_va = extract_features_multiscale_resnet50(model, val_loader,   device, layers=MS_LAYERS)
    X_te, y_te = extract_features_multiscale_resnet50(model, test_loader,  device, layers=MS_LAYERS)

    print(f"Features: train {X_tr.shape}, val {X_va.shape}, test {X_te.shape}")

    use_gpu = torch.cuda.is_available()
    clf = RandomForestClassifier(
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

    clf.fit(X_tr, y_tr)

    # Validation metrics & logs
    y_pred_va = clf.predict(X_va)
    va_acc = (y_pred_va == y_va).mean()
    p, r, f1, support, pm, rm, f1m, cm = compute_prf1_cm(y_va, y_pred_va, num_classes)

    writer.add_scalar("RF/Val/Acc", va_acc, global_step)
    writer.add_scalar("RF/Val/Precision_macro", pm, global_step)
    writer.add_scalar("RF/Val/Recall_macro",    rm, global_step)
    writer.add_scalar("RF/Val/F1_macro",        f1m, global_step)
    for i, cls in enumerate(classes):
        writer.add_scalar(f"RF/Val/F1_per_class/{cls}", f1[i], global_step)

    fig_cm = plot_confusion_matrix(cm, classes, normalize=False)
    writer.add_figure("RF/Val/ConfusionMatrix", fig_cm, global_step); plt.close(fig_cm)
    fig_cmn = plot_confusion_matrix(cm, classes, normalize=True)
    writer.add_figure("RF/Val/ConfusionMatrix_Normalized", fig_cmn, global_step); plt.close(fig_cmn)

    val_time = time.time() - val_start_time
    print(f"Validation time: {val_time:.2f} seconds")
    writer.add_scalar("ValTime", val_time, global_step)

    print(f"[RF] val acc {va_acc:.4f} | macro P/R/F1 {pm:.4f}/{rm:.4f}/{f1m:.4f}")

    # ----- Final test evaluation -----
    test_start_time = time.time()
    y_pred_te = clf.predict(X_te)
    test_acc = (y_pred_te == y_te).mean()
    pt, rt, f1t, _, pmt, rmt, f1mt, cmt = compute_prf1_cm(y_te, y_pred_te, num_classes)
    print(f"[Test RF] acc {test_acc:.4f} | macro P/R/F1 {pmt:.4f}/{rmt:.4f}/{f1mt:.4f}")

    writer.add_scalar("RF/Test/Acc",  test_acc,  global_step)
    writer.add_scalar("RF/Test/Precision_macro", pmt, global_step)
    writer.add_scalar("RF/Test/Recall_macro",    rmt, global_step)
    writer.add_scalar("RF/Test/F1_macro",        f1mt, global_step)
    fig_cmt = plot_confusion_matrix(cmt, classes, normalize=False)
    writer.add_figure("RF/Test/ConfusionMatrix", fig_cmt, global_step); plt.close(fig_cmt)
    fig_cmtn = plot_confusion_matrix(cmt, classes, normalize=True)
    writer.add_figure("RF/Test/ConfusionMatrix_Normalized", fig_cmtn, global_step); plt.close(fig_cmtn)

    test_time = time.time() - test_start_time
    print(f"Test time: {test_time:.2f} seconds")
    writer.add_scalar("TestTime", test_time, global_step)

    # Save artifacts
    with open(OUT_RF,'wb') as f:
        pickle.dump(clf,f)
        
    print(f"  ✓ Saved RF model to {OUT_RF}")

    writer.close()
    print("Done.")

if __name__ == "__main__":
    main()
