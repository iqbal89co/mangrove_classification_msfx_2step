import random
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # for headless environments
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import torch.nn as nn

SEED = 42
# ---------------- Utils ----------------
def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed); torch.backends.cudnn.deterministic = True

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    tot_loss = tot_correct = n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        bs = y.size(0)
        n += bs
        tot_loss += loss.item() * bs
        tot_correct += (out.argmax(1) == y).float().sum().item()
    return tot_loss / n, tot_correct / n

@torch.no_grad()
def evaluate(model, loader, criterion, device, return_preds=False):
    model.eval()
    tot_loss = tot_correct = n = 0
    ys, yhs = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = criterion(out, y)
        bs = y.size(0)
        n += bs
        tot_loss += loss.item() * bs
        preds = out.argmax(1)
        tot_correct += (preds == y).float().sum().item()
        if return_preds:
            ys.append(y.detach().cpu().numpy())
            yhs.append(preds.detach().cpu().numpy())
    if return_preds:
        y_true = np.concatenate(ys) if ys else np.array([])
        y_pred = np.concatenate(yhs) if yhs else np.array([])
        return (tot_loss / n, tot_correct / n, y_true, y_pred)
    return tot_loss / n, tot_correct / n

def freeze_all(model):
    for p in model.parameters(): p.requires_grad = False

def unfreeze_fc(model):
    for p in model.fc.parameters(): p.requires_grad = True

def unfreeze_layer4_and_fc(model):
    for p in model.parameters(): p.requires_grad = False
    for p in model.layer4.parameters(): p.requires_grad = True
    for p in model.fc.parameters():     p.requires_grad = True

def compute_prf1_cm(y_true, y_pred, num_classes):
    labels = list(range(num_classes))
    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    macro_p, macro_r, macro_f1 = p.mean(), r.mean(), f1.mean()
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return (p, r, f1, support, macro_p, macro_r, macro_f1, cm)

def plot_confusion_matrix(cm, class_names, normalize=False):
    if normalize:
        cm = cm.astype(np.float64)
        row_sums = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums!=0)
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names, yticklabels=class_names,
           ylabel='True label', xlabel='Predicted label',
           title='Confusion Matrix' + (' (Normalized)' if normalize else ''))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = cm.max() / 2. if cm.max() > 0 else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            text = f"{cm[i, j]:.2f}" if normalize else f"{int(cm[i, j])}"
            ax.text(j, i, text, ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=8)
    fig.tight_layout()
    return fig

@torch.no_grad()
def extract_features(model, loader, device):
    """
    Extract 512-D features from ResNet18's global average pool (before fc).
    Returns X (N, 512) and y (N,)
    """
    model.eval()
    # Take all layers except the final fc
    backbone = nn.Sequential(*(list(model.children())[:-1])).to(device)
    backbone.eval()

    feats, labels = [], []
    for x, y in loader:
        x = x.to(device)
        f = backbone(x)                 # (B, 512, 1, 1)
        f = torch.flatten(f, 1)         # (B, 512)
        feats.append(f.cpu().numpy())
        labels.append(y.numpy())
    X = np.concatenate(feats, axis=0) if feats else np.empty((0, 512), dtype=np.float32)
    y = np.concatenate(labels, axis=0) if labels else np.empty((0,), dtype=np.int64)
    return X, y

@torch.no_grad()
def extract_features_multiscale(model, loader, device, layers=("layer1","layer2","layer3","layer4")):
    model.eval()
    feats, labels = [], []
    for x, y in loader:
        x = x.to(device)
        z = model.conv1(x); z = model.bn1(z); z = model.relu(z); z = model.maxpool(z)
        f1 = model.layer1(z); f2 = model.layer2(f1); f3 = model.layer3(f2); f4 = model.layer4(f3)
        gap = lambda t: torch.mean(t, dim=(2,3))
        parts = []
        if "layer1" in layers: parts.append(gap(f1))
        if "layer2" in layers: parts.append(gap(f2))
        if "layer3" in layers: parts.append(gap(f3))
        if "layer4" in layers: parts.append(gap(f4))
        f = torch.cat(parts, dim=1)             # (B, 960) by default
        feats.append(f.cpu().numpy()); labels.append(y.numpy())
    import numpy as np
    X = np.concatenate(feats, axis=0)
    y = np.concatenate(labels, axis=0)
    return X, y

@torch.no_grad()
def extract_features_multiscale_resnet50(model, loader, device, layers=("layer1","layer2","layer3","layer4")):
    """
    Extract multi-scale GAP features. Works with:
      - Models exposing model.forward_features(x) -> (B, D)
      - Vanilla torchvision ResNets via manual conv/layers path
    Returns:
      X: (N, D) float32
      y: (N,)   int64
    """
    model.eval()
    feats, labels = [], []

    use_fast_path = hasattr(model, "forward_features") and callable(getattr(model, "forward_features"))

    for x, y in loader:
        x = x.to(device, non_blocking=True)

        if use_fast_path:
            f = model.forward_features(x)                 # (B, D)
        else:
            # Fallback path (vanilla resnet style)
            z = model.conv1(x); z = model.bn1(z); z = model.relu(z); z = model.maxpool(z)
            f1 = model.layer1(z); f2 = model.layer2(f1); f3 = model.layer3(f2); f4 = model.layer4(f3)
            gap = lambda t: torch.mean(t, dim=(2,3))
            parts = []
            if "layer1" in layers: parts.append(gap(f1))
            if "layer2" in layers: parts.append(gap(f2))
            if "layer3" in layers: parts.append(gap(f3))
            if "layer4" in layers: parts.append(gap(f4))
            f = torch.cat(parts, dim=1)                   # (B, D)

        feats.append(f.detach().cpu().numpy().astype(np.float32))
        labels.append(y.numpy().astype(np.int64))

    X = np.concatenate(feats, axis=0) if feats else np.empty((0, 0), dtype=np.float32)
    y = np.concatenate(labels, axis=0) if labels else np.empty((0,), dtype=np.int64)
    return X, y
