import os
from PIL import Image
from torch.utils.data import Dataset

# ---------------- Your dataset class ----------------
class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, class_to_idx=None):
        self.root_dir = root_dir
        self.transform = transform
        if class_to_idx is None:
            classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
            self.class_to_idx = {c: i for i, c in enumerate(classes)}
        else:
            self.class_to_idx = class_to_idx
        self.classes = [c for c, _ in sorted(self.class_to_idx.items(), key=lambda x: x[1])]

        self.images, self.labels = [], []
        for c, idx in self.class_to_idx.items():
            cdir = os.path.join(root_dir, c)
            if not os.path.isdir(cdir):
                continue
            for f in os.listdir(cdir):
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.images.append(os.path.join(cdir, f))
                    self.labels.append(idx)

    def __len__(self): return len(self.images)

    def __getitem__(self, i):
        img = Image.open(self.images[i]).convert('RGB')
        y = self.labels[i]
        if self.transform: img = self.transform(img)
        return img, y