from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset

class HerbDataset(Dataset):
    def __init__(self, txt_path, image_root, transform=None):
        self.txt_path = Path(txt_path)
        self.image_root = Path(image_root)
        self.transform = transform
        self.samples = []

        with open(self.txt_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                img_rel_path, label = line.rsplit(",", 1)
                self.samples.append((img_rel_path, int(label)))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_rel_path, label = self.samples[idx]
        img_path = self.image_root / img_rel_path
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label