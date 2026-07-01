from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from dataset import HerbDataset
from vit_model import VIT

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LAYER_IDX = -1
IMAGE_IDX = 0

display_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
])

model_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

val_dataset = HerbDataset(
    txt_path="data/split/val.txt",
    image_root="data/raw/val",
    transform=model_transform,
)

model = VIT().to(DEVICE)
model.load_state_dict(torch.load("vit_best.pth", map_location=DEVICE))
model.eval()

image_tensor, label = val_dataset[IMAGE_IDX]
image_tensor = image_tensor.unsqueeze(0).to(DEVICE)

img_rel_path, _ = val_dataset.samples[IMAGE_IDX]
original_image = Image.open(Path("data/raw/val") / img_rel_path).convert("RGB")
display_image = display_transform(original_image)

with torch.no_grad():
    logits = model(image_tensor)
    pred = logits.argmax(dim=1).item()

attn = model.blocks[LAYER_IDX].mhsa.last_attn
attn = attn.cpu()

num_heads = attn.shape[1]
fig, axes = plt.subplots(2, 4, figsize=(12, 6))

for head_idx, ax in enumerate(axes.flat):
    if head_idx >= num_heads:
        ax.axis("off")
        continue

    attn_map = attn[0, head_idx, 0, 1:]
    grid_size = int(attn_map.numel() ** 0.5)
    attn_map = attn_map.reshape(1, 1, grid_size, grid_size)

    attn_map = F.interpolate(
        attn_map,
        size=(224, 224),
        mode="bilinear",
        align_corners=False,
    )[0, 0]

    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)

    ax.imshow(display_image)
    ax.imshow(attn_map, cmap="jet", alpha=0.45)
    ax.set_title(f"head {head_idx}")
    ax.axis("off")

plt.suptitle(f"label={label}, pred={pred}, layer={LAYER_IDX}")
plt.tight_layout()
plt.savefig("vit_attention_heads.png", dpi=200)
plt.show()