# %%
from typing import cast

import torch
from numpy._core.numeric import indices
from PIL import Image
from torch import Tensor
from torchvision import models, transforms

# %%
models.list_models()

# %%
vit = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
# %%
vit
# %%
preprocess = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# %%
img = Image.open("./persian-cat.webp")

# %%
print(img)
# %%
img

# %%
image_tensor: Tensor = cast(Tensor, preprocess(img))

# %%
batch_tensor = torch.unsqueeze(image_tensor, 0)

# %%
vit.eval()
# %%
output = vit(batch_tensor)
output
# %%
with open("./imagenet_classes.txt") as f:
    labels = [line.strip() for line in f.readlines()]

# %%
_, index = torch.max(output, 1)
index.item()

# %%
percentage = torch.nn.functional.softmax(output, dim=1)[0] * 100
print(labels[index.item()])
print(percentage[index.item()].item())
# %%
# Let's see other predictions
_, indices = torch.sort(output, descending=True)
[(labels[idx.item()], percentage[idx.item()].item()) for idx in indices[0][:5]]
