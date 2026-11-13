from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from PIL import Image


# Step 1: Locate the image
BASE_DIR = Path(__file__).resolve().parent
IMAGE_PATH = BASE_DIR / "images" / "sample_ab.png"

# Step 2: Open the image
image = Image.open(IMAGE_PATH).convert("RGB")  # Convert to RGB

print("Original image size:", image.size)

# Step 3: Convert to the grayscale
gray_image = image.convert("L")  # Convert to grayscale

# Step 4: Resize the image
resized_image = gray_image.resize(
    (128, 128),
    Image.Resampling.LANCZOS
)

# Step 5: Convert to a NumPy array
image_array = np.array(
    resized_image,
    dtype=np.float32
)

# Normalize from [0, 255] to [0, 1]
image_array /= 255.0


# Step 6: Convert to a PyTorch tensor
tensor = torch.from_numpy(image_array).unsqueeze(0).unsqueeze(0)

print("Tensor shape:", tensor.shape)
print("Minimum value:", tensor.min().item())
print("Maximum value:", tensor.max().item())

# Step 7: Visualize the preprocessed image
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

axes[0].imshow(image)
axes[0].set_title("Original Image")

axes[1].imshow(gray_image, cmap="gray")
axes[1].set_title("Grayscale Image")

axes[2].imshow(image_array, cmap="gray")
axes[2].set_title("Resized 128x128 Image")

for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.show()
