
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch

from PIL import Image, ImageOps


BASE_DIR = Path(__file__).resolve().parent
IMAGE_PATH = BASE_DIR / "images" / "sample_ba.png"

TARGET_SIZE = (128, 128)


# STEP 1: Load the image
image = Image.open(IMAGE_PATH).convert("RGB")

original_width, original_height = image.size

print("Original size:", image.size)


# STEP 2: Convert to grayscale
gray_image = image.convert("L")


# STEP 3: Resize without changing aspect ratio
resized_image = ImageOps.contain(
    gray_image,
    TARGET_SIZE,
    method=Image.Resampling.LANCZOS
)

print("Proportional size:", resized_image.size)


# STEP 4: Add white padding
padded_image = ImageOps.pad(
    gray_image,
    TARGET_SIZE,
    method=Image.Resampling.LANCZOS,
    color=255,
    centering=(0.5, 0.5)
)


# STEP 5: Normalize image
image_array = np.asarray(
    padded_image,
    dtype=np.float32
) / 255.0


# STEP 6: Convert to PyTorch tensor
tensor = torch.from_numpy(image_array)

tensor = tensor.unsqueeze(0).unsqueeze(0)

print("Tensor shape:", tensor.shape)


# STEP 7: Visualize results
fig, axes = plt.subplots(1, 3, figsize=(12, 5))

axes[0].imshow(image)
axes[0].set_title("Original Image")

axes[1].imshow(gray_image, cmap="gray")
axes[1].set_title("Grayscale Image")

axes[2].imshow(padded_image, cmap="gray")
axes[2].set_title("Aspect Ratio Preserved")

for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.show()
