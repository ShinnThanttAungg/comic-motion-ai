import torch
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from cnn_model import ComicCNN


# Step 1: Locate the trained model
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "comic_cnn_weights.pth"

model = ComicCNN()

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only = True
    )
)

model.eval()


# Step 2: Create a synthetic comic-like image for testing
image = np.zeros((32, 32), dtype=np.float32)

# Draw diagonal action line
for i in range(4, 28):
    image[i, i] = 1.0

# Add noise, similar to training experiments
rng = np.random.default_rng(42)

noise = rng.normal(0, 0.75, (32, 32))

image = np.clip(image + noise, 0, 1)
image = image.astype(np.float32)

# Step 3: Convert the image to a tensor and add batch and channel dimensions
input_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)

print("Input tensor shape:", input_tensor.shape)


# Step 4: Extract feature maps
with torch.no_grad():

    # First convolutiona + ReLU
    first_features = model.features[:2](input_tensor)

    # First pooling + second convolution + ReLU
    second_features = model.features[:5](input_tensor)

    # Final predicition
    outputs = model(input_tensor)
    prediction = torch.argmax(outputs, dim=1)

print("First features shape:", first_features.shape)
print("Second features shape:", second_features.shape)
print("Model prediction:", prediction.item())


# Step 5: Convert features maps to numpy for visualization
first_maps = first_features.squeeze(0).numpy()
second_maps = second_features.squeeze(0).numpy()


# Step 6: Visualize the first layer feature maps
fig, axes = plt.subplots(2, 5, figsize=(12, 6))

axes = axes.flatten()

axes[0].imshow(image, cmap="gray_r", vmin=0, vmax=1)
axes[0].set_title("Original Image")

for i in range(8):
    axes[i + 1].imshow(
        first_maps[i], 
        cmap="viridis"
    )

    axes[i + 1].set_title(f"Feature Map {i + 1}")

for ax in axes:
    ax.axis("off")

axes[9].axis("off")

plt.suptitle("First Convolutional Layer")
plt.tight_layout()
plt.show()


# Step 7: Visualize the second layer feature maps
fig, axes = plt.subplots(4, 4, figsize=(10, 10))

for i, ax in enumerate(axes.flatten()):
    ax.imshow(
        second_maps[i], 
        cmap="viridis"
    )

    ax.set_title(f"Feature Map {i + 1}")
    ax.axis("off")

plt.suptitle("Second Convolutional Layer")
plt.tight_layout()
plt.show()