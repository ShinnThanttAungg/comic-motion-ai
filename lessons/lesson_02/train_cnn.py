
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import TensorDataset, DataLoader
from cnn_model import ComicCNN


# STEP 1: Generate synthetic line-art images
def generate_dataset(samples=1500, seed=42):
    rng = np.random.default_rng(seed)

    images = []
    labels = []

    for _ in range(samples):
        image = np.zeros((32, 32), dtype=np.float32)

        label = int(rng.integers(0, 3))
        position = int(rng.integers(8, 24))
        thickness = int(rng.integers(1, 4))

        if label == 0:
            # Horizontal line
            image[
                position:position + thickness, 4:28
            ] = 1.0

        elif label == 1:
            # Vertical line
            image[
                4:28, position:position + thickness
            ] = 1.0

        else:
            # Diagonal line with a small offset
            offset = int(rng.integers(-4, 5))

            for i in range(4, 28):
                col = i + offset

                if 0 <= col < 32:
                    image[
                        i,
                        col:min(col + thickness, 32)
                    ] = 1.0

        # Add background noise
        noise = rng.normal(0, 0.35, (32, 32))
        image = np.clip(image + noise, 0, 1)

        images.append(image)
        labels.append(label)

    X = np.array(images, dtype=np.float32)
    y = np.array(labels, dtype=np.int64)

    # Add channel dimension: N x 1 x 32 x 32
    X = X[:, np.newaxis, :, :]

    return X, y


# STEP 2: Prepare train/test split
X, y = generate_dataset()

rng = np.random.default_rng(123)
indices = rng.permutation(len(X))

split = int(0.8 * len(X))

train_idx = indices[:split]
test_idx = indices[split:]

X_train = torch.from_numpy(X[train_idx])
y_train = torch.from_numpy(y[train_idx])

X_test = torch.from_numpy(X[test_idx])
y_test = torch.from_numpy(y[test_idx])


# STEP 3: Create DataLoaders
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)


# STEP 4: Initialize model
torch.manual_seed(42)

device = torch.device("cpu")

model = ComicCNN().to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.01
)


# STEP 5: Train CNN
epochs = 30
loss_history = []

for epoch in range(epochs):
    model.train()

    running_loss = 0.0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        # Reset gradients
        optimizer.zero_grad()

        # Forward propagation
        outputs = model(images)

        # Calculate loss
        loss = criterion(outputs, labels)

        # Backpropagation
        loss.backward()

        # Update model weights
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(train_dataset)
    loss_history.append(epoch_loss)

    print(
        f"Epoch {epoch + 1}/{epochs} "
        f"| Loss: {epoch_loss:.4f}"
    )


# STEP 6: Evaluate on test dataset
model.eval()

correct = 0
total = 0

all_predictions = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        predictions = torch.argmax(outputs, dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

        all_predictions.extend(
            predictions.cpu().numpy().tolist()
        )


accuracy = 100 * correct / total

print(f"\nTest samples: {total}")
print(f"Correct predictions: {correct}")
print(f"Incorrect predictions: {total - correct}")
print(f"Test Accuracy: {accuracy:.2f}%")


# STEP 7: Save learned parameters
torch.save(
    model.state_dict(),
    "comic_cnn_weights.pth"
)

print("CNN weights saved!")


# STEP 8: Plot training loss
plt.figure(figsize=(8, 5))

plt.plot(
    range(1, epochs + 1),
    loss_history,
    marker="o"
)

plt.title("CNN Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy Loss")
plt.grid(True)

plt.tight_layout()
plt.show()


# STEP 9: Visualize predictions
class_names = [
    "Horizontal",
    "Vertical",
    "Diagonal"
]

fig, axes = plt.subplots(1, 5, figsize=(12, 3))

for i, ax in enumerate(axes):
    image = X_test[i, 0].numpy()

    predicted = all_predictions[i]
    actual = y_test[i].item()

    ax.imshow(image, cmap="gray_r", vmin=0, vmax=1)

    ax.set_title(
        f"Pred: {class_names[predicted]}\n"
        f"True: {class_names[actual]}",
        fontsize=9
    )

    ax.axis("off")

plt.tight_layout()
plt.show()
