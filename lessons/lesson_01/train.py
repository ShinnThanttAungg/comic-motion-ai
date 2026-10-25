import numpy as np
import matplotlib.pyplot as plt

from neural_network import NeuralNetwork

# step 1: Generate a dataset of 8x8 images with random patterns and corresponding labels
def generate_dataset(num_samples=1200, seed=42):
    rng = np.random.default_rng(seed)
    X = []
    y = []

    for _ in range(num_samples):
        image = np.zeros((8, 8), dtype=float)

        # Random class: 0 = horizontal, 1 = vertical
        label = rng.integers(0, 2)

        # Random line position
        position = rng.integers(2, 6)

        if label == 0:  # horizontal line
            image[position, 1:7] = 1.0
        else:  # vertical line
            image[1:7, position] = 1.0

        # Add random noise
        noise = rng.normal(0, 0.35, (8, 8))
        image = np.clip(image + noise, 0, 1)

        X.append(image.flatten())
        y.append(label)

    return np.array(X), np.array(y).reshape(-1, 1)

# step 2: Split the dataset into training and testing sets
X, y = generate_dataset()

rng = np.random.default_rng(123)
indices = rng.permutation(len(X))

split = int(0.8 * len(X))

train_indices = indices[:split]
test_indices = indices[split:]

X_train = X[train_indices]
y_train = y[train_indices]

X_test = X[test_indices]
y_test = y[test_indices]    

# step 3: Initialize the neural network
model = NeuralNetwork(input_size=64, hidden_size=64, learning_rate=1.0)

# step 4: Train the neural network
epochs = 1000
loss_history = []

for epoch in range(epochs):
    loss = model.train_step(X_train, y_train)
    loss_history.append(loss)

    if (epoch + 1) % 50 == 0:
        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"| Loss: {loss:.4f} "
        )

# step 5: Evaluate the model on the test set
predictions = model.predict(X_test)

accuracy = np.mean(predictions == y_test)

print(f"Test samples: {len(y_test)}")
print(f"Correct predictions: {np.sum(predictions == y_test)}")
print(f"Incorrect predictions: {np.sum(predictions != y_test)}")
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# step 6: Save model weights 
np.savez("model_weights.npz", W1=model.W1, b1=model.b1, W2=model.W2, b2=model.b2)

print("\nModel weights saved to 'model_weights.npz'.")

# step 7: Visualize the training loss over epochs
plt.plot(loss_history)
plt.title("Training Loss Over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Binary Cross-Entropy Loss")
plt.grid(True)
plt.show()

# step 8: Display some test images with their predicted labels
fig, axes = plt.subplots(1, 5, figsize=(12, 3))

for i, ax in enumerate(axes):
    image = X_test[i].reshape(8, 8)

    ax.imshow(image, cmap="gray_r", vmin=0, vmax=1)

    predicted = predictions[i, 0]
    actual = y_test[i, 0]

    ax.set_title(f"Pred: {predicted}\nTrue: {actual}")
    ax.axis("off")

plt.tight_layout()
plt.show()

#step 9: Save the test images and their predicted labels to a text file
with open("test_predictions.txt", "w") as f:
    for i in range(len(X_test)):
        image = X_test[i].reshape(8, 8)
        predicted = predictions[i, 0]
        actual = y_test[i, 0]

        f.write(f"Test Image {i + 1}:\n")
        f.write(f"Predicted Label: {predicted}\n")
        f.write(f"Actual Label: {actual}\n")
        f.write("Image Data:\n")
        np.savetxt(f, image, fmt="%.2f")
        f.write("\n\n")

print("\nTest images and their predicted labels saved to 'test_predictions.txt'.")

