from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt


# Path to the image
BASE_DIR = Path(__file__).resolve().parent

IMAGE_PATH = BASE_DIR / "images" / "sample_ab.png"
OUTPUT_DIR = BASE_DIR / "extracted_panels"

OUTPUT_DIR.mkdir(exist_ok=True)

# Load the image
image = cv2.imread(str(IMAGE_PATH))

if image is None:
    raise FileNotFoundError(f"Image not found at {IMAGE_PATH}")

original = image.copy()

height, width = image.shape[:2]

print("Image size:", width, height)

# Convert to grayscale
gray = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2GRAY
)

# Detect dark regions
_, binary = cv2.threshold(
    gray,
    235,
    255,
    cv2.THRESH_BINARY_INV
)

# Join nearby lines
kernel = cv2.getStructuringElement(
    cv2.MORPH_RECT,
    (5, 5)
)

processed = cv2.morphologyEx(
    binary,
    cv2.MORPH_CLOSE,
    kernel,
    iterations=2
)

# Find contours
contours, _ = cv2.findContours(
    processed,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

# Filter panel candidates
panels = []

minimum_area = width * height * 0.02

for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)

    area = w * h

    if area < minimum_area:
        continue

    if w < width * 0.10:
        continue

    if h < height * 0.10:
        continue

    panels.append((x, y, w, h))

# Sort panels by vertical position
panels.sort(key=lambda box: (box[1], box[0]))

# Extract and save panels
for index, (x, y, w, h) in enumerate(panels, 1):
    cropped = original[y:y+h, x:x+w]

    output_path = OUTPUT_DIR / f"panel_{index:02d}.png"

    cv2.imwrite(str(output_path), cropped)

    cv2.rectangle(
        image,
        (x, y),
        (x+w, y+h),
        (0, 255, 0),
        3
    )

    print(f"Panel {index}: x={x}, y={y}, w={w}, h={h}")

# Visualize results
fig, axes = plt.subplots(1, 3, figsize=(15, 6))

axes[0].imshow(
    cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
)
axes[0].set_title("Original Comic")

axes[1].imshow(binary, cmap="gray")
axes[1].set_title("Binary Threshold")

axes[2].imshow(
    cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
)
axes[2].set_title("Detected Panels")

for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.show()
