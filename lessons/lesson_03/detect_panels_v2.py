
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent

IMAGE_PATH = BASE_DIR / "images" / "sample.png"
OUTPUT_DIR = BASE_DIR / "extracted_panels_v2"

OUTPUT_DIR.mkdir(exist_ok=True)


# STEP 1: Load image
image = cv2.imread(str(IMAGE_PATH))

if image is None:
    raise FileNotFoundError(IMAGE_PATH)

height, width = image.shape[:2]

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# STEP 2: Detect near-white pixels
white_mask = gray >= 245


# STEP 3: Find continuous white gutter regions
def find_runs(mask, minimum_length=5):

    runs = []
    start = None

    for index, value in enumerate(mask):

        if value and start is None:
            start = index

        elif not value and start is not None:

            if index - start >= minimum_length:
                runs.append((start, index))

            start = None

    if start is not None:
        if len(mask) - start >= minimum_length:
            runs.append((start, len(mask)))

    return runs


# STEP 4: Find horizontal separators
horizontal_ratio = white_mask.mean(axis=1)

horizontal_gutters = find_runs(
    horizontal_ratio >= 0.95,
    minimum_length=5
)

print("Horizontal gutters:", horizontal_gutters)


# STEP 5: Split image into horizontal bands
def split_intervals(length, gutters):

    intervals = []
    previous = 0

    for start, end in gutters:

        if start > previous:
            intervals.append((previous, start))

        previous = end

    if previous < length:
        intervals.append((previous, length))

    return intervals


bands = split_intervals(height, horizontal_gutters)


# STEP 6: Find vertical gutters in each band
panels = []

for top, bottom in bands:

    band_mask = white_mask[top:bottom, :]

    if band_mask.size == 0:
        continue

    vertical_ratio = band_mask.mean(axis=0)

    vertical_gutters = find_runs(
        vertical_ratio >= 0.95,
        minimum_length=5
    )

    columns = split_intervals(width, vertical_gutters)

    for left, right in columns:

        w = right - left
        h = bottom - top

        if w < width * 0.08:
            continue

        if h < height * 0.08:
            continue

        panels.append((left, top, w, h))


# STEP 7: Sort by reading order
panels.sort(key=lambda p: (p[1], p[0]))


# STEP 8: Save extracted panels
preview = image.copy()

for index, (x, y, w, h) in enumerate(panels, 1):

    crop = image[y:y+h, x:x+w]

    path = OUTPUT_DIR / f"panel_{index:02d}.png"

    cv2.imwrite(str(path), crop)

    cv2.rectangle(
        preview,
        (x, y),
        (x+w, y+h),
        (0, 255, 0),
        3
    )

    cv2.putText(
        preview,
        str(index),
        (x+8, y+30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    print(
        f"Panel {index}: "
        f"x={x}, y={y}, w={w}, h={h}"
    )


print("Total panels detected:", len(panels))


# STEP 9: Visualize
fig, axes = plt.subplots(1, 2, figsize=(12, 7))

axes[0].imshow(
    cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
)
axes[0].set_title("Original Comic")

axes[1].imshow(
    cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
)
axes[1].set_title("Gutter-Based Detection")

for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.show()
