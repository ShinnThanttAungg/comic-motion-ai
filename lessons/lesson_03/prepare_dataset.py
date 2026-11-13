from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_DIR = PROJECT_ROOT / "data" / "processed" / "panels"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "panels_128"

TARGET_SIZE = 128


def resize_with_padding(image, target_size):
    height, width = image.shape[:2]

    # Scale both dimensions equally to preserve the artwork's proportions.
    scale = min(target_size / width, target_size / height)

    new_width = max(1, round(width * scale))
    new_height = max(1, round(height * scale))

    interpolation = cv2.INTER_AREA
    if scale > 1:
        interpolation = cv2.INTER_LINEAR

    resized = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=interpolation
    )

    # Create a white square with three color channels.
    canvas = np.full(
        (target_size, target_size, 3),
        255,
        dtype=np.uint8
    )

    left = (target_size - new_width) // 2
    top = (target_size - new_height) // 2

    canvas[
        top:top + new_height,
        left:left + new_width
    ] = resized

    return canvas


def main():
    image_paths = sorted(INPUT_DIR.glob("*/panel_*.png"))

    if not image_paths:
        raise FileNotFoundError(
            f"No extracted panels found in {INPUT_DIR}"
        )

    for image_path in image_paths:
        image = cv2.imread(str(image_path))

        if image is None:
            raise ValueError(f"Could not read {image_path}")

        prepared = resize_with_padding(image, TARGET_SIZE)

        # Preserve the source-page folder and panel filename.
        relative_path = image_path.relative_to(INPUT_DIR)
        output_path = OUTPUT_DIR / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not cv2.imwrite(str(output_path), prepared):
            raise OSError(f"Could not save {output_path}")

        print(f"{relative_path} -> {prepared.shape}")

    print(f"\nTotal panels prepared: {len(image_paths)}")


if __name__ == "__main__":
    main() 