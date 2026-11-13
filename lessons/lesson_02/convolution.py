import numpy as np
import matplotlib.pyplot as plt


def convolution2d(image, kernel, stride=1, padding=0):
    """
    Apply a 2D cross-correlation operation.
    CNN libraries conventionally call this convolution.
    """

    if stride < 1 or padding < 0:
        raise ValueError("Invalid stride or padding values.")

    # Add zero padding around to the image
    padded_image = np.pad(
        image, 
        ((padding, padding), (padding, padding)),
        mode = 'constant'
    )

    image_height, image_width = padded_image.shape
    kernel_height, kernel_width = kernel.shape

    # Calculate output dimensions
    output_height = (image_height - kernel_height) // stride + 1
    output_width = (image_width - kernel_width) // stride + 1

    if output_height <= 0 or output_width <= 0:
        raise ValueError("Kernel size is too large for the given image dimensions.")

    output = np.zeros((output_height, output_width))

    # Slide the kernel over the image
    for i in range(0, output_height):
        for j in range(0, output_width):

            start_y = i * stride
            start_x = j * stride

            region = padded_image[
                start_y : start_y + kernel_height,
                start_x : start_x + kernel_width
            ]

            output[i, j] = np.sum(region * kernel)

    return output

# Create 32x32 image synthetic comic-like image
image = (np.zeros((32, 32), dtype=float))

# Draw a rectangular outline
image[6:26, 6] = 1.0
image[6:26, 25] = 1.0
image[6, 6:26] = 1.0
image[25, 6:26] = 1.0

# Add a diagonal action line
for i in range(8, 24):
    image[i, i] = 1.0

# Define edge detection filters
vertical_kernel = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
])

horizontal_kernel = np.array([
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]
])

# Apply convolution with vertical and horizontal kernels
vertical_edges = convolution2d(
    image, vertical_kernel, stride=2, padding=1
)

horizontal_edges = convolution2d(
    image, horizontal_kernel, stride=1, padding=1
)

# Visualize the results
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

images = [
        image,
        vertical_edges,
        horizontal_edges
]

titles = [
        "Original Image",
        "Vertical Edges Filter",
        "Horizontal Edges Filter"
]

for ax, img, title in zip(axes, images, titles):
    ax.imshow(img, cmap='gray')
    ax.set_title(title)
    ax.axis('off')

plt.tight_layout()
plt.show()
