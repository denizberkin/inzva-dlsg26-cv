import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from PIL import Image


def apply_sobel(image_path):
    img = Image.open(image_path).convert("L")
    img_array = np.array(img)

    # sobel filter vertical
    #   [[-1, 0, 1],
    #   [-2, 0, 2],
    #   [-1, 0, 1]]
    kernel_v = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    kernel_h = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    grad_v = signal.convolve2d(img_array, kernel_v, mode="same")
    grad_h = signal.convolve2d(img_array, kernel_h, mode="same")

    combined = np.sqrt(grad_v**2 + grad_h**2)  # magnitude

    # plots
    _, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].imshow(img_array, cmap="gray")
    axes[0, 0].set_title("1. Original Image")

    axes[0, 1].imshow(np.abs(grad_v), cmap="gray")
    axes[0, 1].set_title("2. Vertical Edge Detection (Gx)")

    axes[1, 0].imshow(np.abs(grad_h), cmap="gray")
    axes[1, 0].set_title("3. Horizontal Edge Detection (Gy)")

    axes[1, 1].imshow(combined, cmap="gray")
    axes[1, 1].set_title("4. Combined Magnitude (Total Edges)")

    for ax in axes.ravel():
        ax.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    apply_sobel("data/snail.png")
