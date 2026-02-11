import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.signal import convolve2d


def compare_blurs(image_path, kernel_size=11, sigma=2.0):
    img = Image.open(image_path).convert("L")
    img_array = np.array(img)

    box_kernel = np.ones((kernel_size, kernel_size)) / (kernel_size**2)

    ax = np.linspace(-(kernel_size - 1) / 2.0, (kernel_size - 1) / 2.0, kernel_size)
    gauss_1d = np.exp(-0.5 * (ax / sigma) ** 2)
    gauss_kernel = np.outer(gauss_1d, gauss_1d)
    gauss_kernel /= gauss_kernel.sum()

    # convolution
    box_blurred = convolve2d(img_array, box_kernel, mode="same", boundary="fill")
    gauss_blurred = convolve2d(img_array, gauss_kernel, mode="same", boundary="fill")

    _, axes = plt.subplots(1, 3, figsize=(18, 6))

    # original
    axes[0].imshow(img_array, cmap="gray")
    axes[0].set_title("Original Image", fontsize=14)
    axes[0].axis("off")

    # box blur
    axes[1].imshow(box_blurred, cmap="gray")
    axes[1].set_title(f"Box Blur\n({kernel_size}x{kernel_size} Equal Weights)", fontsize=14)
    axes[1].axis("off")

    # gaussian blur
    axes[2].imshow(gauss_blurred, cmap="gray")
    axes[2].set_title(f"Gaussian Blur\n(Sigma={sigma}, Weighted)", fontsize=14)
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()


def get_circle_image():
    x, y = np.ogrid[:50, :50]
    mask = (x - 25) ** 2 + (y - 25) ** 2 < 15**2
    my_image = np.zeros((50, 50))
    my_image[mask] = 1.0
    return my_image


if __name__ == "__main__":
    my_image = get_circle_image()

    compare_blurs("data/snail.png", kernel_size=15, sigma=10.0)
