import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile


def high_pass_filter(path: str, kernel_size: int = 21):
    sample_rate, data = wavfile.read(path)

    if len(data.shape) > 1:
        data = data[:, 0]

    original_signal = data / np.max(np.abs(data))

    low_pass_kernel = np.ones(kernel_size) / kernel_size

    high_pass_kernel = -low_pass_kernel
    high_pass_kernel[kernel_size // 2] += 1

    # Apply convolution
    high_passed_signal = np.convolve(original_signal, high_pass_kernel, mode="same")

    output_path = os.path.splitext(path)[0] + f"_highpassed_{kernel_size}.wav"
    wavfile.write(output_path, sample_rate, high_passed_signal.astype(np.float32))

    # Plotting results
    time_axis = np.linspace(0, len(original_signal) / sample_rate, num=len(original_signal))
    plt.figure(figsize=(12, 6))
    plt.plot(time_axis, original_signal, label="Original", alpha=0.5, color="gray")
    plt.plot(time_axis, high_passed_signal, label="High-Pass (Edges/Noise)", color="blue")
    plt.title(f"High-Pass Filter (Kernel Size: {kernel_size})")
    plt.legend(loc="upper right")
    plt.show()

    return high_passed_signal


if __name__ == "__main__":
    # high_pass_filter("data/cdl.wav", kernel_size=120)
    high_pass_filter("data/airport.wav", kernel_size=120)
