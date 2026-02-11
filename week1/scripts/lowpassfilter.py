import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile


def denoise_from_wav(path: str, kernel_size: int = 21):
    wav_path = path
    sample_rate, data = wavfile.read(wav_path)

    if len(data.shape) > 1:
        data = data[:, 0]

    # normalize to range [-1, 1]
    original_signal = data / np.max(np.abs(data))
    low_pass_kernel = np.ones(kernel_size) / kernel_size  # sums to 1

    smoothed_signal = np.convolve(original_signal, low_pass_kernel, mode="same")

    output_path = os.path.splitext(wav_path)[0] + f"_smoothed_{kernel_size}.wav"
    wavfile.write(output_path, sample_rate, smoothed_signal.astype(np.float32))

    time_axis = np.linspace(0, len(original_signal) / sample_rate, num=len(original_signal))

    plt.figure(figsize=(12, 6))
    plt.plot(time_axis, original_signal, label="Original Signal (Noisy)", alpha=0.5, color="gray")
    plt.plot(time_axis, smoothed_signal, label="Smoothed Signal (Low-pass)", color="red", linewidth=1)

    plt.title("Time Domain: Convolution as a Low-Pass Filter")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Normalized Amplitude")
    plt.legend()
    plt.grid(True)
    plt.show()


def denoise_synthetic_signal(kernel_size: int = 21):
    t = np.linspace(0, 1, 200)
    clean_signal = np.sin(2 * np.pi * 5 * t)
    noise = np.random.normal(0, 0.4, size=t.shape)
    signal = clean_signal + noise

    kernel = np.ones(kernel_size) / kernel_size

    filtered_signal = np.convolve(signal, kernel, mode="same")

    plt.figure(figsize=(12, 6))

    plt.plot(t, signal, label="Noisy Signal (Input)", alpha=0.5, color="gray")
    plt.plot(t, clean_signal, label="Original Sine (Target)", linestyle="--", color="blue")
    plt.plot(t, filtered_signal, label="Filtered Signal (Output)", color="red", linewidth=2)

    plt.title(f"Effect of 1D Convolution (Moving Average Filter, size={kernel_size})")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    kernel_size = 40
    denoise_from_wav("data/cdl.wav", kernel_size=kernel_size)
    # denoise_from_wav("data/airport.wav", kernel_size=kernel_size)
    # denoise_synthetic_signal()
