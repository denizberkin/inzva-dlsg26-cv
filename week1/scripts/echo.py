import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import os


def apply_echo_convolution(wav_path):
    sample_rate, data = wavfile.read(wav_path)

    if len(data.shape) > 1:
        data = data[:, 0]
    original_signal = data.astype(np.float32) / np.max(np.abs(data))

    # kernel
    delay_seconds = 0.1
    delay_samples = int(delay_seconds * sample_rate)

    # 2 echoes
    echo_kernel = np.zeros(delay_samples * 2 + 1)
    echo_kernel[0] = 1.0  # original sound (instant)
    echo_kernel[delay_samples] = 0.6  # first echo (0.5s later)
    echo_kernel[delay_samples * 2] = 0.3  # second echo (1.0s later)

    echoed_signal = np.convolve(original_signal, echo_kernel, mode="full")

    echoed_signal = echoed_signal / np.max(np.abs(echoed_signal))

    output_path = os.path.splitext(wav_path)[0] + "_echoed.wav"
    wavfile.write(output_path, sample_rate, echoed_signal.astype(np.float32))

    if wav_path.endswith("trom.wav"):  # :p hardcoded for demo
        plot_echo_effect(original_signal, echoed_signal, sample_rate)


def plot_echo_effect(original, echoed, sample_rate):
    time_original = np.linspace(0, len(original) / sample_rate, num=len(original))
    time_echoed = np.linspace(0, len(echoed) / sample_rate, num=len(echoed))

    plt.figure(figsize=(15, 6))

    plt.plot(time_echoed, echoed, label="Echoed Signal (Output)", color="skyblue", alpha=0.7)
    plt.plot(time_original, original, label="Original Signal (Input)", color="blue", alpha=0.4)
    plt.title("Full Signal Comparison")
    plt.ylabel("Amplitude")
    plt.legend()

    plt.annotate("Original Hit", xy=(0.1, 0.8), xytext=(0.2, 0.95), arrowprops=dict(facecolor="black", arrowstyle="->"))
    plt.annotate("Echo 1", xy=(0.25, 0.77), xytext=(0.3, 0.65), arrowprops=dict(facecolor="red", arrowstyle="->"))

    plt.title("Zoomed View: Seeing the Convolution Repetitions")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    apply_echo_convolution("data/trom.wav")
