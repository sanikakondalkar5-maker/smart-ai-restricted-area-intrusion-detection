
import numpy as np
import matplotlib.pyplot as plt


def generate_csi_data(samples=100):
    """
    Generate demonstration-only Wi-Fi CSI-like data.
    This is simulated data, not real Wi-Fi CSI.
    """
    time = np.arange(samples)

    normal_signal = (
        1.0
        + 0.05 * np.sin(0.2 * time)
        + np.random.normal(0, 0.02, samples)
    )

    movement = np.zeros(samples)
    movement[40:65] = 0.35

    csi_signal = normal_signal + movement

    return time, csi_signal
    
def detect_movement(csi_signal, threshold=0.15):
    baseline = np.median(csi_signal)
    deviation = np.abs(csi_signal - baseline)
    movement_detected = deviation > threshold
    return movement_detected




if __name__ == "__main__":
    time, csi = generate_csi_data()

    movement = detect_movement(csi)
    print("Movement detected samples:", np.sum(movement))
    print("Movement detected:", np.any(movement))

    plt.figure(figsize=(10, 5))
    plt.plot(time, csi, label="Simulated CSI Signal")
    plt.axvspan(40, 64, alpha=0.2, label="Simulated Movement")
    plt.title("Wi-Fi CSI Intrusion Detection - Demo")
    plt.xlabel("Sample Number")
    plt.ylabel("Signal Amplitude (Simulated)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
