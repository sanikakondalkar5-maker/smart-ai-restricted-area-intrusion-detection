
import numpy as np
import matplotlib.pyplot as plt


def generate_csi_data(samples=100):
    """
    Generate repeatable simulated CSI-like data.
    This is demonstration data, NOT real Wi-Fi CSI.
    """
    rng = np.random.default_rng(42)
    time = np.arange(samples)

    # Simulated normal signal with small noise
    normal_signal = (
        1.0
        + 0.05 * np.sin(0.2 * time)
        + rng.normal(0, 0.02, samples)
    )

    # Simulated movement between samples 40 and 64
    movement = np.zeros(samples)
    movement[40:65] = 0.35

    csi_signal = normal_signal + movement

    return time, csi_signal


def detect_movement(csi_signal, threshold=0.15):
    """
    Detect changes in simulated signal amplitude.
    This does not confirm real human presence.
    """
    baseline = np.median(csi_signal)
    deviation = np.abs(csi_signal - baseline)

    movement_detected = deviation > threshold

    return movement_detected


if __name__ == "__main__":
    time, csi = generate_csi_data()
    movement = detect_movement(csi)

    print("CSI source: SIMULATED")
    print("Total samples:", len(csi))
    print("Movement samples:", int(np.sum(movement)))
    print("Movement detected:", bool(np.any(movement)))

    plt.figure(figsize=(10, 5))
    plt.plot(time, csi, label="Simulated CSI Signal")
    plt.axvspan(
        40, 64,
        alpha=0.2,
        label="Simulated Movement"
    )
    plt.title("Wi-Fi CSI Intrusion Detection - Simulation")
    plt.xlabel("Sample Number")
    plt.ylabel("Simulated Signal Amplitude")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
