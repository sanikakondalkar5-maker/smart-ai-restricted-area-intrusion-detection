
from flask import Flask, render_template_string, request, send_file
import io
import joblib
import numpy as np
from datetime import datetime
from csi_simulator import generate_csi_data, detect_movement

app = Flask(__name__)

model = joblib.load("intrusion_model.joblib")
prediction_history = []

APPLICATIONS = [
    "Electrical Substations",
    "Industrial Safety",
    "Military and Defence Areas",
    "Data Centers and Server Rooms",
    "Warehouses and Storage Areas",
    "Hospitals and Medical Laboratories",
    "Nuclear Power Plants",
    "Research Laboratories",
    "Airports and Restricted Zones",
    "Smart Buildings",
    "Mines and Underground Tunnels",
    "Banks and Secure Vaults",
    "Railway Control Rooms",
    "Power Plants",
    "Government Restricted Buildings"
]

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Smart AI Intrusion Detection</title>

<style>
* {
    box-sizing: border-box;
}

body {
    margin: 0;
    font-family: Arial, sans-serif;
    background: #0b1220;
    color: #edf4ff;
}

header {
    background: #111d32;
    padding: 24px;
    border-bottom: 2px solid #20c9e8;
}

header h1 {
    color: #20c9e8;
    margin: 0 0 8px;
}

header p, .muted {
    color: #a9bad2;
}

main {
    max-width: 1100px;
    margin: auto;
    padding: 24px;
}

/* System Status green card */
.system-status {
    background: #f0fdf4;
    border: 2px solid #22c55e;
    border-radius: 12px;
    padding: 15px 20px;
    margin: 20px auto;
    max-width: 600px;
    text-align: left;
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.08);
}

.system-status h3 {
    color: #166534;
    margin-top: 0;
    text-align: center;
}

.system-status p {
    color: #14532d;
    margin: 10px 0;
}

/* Dashboard cards */
.cards {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
    gap: 16px;
    margin-bottom: 22px;
}

.card, .panel {
    background: #142139;
    border: 1px solid #293c59;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 20px;
}

.card h3 {
    color: #b8cbe5;
}

.value {
    font-size: 19px;
    font-weight: bold;
}

.green {
    color: #5be0a0;
}

.yellow {
    color: #ffd166;
}

/* Form controls */
select, input, button {
    width: 100%;
    padding: 12px;
    margin-top: 8px;
    margin-bottom: 12px;
    border-radius: 7px;
    font-size: 15px;
}

select, input {
    color: white;
    background: #0b1220;
    border: 1px solid #526580;
}

button {
    background: #20c9e8;
    color: #071321;
    border: none;
    font-weight: bold;
    cursor: pointer;
}

button:hover {
    background: #75e6f7;
}

/* Prediction result */
.result {
    padding: 16px;
    margin-top: 18px;
    background: #0b1220;
    border: 1px solid #354760;
    border-radius: 8px;
    overflow-wrap: anywhere;
}

.result h3 {
    margin-top: 0;
}

.result.normal {
    border: 2px solid #5be0a0;
    background: #102b28;
}

.result.suspicious {
    border: 2px solid #ff6262;
    background: #351b25;
}

.alert-normal {
    color: #5be0a0;
    font-weight: bold;
    font-size: 20px;
}

.alert-suspicious {
    color: #ff6262;
    font-weight: bold;
    font-size: 20px;
}

/* Prediction history table */
table {
    width: 100%;
    border-collapse: collapse;
}

th, td {
    padding: 12px;
    border-bottom: 1px solid #293c59;
    text-align: left;
}

footer {
    text-align: center;
    color: #9fb1cb;
    padding: 24px;
}

@media (max-width: 600px) {
    header h1 {
        font-size: 23px;
    }

    main {
        padding: 14px;
    }
}
</style>
</head>

<body>

<header>
    <h1>SMART AI-BASED RESTRICTED AREA INTRUSION DETECTION</h1>
    <p>AI Demonstration Dashboard | Restricted Zone Monitoring</p>

    <div class="system-status">
        <h3>System Status</h3>
        <p>🟢 System: Active</p>
        <p>📡 Wi-Fi CSI: Simulation Running</p>
        <p>🤖 AI Model: Loaded</p>
        <p>🚨 Intrusion Monitoring: Enabled</p>
    </div>
</header>

<main>

    <section class="cards">
        <div class="card">
            <h3>System Status</h3>
            <div class="value green">ONLINE</div>
        </div>

        <div class="card">
            <h3>AI Model</h3>
            <div class="value green">LOADED</div>
        </div>

       
<div class="card">
    <h3>Wi-Fi CSI Sensor</h3>
    <div class="value yellow">SIMULATION</div>
</div>


       
<div class="card">
    <h3>Hardware Integration</h3>
    <div class="value yellow">PENDING</div>
</div>

    </section>

    <section class="panel">
        <h2>Simulated Wi-Fi CSI Signal</h2>
        <p>Samples Generated: 100</p>
        <p>Status: DEMO MODE — No live hardware connected</p>
        <p>CSI-like signal is simulated for demonstration.</p>

        <img src="/csi-graph"
             alt="Simulated Wi-Fi CSI Signal Graph"
             style="width:100%; max-width:700px; border-radius:10px;">

        <p>
            <a href="/csi" target="_blank" style="color:#20c9e8;">
                View CSI Data
            </a>
        </p>

        <hr style="border-color:#293c59; margin:20px 0;">

        <h2>Movement Detection</h2>

        <p id="movement-alert"
           style="font-weight:bold; font-size:18px; color:#5be0a0;">
            CHECKING SIMULATED MOVEMENT...
        </p>

        <p>
            Detection Status:
            <strong id="movement-status">Checking...</strong>
        </p>

        <p>
            Movement Samples:
            <strong id="movement-count">--</strong> / 100
        </p>
    </section>

    <section class="panel">
        <h2>Restricted Area Selection</h2>

        <form method="POST" action="/predict">
            <label for="area">Application Area</label>

            <select name="area" id="area" required>
                {% for item in applications %}
                <option value="{{ item }}"
                    {% if area == item %}selected{% endif %}>
                    {{ item }}
                </option>
                {% endfor %}
            </select>

            <h3>AI Demonstration Input</h3>
            <p class="muted">
                Enter sample feature values from 0 to 1.
                These are not live sensor measurements.
            </p>

            <label for="motion">Motion Level (0–1)</label>
            <input id="motion" name="motion"
                   type="number" min="0" max="1"
                   step="0.1" value="0.1" required>

            <label for="signal">Signal Change (0–1)</label>
            <input id="signal" name="signal"
                   type="number" min="0" max="1"
                   step="0.1" value="0.1" required>

            <label for="zone">Restricted-Zone Activity (0–1)</label>
            <input id="zone" name="zone"
                   type="number" min="0" max="1"
                   step="0.1" value="0.1" required>

            <button type="submit">RUN AI PREDICTION</button>
        </form>
    </section>

    {% if result %}
    <section class="result {% if result == 'NORMAL' %}normal{% else %}suspicious{% endif %}">
        <h3>AI Prediction Result</h3>

        <p>
            <strong>Application Area:</strong> {{ area }}
        </p>

        <p>
            <strong>Model Output:</strong>
            <span class="{% if result == 'NORMAL' %}alert-normal{% else %}alert-suspicious{% endif %}">
                {{ result }}
            </span>
        </p>

        <p><strong>Test Time:</strong> {{ time }}</p>

        <p class="muted">
            Demonstration only: the model uses synthetic training data
            and is not validated for real security use.
        </p>
    </section>
    {% endif %}

    <section class="panel">
        <h2>Intrusion Alert Log</h2>
        <p>Latest AI predictions (maximum 10 records)</p>

        <div style="overflow-x:auto;">
            <table>
                <thead>
                    <tr>
                        <th>Time</th>
                        <th>Restricted Area</th>
                        <th>AI Result</th>
                    </tr>
                </thead>

                <tbody>
                    {% for item in history %}
                    <tr>
                        <td>{{ item.time }}</td>
                        <td>{{ item.area }}</td>
                        <td style="font-weight:bold; color:{% if item.result == 'SUSPICIOUS' %}#ff6262{% else %}#5be0a0{% endif %};">
                            {{ item.result }}
                        </td>
                    </tr>
                    {% else %}
                    <tr>
                        <td colspan="3">
                            No prediction records yet.
                            Run an AI prediction first.
                        </td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
    </section>

</main>

<footer>
    Smart AI-Based Restricted Area Intrusion Detection | Prototype
</footer>

<script>
let previousResult = null;

function playAlertSound() {
    try {
        const audioContext = new (
            window.AudioContext || window.webkitAudioContext
        )();

        const oscillator = audioContext.createOscillator();
        const gain = audioContext.createGain();

        oscillator.type = "sine";
        oscillator.frequency.value = 1000;
        gain.gain.value = 0.3;

        oscillator.connect(gain);
        gain.connect(audioContext.destination);

        oscillator.start();

        setTimeout(() => {
            oscillator.stop();
            audioContext.close();
        }, 700);
    } catch (error) {
        console.log("Audio could not play:", error);
    }
}

function checkLatestPrediction() {
    fetch("/latest-alert")
        .then(response => response.json())
        .then(data => {
            if (
                data.result === "SUSPICIOUS" &&
                previousResult !== "SUSPICIOUS"
            ) {
                playAlertSound();
                alert("WARNING! Suspicious activity detected!");
            }

            previousResult = data.result;
        })
        .catch(error => {
            console.log("Alert check failed:", error);
        });
}

setInterval(checkLatestPrediction, 3000);


function checkMovement() {
    fetch("/movement")
        .then(response => response.json())
        .then(data => {
            document.getElementById("movement-status").textContent =
                data.movement_detected
                    ? "MOVEMENT DETECTED"
                    : "NO MOVEMENT";

            document.getElementById("movement-count").textContent =
                data.movement_samples;

            const alertBox =
                document.getElementById("movement-alert");

            if (data.movement_detected) {
                alertBox.textContent =
                    "MOVEMENT DETECTED (SIMULATED)";
                alertBox.style.color = "#ff6262";
            } else {
                alertBox.textContent =
                    "NO MOVEMENT (SIMULATED)";
                alertBox.style.color = "#5be0a0";
            }
        })
        .catch(() => {
            document.getElementById("movement-status").textContent =
                "Unable to load";

            document.getElementById("movement-alert").textContent =
                "MOVEMENT STATUS UNAVAILABLE";
        });
}

checkMovement();
setInterval(checkMovement, 3000);

</script>

</body>
</html>
"""


@app.route("/", methods=["GET"])
def home():
    return render_template_string(
        HTML,
        applications=APPLICATIONS,
        result=None,
        area=None,
        time=None,
        history=prediction_history
    )


@app.route("/csi")
def csi_demo():
    time_values, csi_values = generate_csi_data()

    return {
        "message": "Simulated CSI data generated",
        "samples": len(csi_values),
        "first_5_values": csi_values[:5].tolist(),
        "note": "These are simulated values, not live Wi-Fi measurements."
    }


@app.route("/predict", methods=["POST"])
def predict():
    try:
        area = request.form["area"]
        motion = float(request.form["motion"])
        signal = float(request.form["signal"])
        zone = float(request.form["zone"])

        if area not in APPLICATIONS:
            return "Invalid application area", 400

        values = [motion, signal, zone]

        if not all(0 <= value <= 1 for value in values):
            return "Values must be between 0 and 1", 400

        features = np.array([values])
        prediction = int(model.predict(features)[0])
        result = "SUSPICIOUS" if prediction == 1 else "NORMAL"

        timestamp = datetime.now().strftime("%d-%m-%Y %H:%M:%S")

        prediction_history.insert(0, {
            "area": area,
            "result": result,
            "time": timestamp
        })

        if len(prediction_history) > 10:
            prediction_history.pop()

        return render_template_string(
            HTML,
            applications=APPLICATIONS,
            result=result,
            area=area,
            time=timestamp,
            history=prediction_history
        )

    except (ValueError, KeyError):
        return "Please enter valid feature values between 0 and 1.", 400


@app.route("/latest-alert")
def latest_alert():
    if prediction_history:
        latest = prediction_history[0]
        return {
            "result": latest["result"],
            "area": latest["area"],
            "time": latest["time"]
        }

    return {
        "result": "NONE",
        "area": "",
        "time": ""
    }


@app.route("/movement")
def movement_demo():
    time_values, csi_values = generate_csi_data()
    movement = detect_movement(csi_values)

    return {
        "movement_detected": bool(movement.any()),
        "movement_samples": int(movement.sum()),
        "total_samples": len(csi_values),
        "note": "Demo uses simulated CSI data, not real Wi-Fi measurements."
    }


@app.route("/csi-graph")
def csi_graph():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    time_values, csi_values = generate_csi_data()

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(time_values, csi_values)
    ax.set_title("Simulated Wi-Fi CSI Signal")
    ax.set_xlabel("Sample Number")
    ax.set_ylabel("CSI Amplitude")
    ax.grid(True)

    image_data = io.BytesIO()
    fig.savefig(image_data, format="png", bbox_inches="tight")
    plt.close(fig)
    image_data.seek(0)

    return send_file(image_data, mimetype="image/png")


if __name__ == "__main__":
    app.run(debug=True)
