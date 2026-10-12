
import joblib
import numpy as np

model = joblib.load("intrusion_model.joblib")

print("\nSMART AI INTRUSION DETECTION - MODEL TEST")
print("Enter values between 0 and 1.")
print("0 = low activity, 1 = high activity\n")

motion = float(input("Enter motion level (0-1): "))
signal = float(input("Enter signal change (0-1): "))
zone_activity = float(input("Enter restricted-zone activity (0-1): "))

if not all(0 <= v <= 1 for v in [motion, signal, zone_activity]):
    print("Error: Values must be between 0 and 1.")
else:
    features = np.array([[motion, signal, zone_activity]])
    prediction = model.predict(features)[0]

    if prediction == 1:
        print("\nMODEL OUTPUT: SUSPICIOUS")
    else:
        print("\nMODEL OUTPUT: NORMAL")

    print("Note: Demonstration only; not validated for real security use.")
