
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Demonstration dataset:
# Features: motion level, signal change, entry-zone activity
# Labels: 0 = Normal, 1 = Suspicious
np.random.seed(42)

normal = np.column_stack([
    np.random.uniform(0, 0.3, 100),
    np.random.uniform(0, 0.3, 100),
    np.random.uniform(0, 0.3, 100)
])

suspicious = np.column_stack([
    np.random.uniform(0.6, 1.0, 100),
    np.random.uniform(0.6, 1.0, 100),
    np.random.uniform(0.6, 1.0, 100)
])

X = np.vstack([normal, suspicious])
y = np.array([0] * 100 + [1] * 100)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=100, random_state=42
)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("AI Demonstration Model Trained Successfully")
print(f"Test accuracy on synthetic data: {accuracy:.2%}")
print("\nClassification Report:")
print(classification_report(
    y_test, predictions,
    target_names=["Normal", "Suspicious"]
))

joblib.dump(model, "intrusion_model.joblib")
print("Model saved as intrusion_model.joblib")
print("Note: This uses synthetic data, not real intrusion data.")
