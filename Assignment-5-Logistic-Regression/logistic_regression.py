import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score

print("=" * 60)
print("ASSIGNMENT 5: LOGISTIC REGRESSION")
print("Student: Deep Shekhar Halder")
print("Roll No: 06/01/2023/063")
print("=" * 60)

# Load data
data = pd.read_csv("Salony_Data.csv")

# Create target column based on Salary threshold
if "Salary" in data.columns:
    data["HighSalary"] = (data["Salary"] >= 50000).astype(int)
else:
    # Target column already exists or different logic needed
    pass

X = data[["YearsExperience"]].values
y = data["HighSalary"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"\n📊 Accuracy Score: {accuracy_score(y_test, y_pred):.4f}")
print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred))

y_proba = model.predict_proba(X_test)[:, 1]
auc_score = roc_auc_score(y_test, y_proba)
fpr, tpr, _ = roc_curve(y_test, y_proba)

plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='crimson', label=f"ROC Curve (AUC = {auc_score:.2f})")
plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Logistic Regression: ROC Curve (Deep Shekhar Halder)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("logistic_roc_curve.png", dpi=150)
print("\n✅ Plot saved as: logistic_roc_curve.png")
plt.show()