import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

print("=" * 60)
print("ASSIGNMENT 9: DECISION TREE REGRESSOR")
print("Student: Deep Shekhar Halder")
print("Roll No: UG/02/BTCSE/2023/063")
print("=" * 60)

# Load data
df = pd.read_csv("Salary_Data.csv")

print("\n📊 Dataset Overview:")
print(df.head())

# Label Encoder for categorical data if any
le = LabelEncoder()
for col in df.columns:
    if df[col].dtype == "object":
        df[col] = le.fit_transform(df[col])

X = df.drop(["Salary"], axis=1)
y = df["Salary"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeRegressor(max_depth=5, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print(f"\n📊 Model Performance:")
print(f"Mean Squared Error: {mse:.4f}")

# Plot Predictions
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='steelblue', label='Predicted vs Actual')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2, label='Perfect Prediction')
plt.xlabel('Actual Salary')
plt.ylabel('Predicted Salary')
plt.title('Decision Tree Regressor: Predictions (Deep Shekhar Halder)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("decision_tree_results.png", dpi=150)
print("\n✅ Plot saved as: decision_tree_results.png")
plt.show()
