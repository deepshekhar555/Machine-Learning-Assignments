import pandas as pd
import matplotlib.pyplot as plt

print("=" * 60)
print("ASSIGNMENT 1: SIMPLE LINEAR REGRESSION")
print("Student: Deep Shekhar Halder")
print("Roll No: 06/01/2023/063")
print("=" * 60)

# Load the data
data = pd.read_csv('test_data.csv')
X = data['SAT']
Y = data['GPA']
n = len(X)

print(f"\n📊 Dataset: test_data.csv")
print(f"📈 Samples: {n}")

# Calculate means
meanX = X.mean()
meanY = Y.mean()

# Calculate Coefficients
numerator = ((X - meanX) * (Y - meanY)).sum()
denominator = ((X - meanX) ** 2).sum()
b1 = numerator / denominator
b0 = meanY - b1 * meanX

print(f"\n🔢 Model Coefficients:")
print(f"   Slope (b1): {b1:.6f}")
print(f"   Intercept (b0): {b0:.6f}")

# Plot
plt.figure(figsize=(10, 6))
plt.scatter(X, Y, color="steelblue", label="Data points", alpha=0.7)
plt.plot(X, b0 + b1 * X, color="crimson", lw=2, label="Regression line")
plt.xlabel("SAT Score")
plt.ylabel("GPA")
plt.title("Simple Linear Regression: SAT vs GPA (Deep Shekhar Halder)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("linear_regression_plot.png", dpi=150)
print("\n✅ Plot saved as: linear_regression_plot.png")
plt.show()