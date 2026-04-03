import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score

print("=" * 60)
print("ASSIGNMENT 3: RIDGE, LASSO & ELASTIC NET")
print("Student: Deep Shekhar Halder")
print("Roll No: 06/01/2023/063")
print("=" * 60)

# Load your dataset (shared with Assignment 1)
try:
    data = pd.read_csv("../Assignment-1-Linear-Regression/test_data.csv")
    X = data[['SAT']].values
    y = data['GPA'].values
    print(f"\n📊 Dataset: test_data.csv (SAT vs GPA)")
except FileNotFoundError:
    # Fallback to generated data if file is missing
    np.random.seed(42)
    X = np.random.randn(200, 3) * 10 + 50
    y = 3.5 * X[:, 0] + 2.1 * X[:, 1] + 1.8 * X[:, 2] + np.random.randn(200) * 10
    print(f"\n📊 Dataset: Generated (test_data.csv not found)")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define models
models = {
    'Linear Regression': LinearRegression(),
    'Ridge (α=1)': Ridge(alpha=1),
    'Ridge (α=10)': Ridge(alpha=10),
    'Lasso (α=0.1)': Lasso(alpha=0.1),
    'Lasso (α=1)': Lasso(alpha=1),
    'Elastic Net (α=0.1)': ElasticNet(alpha=0.1, l1_ratio=0.5),
    'Elastic Net (α=1)': ElasticNet(alpha=1, l1_ratio=0.5)
}

# Train and evaluate
results = []
print("\n📊 Model Performance Comparison:")
print("-" * 80)
print(f"{'Model':<25} | {'Train R²':^10} | {'Test R²':^10} | {'Test MSE':^12}")
print("-" * 80)

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    
    results.append({
        'name': name,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'test_mse': test_mse,
        'model': model
    })
    
    print(f"{name:<25} | {train_r2:^10.4f} | {test_r2:^10.4f} | {test_mse:^12.4f}")

# Create plots
plt.figure(figsize=(12, 6))

# Plot R² Comparison
names = [r['name'] for r in results]
test_r2_scores = [r['test_r2'] for r in results]

plt.bar(names, test_r2_scores, color='steelblue')
plt.xlabel('Models')
plt.ylabel('R² Score')
plt.title('Assignment 3: Regularization Comparison (Deep Shekhar Halder)')
plt.xticks(rotation=45, ha='right')
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('regularization_plot.png', dpi=150)
plt.show()

print(f"\n✅ Plot saved as: regularization_plot.png")
print(f"🏆 Best Model: {results[np.argmax([r['test_r2'] for r in results])]['name']}")
print("=" * 60)