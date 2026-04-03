import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print("=" * 60)
print("ASSIGNMENT 4: KNN FROM SCRATCH")
print("Student: Deep Shekhar Halder")
print("Roll No: 06/01/2023/063")
print("=" * 60)

# Load data
data = pd.read_csv("KNNAlgorithmDataset.csv")

# Clean numeric data for KNN
if "digits" in data.columns:
    X = data.drop(["digits"], axis=1).values
    y = data["digits"].values
else:
    # Fallback to last column as target
    X = data.iloc[:, :-1].select_dtypes(include=[np.number]).values
    y = data.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

def euclidean_distance(x, y):
    return np.sqrt(np.sum((x - y)**2))

def predict(X_train, y_train, X_test_point, k):
    distances = [euclidean_distance(X_test_point, x_t) for x_t in X_train]
    k_indices = np.argsort(distances)[:k]
    k_nearest_labels = [y_train[i] for i in k_indices]
    most_common = Counter(k_nearest_labels).most_common(1)
    return most_common[0][0]

k = 3
y_pred = [predict(X_train, y_train, x, k) for x in X_test]
accuracy = np.sum(y_pred == y_test) / len(y_test)
print(f"\n📊 K Value: {k}")
print(f"✅ Accuracy: {accuracy:.4f}")

# Test different k values for plot
K_values = [1, 3, 5, 7, 9, 11]
accuracies = []
for k_val in K_values:
    preds = [predict(X_train, y_train, x, k_val) for x in X_test]
    accuracies.append(np.sum(preds == y_test) / len(y_test))

plt.figure(figsize=(10, 6))
plt.plot(K_values, accuracies, marker='o', color='steelblue', lw=2)
plt.xlabel("K Values")
plt.ylabel("Accuracy")
plt.title("KNN Analysis: K Value vs Accuracy (Deep Shekhar Halder)")
plt.grid(True, alpha=0.3)
plt.savefig("knn_accuracy_plot.png", dpi=150)
print("\n✅ Plot saved as: knn_accuracy_plot.png")
plt.show()