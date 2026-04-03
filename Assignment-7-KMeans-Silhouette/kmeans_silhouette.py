import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

print("=" * 60)
print("ASSIGNMENT 7: K-MEANS SILHOUETTE SCORE")
print("Student: Deep Shekhar Halder")
print("Roll No: 06/01/2023/063")
print("=" * 60)

# Load data
data = pd.read_csv('diabetes.csv')

# Preprocessing: Select numeric features
X = data.select_dtypes(include=[np.number])
if "Outcome" in X.columns:
    X = X.drop("Outcome", axis=1)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

k_values = [2, 3, 4, 5, 6]
silhouette_scores = []

print("\n📊 Silhouette Results:")
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    silhouette_scores.append(score)
    print(f"   K = {k}, Silhouette Score = {score:.4f}")

plt.figure(figsize=(10, 6))
plt.plot(k_values, silhouette_scores, marker='o', color='crimson', lw=2)
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette Score")
plt.title("K-Means: Silhouette Score Comparison (Deep Shekhar Halder)")
plt.grid(True, alpha=0.3)
plt.savefig("kmeans_silhouette_plot.png", dpi=150)
print("\n✅ Plot saved as: kmeans_silhouette_plot.png")
plt.show()