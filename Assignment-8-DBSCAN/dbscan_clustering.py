import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

print("=" * 60)
print("ASSIGNMENT 8: DBSCAN CLUSTERING")
print("Student: Deep Shekhar Halder")
print("Roll No: UG/02/BTCSE/2023/063")
print("=" * 60)

# Load data
data = pd.read_csv("Advertising.csv")

# Drop the first column if it's an unnamed index
if data.columns[0] == 'Unnamed: 0' or data.columns[0] == '':
    data = data.drop(data.columns[0], axis=1)

print(f"\n📊 Dataset: Advertising.csv")
print(f"📈 Dataset Shape: {data.shape}")

# Prepare features
if 'outcome' in data.columns:
    X = data.drop('outcome', axis=1)
else:
    X = data.copy()

# Handle missing values
X = X.fillna(X.mean())

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply DBSCAN
dbscan = DBSCAN(eps=1.0, min_samples=5)
labels = dbscan.fit_predict(X_scaled)

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = np.sum(labels == -1)

print(f"\n📊 Clustering Results:")
print(f"   Number of clusters: {n_clusters}")
print(f"   Noise points: {n_noise}")

# PCA for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(10, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', s=50, alpha=0.7)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('DBSCAN Clustering Results (Deep Shekhar Halder)')
plt.colorbar(scatter, label='Cluster ID')
plt.grid(True, alpha=0.3)
plt.savefig("dbscan_clusters_plot.png", dpi=150)
print("\n✅ Plot saved as: dbscan_clusters_plot.png")
plt.show()
