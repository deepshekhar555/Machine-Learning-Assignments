import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

print("=" * 60)
print("ASSIGNMENT 6: K-MEANS ELBOW METHOD")
print("Student: Deep Shekhar Halder")
print("Roll No: UG/02/BTCSE/2023/063")
print("=" * 60)

# Load data
df = pd.read_csv('titanic_toy.csv')

# Preprocessing: Fill missing and drop categorical non-numeric if any
df_numeric = df.select_dtypes(include=[np.number])
df_clean = df_numeric.fillna(df_numeric.mean())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_clean)

# Elbow Method
inertia = []
K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(K_range, inertia, marker='o', color='steelblue', lw=2)
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia (Sum of Squares)")
plt.title("K-Means: Elbow Method Analysis (Deep Shekhar Halder)")
plt.grid(True, alpha=0.3)
plt.savefig("kmeans_elbow_plot.png", dpi=150)
print("\n✅ Plot saved as: kmeans_elbow_plot.png")
plt.show()
