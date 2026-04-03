import os

# Create main folders
folders = [
    'Assignment-1-Linear-Regression',
    'Assignment-2-Multiple-Linear-Regression',
    'Assignment-3-Regularization',
    'Assignment-4-KNN',
    'Assignment-5-Logistic-Regression',
    'Assignment-6-KMeans',
    'Assignment-7-KMeans-Silhouette',
    'Assignment-8-DBSCAN',
    'Assignment-9-Decision-Tree'
]

print("Creating assignment folders...")
for folder in folders:
    os.makedirs(folder, exist_ok=True)
    print(f"✅ Created: {folder}")

print("\n✅ All folders created successfully!")
print("\n📌 Next steps:")
print("1. Copy each Python file to its respective folder")
print("2. Copy the corresponding CSV files to each folder")
print("3. Run each assignment using: python filename.py")