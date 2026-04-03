import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class MultipleLinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = 0
        self.mean = None
        self.std = None
        self.cost_history = []

    def normalize_features(self, X):
        if self.mean is None or self.std is None:
            self.mean = np.mean(X, axis=0)
            self.std = np.std(X, axis=0)
        self.std[self.std == 0] = 1
        return (X - self.mean) / self.std

    def add_bias_term(self, X):
        return np.c_[np.ones(X.shape[0]), X]

    def initialize_weights(self, n_features):
        self.weights = np.random.randn(n_features + 1, 1) * 0.01

    def compute_gradient(self, X, y, y_pred):
        m = X.shape[0]
        gradient = (2 / m) * X.T.dot(y_pred - y)
        return gradient

    def cost_function(self, y_pred, y):
        m = y.shape[0]
        return np.mean((y_pred - y) ** 2)

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y).reshape(-1, 1)
        X_normalized = self.normalize_features(X)
        X_with_bias = self.add_bias_term(X_normalized)
        self.initialize_weights(X.shape[1])
        
        m = X_with_bias.shape[0]
        self.cost_history = []
        
        for iteration in range(self.n_iterations):
            y_pred = X_with_bias.dot(self.weights)
            cost = self.cost_function(y_pred, y)
            self.cost_history.append(cost)
            gradient = self.compute_gradient(X_with_bias, y, y_pred)
            self.weights = self.weights - self.learning_rate * gradient
            
            if iteration % 200 == 0:
                print(f"Iteration {iteration}, Cost: {cost:.6f}")

    def predict(self, X):
        X_normalized = (X - self.mean) / self.std
        X_with_bias = self.add_bias_term(X_normalized)
        return X_with_bias.dot(self.weights)

    def score(self, X, y):
        y_pred = self.predict(X)
        y = np.array(y).reshape(-1, 1)
        ss_total = np.sum((y - np.mean(y)) ** 2)
        ss_residual = np.sum((y - y_pred) ** 2)
        r_squared = 1 - (ss_residual / ss_total)
        return r_squared

def main():
    print("=" * 60)
    print("ASSIGNMENT 2: MULTIPLE LINEAR REGRESSION")
    print("Student: Deep Shekhar Halder")
    print("Roll No: 06/01/2023/063")
    print("=" * 60)
    
    np.random.seed(42)
    X = np.random.randn(200, 3) * 10 + 50
    y = 3.5 * X[:, 0] + 2.1 * X[:, 1] + 1.8 * X[:, 2] + np.random.randn(200) * 10
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = MultipleLinearRegression(learning_rate=0.01, n_iterations=2000)
    model.fit(X_train, y_train)
    
    r2 = model.score(X_test, y_test)
    print(f"\nModel Performance:")
    print(f"R-Squared Score: {r2:.4f}")
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(model.cost_history)), model.cost_history, color='steelblue', lw=2)
    plt.xlabel('Iterations')
    plt.ylabel('Cost (MSE)')
    plt.title('Multiple Linear Regression: Cost History (Deep Shekhar Halder)')
    plt.grid(True, alpha=0.3)
    plt.savefig("multiple_regression_plot.png", dpi=150)
    print("\n✅ Plot saved as: multiple_regression_plot.png")
    plt.show()

if __name__ == "__main__":
    main()