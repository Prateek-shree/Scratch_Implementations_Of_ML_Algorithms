import numpy as np

class KNNRegressor:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def _euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2, axis=1))

    def _predict_one(self, x):
        # Compute distances
        distances = self._euclidean_distance(self.X_train, x)

        # Get indices of k nearest neighbors
        k_indices = np.argsort(distances)[:self.k]

        # Get k nearest target values
        k_nearest_values = self.y_train[k_indices]

        # Return mean of neighbors
        return np.mean(k_nearest_values)

    def predict(self, X):
        X = np.array(X)
        predictions = [self._predict_one(x) for x in X]
        return np.array(predictions)


# Training data
X_train = np.array([[1], [2], [3], [4], [5]])
y_train = np.array([2, 4, 6, 8, 10])

# Test data
X_test = np.array([[1.5], [3.5], [6]])

# Create and train model
knn = KNNRegressor(k=3)
knn.fit(X_train, y_train)

# Predict
y_pred = knn.predict(X_test)
print("Predictions:", y_pred)
