"""Model training and prediction logic for Iris dataset."""
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

class IrisModel:
    """Machine learning model for Iris flower classification."""
    def __init__(self):
        data = load_iris()
        features = data["data"]
        label = data["target"]
        self.model = LogisticRegression(max_iter=200)
        self.model.fit(features, label)

    def predict(self, features):
        """Predict the Iris class for given features."""
        return self.model.predict([features])[0]
