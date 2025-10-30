import pandas as pd
import numpy as np
import math
from collections import Counter

def euclidean_distance(row1, row2):
    distance = 0.0
    if len(row1) != len(row2):
        raise ValueError("Feature vectors must have the same dimension.")
        
    for i in range(len(row1)):
        distance += (row1[i] - row2[i])**2
        
    return math.sqrt(distance)


def get_neighbors(X_train, y_train, test_row, k):
    distances = []
    
    for i in range(len(X_train)):
        train_row = X_train[i]
        label = y_train[i]
        dist = euclidean_distance(test_row, train_row)
        distances.append((label, dist))
        
    distances.sort(key=lambda x: x[1])
    neighbors = [item[0] for item in distances[:k]]
    return neighbors


def predict_classification(X_train, y_train, test_row, k):
    neighbors = get_neighbors(X_train, y_train, test_row, k)
    most_common = Counter(neighbors).most_common(1)
    return most_common[0][0]


def knn_algorithm(X_train, y_train, X_test, k):
    predictions = []
    for test_row in X_test:
        output = predict_classification(X_train, y_train, test_row, k)
        predictions.append(output)
        
    return np.array(predictions)


def accuracy_metric(actual, predicted):
    correct = 0
    if len(actual) != len(predicted):
        print("Warning: Actual and predicted arrays have different lengths.")
        return 0.0
        
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
            
    return (correct / float(len(actual))) * 100.0


class KNNClassifier:
    """
    KNN Classifier wrapper class to make the model compatible with SHAP.
    This class stores training data and provides a predict method that works with SHAP's API.
    """
    def __init__(self, k=5):
        self.k = k
        self.X_train = None
        self.y_train = None
        
    def fit(self, X_train, y_train):
        """Store training data"""
        self.X_train = X_train
        self.y_train = y_train
        return self
        
    def predict(self, X_test):
        """Predict class labels for test data"""
        if self.X_train is None or self.y_train is None:
            raise ValueError("Model must be fit before predicting")
        return knn_algorithm(self.X_train, self.y_train, X_test, self.k)
    
    def predict_proba(self, X_test):
        """
        Predict class probabilities for test data.
        Required for SHAP explainability.
        Returns probabilities for each class [P(class=0), P(class=1)]
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError("Model must be fit before predicting")
            
        probas = []
        for test_row in X_test:
            neighbors = get_neighbors(self.X_train, self.y_train, test_row, self.k)
            # Count neighbors for each class
            class_counts = Counter(neighbors)
            # Calculate probabilities
            prob_0 = class_counts.get(0, 0) / self.k
            prob_1 = class_counts.get(1, 0) / self.k
            probas.append([prob_0, prob_1])
            
        return np.array(probas)