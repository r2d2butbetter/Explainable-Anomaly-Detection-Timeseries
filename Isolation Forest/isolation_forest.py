import numpy as np
import pandas as pd
from typing import Union


def c(size):
    if size > 2:
        return 2 * (np.log(size-1)+0.5772156649) - 2*(size-1)/size
    if size == 2:
        return 1
    return 0


class LeafNode:
    def __init__(self, size, data):
        self.size = size
        self.data = data


class DecisionNode:
    def __init__(self, left, right, splitAtt, splitVal):
        self.left = left
        self.right = right
        self.splitAtt = splitAtt
        self.splitVal = splitVal


class IsolationTree:
    def __init__(self, height, height_limit):
        self.height = height
        self.height_limit = height_limit

    def fit(self, X: np.ndarray):
        if self.height >= self.height_limit or X.shape[0] <= 2:
            self.root = LeafNode(X.shape[0], X)
            return self.root

        num_features = X.shape[1]
        splitAtt = np.random.randint(0, num_features)
        splitVal = np.random.uniform(min(X[:, splitAtt]), max(X[:, splitAtt]))

        X_left = X[X[:, splitAtt] < splitVal]
        X_right = X[X[:, splitAtt] >= splitVal]
        
        if X_left.shape[0] == 0 or X_right.shape[0] == 0:
            self.root = LeafNode(X.shape[0], X)
            return self.root

        left = IsolationTree(self.height + 1, self.height_limit)
        right = IsolationTree(self.height + 1, self.height_limit)
        left.fit(X_left)
        right.fit(X_right)
        self.root = DecisionNode(left.root, right.root, splitAtt, splitVal)
        return self.root


class IsolationTreeEnsemble:
    def __init__(self, sample_size, n_trees=10):
        self.sample_size = sample_size
        self.n_trees = n_trees
        self.feature_names = None
        self.trees = []

    def fit(self, X: Union[np.ndarray, pd.DataFrame]):
        self.trees = []
        
        # Store feature names if DataFrame is provided
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
            X = X.values
        else:
            self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            
        n_rows = X.shape[0]
        height_limit = np.ceil(np.log2(self.sample_size))
        
        for i in range(self.n_trees):
            # Use sampling with replacement for better diversity
            data_index = np.random.randint(0, n_rows, self.sample_size)
            X_sub = X[data_index]
            tree = IsolationTree(0, height_limit)
            tree.fit(X_sub)
            self.trees.append(tree)
        return self

    def path_length(self, X:np.ndarray) -> np.ndarray:
        paths = []
        for row in X:
            path = []
            for tree in self.trees:
                node = tree.root
                length = 0
                while isinstance(node, DecisionNode):
                    if row[node.splitAtt] < node.splitVal:
                        node = node.left
                    else:
                        node = node.right
                    length += 1
                leaf_size = node.size
                pathLength = length + c(leaf_size)
                path.append(pathLength)
            paths.append(path)
        paths = np.array(paths)
        return np.mean(paths, axis=1)

    def anomaly_score(self, X:pd.DataFrame) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            X = X.values
        avg_length = self.path_length(X)
        scores = np.array([np.power(2, -l/c(self.sample_size))for l in avg_length])
        return scores

    def predict_from_anomaly_scores(self, scores:np.ndarray, threshold:float) -> np.ndarray:
        return np.array([1 if s >= threshold else 0 for s in scores])