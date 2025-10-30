import numpy as np
import pandas as pd
from typing import Union, Optional, List
import warnings

# Follows algo from https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf

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

    def fit(self, X: np.ndarray, improved=False):
        """
        Given a 2D matrix of observations, create an isolation tree. Set field
        self.root to the root of that tree and return it.

        If you are working on an improved algorithm, check parameter "improved"
        and switch to your new functionality else fall back on your original code.
        """
        if improved:
            self.improved_fit(X)
        else:
            if self.height >= self.height_limit or X.shape[0] <= 2:
                self.root = LeafNode(X.shape[0], X)
                return self.root

            # Choose Random Split Attributes and Value
            num_features = X.shape[1]
            splitAtt = np.random.randint(0, num_features)
            splitVal = np.random.uniform(min(X[:, splitAtt]), max(X[:, splitAtt]))

            X_left = X[X[:, splitAtt] < splitVal]
            X_right = X[X[:, splitAtt] >= splitVal]
            
            # Handle edge case where split results in empty partition
            if X_left.shape[0] == 0 or X_right.shape[0] == 0:
                self.root = LeafNode(X.shape[0], X)
                return self.root

            left = IsolationTree(self.height + 1, self.height_limit)
            right = IsolationTree(self.height + 1, self.height_limit)
            left.fit(X_left, improved=improved)
            right.fit(X_right, improved=improved)
            self.root = DecisionNode(left.root, right.root, splitAtt, splitVal)
            self.n_nodes = self.count_nodes(self.root)
            return self.root

    def improved_fit(self, X: np.ndarray):
        if self.height >= self.height_limit or X.shape[0] <= 2:
            self.root = LeafNode(X.shape[0], X)
            return self.root

        # Choose Best (The Most unbalanced) Random Split Attributes and Value
        num_features = X.shape[1]
        ratio_imp = 0.5  # Initialize the samples ratio after split as 0.5
        
        # Initialize variables to avoid UnboundLocalError
        splitAtt_imp = 0
        splitVal_imp = 0.0
        X_left_imp = X
        X_right_imp = np.array([])

        for i in range(num_features):
            splitAtt = i
            min_val = np.min(X[:, splitAtt])
            max_val = np.max(X[:, splitAtt])
            
            # Skip if all values are the same
            if min_val == max_val:
                continue
                
            for _ in range(10):
                splitVal = np.random.uniform(min_val, max_val)
                X_left = X[X[:, splitAtt] < splitVal]
                X_right = X[X[:, splitAtt] >= splitVal]
                
                # Skip if split results in empty partition
                if X_left.shape[0] == 0 or X_right.shape[0] == 0:
                    continue
                    
                total = X_left.shape[0] + X_right.shape[0]
                ratio = min(X_left.shape[0] / total, X_right.shape[0] / total)
                
                if ratio < ratio_imp:
                    splitAtt_imp = splitAtt
                    splitVal_imp = splitVal
                    X_left_imp = X_left
                    X_right_imp = X_right
                    ratio_imp = ratio

        # Fallback to standard split if no good split was found
        if X_left_imp.shape[0] == X.shape[0] or X_right_imp.shape[0] == 0:
            splitAtt_imp = np.random.randint(0, num_features)
            min_val = np.min(X[:, splitAtt_imp])
            max_val = np.max(X[:, splitAtt_imp])
            if min_val != max_val:
                splitVal_imp = np.random.uniform(min_val, max_val)
                X_left_imp = X[X[:, splitAtt_imp] < splitVal_imp]
                X_right_imp = X[X[:, splitAtt_imp] >= splitVal_imp]
            else:
                # If still can't split, make it a leaf
                self.root = LeafNode(X.shape[0], X)
                return self.root

        left = IsolationTree(self.height + 1, self.height_limit)
        right = IsolationTree(self.height + 1, self.height_limit)
        left.improved_fit(X_left_imp)  # Use improved_fit recursively
        right.improved_fit(X_right_imp)
        self.root = DecisionNode(left.root, right.root, splitAtt_imp, splitVal_imp)
        self.n_nodes = self.count_nodes(self.root)
        return self.root

    def count_nodes(self, root):
        count = 0
        stack = [root]
        while stack:
            node = stack.pop()
            count += 1
            if isinstance(node, DecisionNode):
                stack.append(node.right)
                stack.append(node.left)
        return count


class IsolationTreeEnsemble:
    def __init__(self, sample_size, n_trees=10):
        self.sample_size = sample_size
        self.n_trees = n_trees
        self.feature_names = None
        self.trees = []

    def fit(self, X: Union[np.ndarray, pd.DataFrame], improved=False):
        """
        Given a 2D matrix of observations, create an ensemble of IsolationTree
        objects and store them in a list: self.trees.  Convert DataFrames to
        ndarray objects.
        
        Parameters:
        -----------
        X : np.ndarray or pd.DataFrame
            Training data
        improved : bool, default=False
            Whether to use improved splitting strategy
        
        Returns:
        --------
        self : IsolationTreeEnsemble
        """
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
            tree.fit(X_sub, improved=improved)
            self.trees.append(tree)
        return self

    def path_length(self, X:np.ndarray) -> np.ndarray:
        """
        Given a 2D matrix of observations, X, compute the average path length
        for each observation in X.  Compute the path length for x_i using every
        tree in self.trees then compute the average for each x_i.  Return an
        ndarray of shape (len(X),1).
        """
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
        """
        Given a 2D matrix of observations, X, compute the anomaly score
        for each x_i observation, returning an ndarray of them.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        avg_length = self.path_length(X)
        scores = np.array([np.power(2, -l/c(self.sample_size))for l in avg_length])
        return scores

    def predict_from_anomaly_scores(self, scores:np.ndarray, threshold:float) -> np.ndarray:
        """
        Given an array of scores and a score threshold, return an array of
        the predictions: 1 for any score >= the threshold and 0 otherwise.
        """
        return np.array([1 if s >= threshold else 0 for s in scores])

    def predict(self, X:np.ndarray, threshold:float) -> np.ndarray:
        "A shorthand for calling anomaly_score() and predict_from_anomaly_scores()."
        scores = self.anomaly_score(X)
        prediction = self.predict_from_anomaly_scores(scores, threshold)
        return prediction
    
    def decision_function(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Compute the anomaly score for each sample. This method is compatible
        with SHAP TreeExplainer and follows sklearn conventions.
        
        Parameters:
        -----------
        X : np.ndarray or pd.DataFrame
            Data to score
            
        Returns:
        --------
        scores : np.ndarray
            Anomaly scores for each sample
        """
        return self.anomaly_score(X)
    
    def score_samples(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Compute the anomaly score for each sample (negative of anomaly score
        for sklearn compatibility where more negative = more anomalous).
        
        Parameters:
        -----------
        X : np.ndarray or pd.DataFrame
            Data to score
            
        Returns:
        --------
        scores : np.ndarray
            Negative anomaly scores (lower = more anomalous)
        """
        # Return negative scores so lower values indicate anomalies
        return -self.anomaly_score(X)
    
    def get_feature_names(self) -> List[str]:
        """
        Get the feature names used during training.
        
        Returns:
        --------
        feature_names : list of str
            Feature names
        """
        if self.feature_names is None:
            raise ValueError("Model has not been fitted yet.")
        return self.feature_names
    
    def __call__(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Make the model callable for SHAP compatibility.
        This allows SHAP to use the model as a function.
        
        Parameters:
        -----------
        X : np.ndarray or pd.DataFrame
            Data to score
            
        Returns:
        --------
        scores : np.ndarray
            Anomaly scores for each sample
        """
        return self.anomaly_score(X)


def find_TPR_threshold(y, scores, desired_TPR):
    """
    Start at score threshold 1.0 and work down until we hit desired TPR.
    Step by 0.01 score increments. For each threshold, compute the TPR
    and FPR to see if we've reached to the desired TPR. If so, return the
    score threshold and FPR.
    
    Parameters:
    -----------
    y : array-like
        True labels (1 for anomaly, 0 for normal)
    scores : array-like
        Anomaly scores from the model
    desired_TPR : float
        Desired True Positive Rate (between 0 and 1)
        
    Returns:
    --------
    threshold : float
        Score threshold that achieves desired TPR
    FPR : float
        False Positive Rate at the threshold
    """
    y = np.array(y)
    scores = np.array(scores)
    
    TPR = 0
    FPR = 0
    threshold = 1.0
    
    while TPR < desired_TPR and threshold >= 0:
        threshold -= 0.01
        prediction = (scores > threshold).astype(int)
        
        TP = np.sum((prediction == 1) & (y == 1))
        TN = np.sum((prediction == 0) & (y == 0))
        FP = np.sum((prediction == 1) & (y == 0))
        FN = np.sum((prediction == 0) & (y == 1))
        
        # Avoid division by zero
        TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
        FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
        
        if threshold < 0:
            warnings.warn("The model cannot reach the desired TPR")
            return None, None

    return threshold, FPR