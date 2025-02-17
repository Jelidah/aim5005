import numpy as np
from typing import List, Tuple
### YOU MANY NOT ADD ANY MORE IMPORTS (you may add more typing imports)

class MinMaxScaler:
    def __init__(self):
        self.minimum = None
        self.maximum = None
        
    def _check_is_array(self, x: np.ndarray) -> np.ndarray:
        """
        Try to convert x to a np.ndarray if it is not a np.ndarray and return. 
        If it can't be cast, raise an error.
        """
        if not isinstance(x, np.ndarray):
            x = np.array(x)
            
        assert isinstance(x, np.ndarray), "Expected the input to be a numpy array"
        return x
    
    def fit(self, x: np.ndarray) -> None:
        x = self._check_is_array(x)
        self.minimum = x.min(axis=0)
        self.maximum = x.max(axis=0)
        
    def transform(self, x: np.ndarray) -> np.ndarray:
        """
        MinMax Scale the given vector
        """
        x = self._check_is_array(x)
        diff_max_min = self.maximum - self.minimum
        
        # Handle division by zero (if max == min)
        if np.any(diff_max_min == 0):
            return np.zeros_like(x, dtype=float)
        
        return (x - self.minimum) / diff_max_min  # Fixed formula
    
    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        x = self._check_is_array(x)
        self.fit(x)
        return self.transform(x)

class StandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None  # Standard deviation
        
    def _check_is_array(self, x: np.ndarray) -> np.ndarray:
        """
        Try to convert x to a np.ndarray if it is not a np.ndarray and return. 
        If it can't be cast, raise an error.
        """
        if not isinstance(x, np.ndarray):
            x = np.array(x)
            
        assert isinstance(x, np.ndarray), "Expected the input to be a numpy array"
        return x

    def fit(self, x: np.ndarray) -> None:
        """
        Compute the mean and standard deviation for each feature.
        """
        x = self._check_is_array(x)
        self.mean = x.mean(axis=0)
        self.std = x.std(axis=0, ddof=0)  # Use population std (ddof=0)
        
    def transform(self, x: np.ndarray) -> np.ndarray:
        """
        Standardize the given dataset.
        """
        x = self._check_is_array(x)
        
        if self.mean is None or self.std is None:
            raise ValueError("StandardScaler has not been fitted yet. Call fit() before transform().")
        
        # Handle division by zero (if std is zero)
        std_no_zeros = np.where(self.std == 0, 1, self.std)  # Replace 0 std with 1 to avoid division error
        
        return (x - self.mean) / std_no_zeros
    
    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        """
        Fit to data, then transform it.
        """
        x = self._check_is_array(x)
        self.fit(x)
        return self.transform(x)
     
class LabelEncoder:
    def __init__(self):
        self.classes_ = None  # Stores unique labels
        self.label_map = None  # Dictionary mapping labels to integers

    def fit(self, y: List[str]) -> None:
        """
        Fit the label encoder by identifying unique classes.
        """
        self.classes_ = np.unique(y)
        self.label_map = {label: idx for idx, label in enumerate(self.classes_)}

    def transform(self, y: List[str]) -> np.ndarray:
        """
        Transform labels into numerical format.
        """
        if self.label_map is None:
            raise ValueError("LabelEncoder has not been fitted yet.")

        return np.array([self.label_map[label] for label in y])

    def fit_transform(self, y: List[str]) -> np.ndarray:
        """
        Fit to data and transform it.
        """
        self.fit(y)
        return self.transform(y)

