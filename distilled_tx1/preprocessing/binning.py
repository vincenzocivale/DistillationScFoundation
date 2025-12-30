"""
Expression Binning Module

Discretizes continuous gene expression values into bins for tokenization.
Based on Tahoe X1's expression binning strategy.
"""

import numpy as np
from typing import Union, Optional, List
from scipy import sparse


class ExpressionBinner:
    """
    Discretize continuous gene expression values into bins.
    
    Tahoe X1 uses 51 bins by default, with logarithmic spacing to handle
    the wide dynamic range of gene expression values.
    """
    
    def __init__(
        self,
        n_bins: int = 51,
        strategy: str = "log",
        min_value: float = 0.0,
        max_value: Optional[float] = None,
        bins: Optional[np.ndarray] = None
    ):
        """
        Initialize expression binner.
        
        Args:
            n_bins: Number of bins (default: 51 as in Tahoe X1)
            strategy: Binning strategy ("log", "linear", "quantile", or "custom")
            min_value: Minimum expression value
            max_value: Maximum expression value (auto-computed if None)
            bins: Custom bin edges (only used if strategy="custom")
        """
        self.n_bins = n_bins
        self.strategy = strategy
        self.min_value = min_value
        self.max_value = max_value
        self._bins = bins
        
    def fit(self, X: Union[np.ndarray, sparse.spmatrix]) -> "ExpressionBinner":
        """
        Fit the binner to data (compute bin edges).
        
        Args:
            X: Expression matrix (cells × genes)
            
        Returns:
            self
        """
        if sparse.issparse(X):
            X_dense = X.data
        else:
            X_dense = X.ravel()
        
        # Filter out zeros for statistics
        X_nonzero = X_dense[X_dense > self.min_value]
        
        if self.max_value is None:
            self.max_value = float(np.percentile(X_nonzero, 99.5))
        
        if self.strategy == "log":
            self._bins = self._create_log_bins()
        elif self.strategy == "linear":
            self._bins = np.linspace(self.min_value, self.max_value, self.n_bins + 1)
        elif self.strategy == "quantile":
            self._bins = np.percentile(
                X_nonzero, 
                np.linspace(0, 100, self.n_bins + 1)
            )
        elif self.strategy == "custom":
            if self._bins is None:
                raise ValueError("Must provide bins for custom strategy")
        else:
            raise ValueError(f"Unknown binning strategy: {self.strategy}")
        
        return self
    
    def _create_log_bins(self) -> np.ndarray:
        """
        Create logarithmically-spaced bins.
        
        This is the default strategy used by Tahoe X1 to handle the wide
        dynamic range of gene expression values.
        """
        # Add small epsilon to avoid log(0)
        epsilon = 1e-6
        log_min = np.log1p(self.min_value + epsilon)
        log_max = np.log1p(self.max_value + epsilon)
        
        log_bins = np.linspace(log_min, log_max, self.n_bins + 1)
        bins = np.expm1(log_bins) - epsilon
        bins[0] = self.min_value
        
        return bins
    
    def transform(self, X: Union[np.ndarray, sparse.spmatrix]) -> np.ndarray:
        """
        Bin expression values.
        
        Args:
            X: Expression matrix (cells × genes) or vector
            
        Returns:
            Binned expression values (same shape as input)
        """
        if self._bins is None:
            raise RuntimeError("Must call fit() before transform()")
        
        # Handle sparse matrices
        if sparse.issparse(X):
            X_dense = X.toarray()
        else:
            X_dense = np.asarray(X)
        
        # Digitize into bins
        binned = np.digitize(X_dense, self._bins) - 1
        
        # Clip to valid range [0, n_bins-1]
        binned = np.clip(binned, 0, self.n_bins - 1)
        
        return binned.astype(np.int32)
    
    def fit_transform(self, X: Union[np.ndarray, sparse.spmatrix]) -> np.ndarray:
        """
        Fit to data and transform in one step.
        
        Args:
            X: Expression matrix (cells × genes)
            
        Returns:
            Binned expression values
        """
        return self.fit(X).transform(X)
    
    def inverse_transform(self, binned: np.ndarray) -> np.ndarray:
        """
        Convert binned values back to approximate continuous values.
        
        Uses bin centers as approximate values.
        
        Args:
            binned: Binned expression values
            
        Returns:
            Approximate continuous expression values
        """
        if self._bins is None:
            raise RuntimeError("Must call fit() before inverse_transform()")
        
        # Calculate bin centers
        bin_centers = (self._bins[:-1] + self._bins[1:]) / 2
        
        # Map binned values to centers
        return bin_centers[binned]
    
    @property
    def bin_edges(self) -> Optional[np.ndarray]:
        """Return bin edges"""
        return self._bins
    
    @property
    def bin_centers(self) -> Optional[np.ndarray]:
        """Return bin centers"""
        if self._bins is None:
            return None
        return (self._bins[:-1] + self._bins[1:]) / 2
    
    def save(self, filepath: str):
        """Save binner configuration"""
        import json
        
        config = {
            "n_bins": self.n_bins,
            "strategy": self.strategy,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "bins": self._bins.tolist() if self._bins is not None else None
        }
        
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> "ExpressionBinner":
        """Load binner from saved configuration"""
        import json
        
        with open(filepath, 'r') as f:
            config = json.load(f)
        
        bins = np.array(config["bins"]) if config["bins"] is not None else None
        
        return cls(
            n_bins=config["n_bins"],
            strategy=config["strategy"],
            min_value=config["min_value"],
            max_value=config["max_value"],
            bins=bins
        )
    
    def __repr__(self) -> str:
        return (
            f"ExpressionBinner(n_bins={self.n_bins}, strategy='{self.strategy}', "
            f"min_value={self.min_value}, max_value={self.max_value})"
        )


def normalize_expression(
    X: Union[np.ndarray, sparse.spmatrix],
    method: str = "log1p",
    target_sum: Optional[float] = 1e4
) -> np.ndarray:
    """
    Normalize gene expression values.
    
    Args:
        X: Expression matrix (cells × genes)
        method: Normalization method ("log1p", "zscore", or "minmax")
        target_sum: Target sum for library size normalization (before log)
        
    Returns:
        Normalized expression matrix
    """
    if sparse.issparse(X):
        X = X.toarray()
    
    X = np.asarray(X, dtype=np.float32)
    
    if method == "log1p":
        # Library size normalization + log1p (standard scRNA-seq preprocessing)
        if target_sum is not None:
            lib_sizes = X.sum(axis=1, keepdims=True)
            X = X / lib_sizes * target_sum
        
        X = np.log1p(X)
    
    elif method == "zscore":
        # Z-score normalization (per gene)
        mean = X.mean(axis=0, keepdims=True)
        std = X.std(axis=0, keepdims=True)
        X = (X - mean) / (std + 1e-8)
    
    elif method == "minmax":
        # Min-max normalization (per gene)
        min_val = X.min(axis=0, keepdims=True)
        max_val = X.max(axis=0, keepdims=True)
        X = (X - min_val) / (max_val - min_val + 1e-8)
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return X
