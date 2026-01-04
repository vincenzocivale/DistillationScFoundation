import numpy as np
from typing import Union, Optional
from scipy import sparse


class ExpressionBinner:
    def __init__(
        self,
        n_bins: int = 51,
        strategy: str = "log",
        min_value: float = 0.0,
        max_value: Optional[float] = None,
        bins: Optional[np.ndarray] = None
    ):
        self.n_bins = n_bins
        self.strategy = strategy
        self.min_value = min_value
        self.max_value = max_value
        self._bins = bins
        
    def fit(self, X: Union[np.ndarray, sparse.spmatrix]) -> "ExpressionBinner":
        if sparse.issparse(X):
            # Prendi solo i dati non zero per statistiche
            X_data = X.data
        else:
            X_data = X.ravel()
        
        X_nonzero = X_data[X_data > self.min_value]
        
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
        epsilon = 1e-6
        log_min = np.log1p(self.min_value + epsilon)
        log_max = np.log1p(self.max_value + epsilon)
        
        log_bins = np.linspace(log_min, log_max, self.n_bins + 1)
        bins = np.expm1(log_bins) - epsilon
        bins[0] = self.min_value
        
        return bins
    
    def transform(self, X: Union[np.ndarray, sparse.spmatrix]) -> Union[np.ndarray, sparse.csr_matrix]:
        if self._bins is None:
            raise RuntimeError("Must call fit() before transform()")
        
        if sparse.issparse(X):
            # Operiamo solo sui dati non zero, digitize e clip su di essi
            data_binned = np.digitize(X.data, self._bins) - 1
            data_binned = np.clip(data_binned, 0, self.n_bins - 1).astype(np.int32)
            
            # Costruiamo una nuova matrice sparsa con i valori binned
            X_binned = sparse.csr_matrix((data_binned, X.indices, X.indptr), shape=X.shape)
            return X_binned
        else:
            # Dense fallback
            X_dense = np.asarray(X)
            binned = np.digitize(X_dense, self._bins) - 1
            binned = np.clip(binned, 0, self.n_bins - 1)
            return binned.astype(np.int32)
    
    def fit_transform(self, X: Union[np.ndarray, sparse.spmatrix]) -> Union[np.ndarray, sparse.csr_matrix]:
        return self.fit(X).transform(X)
    
    def inverse_transform(self, binned: Union[np.ndarray, sparse.spmatrix]) -> np.ndarray:
        if self._bins is None:
            raise RuntimeError("Must call fit() before inverse_transform()")
        
        bin_centers = (self._bins[:-1] + self._bins[1:]) / 2
        
        if sparse.issparse(binned):
            # Applichiamo bin centers ai valori non zero
            data_values = bin_centers[binned.data]
            X_approx = sparse.csr_matrix((data_values, binned.indices, binned.indptr), shape=binned.shape)
            return X_approx.toarray()
        else:
            return bin_centers[binned]
    
    @property
    def bin_edges(self) -> Optional[np.ndarray]:
        return self._bins
    
    @property
    def bin_centers(self) -> Optional[np.ndarray]:
        if self._bins is None:
            return None
        return (self._bins[:-1] + self._bins[1:]) / 2
    
    def save(self, filepath: str):
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
    if sparse.issparse(X):
        X = X.toarray()
    
    X = np.asarray(X, dtype=np.float32)
    
    if method == "log1p":
        if target_sum is not None:
            lib_sizes = X.sum(axis=1, keepdims=True)
            X = X / lib_sizes * target_sum
        X = np.log1p(X)
    elif method == "zscore":
        mean = X.mean(axis=0, keepdims=True)
        std = X.std(axis=0, keepdims=True)
        X = (X - mean) / (std + 1e-8)
    elif method == "minmax":
        min_val = X.min(axis=0, keepdims=True)
        max_val = X.max(axis=0, keepdims=True)
        X = (X - min_val) / (max_val - min_val + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return X
