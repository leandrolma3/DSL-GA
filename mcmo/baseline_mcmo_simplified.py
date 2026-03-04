"""
MCMO Baseline Adapter - Simplified Version

No geatpy dependency - uses correlation-based feature selection.
Fully compatible with Google Colab.

Author: Claude Code
Date: 2025-11-24
"""

import numpy as np
import warnings
from typing import Tuple, List
from collections import deque

# Try to import simplified MCMO
try:
    # Use absolute imports for Colab compatibility
    from mcmo.MCMO_simplified import MCMO_Simplified, RiverTreeWrapper, RiverADWINWrapper
    MCMO_AVAILABLE = True
except ImportError as e:
    MCMO_AVAILABLE = False
    IMPORT_ERROR = str(e)
    warnings.warn(f"MCMO dependencies not available: {IMPORT_ERROR}\nPlease install: river scipy scikit-learn")


class MCMOAdapter:
    """MCMO Adapter - Simplified version for Colab."""

    def __init__(
        self,
        n_sources: int = 3,
        initial_beach: int = 200,
        max_pool: int = 5,
        gaussian_number: int = 5,
        feature_selection_ratio: float = 0.7,
        chunk_buffer_size: int = 10,
        fallback_to_baseline: bool = True,
        verbose: bool = False
    ):
        if not MCMO_AVAILABLE:
            raise ImportError(f"MCMO dependencies not available: {IMPORT_ERROR}\nPlease install: river scipy scikit-learn")

        self.n_sources = n_sources
        self.initial_beach = initial_beach
        self.max_pool = max_pool
        self.gaussian_number = gaussian_number
        self.chunk_buffer_size = chunk_buffer_size
        self.fallback_to_baseline = fallback_to_baseline
        self.verbose = verbose

        # Initialize MCMO Simplified
        self.mcmo = MCMO_Simplified(
            base_classifier=RiverTreeWrapper(),
            detector=RiverADWINWrapper(min_num_instances=100),
            source_number=n_sources,
            initial_beach=initial_beach,
            max_pool=max_pool,
            gaussian_number=gaussian_number,
            feature_selection_ratio=feature_selection_ratio
        )

        # Chunk buffer
        self.chunk_buffer = deque(maxlen=chunk_buffer_size)

        # Fallback baseline
        self.baseline = RiverTreeWrapper() if fallback_to_baseline else None

        # Statistics
        self.total_chunks_processed = 0
        self.total_samples_processed = 0

    def _log(self, message: str):
        if self.verbose:
            print(f"[MCMOAdapter] {message}")

    def _predict_baseline(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        if self.baseline is None:
            majority_class = np.bincount(y.astype(int)).argmax()
            return np.full(len(y), majority_class)

        predictions = []
        for i in range(len(X)):
            if self.total_samples_processed == 0:
                pred = 0
            else:
                pred = self.baseline.predict(X[i:i+1])[0]

            predictions.append(pred)
            self.baseline.partial_fit(X[i:i+1], y[i:i+1])

        return np.array(predictions)

    def partial_fit_predict(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if not isinstance(y, np.ndarray):
            y = np.array(y)

        if X.ndim == 1:
            X = X.reshape(1, -1)
        if y.ndim == 0:
            y = np.array([y])

        self._log(f"Processing chunk {self.total_chunks_processed}: {len(X)} samples")

        self.chunk_buffer.append((X, y))

        if len(self.chunk_buffer) <= self.n_sources:
            self._log(f"Not enough chunks yet ({len(self.chunk_buffer)}/{self.n_sources+1}), using baseline")
            predictions = self._predict_baseline(X, y)
        else:
            predictions = self._predict_with_mcmo(X, y)

        self.total_chunks_processed += 1
        self.total_samples_processed += len(X)

        return predictions

    def _predict_with_mcmo(self, X_target: np.ndarray, y_target: np.ndarray) -> np.ndarray:
        buffer_list = list(self.chunk_buffer)
        source_chunks = buffer_list[-(self.n_sources+1):-1]

        self._log(f"Using {len(source_chunks)} source streams for MCMO")

        predictions = []

        for i in range(len(X_target)):
            X_sample = X_target[i:i+1]
            y_sample = y_target[i:i+1]

            # Train source classifiers
            for source_idx, (X_source, y_source) in enumerate(source_chunks):
                if i < len(X_source):
                    X_s = X_source[i:i+1]
                    y_s = y_source[i:i+1]
                else:
                    idx = i % len(X_source)
                    X_s = X_source[idx:idx+1]
                    y_s = y_source[idx:idx+1]

                self.mcmo.source_fit(X=X_s, y=y_s, order=source_idx)

            # Predict on target
            pred = self.mcmo.predict(X=X_sample)
            predictions.append(pred[0])

            # Update MCMO
            self.mcmo.partial_fit(X=X_sample, y=y_sample)

        return np.array(predictions)

    def get_info(self) -> dict:
        return {
            'total_chunks_processed': self.total_chunks_processed,
            'total_samples_processed': self.total_samples_processed,
            'buffer_size': len(self.chunk_buffer),
            'using_mcmo': len(self.chunk_buffer) > self.n_sources,
            'n_sources': self.n_sources,
            'initial_beach': self.initial_beach
        }


class MCMOEvaluator:
    """Evaluator wrapper for MCMO Simplified."""

    def __init__(self, n_sources: int = 3, initial_beach: int = 200, max_pool: int = 5, verbose: bool = False):
        self.adapter = MCMOAdapter(
            n_sources=n_sources,
            initial_beach=initial_beach,
            max_pool=max_pool,
            verbose=verbose
        )

        self.predictions_history = []
        self.labels_history = []

    def evaluate_chunk(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, dict]:
        predictions = self.adapter.partial_fit_predict(X, y)

        accuracy = np.mean(predictions == y)

        self.predictions_history.extend(predictions)
        self.labels_history.extend(y)

        metrics = {
            'accuracy': accuracy,
            'chunk_size': len(y),
            **self.adapter.get_info()
        }

        return predictions, metrics

    def get_global_metrics(self) -> dict:
        if not self.predictions_history:
            return {}

        predictions = np.array(self.predictions_history)
        labels = np.array(self.labels_history)

        accuracy = np.mean(predictions == labels)

        return {
            'global_accuracy': accuracy,
            'total_samples': len(labels),
            **self.adapter.get_info()
        }


def test_mcmo_adapter(X_chunks: List[np.ndarray], y_chunks: List[np.ndarray], n_sources: int = 3, verbose: bool = True) -> dict:
    """Test MCMO adapter on chunks."""
    evaluator = MCMOEvaluator(n_sources=n_sources, verbose=verbose)

    results = {
        'predictions_per_chunk': [],
        'metrics_per_chunk': [],
        'accuracies': []
    }

    for i, (X, y) in enumerate(zip(X_chunks, y_chunks)):
        if verbose:
            print(f"\n=== Chunk {i+1}/{len(X_chunks)} ===")

        predictions, metrics = evaluator.evaluate_chunk(X, y)

        results['predictions_per_chunk'].append(predictions)
        results['metrics_per_chunk'].append(metrics)
        results['accuracies'].append(metrics['accuracy'])

        if verbose:
            print(f"Accuracy: {metrics['accuracy']:.4f}")

    results['global_metrics'] = evaluator.get_global_metrics()
    results['mean_accuracy'] = np.mean(results['accuracies'])

    if verbose:
        print(f"\n=== Global Results ===")
        print(f"Mean Accuracy: {results['mean_accuracy']:.4f}")
        print(f"Total Samples: {results['global_metrics']['total_samples']}")

    return results


if __name__ == "__main__":
    print("MCMO Baseline Adapter (Simplified - No geatpy)")
    print("=" * 50)

    if not MCMO_AVAILABLE:
        print(f"ERROR: {IMPORT_ERROR}")
        print("\nPlease install dependencies:")
        print("  pip install river scipy scikit-learn")
    else:
        print("MCMO Simplified dependencies available ✓")
        print("\nTo use this adapter:")
        print("  from mcmo.baseline_mcmo_simplified import MCMOAdapter, MCMOEvaluator")
