"""
MCMO Baseline Adapter

Adapts MCMO (multistream classification) to work with single-stream data
using temporal splitting strategy.

Strategy:
- Buffer consecutive chunks from the single stream
- Use last N chunks as "source streams" (labeled)
- Use current chunk as "target stream" (unlabeled during prediction)
- This simulates natural covariate shift from temporal drift

Author: Claude Code
Date: 2025-11-24
"""

import numpy as np
import warnings
from typing import Tuple, List, Optional
from collections import deque

# Try to import MCMO, provide helpful error if dependencies missing
try:
    from .MCMO import MCMO
    from skmultiflow.trees import HoeffdingTreeClassifier
    from skmultiflow.drift_detection import DDM
    MCMO_AVAILABLE = True
except ImportError as e:
    MCMO_AVAILABLE = False
    IMPORT_ERROR = str(e)
    warnings.warn(
        f"MCMO dependencies not available: {IMPORT_ERROR}\n"
        "Please install: geatpy==2.7.0, scikit-multiflow==0.5.3"
    )


class MCMOAdapter:
    """
    Adapter to use MCMO with single-stream data via temporal splitting.

    Converts single stream → multistream by treating consecutive chunks
    as separate source streams.

    Parameters
    ----------
    n_sources : int, default=3
        Number of source streams to create from past chunks
    initial_beach : int, default=200
        Number of samples for MCMO initialization (warmup period)
    max_pool : int, default=5
        Maximum size of classifier pool for drift adaptation
    gaussian_number : int, default=5
        Number of Gaussian components in GMM
    chunk_buffer_size : int, default=10
        Maximum number of chunks to keep in buffer
    fallback_to_baseline : bool, default=True
        If True, uses simple baseline when MCMO not ready
    verbose : bool, default=False
        Print debug information
    """

    def __init__(
        self,
        n_sources: int = 3,
        initial_beach: int = 200,
        max_pool: int = 5,
        gaussian_number: int = 5,
        chunk_buffer_size: int = 10,
        fallback_to_baseline: bool = True,
        verbose: bool = False
    ):
        if not MCMO_AVAILABLE:
            raise ImportError(
                f"MCMO dependencies not available: {IMPORT_ERROR}\n"
                "Please install: geatpy==2.7.0, scikit-multiflow==0.5.3"
            )

        self.n_sources = n_sources
        self.initial_beach = initial_beach
        self.max_pool = max_pool
        self.gaussian_number = gaussian_number
        self.chunk_buffer_size = chunk_buffer_size
        self.fallback_to_baseline = fallback_to_baseline
        self.verbose = verbose

        # Initialize MCMO
        self.mcmo = MCMO(
            base_classifier=HoeffdingTreeClassifier(),
            detector=DDM(min_num_instances=100),
            source_number=n_sources,
            initial_beach=initial_beach,
            max_pool=max_pool,
            gaussian_number=gaussian_number
        )

        # Chunk buffer: stores (X, y) tuples
        self.chunk_buffer = deque(maxlen=chunk_buffer_size)

        # Fallback baseline (simple Hoeffding Tree)
        self.baseline = HoeffdingTreeClassifier() if fallback_to_baseline else None

        # Statistics
        self.total_chunks_processed = 0
        self.total_samples_processed = 0

    def _log(self, message: str):
        """Print message if verbose mode enabled."""
        if self.verbose:
            print(f"[MCMOAdapter] {message}")

    def _predict_baseline(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Use simple baseline when MCMO not ready yet.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Features
        y : np.ndarray, shape (n_samples,)
            True labels (for training baseline)

        Returns
        -------
        predictions : np.ndarray, shape (n_samples,)
            Predictions (uses majority class before baseline is trained)
        """
        if self.baseline is None:
            # No fallback, predict majority class from current chunk
            majority_class = np.bincount(y.astype(int)).argmax()
            return np.full(len(y), majority_class)

        # Predict with baseline
        predictions = []
        for i in range(len(X)):
            if self.total_samples_processed == 0:
                # First sample, predict arbitrary class
                pred = 0
            else:
                # Predict with current baseline
                pred = self.baseline.predict(X[i:i+1])[0]

            predictions.append(pred)

            # Update baseline with true label
            self.baseline.partial_fit(X[i:i+1], y[i:i+1])

        return np.array(predictions)

    def partial_fit_predict(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> np.ndarray:
        """
        Train and predict on a chunk using temporal splitting.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Features of current chunk
        y : np.ndarray, shape (n_samples,)
            True labels of current chunk

        Returns
        -------
        predictions : np.ndarray, shape (n_samples,)
            Predictions for the current chunk
        """
        # Convert to numpy arrays if needed
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if not isinstance(y, np.ndarray):
            y = np.array(y)

        # Ensure 2D
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if y.ndim == 0:
            y = np.array([y])

        self._log(f"Processing chunk {self.total_chunks_processed}: {len(X)} samples")

        # Add current chunk to buffer
        self.chunk_buffer.append((X, y))

        # Check if we have enough chunks for MCMO
        if len(self.chunk_buffer) <= self.n_sources:
            self._log(
                f"Not enough chunks yet ({len(self.chunk_buffer)}/{self.n_sources+1}), "
                "using baseline"
            )
            predictions = self._predict_baseline(X, y)
        else:
            # Use temporal splitting with MCMO
            predictions = self._predict_with_mcmo(X, y)

        self.total_chunks_processed += 1
        self.total_samples_processed += len(X)

        return predictions

    def _predict_with_mcmo(
        self,
        X_target: np.ndarray,
        y_target: np.ndarray
    ) -> np.ndarray:
        """
        Predict using MCMO with temporal splitting.

        Strategy:
        - Last n_sources chunks → source streams (with labels)
        - Current chunk → target stream (without labels during prediction)

        Parameters
        ----------
        X_target : np.ndarray
            Features of target chunk
        y_target : np.ndarray
            True labels of target chunk (used for update after prediction)

        Returns
        -------
        predictions : np.ndarray
            Predictions for target chunk
        """
        # Extract source streams (last n_sources chunks, excluding current)
        buffer_list = list(self.chunk_buffer)
        source_chunks = buffer_list[-(self.n_sources+1):-1]

        self._log(f"Using {len(source_chunks)} source streams for MCMO")

        # Process sample-by-sample (MCMO expects online learning)
        predictions = []

        for i in range(len(X_target)):
            X_sample = X_target[i:i+1]
            y_sample = y_target[i:i+1]

            # Train source classifiers with samples from source chunks
            for source_idx, (X_source, y_source) in enumerate(source_chunks):
                # For each source, train with one sample (or multiple if chunk large)
                # To keep it simple, we train with samples in proportion to position
                # More recent sources get more recent samples

                if i < len(X_source):
                    # Use corresponding sample from source
                    X_s = X_source[i:i+1]
                    y_s = y_source[i:i+1]
                else:
                    # If target chunk larger than source, wrap around
                    idx = i % len(X_source)
                    X_s = X_source[idx:idx+1]
                    y_s = y_source[idx:idx+1]

                self.mcmo.source_fit(X=X_s, y=y_s, order=source_idx)

            # Predict on target (current sample)
            pred = self.mcmo.predict(X=X_sample)
            predictions.append(pred[0])

            # Update MCMO with true label (for drift detection)
            self.mcmo.partial_fit(X=X_sample, y=y_sample)

        return np.array(predictions)

    def get_info(self) -> dict:
        """
        Get information about adapter state.

        Returns
        -------
        info : dict
            Dictionary with adapter statistics
        """
        return {
            'total_chunks_processed': self.total_chunks_processed,
            'total_samples_processed': self.total_samples_processed,
            'buffer_size': len(self.chunk_buffer),
            'using_mcmo': len(self.chunk_buffer) > self.n_sources,
            'n_sources': self.n_sources,
            'initial_beach': self.initial_beach
        }


class MCMOEvaluator:
    """
    Evaluator wrapper for MCMO to match the interface of other baselines.

    This class provides a simple interface compatible with the evaluation
    pipeline used in Phase 3 experiments.

    Parameters
    ----------
    n_sources : int, default=3
        Number of source streams
    initial_beach : int, default=200
        Warmup period for MCMO
    max_pool : int, default=5
        Classifier pool size
    verbose : bool, default=False
        Print debug information
    """

    def __init__(
        self,
        n_sources: int = 3,
        initial_beach: int = 200,
        max_pool: int = 5,
        verbose: bool = False
    ):
        self.adapter = MCMOAdapter(
            n_sources=n_sources,
            initial_beach=initial_beach,
            max_pool=max_pool,
            verbose=verbose
        )

        self.predictions_history = []
        self.labels_history = []

    def evaluate_chunk(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Tuple[np.ndarray, dict]:
        """
        Evaluate on a chunk and return predictions + metrics.

        Parameters
        ----------
        X : np.ndarray
            Features
        y : np.ndarray
            True labels

        Returns
        -------
        predictions : np.ndarray
            Predictions for the chunk
        metrics : dict
            Evaluation metrics (accuracy, etc.)
        """
        predictions = self.adapter.partial_fit_predict(X, y)

        # Calculate metrics
        accuracy = np.mean(predictions == y)

        # Store for global metrics
        self.predictions_history.extend(predictions)
        self.labels_history.extend(y)

        metrics = {
            'accuracy': accuracy,
            'chunk_size': len(y),
            **self.adapter.get_info()
        }

        return predictions, metrics

    def get_global_metrics(self) -> dict:
        """
        Get global metrics across all chunks.

        Returns
        -------
        metrics : dict
            Global evaluation metrics
        """
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


# Convenience function for quick testing
def test_mcmo_adapter(
    X_chunks: List[np.ndarray],
    y_chunks: List[np.ndarray],
    n_sources: int = 3,
    verbose: bool = True
) -> dict:
    """
    Test MCMO adapter on a list of chunks.

    Parameters
    ----------
    X_chunks : List[np.ndarray]
        List of feature arrays (one per chunk)
    y_chunks : List[np.ndarray]
        List of label arrays (one per chunk)
    n_sources : int, default=3
        Number of source streams
    verbose : bool, default=True
        Print progress

    Returns
    -------
    results : dict
        Dictionary with predictions and metrics per chunk
    """
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

    # Add global metrics
    results['global_metrics'] = evaluator.get_global_metrics()
    results['mean_accuracy'] = np.mean(results['accuracies'])

    if verbose:
        print(f"\n=== Global Results ===")
        print(f"Mean Accuracy: {results['mean_accuracy']:.4f}")
        print(f"Total Samples: {results['global_metrics']['total_samples']}")

    return results


if __name__ == "__main__":
    print("MCMO Baseline Adapter")
    print("=" * 50)

    if not MCMO_AVAILABLE:
        print(f"ERROR: {IMPORT_ERROR}")
        print("\nPlease install dependencies:")
        print("  pip install geatpy==2.7.0 scikit-multiflow==0.5.3")
    else:
        print("MCMO dependencies available ✓")
        print("\nTo use this adapter:")
        print("  from mcmo.baseline_mcmo import MCMOAdapter, MCMOEvaluator")
        print("\nExample:")
        print("  adapter = MCMOAdapter(n_sources=3)")
        print("  predictions = adapter.partial_fit_predict(X_chunk, y_chunk)")
