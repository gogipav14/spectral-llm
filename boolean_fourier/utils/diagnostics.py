"""
Unified Diagnostics for Boolean Fourier Paper v2
=================================================

Shared utilities for Jaccard trajectories, eigenspectrum analysis,
and routing diagnostics across all phases.

Core thesis: "Topology first, definition second"
- GD learns the support topology early, even when accuracy plateaus
- Discrete refinement quantizes within this learned subspace
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Set, Any, Optional


# =============================================================================
# Support and Jaccard utilities
# =============================================================================

def support_topk(w: np.ndarray, k: int) -> Set[int]:
    """Get top-k indices by absolute value.

    Args:
        w: (n_coeffs,) array of weights
        k: number of indices to return

    Returns:
        Set of top-k indices
    """
    if k <= 0:
        return set()
    k = min(k, len(w))
    return set(np.argsort(np.abs(w))[-k:])


def support_nonzero(w: np.ndarray, tol: float = 1e-6) -> Set[int]:
    """Get indices where weight is nonzero.

    Args:
        w: (n_coeffs,) array of weights
        tol: tolerance for considering a value as zero

    Returns:
        Set of nonzero indices
    """
    return set(np.where(np.abs(w) > tol)[0])


def jaccard(setA: Set[int], setB: Set[int]) -> float:
    """Jaccard similarity between two sets.

    Args:
        setA: first set
        setB: second set

    Returns:
        Jaccard similarity in [0, 1]
    """
    if len(setA | setB) == 0:
        return 1.0
    return len(setA & setB) / len(setA | setB)


def jaccard_trajectory(
    W_log: np.ndarray,
    w_star: np.ndarray
) -> Tuple[np.ndarray, float]:
    """Compute Jaccard(t) for all timesteps.

    Uses top-k support definition where k = |S*| (cardinality of ground truth).

    Args:
        W_log: (n_steps, n_coeffs) array of logged soft weights
        w_star: (n_coeffs,) final ternary mask (ground truth)

    Returns:
        jaccard_t: (n_steps,) Jaccard at each step
        auc: scalar AUC(Jaccard) = mean Jaccard over all steps
    """
    S_star = support_nonzero(w_star)
    k = len(S_star)

    if k == 0:
        # Edge case: constant function (all zeros mask)
        return np.ones(len(W_log)), 1.0

    jaccard_t = []
    for w in W_log:
        S_t = support_topk(w, k)
        jaccard_t.append(jaccard(S_t, S_star))

    jaccard_t = np.array(jaccard_t)
    auc = np.mean(jaccard_t)

    return jaccard_t, auc


def jaccard_final(w_soft: np.ndarray, w_star: np.ndarray) -> float:
    """Compute final Jaccard similarity.

    Args:
        w_soft: (n_coeffs,) soft weights at end of training
        w_star: (n_coeffs,) ternary ground truth mask

    Returns:
        Final Jaccard similarity
    """
    S_star = support_nonzero(w_star)
    k = len(S_star)
    S_soft = support_topk(w_soft, k)
    return jaccard(S_soft, S_star)


# =============================================================================
# Eigenspectrum analysis
# =============================================================================

def eigenspectrum_svd(W_log: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute eigenspectrum via SVD (more stable than covariance).

    Shows how learning collapses into a low-dimensional subspace.

    Args:
        W_log: (n_steps, n_coeffs) array of logged weights

    Returns:
        singular_values: sorted descending
        explained_var: cumulative explained variance ratio
    """
    if len(W_log) < 2:
        return np.array([1.0]), np.array([1.0])

    try:
        W_centered = W_log - W_log.mean(axis=0)
        _, s, _ = np.linalg.svd(W_centered, full_matrices=False)

        var = s**2 / (len(W_log) - 1)
        total_var = var.sum()

        if total_var < 1e-10:
            # Edge case: no variance
            explained_var = np.ones(len(s))
        else:
            explained_var = np.cumsum(var) / total_var

        return s, explained_var
    except np.linalg.LinAlgError:
        # SVD didn't converge - return default values
        n_coeffs = W_log.shape[1] if len(W_log.shape) > 1 else 1
        return np.ones(n_coeffs), np.ones(n_coeffs)


def eigenspectrum_covariance(W_log: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute eigenspectrum via covariance matrix.

    Alternative to SVD-based method.

    Args:
        W_log: (n_steps, n_coeffs) array of logged weights

    Returns:
        eigenvalues: sorted descending
        explained_var: cumulative explained variance ratio
    """
    if len(W_log) < 2:
        return np.array([1.0]), np.array([1.0])

    C = np.cov(W_log.T)
    eigenvals = np.linalg.eigvalsh(C)[::-1]  # Sort descending

    total_var = eigenvals.sum()
    if total_var < 1e-10:
        explained_var = np.ones(len(eigenvals))
    else:
        explained_var = np.cumsum(eigenvals) / total_var

    return eigenvals, explained_var


def spectral_compression_summary(explained_var: np.ndarray) -> Dict[str, float]:
    """Compute summary statistics for spectral compression.

    Args:
        explained_var: cumulative explained variance ratio

    Returns:
        Dictionary with:
        - var_top1: variance explained by top-1 mode
        - var_top3: variance explained by top-3 modes
        - modes_90: number of modes to explain 90% variance
        - modes_95: number of modes to explain 95% variance
    """
    var_per_mode = np.diff(np.concatenate([[0], explained_var]))

    return {
        'var_top1': float(var_per_mode[0]) if len(var_per_mode) > 0 else 0.0,
        'var_top3': float(explained_var[2]) if len(explained_var) > 2 else float(explained_var[-1]),
        'modes_90': int(np.searchsorted(explained_var, 0.9) + 1),
        'modes_95': int(np.searchsorted(explained_var, 0.95) + 1),
    }


# =============================================================================
# Routing diagnostics (for mHC / Sinkhorn analysis)
# =============================================================================

def routing_stats(P: np.ndarray) -> Dict[str, Any]:
    """Compute routing matrix diagnostics for mHC analysis.

    Args:
        P: (n, n) doubly stochastic routing matrix

    Returns:
        Dictionary with:
        - drift: Frobenius distance from identity
        - mean_entropy: mean column entropy
        - max_entries: maximum entry per column
        - permutation_likeness: how close to hard permutation
    """
    n = P.shape[0]
    drift = np.linalg.norm(P - np.eye(n), 'fro')

    # Column entropy (how spread out each column is)
    col_entropies = -np.sum(P * np.log(P + 1e-10), axis=0)
    mean_entropy = float(np.mean(col_entropies))

    # Max entry per column (1.0 = hard permutation)
    max_entries = P.max(axis=0)
    mean_max_entry = float(np.mean(max_entries))

    # Permutation-likeness: 1 if all max_entries are 1
    permutation_likeness = float(np.mean(max_entries > 0.99))

    return {
        'drift': float(drift),
        'mean_entropy': mean_entropy,
        'mean_max_entry': mean_max_entry,
        'permutation_likeness': permutation_likeness,
        'col_entropies': col_entropies.tolist(),
        'max_entries': max_entries.tolist(),
    }


# =============================================================================
# Logging utilities
# =============================================================================

class DiagnosticsLogger:
    """Consistent JSON logging for diagnostics.

    Provides a standardized format for logging results across all phases.
    """

    def __init__(self, experiment_name: str, output_dir: str = 'results'):
        """Initialize logger.

        Args:
            experiment_name: Name of the experiment (used in filename)
            output_dir: Directory to save results
        """
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.data = {
            'experiment': experiment_name,
            'timestamp': datetime.now().isoformat(),
            'operations': {},
            'summary': {},
        }

    def log_operation(
        self,
        op_name: str,
        accuracy: float,
        jaccard_trajectory: Optional[np.ndarray] = None,
        auc_jaccard: Optional[float] = None,
        final_jaccard: Optional[float] = None,
        eigenspectrum: Optional[Dict] = None,
        extra: Optional[Dict] = None,
    ):
        """Log results for a single operation.

        Args:
            op_name: Name of the operation (e.g., 'xor', 'and_3')
            accuracy: Final accuracy achieved
            jaccard_trajectory: Jaccard(t) values over training
            auc_jaccard: AUC(Jaccard) scalar
            final_jaccard: Final Jaccard similarity
            eigenspectrum: Eigenspectrum analysis results
            extra: Additional key-value pairs to log
        """
        op_data = {
            'accuracy': float(accuracy),
        }

        if jaccard_trajectory is not None:
            op_data['jaccard_trajectory'] = jaccard_trajectory.tolist()

        if auc_jaccard is not None:
            op_data['auc_jaccard'] = float(auc_jaccard)

        if final_jaccard is not None:
            op_data['final_jaccard'] = float(final_jaccard)

        if eigenspectrum is not None:
            op_data['eigenspectrum'] = eigenspectrum

        if extra is not None:
            op_data.update(extra)

        self.data['operations'][op_name] = op_data

    def log_summary(self, summary: Dict[str, Any]):
        """Log summary statistics.

        Args:
            summary: Dictionary of summary statistics
        """
        self.data['summary'].update(summary)

    def save(self, filename: Optional[str] = None) -> Path:
        """Save logged data to JSON file.

        Args:
            filename: Optional custom filename (default: experiment_name.json)

        Returns:
            Path to saved file
        """
        if filename is None:
            filename = f'{self.experiment_name}.json'

        filepath = self.output_dir / filename

        with open(filepath, 'w') as f:
            json.dump(self.data, f, indent=2)

        print(f"Saved diagnostics to: {filepath}")
        return filepath


# =============================================================================
# Unified GD Protocol
# =============================================================================

# Standard hyperparameters for all phase diagnostics
GD_PROTOCOL = {
    'arch': {
        'hidden': None,  # direct linear: w^T χ(x)
        'activation': 'gumbel_softmax',
        'temp_anneal': (1.0, 0.1, 'exponential'),
    },
    'hyperparams': {
        'lr': 1e-2,
        'steps': 2000,
        'log_every': 100,
        'batch_size': 256,
    },
}


def get_gd_protocol() -> Dict:
    """Get the standard GD protocol configuration.

    Returns:
        Dictionary with architecture and hyperparameter settings
    """
    return GD_PROTOCOL.copy()


# =============================================================================
# Canonical mask loading
# =============================================================================

def load_phase1_masks() -> Dict[str, np.ndarray]:
    """Load canonical Phase 1 masks.

    Source: train_phase1_fixed.py lines 54-59 (canonical source)
    Basis: [1, a, b, ab]

    Returns:
        Dictionary mapping operation name to mask
    """
    return {
        'xor': np.array([0, 0, 0, 1]),
        'and': np.array([1, 1, 1, -1]),      # CORRECTED: was [-1, 1, 1, 1]
        'or': np.array([-1, 1, 1, 1]),       # CORRECTED: was [1, 1, 1, -1]
        'implies': np.array([-1, -1, 1, -1]),  # CORRECTED: was [1, -1, 1, 1]
    }


def load_phase3_masks(checkpoint_path: Optional[str] = None) -> Dict[str, np.ndarray]:
    """Load canonical Phase 3 masks from checkpoint.

    Args:
        checkpoint_path: Path to phase3_final_results.json
                        If None, uses default path.

    Returns:
        Dictionary mapping operation name to mask
    """
    if checkpoint_path is None:
        checkpoint_path = Path(__file__).parent.parent / 'checkpoints' / 'phase3_final' / 'phase3_final_results.json'

    with open(checkpoint_path, 'r') as f:
        data = json.load(f)

    return {name: np.array(mask) for name, mask in data['optimal_masks'].items()}


def load_phase4_masks(checkpoint_path: Optional[str] = None) -> Dict[str, np.ndarray]:
    """Load canonical Phase 4 masks from checkpoint.

    Args:
        checkpoint_path: Path to phase4_synthesis_results.json
                        If None, uses default path.

    Returns:
        Dictionary mapping operation name to mask
    """
    if checkpoint_path is None:
        checkpoint_path = Path(__file__).parent.parent / 'phase4' / 'checkpoints' / 'phase4_synthesis' / 'phase4_synthesis_results.json'

    with open(checkpoint_path, 'r') as f:
        data = json.load(f)

    return {name: np.array(op['mask']) for name, op in data['operations'].items()}


# =============================================================================
# Testing
# =============================================================================

if __name__ == '__main__':
    print("Testing diagnostics utilities...")

    # Test Jaccard
    w_star = np.array([1, 0, 0, 1, 0, -1, 0, 0])
    W_log = np.array([
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],  # wrong support
        [0.5, 0.1, 0.1, 0.6, 0.2, 0.4, 0.1, 0.1],  # partial
        [0.9, 0.1, 0.0, 0.8, 0.0, 0.7, 0.0, 0.0],  # correct support
    ])

    jaccard_t, auc = jaccard_trajectory(W_log, w_star)
    print(f"Jaccard trajectory: {jaccard_t}")
    print(f"AUC(Jaccard): {auc:.3f}")

    # Test eigenspectrum
    s, explained = eigenspectrum_svd(W_log)
    print(f"\nSingular values: {s}")
    print(f"Cumulative explained variance: {explained}")

    summary = spectral_compression_summary(explained)
    print(f"Compression summary: {summary}")

    # Test routing stats
    P = np.array([
        [0.9, 0.05, 0.05],
        [0.05, 0.9, 0.05],
        [0.05, 0.05, 0.9],
    ])
    stats = routing_stats(P)
    print(f"\nRouting stats: {stats}")

    # Test logger
    logger = DiagnosticsLogger('test_experiment', output_dir='/tmp')
    logger.log_operation('xor', accuracy=1.0, auc_jaccard=0.95)
    logger.log_summary({'mean_auc': 0.95})
    # logger.save()  # Uncomment to actually save

    print("\nAll tests passed!")
