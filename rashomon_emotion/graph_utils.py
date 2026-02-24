# Graph construction utilities.
import numpy as np
from scipy import signal
from scipy.stats import pearsonr
from sklearn.feature_selection import mutual_info_regression


def compute_plv(signal1, signal2):
    """
    Phase Locking Value (PLV) - measures phase synchronization.
    
    Args:
        signal1, signal2: Complex signals or real signals (will use Hilbert transform)
    Returns:
        PLV value between 0 and 1
    """
    # Apply Hilbert transform if signals are real
    if not np.iscomplexobj(signal1):
        signal1 = signal.hilbert(signal1)
    if not np.iscomplexobj(signal2):
        signal2 = signal.hilbert(signal2)
    
    phase_diff = np.angle(signal1) - np.angle(signal2)
    return np.abs(np.mean(np.exp(1j * phase_diff)))


def compute_coherence(signal1, signal2, fs=128):
    """
    Magnitude-squared coherence between two signals.
    
    Args:
        signal1, signal2: Time-domain signals
        fs: Sampling frequency (Hz)
    Returns:
        Mean coherence value
    """
    f, Cxy = signal.coherence(signal1, signal2, fs=fs, nperseg=min(256, len(signal1)))
    return np.mean(Cxy)


def compute_correlation(signal1, signal2):
    """
    Pearson correlation coefficient.
    
    Args:
        signal1, signal2: Time-domain signals
    Returns:
        Absolute correlation value
    """
    if len(signal1) < 2 or len(signal2) < 2:
        return 0.0
    corr, _ = pearsonr(signal1, signal2)
    return np.abs(corr)


def compute_mutual_information(signal1, signal2):
    """
    Mutual Information between two signals.
    
    Args:
        signal1, signal2: Time-domain signals
    Returns:
        MI value (non-negative)
    """
    if len(signal1) < 2:
        return 0.0
    # Reshape for sklearn
    mi = mutual_info_regression(signal1.reshape(-1, 1), signal2)
    return mi[0]


def compute_aec(signal1, signal2):
    """
    Amplitude Envelope Correlation (AEC).
    Correlation between amplitude envelopes of two signals.
    
    Args:
        signal1, signal2: Time-domain signals
    Returns:
        AEC value
    """
    # Get amplitude envelopes using Hilbert transform
    analytic1 = signal.hilbert(signal1)
    analytic2 = signal.hilbert(signal2)
    
    env1 = np.abs(analytic1)
    env2 = np.abs(analytic2)
    
    if len(env1) < 2:
        return 0.0
    corr, _ = pearsonr(env1, env2)
    return np.abs(corr)


def build_connectivity_graph(eeg_data, method='plv', fs=128, threshold=None):
    """
    Build connectivity graph from EEG data using specified method.
    
    Args:
        eeg_data: EEG signals [n_channels, n_samples]
        method: 'plv', 'coherence', 'correlation', 'mi', or 'aec'
        fs: Sampling frequency (for coherence)
        threshold: Optional threshold to binarize edges (keep top % of edges)
    Returns:
        Adjacency matrix [n_channels, n_channels]
    """
    n_channels = eeg_data.shape[0]
    graph = np.zeros((n_channels, n_channels))
    
    # Select connectivity measure
    connectivity_func = {
        'plv': compute_plv,
        'coherence': lambda s1, s2: compute_coherence(s1, s2, fs),
        'correlation': compute_correlation,
        'mi': compute_mutual_information,
        'aec': compute_aec
    }
    
    if method not in connectivity_func:
        raise ValueError(f"Unknown method: {method}. Choose from {list(connectivity_func.keys())}")
    
    func = connectivity_func[method]
    
    # Compute pairwise connectivity
    for i in range(n_channels):
        for j in range(i, n_channels):  # Symmetric, compute upper triangle
            if i == j:
                graph[i, j] = 1.0  # Self-loops
            else:
                connectivity = func(eeg_data[i], eeg_data[j])
                graph[i, j] = connectivity
                graph[j, i] = connectivity  # Symmetric
    
    # Optional thresholding
    if threshold is not None:
        # Keep top threshold% of edges
        flat_values = graph[np.triu_indices(n_channels, k=1)]  # Upper triangle, no diagonal
        cutoff = np.percentile(flat_values, (1 - threshold) * 100)
        graph = np.where(graph >= cutoff, graph, 0)
        # Restore diagonal
        np.fill_diagonal(graph, 1.0)
    
    return graph


def build_multi_graph(eeg_data, methods=['plv', 'coherence', 'correlation', 'mi'], 
                     fs=128, threshold=None):
    """
    Build multiple connectivity graphs using different methods.
    
    Args:
        eeg_data: EEG signals [n_channels, n_samples]
        methods: List of connectivity methods
        fs: Sampling frequency
        threshold: Optional threshold for each graph
    Returns:
        List of adjacency matrices
    """
    graphs = []
    for method in methods:
        graph = build_connectivity_graph(eeg_data, method=method, fs=fs, threshold=threshold)
        graphs.append(graph)
    return graphs
