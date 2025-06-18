import numpy as np
from scipy.stats import pearsonr, zscore
from scipy.signal import butter, filtfilt


def load_timeseries(file_path):
    """
    Loads data from file path.
    """
    data = np.loadtxt(file_path)
    return data

def zscore_timeseries(data):
    """
    Returns timeseries with zero mean and unit variance.
    """
    return zscore(data)

def bandpass_filter(data: np.ndarray, low: float, high: float, fs: float, order: int = 5) -> np.ndarray:
    """
    Applies a Butterworth bandpass filter to each node's time series.
    
    Parameters:
        data (np.ndarray): fmri timeseries (timepoints * nodes).
        low (float): Low cutoff frequency in Hz.
        high (float): High cutoff frequency in Hz.
        fs (float): Sampling frequency (inverse of TR).
        order (int): Order of the Butterworth filter.

    Returns:
        np.ndarray: Bandpass-filtered data, same shape as input.
    """
    nyquist = 0.5 * fs
    low /= nyquist
    high /= nyquist

    b, a = butter(order, [low, high], btype='band')
    filtered_data = filtfilt(b, a, data, axis=1,)
    return filtered_data


def sliding_window(data, window_size=60, step=10, tapered = False):
    """
    Applies sliding window and computes correlation matrix.

    Parameters:
        data (ndarray): fmri timeseries (timepoints * nodes).
        window_size (int): number of timepoints in each window.
        step (int): step size for sliding the window.
        tapered (bool): if true applies Hamming tapering.
        
    Returns:
        correlation matrix array. shape: (n_windows, n_nodes, n_nodes)
        """
    n_timepoints, n_nodes = data.shape
    windows = []

    for start in range(0, n_timepoints - window_size + 1, step):
        window_data = data[start:start + window_size, :]
        if tapered:
            hamming = np.hamming(window_size)[:, np.newaxis]
            window_data *= hamming
        corr_matrix = np.corrcoef(window_data.T)
        windows.append(corr_matrix)

    return np.stack(windows)  # shape: (n_windows, n_nodes, n_nodes)

import numpy as np

def compute_node_features(data: np.ndarray) -> np.ndarray:
    """
    Compute simple features for each node from the whole time series.

    Parameters:
        data (np.ndarray): fmri timeseries (timepoints × nodes).

    Returns:
        np.ndarray: node x feature matrix (n_nodes, 4),
                    features: [mean_activation, std_activation, skewness, kurtosis].
    """
    mean_activation = np.mean(data, axis=0)
    std_activation = np.std(data, axis=0)
    skewness = np.apply_along_axis(lambda x: ((x - x.mean())**3).mean() / (x.std()**3 + 1e-8), 0, data)
    kurtosis = np.apply_along_axis(lambda x: ((x - x.mean())**4).mean() / (x.std()**4 + 1e-8), 0, data)

    features = np.stack([mean_activation, std_activation, skewness, kurtosis], axis=1)
    return features
