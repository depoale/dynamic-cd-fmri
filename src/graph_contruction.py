import numpy as np
import networkx as nx
import os
import pickle
from typing import List, Optional

def apply_threshold(data: np.ndarray, threshold: float, binary: bool = True) -> np.ndarray:
    """
    Applies a threshold to the correlation matrix. 
    If binary = True, creates a binary adjacency matrix.

    Parameters:
        data (np.ndarray): windowed data + correlation matrix of shape (n_windows, n_nodes, n_nodes).
        threshold (float): Threshold value.
        binary (bool): If True, returns a binary adjacency matrix.

    Returns:
        np.ndarray: thresholded matrices (same shape)
    """
    thresholded = np.empty_like(data)

    for i, corr_matrix in enumerate(data):   # loop over windows
        if binary:
            thresholded[i] = (corr_matrix >= threshold).astype(int)
        else:
            thresholded[i] = np.where(corr_matrix >= threshold, corr_matrix, 0)   
    return thresholded


def build_dynamic_graphs(matrices: np.ndarray, node_features: Optional[np.ndarray] = None):
    """
    Converts thresholded connectivity matrices into dynamic NetworkX graphs.

    Parameters:
        matrices (np.ndarray): shape (n_windows, n_nodes, n_nodes)
        node_features (np.ndarray, optional): shape (n_windows, n_nodes, n_features)

    Returns:
        List[nx.Graph]: one graph per window
    """
    n_windows, n_nodes, _ = matrices.shape
    graphs = []

    for i in range(n_windows):
        G = nx.from_numpy_array(matrices[i])
        if node_features is not None:
            for n in G.nodes:
                G.nodes[n]['feature'] = node_features[n]
        graphs.append(G)     # store all snapshots

    return graphs


def save_graphs(graphs: List[nx.Graph], path: str):
    """
    Save dynamic graphs to disk in .gpickle format.

    Parameters:
        graphs (List[nx.Graph]): list of NetworkX graphs
        path (str): directory to save files
    """
    os.makedirs(path, exist_ok=True)
    for i, G in enumerate(graphs):
        pickle.dump(G, open(os.path.join(path, f"graph_{i:03d}.pickle"), 'wb'))
