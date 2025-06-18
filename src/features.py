import numpy as np
import networkx as nx

def compute_dynamic_features(graphs: list) -> np.ndarray:
    """
    Computes nodes' features for each snapshot.

    Parameters:
        graphs (list): List of nx graphs, representing system's snapshots.

    Returns:
        np.ndarray: Array of shape (n_windows, n_features) with computed features.
    """
    features = []

    for G in graphs:
        degree_sequence = np.array([d for n, d in G.degree()])
        clustering_coeffs = np.array(list(nx.clustering(G).values()))
        features.append(np.concatenate((degree_sequence, clustering_coeffs)))

    return np.array(features)  # shape: (n_windows, n_features) 

