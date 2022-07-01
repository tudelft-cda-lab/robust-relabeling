from sklearn.datasets import fetch_openml

from scipy.sparse import csr_matrix

import numpy as np

# Mapping of UCI names to openml identifier and version
name_to_openml_mapping = {
    "Banknote-authentication": ("banknote-authentication", 1),
    "Breast-cancer-diagnostic": ("wdbc", 1),
    "Breast-cancer": ("breast-w", 1),
    "Connectionist-bench-sonar": ("sonar", 1),
    "Ionosphere": ("ionosphere", 1),
    "Parkinsons": ("parkinsons", 1),
    "Pima-Indians-diabetes": ("diabetes", 1),
    "Qsar-biodegradation": ("qsar-biodeg", 1),
    "Spectf-heart": ("SPECTF", 1),
    "Wine": ("wine_quality", 1),
}

name_to_binarize_mapping = {
    "Banknote-authentication": lambda y: np.where(y == "2", 1, 0),
    "Breast-cancer-diagnostic": lambda y: np.where(y == "2", 1, 0),
    "Breast-cancer": lambda y: np.where(y == "malignant", 1, 0),
    "Connectionist-bench-sonar": lambda y: np.where(y == "Rock", 1, 0),
    "Ionosphere": lambda y: np.where(y == "b", 1, 0),
    "Parkinsons": lambda y: np.where(y == "2", 1, 0),
    "Pima-Indians-diabetes": lambda y: np.where(y == "tested_positive", 1, 0),
    "Qsar-biodegradation": lambda y: np.where(y == "2", 1, 0),
    "Spectf-heart": lambda y: y.astype(int),
    "Wine": lambda y: np.where(y >= 6, 0, 1).astype(int),
}

epsilon_mapping = {
    "Banknote-authentication": 0.05,
    "Breast-cancer-diagnostic": 0.05,
    "Breast-cancer": 0.1,
    "Connectionist-bench-sonar": 0.05,
    "Ionosphere": 0.05,
    "Parkinsons": 0.05,
    "Pima-Indians-diabetes": 0.01,
    "Qsar-biodegradation": 0.05,
    "Spectf-heart": 0.005,
    "Wine": 0.025,
}


def load_dataset(name, remove_missing_value_rows=True, binarize_labels=True):
    """
    Loads a dataset from openml.org.

    Parameters
    ----------
    name : str
        Name of the dataset.
    remove_missing_value_rows : bool, optional (default=True)
        Whether to remove rows with missing values.
    binarize_labels : bool, optional (default=True)
        Whether to binarize labels.

    Returns
    -------
    X : numpy.ndarray
        Dataset features.
    y : numpy.ndarray
        Dataset labels.
    """
    if name not in name_to_openml_mapping:
        raise ValueError(f"Unknown dataset: {name}, available datasets: {name_to_openml_mapping.keys()}")
    
    dataset_id, version = name_to_openml_mapping[name]
    dataset = fetch_openml(dataset_id, version=version, return_X_y=False, as_frame=False)
    X = dataset.data
    y = dataset.target

    # Some datasets come in a sparse format, for now we will convert to dense
    # such that this does not give problems later on.
    if isinstance(X, csr_matrix):
        X = X.toarray()

    if remove_missing_value_rows:
        y = y[~np.isnan(X).any(axis=1)]
        X = X[~np.isnan(X).any(axis=1)]

    if binarize_labels:
        y = name_to_binarize_mapping[name](y)

    return X, y

def iterate_datasets():
    """
    Iterates over all datasets.

    Yields
    ------
    name : str
        Name of the dataset.
    X : numpy.ndarray
        Dataset features.
    y : numpy.ndarray
        Dataset labels (only values 0 and 1).
    """
    for name in name_to_openml_mapping:
        X, y = load_dataset(name)
        yield name, X, y
