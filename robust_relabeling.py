from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
from scipy.spatial import distance_matrix

from collections import defaultdict

import numpy as np

from copy import deepcopy

from numba import njit

from tqdm import tqdm


def _extract_bounding_boxes(tree, bounds):
    if "leaf" in tree:
        return [(deepcopy(bounds), tree)]
    else:
        leaves = []

        # If the split's new right threshold (so the node on the left) is more specific
        # than the previous one, update the bound and recurse
        old_bound = bounds[tree["split"]][1]
        if (
            tree["split_condition"] <= bounds[tree["split"]][1]
            and tree["split_condition"] >= bounds[tree["split"]][0]
        ):
            bounds[tree["split"]][1] = tree["split_condition"]

        if tree["split_condition"] >= bounds[tree["split"]][0]:
            for subtree in tree["children"]:
                if subtree["nodeid"] == tree["yes"]:
                    leaves.extend(_extract_bounding_boxes(subtree, bounds))

        bounds[tree["split"]][1] = old_bound

        # If the split's new left threshold (so the node on the right) is more specific
        # than the previous one, update the bound and recurse
        old_bound = bounds[tree["split"]][0]
        if (
            tree["split_condition"] >= bounds[tree["split"]][0]
            and tree["split_condition"] <= bounds[tree["split"]][1]
        ):
            bounds[tree["split"]][0] = tree["split_condition"]

        if tree["split_condition"] < bounds[tree["split"]][1]:
            for subtree in tree["children"]:
                if subtree["nodeid"] == tree["no"]:
                    leaves.extend(_extract_bounding_boxes(subtree, bounds))

        bounds[tree["split"]][0] = old_bound

        return leaves


def _dfs_alternating_paths(G, L, v, matched_edges, U):
    stack = [v]
    while stack:
        v = stack.pop()
        if v in L:
            for u in G[v]:
                if u in U:
                    continue

                if (u, v) not in matched_edges:
                    U.add(u)
                    stack.append(u)
        else:
            for u in G[v]:
                if u in U:
                    continue

                if (u, v) in matched_edges:
                    U.add(u)
                    stack.append(u)


def _connected_by_alternating_paths(G, L, matching, targets):
    matched_edges = {(u, v) for u, v in matching.items()}

    U = set(targets)
    for v in targets:
        _dfs_alternating_paths(G, L, v, matched_edges, U)

    return U


def _networkx_matching_to_cover(G, matching, i_0, i_1):
    L = set(i_0)
    R = set(i_1)

    unmatched_vertices = set(G) - set(matching)
    U = unmatched_vertices & L

    Z = _connected_by_alternating_paths(G, L, matching, U)

    return (L - Z) | (R & Z)


@njit
def _reach_samples_rowcol(i_0, i_1, reaches):
    # Mark the samples with different labels that reach the same leaf
    # with 1 (these cannot both be correctly classified)
    row_indices = []
    col_indices = []
    for i, original_index_0 in enumerate(i_0):
        sample_0_reaches = reaches[original_index_0]

        for j, original_index_1 in enumerate(i_1):
            for a, b in zip(sample_0_reaches, reaches[original_index_1]):
                if a & b:
                    row_indices.append(i)
                    col_indices.append(j)
                    break

    return row_indices, col_indices


@njit
def _reach_leaves(X, bounding_boxes, epsilon):
    # Compute the distance from each sample to each leaf
    # and mark by a 1 the the sample reaches the leaf
    reaches = np.zeros((len(X), len(bounding_boxes)), dtype=np.bool8)
    for i, sample in enumerate(X):
        for j, bounding_box in enumerate(bounding_boxes):
            nearest_point = np.empty(sample.shape)
            for k in range(len(sample)):
                nearest_point[k] = max(
                    bounding_box[k, 0], min(sample[k], bounding_box[k, 1])
                )
            distance = np.linalg.norm(nearest_point - sample, ord=np.inf)

            if distance <= epsilon:
                reaches[i, j] = 1
    return reaches


def relabel_tree(json_tree, X, y, epsilon):
    bounds = defaultdict(lambda: np.array([-np.inf, np.inf]))
    boxes_and_leaves = _extract_bounding_boxes(json_tree, bounds)

    bound_dicts, leaves = zip(*boxes_and_leaves)

    bounding_boxes = []
    for bound_dict in bound_dicts:
        bounding_box = np.tile(np.array([-np.inf, np.inf]), (X.shape[1], 1))
        for i, bound in bound_dict.items():
            bounding_box[i] = bound
        bounding_boxes.append(bounding_box)
    bounding_boxes = np.array(bounding_boxes)

    leaves_in_reach = _reach_leaves(X, bounding_boxes, epsilon)

    X_0 = X[y == 0]
    X_1 = X[y == 1]
    i_0 = np.where(y == 0)[0]
    i_1 = np.where(y == 1)[0]

    row_indices, col_indices = _reach_samples_rowcol(i_0, i_1, leaves_in_reach)
    samples_in_reach = coo_matrix(
        (np.ones(len(row_indices)), (row_indices, col_indices)),
        shape=(len(X_0), len(X_1)),
    )

    del row_indices
    del col_indices

    maximum_matching_arr = maximum_bipartite_matching(
        csr_matrix(samples_in_reach), perm_type="column"
    )
    maximum_matching = {}
    for i, j in enumerate(maximum_matching_arr):
        if j != -1:
            index_0 = i_0[i]
            index_1 = i_1[j]
            maximum_matching[index_0] = index_1
            maximum_matching[index_1] = index_0

    G = {}
    for i in range(len(y)):
        G[i] = set()
    for i, j in zip(samples_in_reach.row, samples_in_reach.col):
        G[i_0[i]].add(i_1[j])
        G[i_1[j]].add(i_0[i])

    cover = _networkx_matching_to_cover(G, maximum_matching, i_0, i_1)

    correct_samples = np.setdiff1d(np.arange(len(X)), list(cover))

    for i, label in zip(correct_samples, y[correct_samples]):
        for leaf_i in np.where(leaves_in_reach[i])[0]:
            leaves[leaf_i]["leaf"] = 1.0 if label else -1.0

    while True:
        if not prune_tree_rec(json_tree):
            break


def relabel_model(model, X, y, epsilon):
    verbose = len(model.json_model) > 1
    for json_tree in tqdm(model.json_model, disable=not verbose):
        relabel_tree(json_tree, X, y, epsilon)


node_keys = (
    "yes",
    "no",
    "missing",
    "depth",
    "split_condition",
    "split",
    "leaf",
    "children",
)


def prune_tree_rec(tree):
    if "children" in tree:
        children = tree["children"]
        if all("leaf" in child for child in children):
            first_leaf_value = children[0]["leaf"]
            if all(child["leaf"] == first_leaf_value for child in children):
                for key in node_keys:
                    if key in tree:
                        del tree[key]

                tree["leaf"] = first_leaf_value

                # Return True to indicate that the tree was pruned
                return True
        else:
            return any(prune_tree_rec(child) for child in tree["children"])

    return False


def prune_model(model):
    for json_tree in model.json_model:
        while True:
            if not prune_tree_rec(json_tree):
                break


def count_leaves_tree_rec(tree):
    if "leaf" in tree:
        return 1
    else:
        return sum(count_leaves_tree_rec(child) for child in tree["children"])


def count_leaves_model(model):
    return sum(count_leaves_tree_rec(tree) for tree in model.json_model)


def yang_adv_pruning_dataset(X, y, epsilon, verbose=False):
    X_0 = X[y == 0]
    X_1 = X[y == 1]
    i_0 = np.where(y == 0)[0]
    i_1 = np.where(y == 1)[0]

    pairwise_distances = distance_matrix(X_0, X_1, np.inf)
    samples_in_reach = coo_matrix(pairwise_distances <= 2 * epsilon)

    maximum_matching_arr = maximum_bipartite_matching(
        csr_matrix(samples_in_reach), perm_type="column"
    )
    maximum_matching = {}
    for i, j in enumerate(maximum_matching_arr):
        if j != -1:
            index_0 = i_0[i]
            index_1 = i_1[j]
            maximum_matching[index_0] = index_1
            maximum_matching[index_1] = index_0

    G = {}
    for i in range(len(y)):
        G[i] = set()
    for i, j in zip(samples_in_reach.row, samples_in_reach.col):
        G[i_0[i]].add(i_1[j])
        G[i_1[j]].add(i_0[i])

    cover = _networkx_matching_to_cover(G, maximum_matching, i_0, i_1)

    correct_samples = np.setdiff1d(np.arange(len(X)), list(cover))

    if verbose:
        print(f"Keeping {len(correct_samples)} samples out of {len(X)}")

    return X[correct_samples], y[correct_samples]
