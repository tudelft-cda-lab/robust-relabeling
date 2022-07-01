import numbers
import numpy as np
from numpy.lib.function_base import iterable
from sklearn.base import ClassifierMixin
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils import check_random_state
from groot.model import BaseGrootTree, CompiledTree, Node, NumericalNode

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

# Much of this code is adapted from the GROOT code at:
# https://github.com/tudelft-cda-lab/GROOT/blob/main/groot/model.py

_TREE_LEAF = -1
_TREE_UNDEFINED = -2

LEFT = 0
LEFT_INTERSECT = 1
RIGHT_INTERSECT = 2
RIGHT = 3

NOGIL = True

def _attack_model_to_tuples(attack_model, n_features):
    if isinstance(attack_model, numbers.Number):
        return [(attack_model, attack_model) for _ in range(n_features)]
    elif iterable(attack_model):
        new_attack_model = []
        for attack_mode in attack_model:
            if attack_mode == "":
                new_attack_model.append((0, 0))
            elif attack_mode == ">":
                new_attack_model.append((0, 10e9))
            elif attack_mode == "<":
                new_attack_model.append((10e9, 0))
            elif attack_mode == "<>":
                new_attack_model.append((10e9, 10e9))
            elif isinstance(attack_mode, numbers.Number):
                new_attack_model.append((attack_mode, attack_mode))
            elif isinstance(attack_mode, tuple) and len(attack_mode) == 2:
                new_attack_model.append(attack_mode)
            else:
                raise Exception("Unknown attack model spec:", attack_mode)
        return new_attack_model
    else:
        raise Exception(
            "Unknown attack model spec, needs to be perturbation radius or perturbation"
            " per feature:",
            attack_model,
        )

def _scan_numerical_feature_matching(
    samples,
    y,
    dec,
    inc,
    left_bound,
    right_bound,
    samples_reachable,
    leaves_reachable,
    original_leaf,
    new_leaf,
    i_0,
    i_1,
):
    sort_order = samples.argsort()
    sorted_labels = y[sort_order]
    sample_queue = samples[sort_order]
    dec_queue = sample_queue - dec
    inc_queue = sample_queue + inc

    # Initialize queue values and indices
    dec_i = inc_i = 0
    dec_val = dec_queue[0]
    inc_val = inc_queue[0]

    best_score = 10e9
    best_split = None
    score = None
    while True:
        # Find the current point and label from the queue with smallest value.
        if dec_val < inc_val:
            point = dec_val
            label = sorted_labels[dec_i]

            # If the sample could reach the original leaf before,
            # then now it can also reach the new leaf
            orig_sample_i = sort_order[dec_i]
            leaves_reachable[orig_sample_i, new_leaf] = leaves_reachable[
                orig_sample_i, original_leaf
            ]

            # Update dec_val and i to the values belonging to the next
            # sample in queue. If we reached the end of the queue then store
            # a high number to make sure the dec_queue does not get picked
            if dec_i < dec_queue.shape[0] - 1:
                dec_i += 1
                dec_val = dec_queue[dec_i]
            else:
                dec_val = 10e9
        else:
            point = inc_val
            label = sorted_labels[inc_i]

            # If the sample cannot reach the original leaf anymore are it
            # is too far from the decision boundary
            orig_sample_i = sort_order[inc_i]
            leaves_reachable[orig_sample_i, original_leaf] = False

            # Update inc_val and i to the values belonging to the next
            # sample in queue. If we reached the end of the queue then store
            # a high number to make sure the inc_queue does not get picked
            if inc_i < inc_queue.shape[0] - 1:
                inc_i += 1
                inc_val = inc_queue[inc_i]
            else:
                inc_val = 10e9

        if label == 0:
            # If the label of this sample is 0 then update its reachability
            # to all samples of label 1
            i_curr_sample = np.searchsorted(i_0, orig_sample_i)

            # assert orig_sample_i in i_0

            curr_sample_reachable_leaves = leaves_reachable[orig_sample_i, :]
            for j, original_index_1 in enumerate(i_1):
                if np.any(
                    curr_sample_reachable_leaves & leaves_reachable[original_index_1, :]
                ):
                    samples_reachable[i_curr_sample, j] = 1
                else:
                    samples_reachable[i_curr_sample, j] = 0
        else:
            # If the label of this sample is 1 then update its reachability
            # to all samples of label 0
            i_curr_sample = np.searchsorted(i_1, orig_sample_i)

            # assert orig_sample_i in i_1

            curr_sample_reachable_leaves = leaves_reachable[orig_sample_i, :]
            for i, original_index_0 in enumerate(i_0):
                if np.any(
                    curr_sample_reachable_leaves & leaves_reachable[original_index_0, :]
                ):
                    samples_reachable[i, i_curr_sample] = 1
                else:
                    samples_reachable[i, i_curr_sample] = 0

        # print(list(leaves_reachable[:, 0]), list(leaves_reachable[:, 1]))
        # print(samples_reachable.todense())

        if point >= right_bound:
            break

        if point < left_bound:
            continue

        # If the next point is not the same as this one
        next_point = min(dec_val, inc_val)
        if next_point != point:
            maximum_matching = maximum_bipartite_matching(csr_matrix(samples_reachable))
            score = np.sum(maximum_matching != -1)

            # print(point, score)

            # Maximize the margin of the split
            split = (point + next_point) * 0.5

            if score is not None and score < best_score:
                best_score = score
                best_split = split

    return best_score, best_split


class RelabelingCriterionTreeClassifier(BaseGrootTree, ClassifierMixin):
    """
    A robust decision tree for binary classification.
    """

    def __init__(
        self,
        max_depth=5,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=None,
        robust_weight=1.0,
        attack_model=None,
        one_adversarial_class=False,
        chen_heuristic=False,
        compile=True,
        random_state=None,
    ):
        """
        Parameters
        ----------
        max_depth : int, optional
            The maximum depth for the decision tree once fitted.
        min_samples_split : int, optional
            The minimum number of samples required to split a node.
        min_samples_leaf : int, optional
            The minimum number of samples required to make a leaf.
        max_features : int or {"sqrt", "log2"}, optional
            The number of features to consider while making each split, if None then all features are considered.
        robust_weight : float, optional
            The ratio of samples that are actually moved by an adversary.
        attack_model : array-like of shape (n_features,), optional
            Attacker capabilities for perturbing X. By default, all features are considered not perturbable.
        one_adversarial_class : bool, optional
            Whether one class (malicious, 1) perturbs their samples or if both classes (benign and malicious, 0 and 1) do so.
        chen_heuristic : bool, optional
            Whether to use the heuristic for the adversarial Gini impurity from Chen et al. (2019) instead of GROOT's adversarial Gini impurity.
        compile : bool, optional
            Whether to compile the tree for faster predictions.
        random_state : int, optional
            Controls the sampling of the features to consider when looking for the best split at each node.

        Attributes
        ----------
        classes_ : ndarray of shape (n_classes,)
            The class labels.
        max_features_ : int
            The inferred value of max_features.
        n_samples_ : int
            The number of samples when `fit` is performed.
        n_features_ : int
            The number of features when `fit` is performed.
        root_ : Node
            The root node of the tree after fitting.
        compiled_root_ : CompiledTree
            The compiled root node of the tree after fitting.
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.robust_weight = robust_weight
        self.attack_model = attack_model
        self.one_adversarial_class = one_adversarial_class
        self.chen_heuristic = chen_heuristic
        self.compile = compile
        self.random_state = random_state

    def _check_target(self, y):
        target_type = type_of_target(y)
        if target_type != "binary":
            raise ValueError(
                "Unknown label type: classifier only supports binary labels but found"
                f" {target_type}"
            )

        self.classes_, y = np.unique(y, return_inverse=True)
        self.n_classes_ = len(self.classes_)

        return y

    def _create_leaf(self, y):
        """
        Create a leaf object that predicts according to the ratio of benign
        and malicious labels in the array y.
        """

        # Count the number of points that fall into this leaf including
        # adversarially moved points
        label_counts = np.bincount(y, minlength=2)

        # Set the leaf's prediction value to the weighted average of the
        # prediction with and without moving points
        value = label_counts / np.sum(label_counts)

        return Node(_TREE_UNDEFINED, _TREE_LEAF, _TREE_LEAF, value)

    def fit(self, X, y, check_input=True):
        """
        Build a robust and fair binary decision tree from the training set
        (X, y) using greedy splitting according to the weighted adversarial
        Gini impurity and fairness impurity.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training samples.
        y : array-like of shape (n_samples,)
            The class labels as integers 0 (benign) or 1 (malicious)

        Returns
        -------
        self : object
            Fitted estimator.
        """

        if check_input:
            X, y = check_X_y(X, y)
            y = self._check_target(y)

        self.n_samples_, self.n_features_in_ = X.shape

        if self.attack_model is None:
            attack_model = [""] * X.shape[1]
        else:
            attack_model = self.attack_model

        # Turn numerical features in attack model into tuples to make fitting
        # code simpler
        self.attack_model_ = np.array(
            _attack_model_to_tuples(attack_model, X.shape[1]), dtype=X.dtype
        )

        self.random_state_ = check_random_state(self.random_state)

        if self.max_features == "sqrt":
            self.max_features_ = int(np.sqrt(self.n_features_in_))
        elif self.max_features == "log2":
            self.max_features_ = int(np.log2(self.n_features_in_))
        elif self.max_features is None:
            self.max_features_ = self.n_features_in_
        else:
            self.max_features_ = self.max_features

        if self.max_features_ == 0:
            self.max_features_ = 1

        # Keep track of the minimum and maximum split value for each feature
        constraints = np.concatenate(
            (np.min(X, axis=0).reshape(-1, 1), np.max(X, axis=0).reshape(-1, 1)), axis=1
        )

        # samples_reachable = lil_array(np.ones(tuple(np.bincount(y)), dtype=bool))
        samples_reachable = np.ones(tuple(np.bincount(y)), dtype=bool)
        leaves_reachable = np.zeros(
            shape=(self.n_samples_, 2**self.max_depth), dtype=np.bool
        )
        leaves_reachable[:, -1] = True
        i_0 = np.where(y == 0)[0]
        i_1 = np.where(y == 1)[0]
        self.score_ = np.min(np.bincount(y))
        self.root_ = self.__fit_recursive(
            X, y, constraints, samples_reachable, leaves_reachable, i_0, i_1
        )

        # Compile the tree into a representation that is faster when predicting
        if self.compile:
            self.compiled_root_ = CompiledTree(self.root_)

        return self

    def __fit_recursive(
        self,
        X,
        y,
        constraints,
        samples_reachable,
        leaves_reachable,
        i_0,
        i_1,
        depth=0,
        node_id=1,
    ):
        """
        Recursively fit the decision tree on the training dataset (X, y).

        The constraints make sure that leaves are well formed, e.g. don't
        cross an earlier split. Stop when the depth has reached self.max_depth,
        when a leaf is pure or when the leaf contains too few samples.
        """
        if (
            (self.max_depth is not None and depth == self.max_depth)
            or len(y) < self.min_samples_split
            or np.all(y == y[0])
        ):
            return self._create_leaf(y)

        original_leaf = node_id
        # Follow all the right decisions for nodes that do not exist yet,
        # all the way to the leaves
        for _ in range(self.max_depth - depth):
            original_leaf = original_leaf * 2 + 1
        original_leaf -= 2**self.max_depth

        new_leaf = 2 * node_id
        for _ in range(self.max_depth - depth - 1):
            new_leaf = new_leaf * 2 + 1
        new_leaf -= 2**self.max_depth
        # print(depth, self.max_depth, original_leaf, new_leaf)

        rule, feature, split_score = self.__best_adversarial_decision(
            X,
            y,
            constraints,
            samples_reachable,
            leaves_reachable,
            original_leaf,
            new_leaf,
            i_0,
            i_1,
        )

        score_gain = self.score_ - split_score

        # print(f"Score gain: {score_gain} ({split_score})")

        if rule is None or score_gain <= 0.00:
            return self._create_leaf(y)

        self.score_ = split_score

        # Assert that the split obeys constraints made by previous splits
        assert rule >= constraints[feature][0]
        assert rule < constraints[feature][1]

        self.__update_reachability(
            X,
            samples_reachable,
            leaves_reachable,
            original_leaf,
            new_leaf,
            i_0,
            i_1,
            rule,
            feature,
        )

        # Set the right bound and store old one for after recursion
        old_right_bound = constraints[feature][1]
        constraints[feature][1] = rule

        left_node = self.__fit_recursive(
            X,
            y,
            constraints,
            samples_reachable,
            leaves_reachable,
            i_0,
            i_1,
            depth + 1,
            node_id * 2,
        )

        # Reset right bound, set left bound, store old one for after recursion
        constraints[feature][1] = old_right_bound
        old_left_bound = constraints[feature][0]
        constraints[feature][0] = rule

        right_node = self.__fit_recursive(
            X,
            y,
            constraints,
            samples_reachable,
            leaves_reachable,
            i_0,
            i_1,
            depth + 1,
            node_id * 2 + 1,
        )

        # Reset the left bound
        constraints[feature][0] = old_left_bound

        node = NumericalNode(feature, rule, left_node, right_node, _TREE_UNDEFINED)

        return node

    def __best_adversarial_decision(
        self,
        X,
        y,
        constraints,
        samples_reachable,
        leaves_reachable,
        original_leaf,
        new_leaf,
        i_0,
        i_1,
    ):
        """
        Find the best split by iterating through each feature and scanning
        it for that feature's optimal split.
        """

        best_score = 10e9
        best_rule = None
        best_feature = None

        # If there is a limit on features to consider in a split then choose
        # that number of random features.
        all_features = np.arange(self.n_features_in_)
        features = self.random_state_.choice(
            all_features, size=self.max_features_, replace=False
        )

        for feature in features:
            # score, decision_rule = self._scan_feature(X, y, feature, constraints)

            samples = X[:, feature]
            attack_mode = self.attack_model_[feature]
            constraint = constraints[feature]

            score, decision_rule = _scan_numerical_feature_matching(
                samples,
                y,
                *attack_mode,
                *constraint,
                samples_reachable.copy(),
                leaves_reachable.copy(),
                original_leaf,
                new_leaf,
                i_0,
                i_1,
            )

            # print(feature, score, decision_rule)

            if decision_rule is not None and score < best_score:
                best_score = score
                best_rule = decision_rule
                best_feature = feature

        return best_rule, best_feature, best_score

    def __update_reachability(
        self,
        X,
        samples_reachable,
        leaves_reachable,
        original_leaf,
        new_leaf,
        i_0,
        i_1,
        rule,
        feature,
    ):
        dec, inc = self.attack_model_[feature]
        for i, feature_value in enumerate(X[:, feature]):
            if leaves_reachable[i, original_leaf]:
                if feature_value - dec <= rule:
                    leaves_reachable[i, new_leaf] = True

                if feature_value + inc <= rule:
                    leaves_reachable[i, original_leaf] = False

        for i, original_index_0 in enumerate(i_0):
            for j, original_index_1 in enumerate(i_1):
                if np.any(
                    leaves_reachable[original_index_0, :]
                    & leaves_reachable[original_index_1, :]
                ):
                    samples_reachable[i, j] = 1
                else:
                    samples_reachable[i, j] = 0

    def predict_proba(self, X):
        """
        Predict class probabilities of the input samples X.

        The class probability is the fraction of samples of the same class in
        the leaf.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples to predict.

        Returns
        -------
        proba : array of shape (n_samples,)
            The probability for each input sample of being malicious.
        """

        check_is_fitted(self, "root_")

        X = check_array(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                "Received different number of features during predict than during fit"
            )

        # If model has been compiled, use compiled predict_proba
        if self.compile:
            return self.compiled_root_.predict_classification_proba(X)

        predictions = []
        for sample in X:
            predictions.append(self.root_.predict(sample))

        predictions = np.array(predictions)
        predictions /= np.sum(predictions, axis=1)[:, np.newaxis]

        return predictions

    def predict(self, X):
        """
        Predict the classes of the input samples X.

        The predicted class is the most frequently occuring class label in a
        leaf.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples to predict.

        Returns
        -------
        y : array-like of shape (n_samples,)
            The predicted class labels
        """

        # If model has been compiled, use compiled predict
        if self.compile:
            check_is_fitted(self, "root_")

            X = check_array(X)
            if X.shape[1] != self.n_features_in_:
                raise ValueError(
                    "Received different number of features during predict than"
                    " during fit"
                )

            return self.classes_.take(self.compiled_root_.predict_classification(X))

        y_pred_proba = self.predict_proba(X)

        return self.classes_.take(np.argmax(y_pred_proba, axis=1))
