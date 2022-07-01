from datasets import iterate_datasets, epsilon_mapping

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier

from groot.model import GrootTreeClassifier, GrootRandomForestClassifier
from groot.toolbox import Model

from robust_relabeling import relabel_model, yang_adv_pruning_dataset

from model import RelabelingCriterionTreeClassifier

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np

import os

import argparse


def train_decision_tree(X, y):
    tree = DecisionTreeClassifier(max_depth=args.max_depth, random_state=1)
    tree.fit(X, y)
    model = Model.from_sklearn(tree)
    return model


def train_groot_tree(X, y):
    tree = GrootTreeClassifier(
        max_depth=args.max_depth, attack_model=epsilon, random_state=1
    )
    tree.fit(X, y)
    model = Model.from_groot(tree)
    return model


def train_random_forest(X, y):
    forest = RandomForestClassifier(
        n_estimators=args.n_estimators,
        random_state=1,
    )
    forest.fit(X, y)
    model = Model.from_sklearn(forest)
    return model


def train_boosting(X, y):
    forest = GradientBoostingClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=1,
    )
    forest.fit(X, y)
    model = Model.from_sklearn(forest)
    return model


def train_groot_forest(X, y):
    forest = GrootRandomForestClassifier(
        n_estimators=args.n_estimators,
        attack_model=epsilon,
        random_state=1,
    )
    forest.fit(X, y)
    model = Model.from_groot(forest)
    return model


def train_relabeling_criterion(X, y):
    tree = RelabelingCriterionTreeClassifier(
        max_depth=args.max_depth,
        attack_model=epsilon,
        random_state=1,
    )
    tree.fit(X, y)
    model = Model.from_groot(tree)
    relabel_model(model, X, y, epsilon)
    return model


def train_adversarial_pruning_tree(X, y):
    X_aug, y_aug = yang_adv_pruning_dataset(X, y, epsilon)
    tree = DecisionTreeClassifier(
        max_depth=args.max_depth,
        random_state=1,
    )
    tree.fit(X_aug, y_aug)
    model = Model.from_sklearn(tree)
    return model


def train_adversarial_pruning_forest(X, y):
    X_aug, y_aug = yang_adv_pruning_dataset(X, y, epsilon)
    tree = RandomForestClassifier(
        n_estimators=args.n_estimators,
        random_state=1,
    )
    tree.fit(X_aug, y_aug)
    model = Model.from_sklearn(tree)
    return model


def train_adversarial_pruning_boosting(X, y):
    X_aug, y_aug = yang_adv_pruning_dataset(X, y, epsilon)
    tree = GradientBoostingClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=1,
    )
    tree.fit(X_aug, y_aug)
    model = Model.from_sklearn(tree)
    return model


parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", type=str, default="out/trees/")
parser.add_argument("--results_dir", type=str, default="out/results/")
parser.add_argument("--max_depth", type=int, default=5)
parser.add_argument("--n_estimators", type=int, default=100)
parser.add_argument("--k_folds", type=int, default=5)
parser.add_argument("--max_relabel_samples", type=int, default=10000)
parser.add_argument("--max_test_samples_wine", type=int, default=100)
parser.add_argument("--ignore_relabeling_criterion", action="store_true")
args = parser.parse_args()

# If the output directories do not exist, make them
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

if not os.path.exists(args.results_dir):
    os.makedirs(args.results_dir)

results = []
for name, X, y in iterate_datasets():
    print(f"Starting dataset: {name}")

    epsilon = epsilon_mapping[name]

    # Create a stratified k-fold cross validation object
    skf = StratifiedKFold(n_splits=args.k_folds, shuffle=True, random_state=1)
    for fold_i, (train_index, test_index) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Exact robustness evaluation of large ensembles on the Wine
        # dataset takes a lot of time so we limit the number of test samples
        if name == "Wine":
            X_test = X_test[: args.max_test_samples_wine]
            y_test = y_test[: args.max_test_samples_wine]

        for model_name, train_model in zip(
            (
                "Decision tree",
                "GROOT tree",
                "Random forest",
                "Boosting",
                "GROOT forest",
                "Robust relabeling criterion",
                "Adversarial pruning tree",
                "Adversarial pruning forest",
                "Adversarial pruning boosting",
            ),
            (
                train_decision_tree,
                train_groot_tree,
                train_random_forest,
                train_boosting,
                train_groot_forest,
                train_relabeling_criterion,
                train_adversarial_pruning_tree,
                train_adversarial_pruning_forest,
                train_adversarial_pruning_boosting,
            ),
        ):
            # If the user wants to skip the relabeling criterion then ignore this method
            if (
                args.ignore_relabeling_criterion
                and model_name == "Robust relabeling criterion"
            ):
                continue

            # Robust relabeling criterion takes too long on
            # the Wine dataset, so we skip it
            if model_name == "Robust relabeling criterion" and name == "Wine":
                continue

            model = train_model(X_train, y_train)

            # Evaluate the model and store the results
            accuracy = model.accuracy(X_test, y_test)
            adversarial_accuracy = model.adversarial_accuracy(
                X_test, y_test, attack="auto", epsilon=epsilon
            )
            results.append(
                (
                    name,
                    fold_i,
                    model_name,
                    accuracy,
                    adversarial_accuracy,
                )
            )

            # Save the model
            model.to_json(
                os.path.join(
                    args.output_dir,
                    f"{name}_{model_name.replace(' ', '_')}_{args.max_depth}_{fold_i}.json",
                )
            )

            # Do not apply robust relabeling for these models
            if model_name in {
                "Robust relabeling criterion",
                "Adversarial pruning tree",
                "Adversarial pruning forest",
                "Adversarial pruning boosting",
            }:
                continue

            # Relabel the tree model in-place
            relabel_model(
                model,
                X_train[: args.max_relabel_samples],
                y_train[: args.max_relabel_samples],
                epsilon=epsilon,
            )
            model_name += " relabeled"

            # Evaluate the relabeled model and store the results
            accuracy = model.accuracy(X_test, y_test)
            adversarial_accuracy = model.adversarial_accuracy(
                X_test, y_test, attack="auto", epsilon=epsilon
            )
            results.append(
                (
                    name,
                    fold_i,
                    model_name,
                    accuracy,
                    adversarial_accuracy,
                )
            )

            # Save the relabeled model
            model.to_json(
                os.path.join(
                    args.output_dir,
                    f"{name}_{model_name.replace(' ', '_')}_{args.max_depth}_{fold_i}.json",
                )
            )

        # Add the scores for a dummy classifier
        majority_class = np.argmax(np.bincount(y_train))
        dummy_score = np.sum(y_test == majority_class) / len(y_test)
        results.append(
            (
                name,
                fold_i,
                "Dummy model",
                dummy_score,
                dummy_score,
            )
        )

    # Output the results after every dataset (this adds only a few seconds overhead)
    results_df = pd.DataFrame(
        results,
        columns=["dataset", "fold", "model", "accuracy", "adversarial accuracy"],
    )
    results_df.to_csv(
        os.path.join(
            args.results_dir,
            f"robustness_{args.max_depth}_{args.n_estimators}_{args.k_folds}.csv",
        ),
        index=False,
    )

print(results_df)

sns.set_theme(style="whitegrid", palette="colorblind", font="sans-serif")

sns.barplot(x="dataset", y="adversarial accuracy", hue="model", data=results_df)
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(
    os.path.join(
        args.results_dir,
        f"robustness_{args.max_depth}_{args.n_estimators}_{args.k_folds}.png",
    )
)
plt.savefig(
    os.path.join(
        args.results_dir,
        f"robustness_{args.max_depth}_{args.n_estimators}_{args.k_folds}.pdf",
    )
)
plt.close()

sns.barplot(x="dataset", y="accuracy", hue="model", data=results_df)
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(
    os.path.join(
        args.results_dir,
        f"accuracy_{args.max_depth}_{args.n_estimators}_{args.k_folds}.png",
    )
)
plt.savefig(
    os.path.join(
        args.results_dir,
        f"accuracy_{args.max_depth}_{args.n_estimators}_{args.k_folds}.pdf",
    )
)
plt.close()
