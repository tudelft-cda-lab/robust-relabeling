from groot.toolbox import Model

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import pandas as pd

from datasets import iterate_datasets, load_dataset, epsilon_mapping

from robust_relabeling import relabel_model

import argparse


def plot_pruning(x_label, method, ylim=None):
    method_filename = method.replace(" ", "_")
    method_df = results_df[results_df["method"] == method]
    sns.lineplot(x="parameter", y="train accuracy", data=method_df, marker="o")
    sns.lineplot(x="parameter", y="test accuracy", data=method_df, marker="o")
    plt.xlabel(x_label)
    plt.tight_layout()

    if ylim is None:
        ylim = plt.gca().get_ylim()
    else:
        plt.ylim(ylim)

    plt.savefig(f"{args.results_dir}/pruning_{method_filename}_{dataset_name}.pdf")
    plt.savefig(f"{args.results_dir}/pruning_{method_filename}_{dataset_name}.png")
    plt.close()

    return ylim


def plot_side_by_side_comparison(score_name):
    first_color = sns.color_palette()[0]
    second_color = sns.color_palette()[1]

    results_ccp_df = results_df[results_df["method"] == "Cost complexity pruning"]
    results_relabel_df = results_df[results_df["method"] == "Robust relabeling"]

    best_mean_ccp_score = results_ccp_df.groupby("parameter").mean()[score_name].max()
    best_mean_relabel_score = (
        results_relabel_df.groupby("parameter").mean()[score_name].max()
    )

    score_name_filename = score_name.replace(" ", "_")

    _, ax = plt.subplots(1, 2, sharey=True)
    sns.lineplot(
        x="parameter",
        y=score_name,
        data=results_ccp_df,
        color=first_color,
        marker="o",
        ax=ax[0],
    )
    ax[0].axhline(best_mean_ccp_score, color=first_color, linestyle="--")
    ax[0].set_xlabel("$\\alpha$")
    ax[0].set_title("Cost complexity pruning")
    sns.lineplot(
        x="parameter",
        y=score_name,
        data=results_relabel_df,
        color=second_color,
        marker="o",
        ax=ax[1],
    )
    ax[1].axhline(best_mean_relabel_score, color=second_color, linestyle="--")
    ax[1].set_xlabel("$\\epsilon$")
    ax[1].set_title("Robust relabeling")
    plt.tight_layout()
    plt.savefig(f"{args.results_dir}/pruning_{dataset_name}_{score_name_filename}.pdf")
    plt.savefig(f"{args.results_dir}/pruning_{dataset_name}_{score_name_filename}.png")
    plt.close()


parser = argparse.ArgumentParser()
parser.add_argument("--results_dir", type=str, default="out/results/")
args = parser.parse_args()

alphas = np.linspace(0, 0.05, 51)
epsilons = np.linspace(0, 0.3, 51)

options = {"disable_progress_bar": True}

for dataset_name, X, y in iterate_datasets():
    attack_epsilon = epsilon_mapping[dataset_name]

    results = []

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        full_tree = DecisionTreeClassifier(max_depth=None, random_state=1)
        full_tree.fit(X_train, y_train)

        train_majority_class_pct = np.max(np.bincount(y_train)) / len(y_train)

        for ccp_alpha in alphas:
            tree = DecisionTreeClassifier(
                max_depth=None, ccp_alpha=ccp_alpha, random_state=1
            )
            tree.fit(X_train, y_train)

            train_acc = tree.score(X_train, y_train)
            test_acc = tree.score(X_test, y_test)

            model = Model.from_sklearn(tree)
            test_adv_acc = model.adversarial_accuracy(
                X_test, y_test, epsilon=attack_epsilon, options=options
            )

            results.append(
                (
                    "Cost complexity pruning",
                    ccp_alpha,
                    train_acc,
                    test_acc,
                    test_adv_acc,
                )
            )

        for epsilon in epsilons:
            model = Model.from_sklearn(full_tree)

            relabel_model(model, X_train, y_train, epsilon)

            train_acc = model.accuracy(X_train, y_train)
            test_acc = model.accuracy(X_test, y_test)

            test_adv_acc = model.adversarial_accuracy(
                X_test, y_test, epsilon=attack_epsilon, options=options
            )

            results.append(
                ("Robust relabeling", epsilon, train_acc, test_acc, test_adv_acc)
            )

            # If we cannot do better than predict the majority class
            # then stop trying higher values of epsilon
            if train_acc == train_majority_class_pct:
                break

    results_df = pd.DataFrame(
        results,
        columns=[
            "method",
            "parameter",
            "train accuracy",
            "test accuracy",
            "test adversarial accuracy",
        ],
    )

    sns.set_theme(style="whitegrid", palette="colorblind")

    plot_side_by_side_comparison("test accuracy")
    plot_side_by_side_comparison("test adversarial accuracy")
