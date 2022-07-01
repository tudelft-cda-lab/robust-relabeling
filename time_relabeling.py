from groot.toolbox import Model

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import check_random_state

from model import RelabelingCriterionTreeClassifier

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from datasets import load_dataset, epsilon_mapping

from robust_relabeling import relabel_model, count_leaves_model

import time

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, default="Wine")
parser.add_argument("--max_depth", type=int, default=5)
parser.add_argument("--n_estimators", type=int, default=100)
parser.add_argument("--results_dir", type=str, default="out/results/")
parser.add_argument("--step_size", type=int, default=500)
parser.add_argument("--max_samples_relabeling", type=int, default=50000)
parser.add_argument("--repetitions", type=int, default=3)
args = parser.parse_args()

X, y = load_dataset(args.dataset_name)
epsilon = epsilon_mapping[args.dataset_name]

tree = DecisionTreeClassifier(max_depth=args.max_depth, random_state=1)
tree.fit(X, y)

forest = RandomForestClassifier(
    n_estimators=args.n_estimators, max_depth=None, random_state=1
)
forest.fit(X, y)

boosting = GradientBoostingClassifier(
    n_estimators=args.n_estimators, max_depth=args.max_depth, random_state=1
)
boosting.fit(X, y)

# Perform relabeling once to get rid of the JIT overhead
# (otherwise variance is exaggerated for low n_samples)
model = Model.from_sklearn(tree)
relabel_model(model, X[:10], y[:10], epsilon)

classifier_names = ("Decision tree", "Random forest", "Gradient boosting")
classifiers = (tree, forest, boosting)
results = []
sample_sizes = np.arange(
    args.step_size, min(args.max_samples_relabeling, len(y)) + 1, args.step_size
)

# Add another run on 10 samples to start the plot
# (pruning on 0 samples is not possible)
sample_sizes = np.insert(sample_sizes, 0, 10)

last_criterion_time = 0
criterion_results = []

for n_samples in sample_sizes:
    for classifier_name, classifier in zip(
        classifier_names,
        classifiers,
    ):
        # Reset the random state for each classifier to make sure
        # we use the same data for each of them
        random_state = check_random_state(1)

        for _ in range(args.repetitions):
            random_sample = random_state.choice(len(X), n_samples, replace=False)
            X_sample = X[random_sample]
            y_sample = y[random_sample]

            # Create a new model object each time since the relabeling
            # updates it in-place
            model = Model.from_sklearn(classifier)

            start_time = time.time()
            relabel_model(model, X_sample, y_sample, epsilon)
            runtime = time.time() - start_time

            results.append((n_samples, classifier_name, runtime))

            print(f"{classifier_name} with {n_samples} samples: {runtime} seconds")

        if last_criterion_time < 3600:
            start_time = time.time()
            tree = RelabelingCriterionTreeClassifier(
                max_depth=5,
                attack_model=epsilon,
                random_state=1,
            )
            tree.fit(X_sample, y_sample)
            last_criterion_time = time.time() - start_time

            criterion_results.append((n_samples, last_criterion_time))

            print(
                f"Relabeling criterion with {n_samples} samples: {last_criterion_time} seconds"
            )

    results_df = pd.DataFrame(results, columns=["samples", "model", "runtime (s)"])

    # Export results csv
    results_df.to_csv(
        f"{args.results_dir}/{args.dataset_name}_time_relabeling.csv", index=False
    )

    n_leaves = {
        classifier_name: count_leaves_model(Model.from_sklearn(classifier))
        for classifier_name, classifier in zip(classifier_names, classifiers)
    }
    results_df["model"] = results_df["model"].apply(
        lambda x: f"{x} ({n_leaves[x]} leaves)"
    )

    sns.set_theme(style="whitegrid", palette="colorblind")

    sns.lineplot(
        x="samples",
        y="runtime (s)",
        hue="model",
        data=results_df,
        marker="o",
    )
    plt.tight_layout()
    plt.savefig(f"{args.results_dir}/time_relabeling.png")
    plt.savefig(f"{args.results_dir}/time_relabeling.pdf")
    plt.close()

    sns.lineplot(
        x="samples",
        y="runtime (s)",
        hue="model",
        data=results_df,
        marker="o",
    )
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig(f"{args.results_dir}/time_relabeling_log.png")
    plt.savefig(f"{args.results_dir}/time_relabeling_log.pdf")
    plt.close()

    criterion_results_df = pd.DataFrame(
        criterion_results, columns=["samples", "criterion runtime (s)"]
    )
    criterion_results_df.to_csv(
        f"{args.results_dir}/{args.dataset_name}_criterion_time_relabeling.csv",
        index=False,
    )

    sns.lineplot(
        x="samples",
        y="criterion runtime (s)",
        data=criterion_results_df,
        marker="o",
    )
    plt.xlim(0, max(sample_sizes))
    plt.tight_layout()
    plt.savefig(f"{args.results_dir}/criterion_time_relabeling.png")
    plt.savefig(f"{args.results_dir}/criterion_time_relabeling.pdf")
    plt.close()

    sns.lineplot(
        x="samples",
        y="criterion runtime (s)",
        data=criterion_results_df,
        marker="o",
    )
    plt.xlim(0, max(sample_sizes))
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig(f"{args.results_dir}/criterion_time_relabeling_log.png")
    plt.savefig(f"{args.results_dir}/criterion_time_relabeling_log.pdf")
    plt.close()
