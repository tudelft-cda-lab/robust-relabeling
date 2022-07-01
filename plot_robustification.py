from sklearn.datasets import make_moons, make_circles
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import check_random_state

from groot.toolbox import Model
from groot.visualization import plot_estimator

from robust_relabeling import relabel_model

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import argparse


def make_tree(n_samples, random_state=None):
    random_state = check_random_state(random_state)

    def ground_truth_tree(sample):
        if sample[0] <= 0.5:
            if sample[1] <= 0.5:
                return 0
            else:
                return 1
        else:
            if sample[1] <= 0.5:
                return 1
            else:
                return 0

    X = random_state.uniform(size=(n_samples, 2))
    y = np.apply_along_axis(ground_truth_tree, 1, X)

    return X, y


parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", type=str, default="out/")
parser.add_argument("--epsilon", type=float, default=0.04)
args = parser.parse_args()

sns.set_theme(style="whitegrid", palette="colorblind")

for name, (X, y), (X_test, y_test) in zip(
    ("moons", "circles", "tree"),
    (
        make_moons(n_samples=200, noise=0.3, random_state=1),
        make_circles(n_samples=200, factor=0.5, noise=0.2, random_state=1),
        make_tree(n_samples=200, random_state=1),
    ),
    (
        make_moons(n_samples=200, noise=0.3, random_state=2),
        make_circles(n_samples=200, factor=0.5, noise=0.2, random_state=2),
        make_tree(n_samples=200, random_state=2),
    ),
):
    X = MinMaxScaler().fit_transform(X)

    random_state = check_random_state(1)
    y = np.where(random_state.rand(len(y)) > 0.95, 1 - y, y)

    for model_name, classifier in zip(
        ("tree", "forest", "boosting"),
        (
            DecisionTreeClassifier(max_depth=5, min_samples_leaf=3, random_state=1),
            RandomForestClassifier(n_estimators=100, random_state=1),
            GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=1),
        ),
    ):
        classifier.fit(X, y)

        model = Model.from_sklearn(classifier)

        plot_estimator(X, y, model, steps=500)
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.savefig(f"{args.output_dir}{name}_{model_name}.png", dpi=200)
        plt.savefig(f"{args.output_dir}{name}_{model_name}.pdf")
        plt.close()

        relabel_model(model, X, y, args.epsilon)

        plot_estimator(X, y, model, steps=500)
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.savefig(f"{args.output_dir}{name}_{model_name}_relabeled.png", dpi=200)
        plt.savefig(f"{args.output_dir}{name}_{model_name}_relabeled.pdf")
        plt.close()
