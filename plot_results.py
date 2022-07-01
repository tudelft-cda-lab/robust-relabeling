import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import os

import argparse

def export_to_latex(
    data_alg_table_mean, data_alg_table_std, alg_columns, column_sort_order
):
    latex_table = data_alg_table_mean.copy()
    latex_table_std = data_alg_table_std.copy()

    rename_columns = {
        "Decision tree": "tree",
        "Decision tree relabeled": "tree rel.",
        "GROOT tree": "GROOT",
        "GROOT tree relabeled": "GROOT rel.",
        "Dummy model": "dummy",
        "Random forest": "forest",
        "Random forest relabeled": "forest rel.",
        "Boosting": "boosting",
        "Boosting relabeled": "boosting rel.",
        "GROOT forest": "GROOT forest",
        "GROOT forest relabeled": "GROOT forest rel.",
        "Adversarial pruning tree": "adv. pruning tree",
        "Adversarial pruning forest": "adv. pruning forest",
        "Adversarial pruning boosting": "adv. pruning boosting",
        "Robust relabeling criterion": "relabeling criterion",
    }
    latex_table = latex_table.rename(columns=rename_columns)
    latex_table_std = latex_table_std.rename(columns=rename_columns)

    column_sort_order = [
        rename_columns[col] if col in rename_columns else col
        for col in column_sort_order
    ]
    export_columns = ["dataset"] + [
        col for col in column_sort_order if col in alg_columns
    ]

    latex_table = latex_table[export_columns]
    latex_table_std = latex_table_std[export_columns]

    max_alg_values = latex_table.max(axis=1).round(3)

    if "dummy" in alg_columns:
        i_dummy = latex_table.columns.get_loc("dummy")
    else:
        i_dummy = None

    for i in range(len(latex_table.index)):
        for j in range(1, len(latex_table.columns)):
            if latex_table.iloc[i, j] == max_alg_values[i]:
                latex_table.iloc[i, j] = f"\\textbf{{{latex_table.iloc[i, j]}}}"

            if j != i_dummy:
                latex_table.iloc[
                    i, j
                ] = f"{latex_table.iloc[i, j]} \\tiny $\\pm$ {latex_table_std.iloc[i, j]}"

            latex_table.iloc[i, j] = str(latex_table.iloc[i, j]).replace("0.", ".")

    table_text = latex_table[export_columns].to_latex(index=False, escape=False)
    while True:
        new_table_text = table_text.replace("  ", " ")

        if new_table_text == table_text:
            table_text = new_table_text
            break

        table_text = new_table_text
    print(table_text)

parser = argparse.ArgumentParser()
parser.add_argument("--results_dir", type=str, default="out/results/")
parser.add_argument("--max_depth", type=int, default=5)
parser.add_argument("--k_folds", type=int, default=5)
parser.add_argument("--n_estimators", type=int, default=100)
parser.add_argument("--trade_off_dataset", type=str, default="Banknote-authentication")
args = parser.parse_args()

sns.set_theme(style="whitegrid", palette="colorblind")

filename = os.path.join(
    args.results_dir,
    f"robustness_{args.max_depth}_{args.n_estimators}_{args.k_folds}.csv",
)

results_df = pd.read_csv(filename)

results_mean = results_df.groupby(["dataset", "model"]).mean().reset_index()
results_std = results_df.groupby(["dataset", "model"]).std().reset_index()

def export_results(metric):
    data_alg_table_mean = (
        pd.pivot_table(
            results_mean,
            index="dataset",
            columns="model",
            values=metric,
        )
        .round(3)
        .reset_index()
    )

    data_alg_table_std = (
        pd.pivot_table(
            results_std,
            index="dataset",
            columns="model",
            values=metric,
        )
        .round(3)
        .reset_index()
    )
    
    print(f"Single trees ({metric})")
    export_to_latex(
        data_alg_table_mean,
        data_alg_table_std,
        {"tree", "tree rel.", "GROOT", "GROOT rel.", "relabeling criterion"},
        ["dataset", "tree", "tree rel.", "GROOT", "GROOT rel.", "relabeling criterion"],
    )

    print(f"Tree ensembles ({metric})")
    export_to_latex(
        data_alg_table_mean,
        data_alg_table_std,
        {
            "forest",
            "forest rel.",
            "boosting",
            "boosting rel.",
            "GROOT forest",
            "GROOT forest rel.",
        },
        [
            "boosting",
            "forest",
            "GROOT forest",
            "boosting rel.",
            "forest rel.",
            "GROOT forest rel.",
        ],
    )

    print(f"Adversarial pruning vs robust relabeling ({metric})")
    export_to_latex(
        data_alg_table_mean,
        data_alg_table_std,
        {
            "adv. pruning tree",
            "tree rel.",
            "adv. pruning boosting",
            "boosting rel.",
            "adv. pruning forest",
            "forest rel.",
        },
        [
            "adv. pruning tree",
            "tree rel.",
            "adv. pruning boosting",
            "boosting rel.",
            "adv. pruning forest",
            "forest rel.",
        ],
    )

    print(f"Regular vs robust relabeling ({metric})")
    export_to_latex(
        data_alg_table_mean,
        data_alg_table_std,
        {
            "tree",
            "tree rel.",
            "boosting",
            "boosting rel.",
            "forest",
            "forest rel.",
        },
        [
            "tree",
            "tree rel.",
            "boosting",
            "boosting rel.",
            "forest",
            "forest rel.",
        ],
    )

export_results("adversarial accuracy")
export_results("accuracy")

for trade_off_dataset in results_df["dataset"].unique():
    tradeoff_results = results_df[results_df["dataset"] == trade_off_dataset]
    tradeoff_results = tradeoff_results[tradeoff_results["model"] != "Dummy model"]

    tradeoff_results = tradeoff_results.groupby("model").mean().reset_index()

    models = {
        "Decision tree",
        "Decision tree relabeled",
        "Random forest",
        "Random forest relabeled",
        "Boosting",
        "Boosting relabeled",
    }
    tradeoff_results = tradeoff_results[tradeoff_results["model"].isin(models)]

    sns.scatterplot(
        x="accuracy",
        y="adversarial accuracy",
        hue="model",
        data=tradeoff_results,
    )
    plt.tight_layout()
    plt.savefig(f"{args.results_dir}/robustness_tradeoff_{trade_off_dataset}.png")
    plt.savefig(f"{args.results_dir}/robustness_tradeoff_{trade_off_dataset}.pdf")
    plt.close()
