import numpy as np

import pandas as pd

from datasets import iterate_datasets, epsilon_mapping

table = []
for name, X, y in iterate_datasets():
    table.append(
        (
            name,
            epsilon_mapping[name],
            X.shape[0],
            X.shape[1],
            np.bincount(y).max() / len(y),
        )
    )

table_df = pd.DataFrame(
    table, columns=["Dataset", "$\epsilon$", "Samples", "Features", "Majority class"]
)
table_df["Majority class"] = table_df["Majority class"].map("{:.2f}".format)
print(table_df.to_latex(index=False, escape=False).replace("0.", "."))
