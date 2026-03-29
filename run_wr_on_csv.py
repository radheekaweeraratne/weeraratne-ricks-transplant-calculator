import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def compute_wr_scores(df):
    # Dependent term
    dep_term = np.where(
        df["D"] == 0,
        0,
        df["D"] * (np.log(df["a"] + 1) / 2 + 1)
    )

    success_term = (df["S"] + df["S_prev"]) / 100
    transplant_term = df["n"] + 1
    age_penalty = 0.01 * np.abs(25 - df["A"]) + 1

    scores = (dep_term * success_term / transplant_term) * (df["Q"] * df["L"] * df["C"] / age_penalty) - df["R"]
    return np.maximum(scores, 0)


def main():
    df = pd.read_csv("synthetic_100k_general.csv")

    df["WR_score"] = compute_wr_scores(df)

    df_sorted = df.sort_values("WR_score", ascending=False)

    print("Patients ranked by Weeraratne-Ricks score:")
    print(df_sorted[["id", "WR_score"]])

    # Simple visualization: score vs age
    plt.scatter(df["A"], df["WR_score"])
    plt.xlabel("General")
    plt.ylabel("WR Score")
    plt.title("Weeraratne-Ricks Score General")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()