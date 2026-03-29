import numpy as np
import pandas as pd

def compute_wr_scores_df(df):
    """
    Compute Weeraratne-Ricks scores for a DataFrame.

    Expected columns in df (you will map these from real data):
        D       -> number of dependents
        a       -> mean age of dependents (0 if none)
        S       -> success probability current transplant (0-100)
        S_prev  -> success probability previous transplants (0-100)
        n       -> number of previous transplants
        R       -> lifestyle risk (0-20)
        Q       -> quality of life after transplant (0.0-1.0)
        L       -> years of life post-transplant
        C       -> critical need factor
        A       -> age of patient
    """

    # Ensure no negative dependents or ages
    D = df["D"].clip(lower=0)
    a = df["a"].clip(lower=0)
    S = df["S"].clip(lower=0, upper=100)
    S_prev = df["S_prev"].clip(lower=0, upper=100)
    n = df["n"].clip(lower=0)
    R = df["R"].clip(lower=0)
    Q = df["Q"].clip(lower=0.0)
    L = df["L"].clip(lower=0.0)
    C = df["C"].clip(lower=0.0)
    A = df["A"].clip(lower=0.0)

    # Dependent term
    dep_term = np.where(
        D <= 0,
        0.0,
        D * (np.log(a + 1.0) / 2.0 + 1.0)
    )

    # Success term
    success_term = (S + S_prev) / 100.0

    # Transplant history normalization
    transplant_term = n + 1.0

    # Age penalty
    age_penalty = 0.01 * np.abs(25.0 - A) + 1.0

    # Raw score
    raw_scores = (dep_term * success_term / transplant_term) * (Q * L * C / age_penalty) - R

    # Clamp to non-negative
    final_scores = np.maximum(raw_scores, 0.0)

    return final_scores


def main():
    # Example placeholder: replace with your real file when it arrives
    df = pd.read_csv("real_optn_data.csv")

    # TODO: map real columns to model variables here, e.g.:
    # df["D"] = df["num_dependents"]
    # df["a"] = df["mean_dependent_age"]
    # df["S"] = df["immunology_success_prob"]
    # df["S_prev"] = df["prior_transplant_success_prob"]
    # df["n"] = df["num_prior_transplants"]
    # df["R"] = df["lifestyle_risk_score"]
    # df["Q"] = df["post_tx_quality_of_life_score"]
    # df["L"] = df["expected_years_post_tx"]
    # df["C"] = df["critical_need_factor"]
    # df["A"] = df["age"]

    df["WR_score"] = compute_wr_scores_df(df)

    # Sort by priority
    df_sorted = df.sort_values("WR_score", ascending=False)

    # Save or inspect
    df_sorted.to_csv("real_optn_with_wr_scores.csv", index=False)
    print("Computed WR scores and saved to real_optn_with_wr_scores.csv")
    print(df_sorted[["WR_score"]].describe())


if __name__ == "__main__":
    main()