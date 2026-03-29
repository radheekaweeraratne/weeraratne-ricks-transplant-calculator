import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

# -----------------------------
# WR Formula (vectorized version)
# -----------------------------
def compute_wr_scores(df):
    dep_term = np.where(
        df["D"] == 0,
        0,
        df["D"] * (np.log(df["a"] + 1) / 2 + 1)
    )

    success_term = (df["S"] + df["S_prev"]) / 100
    transplant_term = df["n"] + 1
    age_penalty = 0.01 * np.abs(25 - df["A"]) + 1

    scores = (dep_term * success_term / transplant_term) * (
        df["Q"] * df["L"] * df["C"] / age_penalty
    ) - df["R"]

    return scores


# -----------------------------
# Generate synthetic dataset
# -----------------------------
def generate_synthetic_patients(n=100000):
    np.random.seed(42)  # reproducibility

    df = pd.DataFrame({
        "id": np.arange(1, n + 1),

        # Age: realistic transplant ages (skewed older)
        "A": np.random.normal(loc=50, scale=15, size=n).clip(1, 85),

        # Dependents: 0–6, skewed toward fewer
        "D": np.random.choice([0,1,2,3,4,5,6], size=n, p=[0.35,0.25,0.2,0.1,0.05,0.03,0.02]),

        # Mean age of dependents: if no dependents → 0
        "a": np.random.normal(loc=12, scale=8, size=n).clip(0, 80),
        
        # Success probability (S): 50–100
        "S": np.random.normal(loc=75, scale=10, size=n).clip(50, 100),

        # Previous success (S'): mostly 0
        "S_prev": np.random.choice([0, 60, 80, 90], size=n, p=[0.9, 0.05, 0.03, 0.02]),

        # Number of previous transplants: mostly 0
        "n": np.random.choice([0,1,2], size=n, p=[0.85, 0.12, 0.03]),

        # Lifestyle risk (R): 0–20
        "R": np.random.choice(range(0, 21), size=n, p=np.linspace(0.1, 0.9, 21)/np.sum(np.linspace(0.1, 0.9, 21))),

        # Quality of life (Q): 0.4–1.0
        "Q": np.random.uniform(0.4, 1.0, size=n),

        # Years of life post-transplant (L): 1–40
        "L": np.random.normal(loc=15, scale=8, size=n).clip(1, 40),

        # Critical need factor (C): weighted toward lower urgency
        "C": np.random.choice([1.0,1.1,1.2,1.3,1.4,1.5,2.0], size=n,
                              p=[0.25,0.25,0.2,0.15,0.1,0.04,0.01])
    })

    # If D = 0, set a = 0
    df.loc[df["D"] == 0, "a"] = 0

    return df


# -----------------------------
# Main execution
# -----------------------------
def main():
    print("Generating 100,000 synthetic patients...")
    df = generate_synthetic_patients(100000)

    print("Computing WR scores...")
    df["WR_score"] = compute_wr_scores(df)

    # Rank patients
    df_sorted = df.sort_values("WR_score", ascending=False)

    print("\nTop 10 highest-priority patients:")
    print(df_sorted.head(10)[["id", "WR_score", "A", "D", "S", "Q", "L", "C"]])

    print("\nScore summary statistics:")
    print(df["WR_score"].describe())

    # Plot distribution
    plt.hist(df["WR_score"], bins=100, color="skyblue", edgecolor="black")
    plt.title("Distribution of Weeraratne-Ricks Scores (100,000 Patients)")
    plt.xlabel("WR Score")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()

    # Save dataset
    df.to_csv("synthetic_100k_patients.csv", index=False)
    print("\nSaved synthetic dataset as synthetic_100k_patients.csv")


if __name__ == "__main__":
    main()