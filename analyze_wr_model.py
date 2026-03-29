import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Load data
# -----------------------------
def load_data(path="synthetic_100k_patients.csv"):
    df = pd.read_csv(path)
    return df

# -----------------------------
# A) Basic visualizations
# -----------------------------
def plot_distributions(df):
    plt.figure(figsize=(10, 5))
    plt.hist(df["WR_score"], bins=100, color="skyblue", edgecolor="black")
    plt.title("Distribution of Weeraratne-Ricks Scores")
    plt.xlabel("WR Score")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(6, 5))
    plt.scatter(df["A"], df["WR_score"], alpha=0.2, s=5)
    plt.xlabel("Age")
    plt.ylabel("WR Score")
    plt.title("WR Score vs Age")
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(6, 5))
    plt.scatter(df["D"], df["WR_score"], alpha=0.2, s=5)
    plt.xlabel("Number of Dependents")
    plt.ylabel("WR Score")
    plt.title("WR Score vs Dependents")
    plt.grid(True)
    plt.show()

# -----------------------------
# B) Sensitivity analysis
# -----------------------------
def sensitivity_analysis(df):
    vars_of_interest = ["A", "D", "a", "S", "S_prev", "n", "R", "Q", "L", "C"]
    corr = df[vars_of_interest + ["WR_score"]].corr()["WR_score"].sort_values(ascending=False)
    print("\nCorrelation of variables with WR_score:")
    print(corr)

# -----------------------------
# C) Fairness checks
# -----------------------------
def fairness_checks(df):
    # Age groups
    bins = [0, 18, 30, 50, 65, 100]
    labels = ["0-18", "19-30", "31-50", "51-65", "66+"]

    df["age_group"] = pd.cut(df["A"], bins=bins, labels=labels, right=False)

    group_stats = df.groupby("age_group")["WR_score"].describe()
    print("\nWR_score by age group:")
    print(group_stats)

    # Risk groups
    risk_bins = [0, 5, 10, 15, 21]
    risk_labels = ["0-4", "5-9", "10-14", "15-20"]
    df["risk_group"] = pd.cut(df["R"], bins=risk_bins, labels=risk_labels, right=False)

    risk_stats = df.groupby("risk_group")["WR_score"].describe()
    print("\nWR_score by lifestyle risk group:")
    print(risk_stats)

# -----------------------------
# D) Simple allocation simulation
# -----------------------------
def simulate_allocation(df, num_organs=1000):
    # Sort by WR_score descending
    df_sorted = df.sort_values("WR_score", ascending=False)

    # "Allocated" = top N
    allocated = df_sorted.head(num_organs)
    not_allocated = df_sorted.iloc[num_organs:]

    print(f"\nSimulated allocation of {num_organs} organs:")
    print("Average WR_score of allocated:", allocated["WR_score"].mean())
    print("Average WR_score of not allocated:", not_allocated["WR_score"].mean())

    # Compare age distribution
    print("\nAge distribution (allocated vs not allocated):")
    print("Allocated ages:")
    print(allocated["A"].describe())
    print("\nNot allocated ages:")
    print(not_allocated["A"].describe())

    # Compare dependents
    print("\nDependents (allocated vs not allocated):")
    print("Allocated dependents:")
    print(allocated["D"].describe())
    print("\nNot allocated dependents:")
    print(not_allocated["D"].describe())

# -----------------------------
# Main
# -----------------------------
def main():
    df = load_data()

    print("Data loaded. Rows:", len(df))
    print(df.head())

    # A) Visualizations
    plot_distributions(df)

    # B) Sensitivity
    sensitivity_analysis(df)

    # C) Fairness
    fairness_checks(df)

    # D) Allocation simulation
    simulate_allocation(df, num_organs=1000)


if __name__ == "__main__":
    main()