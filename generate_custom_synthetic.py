import numpy as np
import pandas as pd

def generate_dataset(n=100000, mode="general"):
    np.random.seed(42)

    # -----------------------------
    # Different population modes
    # -----------------------------
    if mode == "young":
        age = np.random.normal(25, 8, n).clip(1, 60)
        dependents = np.random.choice([0,1,2,3], size=n, p=[0.4,0.3,0.2,0.1])
        risk = np.random.randint(0, 15, n)

    elif mode == "older":
        age = np.random.normal(60, 10, n).clip(30, 85)
        dependents = np.random.choice([0,1,2], size=n, p=[0.6,0.3,0.1])
        risk = np.random.randint(0, 20, n)

    elif mode == "high_risk":
        age = np.random.normal(50, 15, n).clip(1, 85)
        dependents = np.random.choice([0,1,2,3,4], size=n, p=[0.3,0.25,0.2,0.15,0.1])
        risk = np.random.randint(10, 21, n)

    elif mode == "many_dependents":
        age = np.random.normal(40, 12, n).clip(1, 85)
        dependents = np.random.choice([2,3,4,5,6], size=n, p=[0.2,0.3,0.25,0.15,0.1])
        risk = np.random.randint(0, 10, n)

    elif mode == "random":
        age = np.random.randint(1, 85, n)
        dependents = np.random.randint(0, 7, n)
        risk = np.random.randint(0, 21, n)

    else:  # general population
        age = np.random.normal(50, 15, n).clip(1, 85)
        dependents = np.random.choice([0,1,2,3,4,5,6], size=n,
                                      p=[0.35,0.25,0.2,0.1,0.05,0.03,0.02])
        risk = np.random.randint(0, 21, n)

    # -----------------------------
    # Other variables (same across modes)
    # -----------------------------
    df = pd.DataFrame({
        "id": np.arange(1, n + 1),
        "A": age,
        "D": dependents,
        "a": np.random.normal(12, 8, n).clip(0, 80),
        "S": np.random.normal(75, 10, n).clip(50, 100),
        "S_prev": np.random.choice([0, 60, 80, 90], size=n, p=[0.9, 0.05, 0.03, 0.02]),
        "n": np.random.choice([0,1,2], size=n, p=[0.85, 0.12, 0.03]),
        "R": risk,
        "Q": np.random.uniform(0.4, 1.0, n),
        "L": np.random.normal(15, 8, n).clip(1, 40),
        "C": np.random.choice([1.0,1.1,1.2,1.3,1.4,1.5,2.0], size=n,
                              p=[0.25,0.25,0.2,0.15,0.1,0.04,0.01])
    })

    df.loc[df["D"] == 0, "a"] = 0
    return df


def main():
    # Change mode here:
    mode = "general"  # options: young, older, high_risk, many_dependents, random, general

    print(f"Generating dataset mode: {mode}")
    df = generate_dataset(100000, mode=mode)

    filename = f"synthetic_100k_{mode}.csv"
    df.to_csv(filename, index=False)

    print(f"Saved: {filename}")


if __name__ == "__main__":
    main()