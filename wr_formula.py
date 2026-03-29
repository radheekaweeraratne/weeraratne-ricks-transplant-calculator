import math

def wr_allocation_score(
    D,      # number of dependents
    a,      # mean age of dependents (0 if none)
    S,      # success probability current transplant (0-100)
    S_prev, # success probability previous transplants (0-100)
    n,      # number of previous transplants
    R,      # lifestyle risk (0-20)
    Q,      # quality of life after transplant (0.0-1.0)
    L,      # years of life post-transplant
    C,      # critical need factor
    A       # age of patient
):
    """
    Computes the Weeraratne-Ricks Transplant Allocation Score
    using the revised age-normalized formula.
    """

    # Dependent term
    if D == 0:
        dependent_term = 0
    else:
        dependent_term = D * (math.log(a + 1) / 2 + 1)

    # Success term
    success_term = (S + S_prev) / 100

    # Transplant history normalization
    transplant_term = n + 1

    # Age penalty (revised formula)
    age_penalty = 0.01 * abs(25 - A) + 1

    # Final score
    score = (dependent_term * success_term / transplant_term) * (Q * L * C / age_penalty) - R

    return max(0, score)


if __name__ == "__main__":
    # Quick test with two hypothetical patients

    patient_A = wr_allocation_score(
        D=3, a=14, S=62, S_prev=0, n=0,
        R=0, Q=0.94, L=18, C=1.3, A=40
    )

    patient_B = wr_allocation_score(
        D=6, a=8, S=95, S_prev=0, n=0,
        R=5, Q=0.89, L=37, C=1.4, A=28
    )

    print("Patient A Score:", patient_A)
    print("Patient B Score:", patient_B)