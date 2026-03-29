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
    Final Weeraratne-Ricks Transplant Allocation Score
    (revised age-normalized formula, non-negative clamped).

    Score = max(0,
        [ D * (log(a+1)/2 + 1) * (S + S_prev)/100 / (n+1) ] *
        [ Q * L * C / (0.01*|25 - A| + 1) ]
        - R
    )
    """

    # Handle dependents
    if D <= 0:
        dependent_term = 0.0
    else:
        dependent_term = D * (math.log(a + 1) / 2.0 + 1.0)

    # Success term
    success_term = (S + S_prev) / 100.0

    # Transplant history normalization
    transplant_term = n + 1.0

    # Age penalty (revised)
    age_penalty = 0.01 * abs(25 - A) + 1.0

    # Core score
    raw_score = (dependent_term * success_term / transplant_term) * (Q * L * C / age_penalty) - R

    # Clamp to non-negative
    final_score = max(0.0, raw_score)

    return final_score