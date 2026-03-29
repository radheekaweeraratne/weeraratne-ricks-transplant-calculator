from flask import Flask, render_template, request
import math

app = Flask(__name__)

def wr_allocation_score(D, a, S, S_prev, n, R, Q, L, C, A):
    # Dependents term: D * (log(a+1)/2) + 1
    # This gives a baseline of 1 even when D = 0
    dependent_term = D * (math.log(a + 1.0) / 2.0) + 1.0

    # Success term
    success_term = (S + S_prev) / 100.0

    # Transplant history normalization
    transplant_term = n + 1.0

    # Age penalty (revised formula)
    age_penalty = 0.01 * abs(25.0 - A) + 1.0

    # Core score
    raw_score = (dependent_term * success_term / transplant_term) * (Q * L * C / age_penalty) - R

    # Clamp to non-negative
    final_score = max(0.0, raw_score)

    return round(final_score, 4)


@app.route("/", methods=["GET", "POST"])
def index():
    score = None

    if request.method == "POST":
        try:
            D = float(request.form.get("D", 0))
            a = float(request.form.get("a", 0))
            S = float(request.form.get("S", 0))
            S_prev = float(request.form.get("S_prev", 0))
            n = float(request.form.get("n", 0))
            R = float(request.form.get("R", 0))
            Q = float(request.form.get("Q", 0))
            L = float(request.form.get("L", 0))
            C = float(request.form.get("C", 1))
            A = float(request.form.get("A", 0))

            score = wr_allocation_score(D, a, S, S_prev, n, R, Q, L, C, A)

        except Exception as e:
            score = f"Error: {e}"

    return render_template("index.html", score=score)


if __name__ == "__main__":
    app.run(debug=True)