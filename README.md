# propensity-gbdt
### A Unified Framework for Robust Causal Inference
**Enforcing Common Support in Causal Inference with Gradient Boosted Trees.**

The credibility of causal inference methods like Synthetic Controls (SCM), Difference-in-Differences (DiD), and Inverse Propensity Weighting (IPW) hinges on the "common support" assumption—that the treated and control groups must be comparable. While traditional methods rely on subjective selection or linear models like logistic regression which may fail to capture complex, non-linear relationships, `propensity-gbdt` uses the power of gradient boosted trees to objectively ensure that your control units are truly comparable.

The vision of this repository is to provide a unified, machine learning-driven framework for these techniques. Our central thesis is that modern gradient boosting models are exceptionally good at estimating propensity scores, and that these scores are a powerful tool to **detect and eliminate off-support control units**.

#### The Unifying Principle: **prune before you estimate**.

For each causal method, our workflow enforces common support before the final estimation:

1.  **Estimate Propensity Scores:** Use a flexible gradient boosting model (like XGBoost or LightGBM) to predict the probability of treatment based on pre-intervention data: outcomes and/or covariates.
2.  **Identify and Prune:** Use these scores to identify and remove control units that are **off-support**—those that don't overlap with the treated group's characteristics, ensuring that the common support assumption is met.
3.  **Perform Causal Analysis:** Run the final analysis (SCM, DiD, etc.) using only the pruned, high-quality set of on-support control units.

This approach ensures that the final effect estimation is more robust, credible, and less prone to bias from invalid comparisons.

#### Current and Future Implementations

*   **✓ Synthetic Control Methods (SCM):** The current implementation offers a two-stage approach: first, it uses the GBDT-based pruning to create a small, causally valid donor pool. Second, it offers a **Bayesian SCM implementation** to estimate the final weights and their corresponding uncertainty, providing a full posterior distribution for the treatment effect.

*   **→ Difference-in-Differences (DiD):** Future work will apply this pruning method to refine control groups for DiD models, ensuring parallel trends are evaluated on a more comparable set of units. This will provide a Python implementation of the robust methods found in modern packages like the R `did` library.

*   **→ Inverse Propensity Weighting (IPW):** The framework will be extended to use this pre-estimation pruning step before applying standard IPW, helping to prevent extreme weights and improve model stability.

