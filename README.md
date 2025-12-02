# propensity-gbdt
### A Unified Framework for Robust Causal Inference
**Enforcing Common Support in Causal Inference with Gradient Boosted Trees.**

The credibility of causal inference methods like Synthetic Controls (SCM), Difference-in-Differences (DiD), and Inverse Propensity Weighting (IPW) hinges on the "common support" assumption—that the treated and control groups must be comparable. While traditional methods rely on subjective selection or linear models like logistic regression which may fail to capture complex, non-linear relationships, `propensity-gbdt` uses the power of gradient boosted trees to objectively ensure that your control units are truly comparable.

The vision of this repository is to provide a unified, machine learning-driven framework for these techniques. Our central thesis is that modern gradient boosting models are exceptionally good at estimating propensity scores, and that these scores are a powerful tool to **detect and eliminate off-support control units**.

#### The Unifying Principle: **prune before you estimate**.

For each causal method, our workflow enforces common support before the final estimation:

1.  **Estimate Propensity Scores:** Use a flexible gradient boosting model to predict the probability of treatment based on pre-intervention data: outcomes variables.
2.  **Identify and Prune:** Use these scores to identify and remove control units that are **off-support**—those that don't overlap with the treated group's characteristics, ensuring that the common support assumption is met.
3.  **Perform Causal Analysis:** Run the final analysis (SCM, DiD, etc.) using only the pruned, high-quality set of on-support control units.

This approach ensures that the final effect estimation is more robust, credible, and less prone to bias from invalid comparisons.

#### Current and Future Implementations

*   **✓ Synthetic Control Methods (SCM):** The current implementation offers a two-stage approach: first, it uses the GBDT-based pruning to create a small, causally valid donor pool. Second, it offers a **Bayesian SCM implementation** to estimate the final weights and their corresponding uncertainty, providing a full posterior distribution for the treatment effect.

*   **→ Difference-in-Differences (DiD):** Future work will apply this pruning method to refine control groups for DiD models, ensuring parallel trends are evaluated on a more comparable set of units. This will provide a Python implementation of the robust methods found in modern packages like the R `did` library.

*   **→ Inverse Propensity Weighting (IPW):** The framework will be extended to use this pre-estimation pruning step before applying standard IPW, helping to prevent extreme weights and improve model stability.


## Data Schema

The input DataFrame (`all_units`) must be in a **long format**, where each row represents a single observation for a specific unit, variable, and time period. The table should have the following structure:

| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| `unitname` | str | Orderable identifier for the unit (e.g., city ID, user ID). |
| `tname` | str | Orderable time period identifier (e.g., year, year_month). |
| `yname` | str | Name of the outcome variable. |
| `value` | float | The numeric value for the given `yname`. |
| `treatment` | int | Binary indicator: `1` for the treated unit, `0` for control units. |
| `pre_intervention` | int | Binary indicator: `1` for the pre-intervention period, `0` for post-intervention. |

Here, the column names (`unitname`, `tname`, etc.) are placeholders. You will map your DataFrame's column names to these roles using the arguments in the `donor_selection.search()` function.

## Installation

```bash
pip install git+https://github.com/fredguinog/propensity-gbdt.git
```

## Usage

```python
import numpy as np
import pandas as pd
from propensitygbdt.scm import donor_selection

# https://www.ers.usda.gov/data-products/county-level-data-sets/county-level-data-sets-download-data
data = pd.read_csv("https://ers.usda.gov/sites/default/files/_laserfiche/DataFiles/48747/Unemployment2023.csv")

data['FIPS_Code'] = data['FIPS_Code'].astype(str).str.zfill(5)
data = data[~data['FIPS_Code'].astype(str).str.endswith('0')].reset_index(drop=True)

data['year'] = data['Attribute'].str[-4:]
data['indicator'] = data['Attribute'].str[:-5]

indicators_to_keep = [
    "Civilian_labor_force",
    "Unemployment_rate"
]
data = data[data['indicator'].isin(indicators_to_keep)]

columns_ordered = [
    "year",
    "FIPS_Code",
    "State",
    "Area_Name",
    "indicator",
    "Value"
]
data = data[columns_ordered]

data = data.dropna(subset=['Value']).reset_index(drop=True)

print(data.head())
print(data.info())

data['treatment'] = np.where(data['FIPS_Code'] == "12001", 1, 0)
data['pre_intervention'] = np.where(data['year'].isin(['2000', '2001', '2002', '2003','2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015']), 1, 0)

donor_selection.search(
    all_units = data,
    yname = "indicator",
    unitname = "FIPS_Code",
    tname = "year",
    value = "Value",
    treatment = "treatment",
    pre_intervention = "pre_intervention",
    workspace_folder = 'C:/test_propensitygbdt_scm_donor_selection/'
)

from propensitygbdt.scm import bayesian_scm

bayesian_scm.estimate(
    timeid_previous_intervention = '2015',
    workspace_folder = 'C:/test_propensitygbdt_scm_donor_selection/'
)
```

## donor_selection.search Parameters

### Main Parameters

*   `all_units` : pd.DataFrame. The panel data containing columns for time, unit, outcome, treatment indicator, and values.
*   `yname` : str. Name of the column containing the outcome variable names/labels.
*   `unitname` : str. Name of the column containing unit identifiers.
*   `tname` : str. Name of the column containing time identifiers.
*   `value` : str. Name of the column containing the metric values.
*   `treatment` : str. Name of the column indicating treatment status (1 for treated unit, 0 for control).
*   `pre_intervention` : str. Name of the column indicating the pre-intervention period (1 for pre, 0 for post).
*   `workspace_folder` : str. Path to the directory where intermediate results and the solution CSV will be saved.

### Optional Parameters

*   `temporal_cross_search_splits` : list, optional. List of time-IDs defining the cutoffs for the expanding window cross-validation. If None, splits are calculated automatically based on ratios.
*   `seed` : int, default=111. Random seed for reproducibility in sampling and model training.
*   `maximum_control_sd_times_treatment_sd `: int, default=5.0. Threshold for filtering control units based on variance comparison with the treated unit.
*   `maximum_num_units_on_attipw_support` : int, default=50. Maximum number of control units to select based on IPW ranking before Gram condition selection then fitting SCM.
*   `maximum_gram_cond_train` : float, default=500.0. Maximum allowable condition number for the Gram matrix of selected donors to ensure linear independence (mitigates multicollinearity).
*   `minimum_donor_selection` : int, default=3. Minimum number of donor units required to form a valid synthetic control.
*   `maximum_control_unit_weight_train` : float, default=0.5. Constraint to ensure no single donor dominates the synthetic control (max weight < 0.5).
*   `synthetic_control_bias_removal_period` : Literal, default='pre_intervention'. Strategy for centering/scaling control units relative to the treated unit (e.g., based on the full pre-period).
*   `function_aggregate_outcomes_error` : Literal, default='mean'. Metric to aggregate errors across multiple outcomes ('mean' or 'max').
*   `save_solution_period_error` : Literal, default='pre_intervention'. Determines which period's error is checked against `save_solution_maximum_error` to decide if a solution is saved.
*   `save_solution_maximum_error` : float, default=0.15. The maximum allowable RMSPE (normalized) for a candidate solution to be saved to disk.
*   `alpha_exponential_decay` : float, default=0.00. Decay factor for time-relevance weights; higher values give more weight to recent time periods.
*   `optuna_optimization_target` : Literal, default='pre_intervention'. The error metric Optuna attempts to minimize ('pre_intervention' or 'validation_folder').
*   `optuna_number_trials` : int, default=1000. Number of hyperparameter optimization trials to run per temporal split.
*   `optuna_timeout_cycle` : int, default=900. Time limit (in seconds) for the Optuna optimization cycle.

## bayesian_scm.estimate Parameters

### Main Parameters

*   `timeid_previous_intervention` : str. The identifier for the final time period before the intervention starts. This value marks the end of the pre-treatment window.
*   `workspace_folder` : str. The file path to the root directory that contains the input data file ('scm_donor_selection_candidate_units_data.csv') and will store all output files (CSVs, plots).

### Optional Parameters

*   `solution_id` : int, default=None.  The specific ID for a pre-selected group of control units (the donor pool). If set to None, the function will automatically choose the solution that demonstrates the best balance between pre-intervention fit and impact score.
*   `period_effect_format` : str, default='{:.2f}'. A format string used to display numeric results in the plot annotations, such as the average treatment effect.
*   `seed` : int, default=222. A random seed to ensure the reproducibility of the MCMC sampling and any other stochastic processes.
*   `maximum_gram_cond` : float, default=100.0. The maximum allowable value for the Gram matrix condition number. This is used as a threshold to detect multicollinearity among control units; solutions exceeding it are flagged.
*   `maximum_mean_gini_weights` : float, default=0.75. The maximum permissible mean gini of the weight distribution for the synthetic control. This helps prevent the model from relying too heavily on one control unit.

## Citation

The methodology implemented in this package is described in the following article. If you use this software in your research, please cite it.

Nogueira, Frederico. "A New Lens for Donor Selection: ATT/IPW-Based Ranking." *Medium*, 2025. [https://medium.com/@frederico.nogueira/a-new-lens-for-donor-selection-att-ipw-based-ranking-198b9d30bc69](https://medium.com/@frederico.nogueira/a-new-lens-for-donor-selection-att-ipw-based-ranking-198b9d30bc69)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
