# propensity-gbdt
### A Unified Framework for Robust Causal Inference
**Enforcing Common Support in Causal Inference with Gradient Boosted Trees.**

The credibility of causal inference methods like Synthetic Controls (SCM), Difference-in-Differences (DiD), and Inverse Propensity Weighting (IPW) hinges on the "common support" assumption—that the treated and control groups must be comparable. While traditional methods rely on subjective selection or linear models like logistic regression which may fail to capture complex, non-linear relationships, `propensity-gbdt` uses the power of gradient boosted trees to objectively ensure that your control units are truly comparable.

The vision of this repository is to provide a unified, machine learning-driven framework for these techniques. Our central thesis is that modern gradient boosting models are exceptionally good at estimating propensity scores, and that these scores are a powerful tool to **detect and eliminate off-support control units**.

#### The Unifying Principle: **prune before you estimate**.

For each causal method, our workflow enforces common support before the final estimation:

1.  **Estimate Propensity Scores:** Use a flexible gradient boosting model to predict the probability of treatment based on pre-intervention data: outcomes and/or covariates.
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
| `tname` | str | Orderable time period identifier (e.g., year, month, 'covariate' for timeless covariables). |
| `yname` | str | Name of the outcome variable or covariate. |
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

# for .xlsx files, you may need to install (`pip install openpyxl`).
data = pd.read_excel("https://ers.usda.gov/sites/default/files/_laserfiche/DataFiles/48747/Unemployment2023.xlsx", header=4)

data = data.melt(
    id_vars=["FIPS_Code", "State", "Area_Name"],
    var_name="indicator_year",
    value_name="value"
)

data['year'] = data['indicator_year'].str[-4:]
data['indicator'] = data['indicator_year'].str[:-5]

indicators_to_keep = [
    "Civilian_labor_force",
    "Unemployed",
    "Unemployment_rate",
    "Urban_Influence_Code"
]
dataset_mask = data['indicator'].isin(indicators_to_keep)
data = data[dataset_mask]

columns_ordered = [
    "year",
    "FIPS_Code",
    "State",
    "Area_Name",
    "indicator",
    "value"
]
data = data[columns_ordered]
data.loc[data['indicator'] == "Urban_Influence_Code", 'year'] = "covariate"

data = data.dropna(subset=['value']).reset_index(drop=True)

print(data.head())
print(data.info())

data['indicator'] = data['indicator'].astype(str)
data['FIPS_Code'] = data['FIPS_Code'].astype(str)
data['year'] = data['year'].astype(str)
data['treatment'] = np.where(data['FIPS_Code'] == "12001", 1, 0)
data['pre_intervention'] = np.where(data['year'].isin(['2000', '2001', '2002', '2003','2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015']), 1, 0)

donor_selection.search(
    all_units = data,
    yname = "indicator",
    unitname = "FIPS_Code",
    tname = "year",
    value = "value",
    treatment = "treatment",
    pre_intervention = "pre_intervention",
    temporal_cross_search = ['2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011'],
    workspace_folder = 'C:/test_propensitygbdt_scm_donor_selection/',
    tname_covariate = "covariate",
    # include_error_post_intervention_in_optuna_objective = True
)
```

## Parameters

### Main Parameters

*   `all_units` (pd.DataFrame): The input DataFrame containing the entire dataset.
*   `yname` (str): The name of the column indicating the outcome variable.
*   `unitname` (str): The name of the column for the unit identifier.
*   `tname` (str): The name of the column for the time period.
*   `value` (str): The name of the column for the value of the outcome variable.
*   `treatment` (str): The name of the column indicating treatment status (1 for treated, 0 for control).
*   `pre_intervention` (str): The name of the column indicating the pre-intervention period (1 for pre-intervention, 0 for post-intervention).
*   `temporal_cross_search` (list): A list of pre-intervention splits to be used for cross-validation.
*   `workspace_folder` (str): The path to the folder where the output files will be saved.

### Optional Parameters

*   `tname_covariate` (str, optional): The name of the time period that identifies covariates. Defaults to `None`.
*   `seed` (int, optional): The random seed for reproducibility. Defaults to `111`.
*   `maximum_num_units_on_support_first_filter` (int, optional): The maximum number of units to be considered "on support" in the first filtering stage. Defaults to `50`.
*   `maximum_error_pre_intervention` (float, optional): The maximum allowable error in the pre-treatment period for a donor pool to be considered valid. Defaults to `0.15`.
*   `maximum_error_covariates` (float, optional): The maximum allowable error for covariates. Defaults to `0.15`.
*   `proportion_pre_intervention_period_outcomes_donor` (int, optional): The desired ratio of pre-intervention data points of all outcomes to the maximum size of the donor pool. Defaults to `10`.
```python
    superior_limit_maximum_donor_pool_size = int(num_pre_intervention_periods_per_outcome['timeid'].sum() / proportion_pre_intervention_period_outcomes_donor)
    if superior_limit_maximum_donor_pool_size < 2:
        superior_limit_maximum_donor_pool_size = 2
```
*   `inferior_limit_maximum_donor_pool_size` (int, optional): The minimum size of the donor pool. Defaults to `2`.
*   `on_support_first_filter` (str, optional): The strategy for the first on-support filter. Can be `'max_weight'`, `'maximum_num_units_on_support_first_filter'`, or `'bigger_than_min_weight'`. Defaults to `'max_weight'`.
*   `on_support_second_filter` (str, optional): The strategy for the second on-support filter, applyied after the first one. Can be `'randomN'` or `'all'`. N being a number between inferior_limit_maximum_donor_pool_size and superior_limit_maximum_donor_pool_size. Defaults to `'randomN'`.
*   `include_error_post_intervention_in_optuna_objective` (bool, optional): Flag to include or not the post-intervention error in the Optuna's search criteria. Defaults to `False`.
*   `number_optuna_trials` (int, optional): The number of trials for the Optuna hyperparameter optimization. Defaults to `300`.
*   `timeout_optuna_cycle` (int, optional): The timeout in seconds for each Optuna optimization cycle. Defaults to `900`.
*   `time_serie_covariate_metric` (str, optional): The metric to be used for time-series covariate balancing. Can be `'rmse'` or `'mae'`. Defaults to `'rmse'`.

## Citation

The methodology implemented in this package is described in the following article. If you use this software in your research, please cite it.

Nogueira, Frederico. "A New Lens for Donor Selection: ATT/IPW-Based Ranking." *Medium*, 2025. [https://medium.com/@frederico.nogueira/a-new-lens-for-donor-selection-att-ipw-based-ranking-198b9d30bc69](https://medium.com/@frederico.nogueira/a-new-lens-for-donor-selection-att-ipw-based-ranking-198b9d30bc69)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
