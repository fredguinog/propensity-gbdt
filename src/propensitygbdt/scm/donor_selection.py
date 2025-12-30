# -*- coding: utf-8 -*-

# Copyright (c) 2025 Frederico Guilherme Nogueira (frederico.nogueira@gmail.com)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
This script implements a two-step, machine learning-driven framework for donor
selection in Synthetic Control Methods (SCM), as described in the Medium article
"A New Lens for Donor Selection: ATT/IPW-Based Ranking" by Frederico Nogueira.

The primary goal of this algorithm is to address the "Curse of Dimensionality"
in SCM by providing a rigorous and objective process for selecting a valid
donor pool, thus minimizing the risks of overfitting and p-hacking.

The framework consists of two main steps:
1.  **Donor Selection:** [THIS SCRIPT] A carefully tuned XGBoost model is used to identify a
    small, causally valid donor units. This is achieved through a
    sophisticated multi-loop search process that leverages ATT/IPW-based
    ranking and cross-temporal validation.
2.  **Optimization:** The selected donors are then used in a
    Bayesian SCM algorithm to estimate the effect and its uncertainty
    associated with the synthetic control.

This implementation aims to transform the donor selection process from a
subjective "dark art" into a more scientific and trustworthy methodology for
causal inference.

.. _Medium Article:
    https://medium.com/@frederico.nogueira/a-new-lens-for-donor-selection-att-ipw-based-ranking-198b9d30bc69

"""

from importlib import resources
import os
import shutil
import numpy as np
import pandas as pd
import math
from scipy.optimize import minimize
from scipy.linalg import qr
from typing import Literal, get_args
import xgboost as xgb
xgb.set_config(verbosity=0)
import optuna
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

"""
Calculates the Gram condition number of the shapes of time series for multiple outcomes.
This function isolates the shape of the time series by taking the first difference.
It then computes the condition number of the Gram matrix of these normalized differenced series
for each outcome, providing a measure of shape multicollinearity among control units.
Differencing removes levels; per-unit standardization of differences removes scales, focusing purely on shape.
Uses QR orthogonalization with pivoting for robust condition number estimation.

Args:
    df: A long format pandas DataFrame with the following columns:
        'treatment': A binary indicator (0 for control, 1 for treated).
        'timeid': The time period identifier.
        'unitid': The unit identifier.
        'outcome': The name of the outcome variable.
        'value': The value of the outcome.

Returns:
    A pandas DataFrame with the Gram condition number for each outcome.
"""
def calculate_gram_cond_by_shape(df: pd.DataFrame) -> np.ndarray:
    # Ensure the DataFrame is sorted for differencing
    df = df.sort_values(by=['outcome', 'unitid', 'timeid']).reset_index(drop=True)
    # Isolate control units
    control_df = df[df['treatment'] == 0].copy()
    # Calculate the first difference of the 'value' for each unit and outcome
    # This transformation focuses on the shape of the time series
    control_df['value_diff'] = control_df.groupby(['unitid', 'outcome'])['value'].diff()
    # Pivot the table to have time series of control units as columns for each outcome
    # Drop rows with NaN values resulting from the differencing
    pivot_df = control_df.dropna().pivot_table(
        index=['outcome', 'timeid'],
        columns='unitid',
        values='value_diff'
    )
    # Group by outcome to calculate the Gram condition number for each
    results = []
    for outcome, group_df in pivot_df.groupby(level='outcome'):
        # The matrix of differenced time series for the control units
        X = group_df.values
       
        # Normalize to ignore scales: standardize per unit (column)
        if X.shape[1] > 0:
            stds = np.std(X, axis=0)
            # Handle zero std (constant series after diff)
            stds[stds == 0] = 1.0  # Treat as unit variance to avoid div-by-zero
            X_normalized = X / stds[None, :]  # Unit-variance columns, pure shape
           
            # Robust cond via QR orthogonalization with pivoting
            _, R, _ = qr(X_normalized, mode='economic', pivoting=True)
            if R.size > 0:
                cond_r = np.linalg.cond(R)
                cond_number = cond_r ** 2
            else:
                cond_number = 1.0
            if cond_number == float('inf'):
                cond_number = 9999999.0
            results.append({'outcome': outcome, 'gram_cond': cond_number})
    return pd.DataFrame(results)['gram_cond'].to_numpy()

"""
Robust per-outcome SNR estimator using differenced noise.

Signal: median absolute level
Noise: MAD of first differences (scaled)

Parameters
----------
y_treated : array-like, shape (T, M)
    Pre-treatment treated outcomes.
eps : float
    Numerical floor to prevent division by zero.

Returns
-------
snr : np.ndarray, shape (M,)
    Estimated SNR per outcome.
"""
def estimate_snr_per_outcome(
    y_treated: np.ndarray,
    eps: float = 1e-8
) -> np.ndarray:
    Y = np.atleast_2d(y_treated)
    T, M = Y.shape

    if T < 3:
        raise ValueError("Need at least 3 pre-periods for SNR estimation.")

    snr = np.zeros(M)

    for m in range(M):
        ts = Y[:, m]

        # --- Signal: robust scale of levels ---
        signal = np.median(np.abs(ts))

        # --- Noise: robust scale of increments ---
        diffs = np.diff(ts)
        mad_diff = np.median(np.abs(diffs - np.median(diffs)))

        # Gaussian-consistent sigma of noise increments
        sigma_noise = 1.4826 * mad_diff / np.sqrt(2)

        if sigma_noise > eps:
            snr[m] = (signal / sigma_noise) ** 2
        else:
            snr[m] = (eps / eps) ** 2

    return snr

"""
Adaptive Gram condition number threshold.

Parameters
----------
tpre : int
    Number of pre-treatment periods.
snr : float
    Estimated SNR for the outcome.

Returns
-------
threshold : float
    Maximum admissible Gram condition number.
"""
def gram_cond_threshold(
    tpre: int,
    snr: float
) -> float:
    if tpre <= 0 or snr <= 0:
        raise ValueError("Invalid inputs.")

    precision = np.sqrt(tpre * snr)
    trust = precision / (1.0 + precision)

    return float(100.0 * np.exp(1.0 / trust))

"""
Greedy donor selection with per-outcome Gram condition enforcement.

- Outcome-specific Gram thresholds are computed internally.
- Stoppage occurs when ANY outcome violates its threshold.
- No outcome averaging, no geometry leakage, no redundant parameters.

Parameters
----------
X : array, shape (T_pre, K * M)
    Donor pre-treatment series.
y_treated : array, shape (T_pre, M)
    Treated unit pre-treatment outcomes.
max_donors : int, optional
    Hard cap on number of selected donors.
seed : int, optional
    Random seed for reproducibility.

Returns
-------
selected_indices : list[int]
    Selected donor indices.
final_max_cond : float
    Worst-outcome Gram condition of final donor set.
"""
def build_low_cond_set_greedy_robust(
    X: np.ndarray,
    gram_thresholds: np.ndarray,
    M: int,
    max_donors: int | None = None,
    seed: int | None = None
):
    if seed is not None:
        np.random.seed(seed)

    T, total_cols = X.shape

    if T < 2:
        raise ValueError("Need at least two pre-periods for differencing.")
    if total_cols % M != 0:
        raise ValueError("X columns incompatible with number of outcomes.")

    K = total_cols // M

    selected = set()
    available = list(range(K))
    if seed is not None:
        np.random.shuffle(available)

    final_max_cond = 0.0

    while available:
        best_idx = None
        best_worst_cond = np.inf

        for idx in available:
            temp = sorted(selected | {idx})
            worst_cond = 0.0
            violated = False

            # ---- per-outcome geometry (fail-fast) ----
            for m in range(M):
                cols_m = [X[:, j * M + m] for j in temp]
                Xm = np.column_stack(cols_m)      # (T, |temp|)
                dXm = np.diff(Xm, axis=0)         # (T-1, |temp|)

                if dXm.shape[1] == 0:
                    continue

                stds = np.std(dXm, axis=0)
                stds[stds == 0.0] = 1.0
                dXm /= stds[None, :]

                _, R, _ = qr(dXm, mode="economic", pivoting=True)
                cond_m = np.linalg.cond(R) ** 2 if R.size else 1.0

                # ---- strict per-outcome stoppage ----
                if cond_m >= gram_thresholds[m]:
                    violated = True
                    break

                if cond_m > worst_cond:
                    worst_cond = cond_m

            if violated:
                continue

            if worst_cond < best_worst_cond:
                best_worst_cond = worst_cond
                best_idx = idx

        # ---- global stoppage ----
        if best_idx is None:
            break

        selected.add(best_idx)
        available.remove(best_idx)
        final_max_cond = best_worst_cond

        if max_donors is not None and len(selected) >= max_donors:
            break

    return sorted(selected), final_max_cond

"""
Calculates a single set of optimal weights for a synthetic control group
by simultaneously minimizing the error across multiple, standardized outcome variables.

This function first standardizes the data for each outcome variable (mean=0, std=1)
before performing the optimization. This is crucial when outcomes are on different scales.

Args:
    treated_pre_intervention (np.ndarray): A 2D array where each row represents a
                                            pre-intervention time period and each
                                            column represents an outcome variable for
                                            the treated unit. (Shape: periods x outcomes)
    control_pre_intervention (np.ndarray): A 3D array where the first dimension
                                            represents pre-intervention time periods,
                                            the second represents control units, and
                                            the third represents outcome variables.
                                            (Shape: periods x units x outcomes)
    max_weight (float, optional): The maximum acceptable weight for any single control unit.
                                   Defaults to 1.0 (no additional restriction beyond the sum-to-1 constraint).

Returns:
    np.ndarray: A 1D array of the optimal weights for the control units.
"""
def fast_synthetic_control_fitting(treated_pre_intervention, control_pre_intervention, max_weight=1.0):
    num_control_units = control_pre_intervention.shape[1]
    num_outcomes = treated_pre_intervention.shape[1]

    # --- Data Standardization Step ---
    # Create copies to avoid modifying the original data
    standardized_treated = np.copy(treated_pre_intervention).astype(float)
    standardized_control = np.copy(control_pre_intervention).astype(float)

    for i in range(num_outcomes):
        # Extract all data for the current outcome
        all_outcome_data = np.concatenate(
            [treated_pre_intervention[:, i], control_pre_intervention[:, :, i].flatten()]
        )

        # Calculate mean and standard deviation for the current outcome
        mean = np.mean(all_outcome_data)
        std = np.std(all_outcome_data)

        # Standardize, avoiding division by zero
        if std > 0:
            standardized_treated[:, i] = (standardized_treated[:, i] - mean) / std
            standardized_control[:, :, i] = (standardized_control[:, :, i] - mean) / std

    # --- Optimization using Standardized Data ---
    def objective_function(weights):
        """
        The objective function to minimize using standardized data.

        Calculates the synthetic control for all outcomes and returns the total
        sum of squared errors.
        """
        # Sum over the 'units' axis (1) of the control data and the 'weights' axis (0)
        synthetic_control = np.tensordot(standardized_control, weights, axes=([1], [0]))

        # Return the sum of squared errors across all periods and outcomes
        return np.sum((standardized_treated - synthetic_control)**2)

    # Constraints: weights must sum to 1.
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})

    # Bounds: weights must be between 0 and max_weight.
    bounds = tuple((0, max_weight) for _ in range(num_control_units))

    # Initial guess for the weights (equal weighting, capped if necessary).
    initial_weights = np.ones(num_control_units) / num_control_units
    initial_weights = np.minimum(initial_weights, max_weight)  # Ensure initial guess respects max_weight
    initial_weights /= np.sum(initial_weights)  # Renormalize if needed
    # Solve the optimization problem.
    result = minimize(objective_function, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)

    return result.x

"""
Theoretical upper bound for admissible pre-period NRMSE.

Parameters
----------
snr : float
    Estimated SNR for the outcome.
tpre : int
    Number of pre-treatment periods.
k_controls : int
    Number of selected control units.

Returns
-------
bound : float
    Maximum admissible NRMSE.
"""
def nrmse_upper_bound(
    snr: float,
    tpre: int,
    k_controls: int,
    resolution_floor: float = 0.15
) -> float:
    if snr < 0 or tpre <= 0 or k_controls >= tpre:
        raise ValueError("Invalid inputs.")

    noise_bound = (1.0 / np.sqrt(snr + 1.0)) * np.sqrt(1.0 - k_controls / tpre)
    concentration_bound = 1.0 / np.sqrt(tpre)

    identifiability_floor = max(concentration_bound, noise_bound)

    return max(resolution_floor, identifiability_floor)

"""
Adaptive lower bound on weights to enforce DGP separation.

Parameters
----------
tpre : int
snr : float
k_controls : int
gram_cond : float
    Current Gram condition number.

Returns
-------
w_min : float
"""
def w_min_bound(
    tpre: int,
    snr: float,
    k_controls: int,
    w_min_low: float = 0.01,
    w_min_high: float = 0.1
) -> float:
    if tpre <= 0 or snr <= 0 or k_controls <= 0:
        raise ValueError("Invalid inputs.")

    precision = np.sqrt(tpre * snr)
    info = precision / (1.0 + precision)

    w_min_ideal = w_min_low + (w_min_high - w_min_low) * info

    return float(np.clip(w_min_low + w_min_ideal, w_min_low, w_min_high))

"""
Adaptive upper bound on weights to prevent dominance.

Parameters
----------
tpre : int
k_controls : int

Returns
-------
w_max : float
"""
def w_max_bound(
    tpre: int,
    k_controls: int,
    min_bound: float = 0.4
) -> float:
    if tpre <= 0 or k_controls <= 0:
        raise ValueError("Invalid inputs.")
    if min_bound < 0 or min_bound > 1:
        raise ValueError("min_bound must be between 0 and 1.")

    adaptive_bound = np.sqrt(tpre / k_controls)
    return float(max(min_bound, min(1.0, adaptive_bound)))

"""
Finite-sample robust power estimator for smooth mean-shift
treatment effects in SCM.

Statistic:
    T = mean(post gaps) / (sd(pre gaps) / sqrt(T_post))

Inference:
    Pre-period-only circular block bootstrap.

Returns:
    power_by_outcome, aggregate_power

Interpretation:

🔴 0-10% — Effect statistically invisible
Detection is extremely unlikely.
Effect is too small, post period too short, or statistic is misaligned.
Common for RMSPE-ratio under smooth, stable effects.
Honest but uninformative.

🟠 10-30% — Weak detectability
Effect exists but rarely distinguishable from noise.
High sensitivity to sample path and serial correlation.
Typical for variance-based tests in well-behaved SCMs.

🟡 30-50% — Borderline informativeness
Detection roughly a coin flip.
Small perturbations flip inference.
Dangerous regime if over-interpreted.
→ Consider: longer post-period, multiple outcomes, 
   or treat as exploratory rather than confirmatory.

🟢 50-70% — Moderate, honest power (Ideal DGP sweet spot)

Effect detected more often than not.
SCM extrapolates smoothly; uncertainty remains.
Indicates realistic, finite-sample causal inference.
Signals methodological maturity.

🟢🟢 70-85% — Strong detectability
Effect reliably detected.
Requires sizable effect, longer post period, or aligned statistic.
Convincing but still credible.

🔵 85-95% — Near-deterministic detection
Effect almost always detected.
Triggers scrutiny: check conditioning and null construction.
Must be carefully justified.
→ Verify: No post-period leakage? Donor pool large enough?
   Pre-trends genuinely parallel? Effect isn't mechanical?

🚨 95-100% — Suspicious
Inference behaves like classification, not statistics.
Often indicates leakage, misspecified null, or overfitting.
Red flag in SCM simulations.
"""
def estimate_scm_power_mean_shift(
    data_pre,
    data_post,
    effect_sizes_sd_per_outcome,          # standardized effects (Cohen's d per outcome)
    n_sim=1000,
    sig_level=0.05,
    min_block_length=3
):
    rng = np.random.default_rng()
    outcomes = data_pre["outcome"].unique()

    if len(effect_sizes_sd_per_outcome) != len(outcomes):
        raise ValueError("effect_sizes must match number of outcomes.")

    power_by_outcome = {}

    for i, outcome in enumerate(outcomes):
        # Extract data
        pre = data_pre.loc[data_pre["outcome"] == outcome, "value"].to_numpy()
        post = data_post.loc[data_post["outcome"] == outcome, "value"].to_numpy()

        T_pre, T_post = len(pre), len(post)

        if T_pre < 5 or T_post < 2:
            power_by_outcome[outcome] = 0.0
            continue

        # Standardize using pre-period
        mu = pre.mean()
        sd = pre.std()
        if sd == 0:
            power_by_outcome[outcome] = 0.0
            continue

        pre_std = (pre - mu) / sd

        # Vectorized circular block bootstrap (KEEP THIS - it's correct and fast)
        block_len = max(min_block_length, int(math.ceil(T_pre ** (1 / 3))))
        n_blocks = int(math.ceil(T_post / block_len))

        starts = rng.integers(0, T_pre, size=(n_sim, n_blocks))
        offsets = np.arange(block_len)
        indices = (starts[:, :, None] + offsets) % T_pre
        boot_paths = pre_std[indices].reshape(n_sim, -1)[:, :T_post]

        # H0: No effect
        mean_h0 = boot_paths.mean(axis=1)
        
        # FIX: Correct test statistic with proper denominator
        # Since pre_std has unit variance, sd = 1.0
        stat_h0 = mean_h0 / (1.0 / np.sqrt(T_post))
        
        # Critical value
        crit = np.quantile(stat_h0, 1 - sig_level)

        # H1: Effect present (standardized)
        delta = effect_sizes_sd_per_outcome[i]
        boot_paths_h1 = boot_paths + delta
        mean_h1 = boot_paths_h1.mean(axis=1)
        stat_h1 = mean_h1 / (1.0 / np.sqrt(T_post))

        # Power = P(reject H0 | H1 true)
        power_by_outcome[outcome] = np.mean(stat_h1 >= crit)

    return np.array(list(power_by_outcome.values()))

"""
Compare observed per-outcome values against per-outcome adaptive limits.

Stops on any violation.

Parameters
----------
observed : array-like, shape (M,)
    Observed statistics per outcome (e.g., NRMSE, RMSPE, Gram cond).
limits : array-like, shape (M,)
    Adaptive admissible limits per outcome.
strict : bool, default=True
    If True: violation occurs when observed > limit.
    If False: allows equality (observed <= limit).

Returns
-------
bool
    True if all outcomes satisfy their limits, False otherwise.
"""
def check_per_outcome_limits(
    observed: np.ndarray,
    limits: np.ndarray,
    strict: bool = True
) -> bool:
    observed = np.asarray(observed, dtype=float)
    limits = np.asarray(limits, dtype=float)

    if observed.shape != limits.shape:
        raise ValueError("observed and limits must have the same shape.")

    if np.any(limits <= 0):
        raise ValueError("All limits must be positive.")

    if strict:
        return bool(np.all(observed < limits))
    else:
        return bool(np.all(observed <= limits))

"""

This function executes a multi-loop search to identify optimal donor pools for Synthetic
Control Methods (SCM).

This function combines machine learning (XGBoost) with causal inference (SCM) to construct 
a synthetic counterfactual for a treated unit. It employs an expanding-window cross-validation 
strategy (temporal cross-validation) to find a set of control units that best match the 
treated unit's pre-intervention trajectory.

The algorithm works as follows:
1. **Data Preparation**: Standardizes column names, filters for balanced panels, and validates data integrity.
2. **Temporal Splitting**: divides the pre-intervention period into training and validation splits 
   to prevent overfitting to the immediate pre-treatment noise.
3. **Hyperparameter Optimization (Optuna)**: Searches for the best XGBoost hyperparameters.
4. **Propensity Scoring**: Uses XGBoost to learn a propensity score based on outcome trajectories.
5. **Donor Selection 1 (IPW)**: Converts propensity scores to ATT Inverse Probability Weights (ATT/IPW) 
   to rank and select the top N candidate control units.
6. **Donor Slection 2 (Gram condition)** Build a low-condition-number donor set via greedy forward selection.
7. **SCM Fitting**: Applies standard Synthetic Control optimization on the selected subset of donors to
   calculate exact weights.
8. **Evaluation**: Logs valid solutions (those meeting error and stability thresholds) to a CSV file.

Parameters
----------
all_units : pd.DataFrame
    The panel data containing columns for time, unit, outcome, treatment indicator, and values.
yname : str
    Name of the column containing the outcome variable names/labels.
unitname : str
    Name of the column containing unit identifiers.
tname : str
    Name of the column containing time identifiers.
value : str
    Name of the column containing the metric values.
treatment : str
    Name of the column indicating treatment status (1 for treated unit, 0 for control).
pre_intervention : str
    Name of the column indicating the pre-intervention period (1 for pre, 0 for post).
workspace_folder : str
    Path to the directory where intermediate results and the solution CSV will be saved.
temporal_cross_search_splits : list, optional
    List of time-IDs defining the cutoffs for the expanding window cross-validation. 
    If None, splits are calculated automatically based on ratios.
seed : int, default=111
    Random seed for reproducibility in sampling and model training.
maximum_ratio_var_treated_var_donor : int, default=10.0
    The max ratio (Var_Treated / Var_Donor) allowed.
maximum_num_units_on_attipw_support : int, default=50
    Maximum number of control units to select based on IPW ranking before Gram condition selection then fitting SCM.
maximum_gram_cond_per_outcome_train : float, default=100.0
    Maximum allowable condition number for the Gram matrix of selected donors to ensure 
    linear independence (mitigates multicollinearity).
minimum_donor_selection : int, default=3
    Minimum number of donor units required to form a valid synthetic control.
maximum_control_unit_weight_train : float, default=None
    Constraint to ensure no single donor dominates the synthetic control. None makes the value adaptative.
minimum_control_unit_weight_train : float, default=None
    Constraint to ensure no single donor is meaningless. None makes the value adaptative.
synthetic_control_bias_removal_period : Literal, default='pre_intervention'
    Strategy for centering/scaling control units relative to the treated unit (e.g., based on the full pre-period).
save_solution_period_error : Literal, default='pre_intervention'
    Determines which period's error is checked against `save_solution_maximum_error` to decide if a solution is saved.
save_solution_maximum_error : float, default=None
    The maximum allowable RMSPE (normalized) for a candidate solution to be saved to disk.
maximum_gram_cond_per_outcome_pre : float, Defaults to 100.0. The maximum allowable value for the Gram matrix condition number.
    This is used as a threshold to detect multicollinearity among control units; solutions exceeding it are flagged.
alpha_exponential_decay : float, default=0.00
    Decay factor for time-relevance weights; higher values give more weight to recent time periods.
optuna_optimization_target : Literal, default='pre_intervention'
    The error metric Optuna attempts to minimize ('pre_intervention' or 'validation_folder').
optuna_number_trials : int, default=100
    Number of hyperparameter optimization trials to run per temporal split.
optuna_timeout_cycle : int, default=900
    Time limit (in seconds) for the Optuna optimization cycle.

Returns
-------
None
    The function does not return a value. Instead, it appends valid candidate solutions 
    to 'scm_donor_selection_candidate_units_data.csv' within the `workspace_folder`.
"""
type_synthetic_control_bias_removal_period = Literal['pre_intervention', 'validation_folder', 'last_pre_period_timeid']
type_save_solution_period_error = Literal['pre_intervention', 'validation_folder']
type_optuna_optimization_target = Literal['pre_intervention', 'validation_folder']
def search(
    all_units: pd.DataFrame,
    yname: str,
    unitname: str,
    tname: str,
    value: str,
    treatment: str,
    pre_intervention: str,
    workspace_folder: str,
    temporal_cross_search_splits: list = None,
    seed: int = 111,
    maximum_ratio_var_treated_var_donor: int = 10.0,
    maximum_num_units_on_attipw_support: int = 50,
    maximum_gram_cond_per_outcome_train: float = None,
    minimum_donor_selection: int = 2,
    maximum_control_unit_weight_train: float = None,
    minimum_control_unit_weight_train: float = None,
    synthetic_control_bias_removal_period: type_synthetic_control_bias_removal_period = 'pre_intervention',
    save_solution_period_error: type_save_solution_period_error = 'pre_intervention',
    save_solution_maximum_error: float = None,
    maximum_gram_cond_per_outcome_pre: float = None,
    effect_sizes_sd_per_outcome: list = None,
    alpha_exponential_decay: float = 0.00,
    optuna_optimization_target: type_optuna_optimization_target = 'pre_intervention',
    optuna_number_trials: int = 100,
    optuna_timeout_cycle: int = 900
):
    # RENAME AND INFORM ERRORS
    if tname in all_units.columns:
        all_units.rename(columns={tname: 'timeid'}, inplace=True)
    else:
        raise ValueError(f"The '{tname}' column does not exist in the dataframe.")

    if unitname in all_units.columns:
        all_units.rename(columns={unitname: 'unitid'}, inplace=True)
    else:
        raise ValueError(f"The '{unitname}' column does not exist in the dataframe.")

    if yname in all_units.columns:
        all_units.rename(columns={yname: 'outcome'}, inplace=True)
    else:
        raise ValueError(f"The '{yname}' column does not exist in the dataframe.")

    if value in all_units.columns:
        all_units.rename(columns={value: 'value'}, inplace=True)
    else:
        raise ValueError(f"The '{value}' column does not exist in the dataframe.")

    if treatment in all_units.columns:
        all_units.rename(columns={treatment: 'treatment'}, inplace=True)
    else:
        raise ValueError(f"The '{treatment}' column does not exist in the dataframe.")

    if pre_intervention in all_units.columns:
        all_units.rename(columns={pre_intervention: 'pre_intervention'}, inplace=True)
    else:
        raise ValueError(f"The '{pre_intervention}' column does not exist in the dataframe.")

    # temporal_cross_search_splits MUST BE A SUBSET OF pre_intervention
    if temporal_cross_search_splits is None:
        sorted_timeids = sorted(all_units[all_units['pre_intervention'] == 1]['timeid'].unique().tolist())
        n = len(sorted_timeids)
        ratios = [0.5, 0.6, 0.7, 0.8, 0.9]
        temporal_cross_search_splits= []
        previous_index = -1

        for ratio in ratios:
            target_count = int(n * ratio)
            current_index = target_count - 1
            if current_index <= previous_index:
                current_index = previous_index + 1
            if current_index <= n:
                cutoff_item = sorted_timeids[current_index]
                temporal_cross_search_splits.append(cutoff_item)
            previous_index = current_index
        print(f"Seleted temporal_cross_search_splits: {temporal_cross_search_splits}")
    # CHECK pre_intervention COULUMN AGAINST temporal_cross_search_splits
    elif not set(temporal_cross_search_splits).issubset(set(all_units[all_units['pre_intervention'] == 1]['timeid'].unique().tolist())):
        print(f"temporal_cross_search_splits: {sorted(temporal_cross_search_splits)}")
        print(f"pre_intervention: {sorted(all_units[all_units['pre_intervention'] == 1]['timeid'].unique().tolist())}")
        raise ValueError("ERROR: temporal_cross_search_splits MUST BE A SUBSET OF pre_intervention")
        
    # CHECH DIRECTORY EXISTS AND CREATE IT IF NOT
    if not os.path.exists(workspace_folder):
        os.makedirs(workspace_folder)
        print(f"Created workspace folder: {workspace_folder}")

    if save_solution_period_error != 'pre_intervention' and save_solution_period_error != 'validation_folder':
        raise ValueError("ERROR: save_solution_period_error MUST BE 'pre_intervention' or 'validation_folder'")

    if not isinstance(maximum_ratio_var_treated_var_donor, float) or maximum_ratio_var_treated_var_donor <= 0:
        raise ValueError(f"maximum_ratio_var_treated_var_donor must be a positive real, got {maximum_ratio_var_treated_var_donor}")

    if not isinstance(maximum_num_units_on_attipw_support, int) or maximum_num_units_on_attipw_support <= 0:
        raise ValueError(f"maximum_num_units_on_attipw_support must be a positive integer, got {maximum_num_units_on_attipw_support}")

    if maximum_gram_cond_per_outcome_train is not None and (not isinstance(maximum_gram_cond_per_outcome_train, float) or maximum_gram_cond_per_outcome_train <= 0):
        raise ValueError(f"maximum_gram_cond_per_outcome_train must be a positive real, got {maximum_gram_cond_per_outcome_train}")

    if not isinstance(minimum_donor_selection, int) or minimum_donor_selection <= 0:
        raise ValueError(f"minimum_donor_selection must be a positive integer, got {minimum_donor_selection}")

    if maximum_control_unit_weight_train is not None and (not isinstance(maximum_control_unit_weight_train, float) or maximum_control_unit_weight_train <= 0):
        raise ValueError(f"maximum_control_unit_weight_train must be a positive real, got {maximum_control_unit_weight_train}")
        
    if minimum_control_unit_weight_train is not None and (not isinstance(minimum_control_unit_weight_train, float) or minimum_control_unit_weight_train <= 0):
        raise ValueError(f"minimum_control_unit_weight_train must be a positive real, got {minimum_control_unit_weight_train}")

    valid_options = get_args(type_synthetic_control_bias_removal_period)
    if synthetic_control_bias_removal_period not in valid_options:
        raise ValueError(f"Invalid mode: '{synthetic_control_bias_removal_period}'. Expected one of: {valid_options}")

    valid_options = get_args(type_save_solution_period_error)
    if save_solution_period_error not in valid_options:
        raise ValueError(f"Invalid mode: '{save_solution_period_error}'. Expected one of: {valid_options}")

    if save_solution_maximum_error is not None and not isinstance(save_solution_maximum_error, float) and save_solution_maximum_error <= 0:
        raise ValueError(f"save_solution_maximum_error must be a positive real, got {save_solution_maximum_error}")

    if maximum_gram_cond_per_outcome_pre is not None and (not isinstance(maximum_gram_cond_per_outcome_pre, float) or maximum_gram_cond_per_outcome_pre <= 0):
        raise ValueError(f"maximum_gram_cond_per_outcome_pre must be a positive real, got {maximum_gram_cond_per_outcome_pre}")

    if not isinstance(alpha_exponential_decay, float) or alpha_exponential_decay < 0:
        raise ValueError(f"alpha_exponential_decay must be zero or a positive real, got {alpha_exponential_decay}")

    valid_options = get_args(type_optuna_optimization_target)
    if optuna_optimization_target not in valid_options:
        raise ValueError(f"Invalid mode: '{optuna_optimization_target}'. Expected one of: {valid_options}")

    if not isinstance(optuna_number_trials, int) or optuna_number_trials <= 0:
        raise ValueError(f"optuna_number_trials must be a positive integer, got {optuna_number_trials}")

    if not isinstance(optuna_timeout_cycle, int) or optuna_timeout_cycle <= 0:
        raise ValueError(f"optuna_timeout_cycle must be a positive integer, got {optuna_timeout_cycle}")

    file_path = resources.files('propensitygbdt.data').joinpath('scm_donor_selection_candidate_units_data.xlsx')
    try:
        with resources.as_file(file_path) as source_path:
            shutil.copy(source_path, workspace_folder)
    except (FileNotFoundError, ModuleNotFoundError):
        print("Error: Could not find the source file or the 'propensitygbdt.data' package.")
        print("Please ensure the package is correctly installed.")

    all_units = all_units[['timeid', 'pre_intervention', 'unitid', 'treatment', 'outcome', 'value']]

    all_units_std = all_units.groupby(['unitid', 'treatment', 'outcome']).agg({'value': 'var'}).reset_index()
    control_units_std = pd.merge(all_units_std[all_units_std['treatment'] == 1], all_units_std[all_units_std['treatment'] == 0], how='left', on='outcome')
    control_units_std['acceptable_ratio_var_treated_var_donor'] = np.where(
        (control_units_std['value_x'] / control_units_std['value_y']) <= maximum_ratio_var_treated_var_donor,
        1,
        0
    )
    all_units = all_units[(all_units['treatment'] == 1) | (all_units['unitid'].isin(control_units_std[control_units_std['acceptable_ratio_var_treated_var_donor']  == 1]['unitid_y'].tolist()))]

    # MAKE THE PANEL DATA BALANCED BY REMOVING ALL UNITS WHICH HAVE NOT THE SAME NUMBER OF timeids PER outcome EQUAL TO THEN TREATMENT UNIT
    print(all_units.shape)
    all_units = all_units[all_units['value'].notna()]

    check_timeid_all_outcome = all_units.groupby(['timeid', 'outcome']).size().unstack(fill_value=0)
    valid_timeids = check_timeid_all_outcome[(check_timeid_all_outcome != 0).all(axis=1)].index.tolist()
    all_units = all_units[all_units['timeid'].isin(valid_timeids)]
    dt1 = all_units[['unitid', 'treatment', 'timeid', 'outcome']].copy()

    treatment_indicators = dt1[dt1['treatment'] == 1]['outcome'].unique()
    dt2 = dt1[dt1['outcome'].isin(treatment_indicators)][['unitid', 'treatment', 'timeid', 'outcome']]
    dt2 = dt2.groupby(['unitid', 'treatment', 'timeid']).agg({'outcome': 'count'}).sort_values('outcome').reset_index()
    dt2 = dt2[dt2['outcome'] == len(treatment_indicators)][['unitid', 'treatment', 'timeid']]

    treatment_timeids = dt2[dt2['treatment'] == 1]['timeid'].unique()
    dt3 = dt2[dt2['timeid'].isin(treatment_timeids)][['unitid', 'treatment', 'timeid']]
    dt3 = dt3.groupby(['unitid', 'treatment']).agg({'timeid': 'count'}).sort_values('timeid').reset_index()
    valid_units = dt3[dt3['timeid'] == len(treatment_timeids)]['unitid'].unique()
    invalid_units = all_units[~all_units['unitid'].isin(valid_units)]['unitid'].unique()
    print(f"Number of dropped invalid unitids: {len(invalid_units)}")
    print("Invalid units: ", invalid_units)
    all_units = all_units[all_units['unitid'].isin(valid_units)]
    print(all_units.shape)

    print(all_units.groupby(['timeid', 'outcome']).size().unstack(fill_value=0))
    print(all_units.head())
    print(all_units.info())

    all_units['timeid_relevance'] = pd.Categorical(all_units['timeid'])
    mapping_timeid = all_units['timeid_relevance'].cat.categories
    all_units['timeid_relevance'] = all_units['timeid_relevance'].cat.codes
    all_units['timeid_relevance']= np.exp(-alpha_exponential_decay * (len(mapping_timeid) - all_units['timeid_relevance']))

    treatment_unitid = all_units[all_units['treatment'] == 1]['unitid'].iloc[0]
    outcomes = all_units['outcome'].sort_values().unique().tolist()

    if effect_sizes_sd_per_outcome  is None:
        effect_sizes_sd_per_outcome = np.repeat(0.4, len(outcomes)).tolist() # Moderate effect sd
    elif effect_sizes_sd_per_outcome is not None and not isinstance(effect_sizes_sd_per_outcome, list) and len(effect_sizes_sd_per_outcome) != len(outcomes):
        raise ValueError(f"effect_sizes_sd_per_outcome must be a positive real, got {effect_sizes_sd_per_outcome}")

    # timeids = all_units['timeid'].sort_values().unique().tolist()
    timeid_pre_intervention = all_units[all_units['pre_intervention'] == 1]['timeid'].sort_values().unique().tolist()
    timeid_post_intervention = all_units[all_units['pre_intervention'] == 0]['timeid'].sort_values().unique().tolist()

    scm_donor_selection_candidate_units_data_file_path = workspace_folder + 'scm_donor_selection_candidate_units_data.csv'
    if os.path.exists(scm_donor_selection_candidate_units_data_file_path):
        os.remove(scm_donor_selection_candidate_units_data_file_path)

    all_units.sort_values(by=['unitid', 'timeid', 'outcome'], inplace=True)
    all_units['weight'] = 1.0

    amplitude = all_units[(all_units['treatment'] == 1) & all_units['pre_intervention'] == 1].groupby(['outcome']).apply(
        lambda x : pd.Series({
            'amplitude': 1 if x['value'].max() == x['value'].min() else x['value'].max() - x['value'].min()
        })
    ).reset_index()

    all_units.drop('pre_intervention', axis=1, inplace=True)

    treatment = all_units[all_units['treatment'] == 1]
    treatment.loc[:, 'unitid'] = treatment_unitid

    # Initialize CSV files for storing the donor candidates data with headers.
    scm_donor_selection_candidate_units_data = treatment[['outcome', 'timeid', 'value', 'treatment', 'weight', 'unitid']].copy()
    scm_donor_selection_candidate_units_data['valor_m_weight'] = scm_donor_selection_candidate_units_data['value'] * scm_donor_selection_candidate_units_data['weight']
    scm_donor_selection_candidate_units_data['id'] = None
    scm_donor_selection_candidate_units_data['cycle'] = None
    scm_donor_selection_candidate_units_data['trial'] = None
    scm_donor_selection_candidate_units_data['solution_id'] = None
    scm_donor_selection_candidate_units_data['num_units_on_attipw_support_train'] = None
    scm_donor_selection_candidate_units_data['gram_cond_train'] = None
    scm_donor_selection_candidate_units_data['max_weight_train'] = None
    scm_donor_selection_candidate_units_data['error_train'] = None
    scm_donor_selection_candidate_units_data['error_valid'] = None
    scm_donor_selection_candidate_units_data['error_pre_intervention'] = None
    scm_donor_selection_candidate_units_data['error_post_intervention'] = None
    scm_donor_selection_candidate_units_data['stat_power'] = None
    scm_donor_selection_candidate_units_data['gram_cond_pre'] = None
    scm_donor_selection_candidate_units_data.to_csv(scm_donor_selection_candidate_units_data_file_path, mode='w', header=True, index=False)

    class Dataset:
        """
        This class handles the data for the training and evaluation of the model.
        It's responsible for creating the training set, evaluating the IPW-based metric,
        and calculating the outcome metrics for the synthetic control.
        """
        def __init__(self, data, timeid_train, timeid_valid, timeid_post_intervention, state_context, config_context):
            self.timeid_train = timeid_train
            self.timeid_valid = timeid_valid
            self.timeid_post_intervention = timeid_post_intervention
            
            # Store contexts for state (counters) and config (hyperparameters)
            self.state = state_context
            self.config = config_context

            # The article explains that the data is transposed so that units are rows and time periods are features.
            # This is the ideal format for machine learning models like XGBoost.
            dataset_train = data[data['timeid'].isin(timeid_train)].pivot(
                index=['unitid', 'treatment', 'weight'],
                columns=['outcome', 'timeid'],
                values='value'
            )
            dataset_train.shape
            dataset_train.columns = [f'{col[0]}_{col[1]}' if col[1] else col[0] for col in dataset_train.columns]
            dataset_train = dataset_train.reset_index(level=['unitid', 'treatment', 'weight'])

            # CRIA DATASET DE DEPARA MUNICIPIO - ID PARA SER USADO METRICA DEFINIDA POR FUNCAO CUSTOMIZADE
            dataset_train.sort_values(by=['unitid'], inplace=True)
            dataset_train['id'] = np.arange(len(dataset_train))
            self.dataset_train = dataset_train
            self.from_to = dataset_train[['unitid', 'id']].copy()

            self.full_data = data.copy()
            self.full_data.drop('weight', axis=1, inplace=True)
            self.full_data2 = data.copy()
           
            if maximum_gram_cond_per_outcome_pre is None:
                treatment_df_pre = data[(data['treatment'] == 1) & (data['timeid'].isin(timeid_train + timeid_valid))]
                pivot_treatment_pre = treatment_df_pre.pivot(index='timeid', columns=['outcome'], values='value')
                treatment_ts_pre = pivot_treatment_pre.values
                self.snr_pre = estimate_snr_per_outcome(treatment_ts_pre)
                self.ts_pre_length = len(treatment_ts_pre)
                self.gram_thresholds_per_outcome_pre = np.array([
                    gram_cond_threshold(self.ts_pre_length, self.snr_pre[m]) for m in range(len(outcomes))
                ])
            else:
                self.gram_thresholds_per_outcome_pre = maximum_gram_cond_per_outcome_pre

            if maximum_gram_cond_per_outcome_train is None:
                treatment_df_train = data[(data['treatment'] == 1) & (data['timeid'].isin(timeid_train + timeid_valid))]
                pivot_treatment_train = treatment_df_train.pivot(index='timeid', columns=['outcome'], values='value')
                treatment_ts_train = pivot_treatment_train.values
                self.snr_train = estimate_snr_per_outcome(treatment_ts_train)
                self.ts_train_length = len(treatment_ts_train)
                self.gram_thresholds_per_outcome_train = np.array([
                    gram_cond_threshold(self.ts_train_length, self.snr_train[m]) for m in range(len(outcomes))
                ])
            else:
                self.gram_thresholds_per_outcome_train = maximum_gram_cond_per_outcome_train

        def pre_intervention_scaling(self, data, train_period, valid_period):
            """
            APPLY TRANSFORMATIONS TO THE CONTROL TIME SERIE TO GUARRANTEE EXCLUSIVE SHAPE COMPARISON TO THE TREATMENT ONE
            This function transforms the aggregated control group data using the mean and standard 
            deviation of the treatment group from the pre-intervation period defined by the parameter period.
            By handling this scaling and re-centering step explicitly, we simplify the model's task.
            Instead of having to learn weights that adjust for shape, level and scale, the model can focus
            solely on identifying control units with similar temporal shape to the treatment unit.
            """
            data = data[data['weight'].notna()]
            # Generate the times series created by the weighted average of the units in the dataframe in the defined period 
            aggregated = data.groupby(['outcome', 'timeid', 'treatment'], dropna=False).apply(
                lambda x : pd.Series({
                    'value' : np.ma.filled(np.ma.average(np.ma.masked_invalid(x['value']), weights=x['weight']), fill_value=np.nan)
                })
            ).reset_index()

            # Calculate the average and standard deviation of the time series in the given period
            if synthetic_control_bias_removal_period == 'pre_intervention':
                standardization = aggregated[aggregated['timeid'].isin(train_period + valid_period)].sort_values(['timeid'], ascending=True).groupby(['outcome', 'treatment'], dropna=False).agg(
                    avg = ('value', lambda x: np.mean(x)),
                    std = ('value', lambda x: np.std(x, ddof=1))
                ).reset_index()
            elif synthetic_control_bias_removal_period == 'validation_folder':
                standardization = aggregated[aggregated['timeid'].isin(train_period + valid_period)].sort_values(['timeid'], ascending=True).groupby(['outcome', 'treatment'], dropna=False).agg(
                    avg = ('value', lambda x: np.mean(x[-len(valid_period):] if len(x) >= len(valid_period) else x)),
                    std = ('value', lambda x: np.std(x, ddof=1))
                ).reset_index()
            elif synthetic_control_bias_removal_period == 'last_pre_period_timeid':
                standardization = aggregated[aggregated['timeid'].isin(train_period + valid_period)].sort_values(['timeid'], ascending=True).groupby(['outcome', 'treatment'], dropna=False).agg(
                    avg = ('value', lambda x: np.mean(x[-1:])),
                    std = ('value', lambda x: np.std(x, ddof=1))
                ).reset_index()
            standardization['std'] = np.where((standardization['std'].isna()) | (standardization['std'] == 0.0), 1.0, standardization['std'])

            # Standardize the control times series created by the weighted average of the control unit in the dataframe in the defined period
            data = pd.merge(data, standardization, on=['outcome', 'treatment'], how='left')
            data['value'] = np.where(data['treatment'] == 0, (data['value'] - data['avg']) / data['std'], data['value'])
            data.drop(['avg', 'std'], axis=1, inplace=True)

            # Apply the treatment's time series mean and standard deviation to the standardized control units.
            # Positioning them in level and amplitude guaranteeing the only difference between them is the time series' shape.
            data = pd.merge(data, standardization[standardization['treatment'] == 1][['outcome', 'avg', 'std']], on=['outcome'], how='left')
            data['value'] = np.where(data['treatment'] == 0, (data['value'] * data['std']) + data['avg'], data['value'])
            data.drop(['avg', 'std'], axis=1, inplace=True)

            return data

        def evaluate_outcomes_metric(self, data, max_error):
            """
            Evaluates the performance of the synthetic control based on the provided weights.
            The article describes this as evaluating "Causal Fitness", which includes pre-intervention
            outcome balance, and post-intervention "null" balance.
            """
            aggregated3 = data.groupby(['outcome', 'timeid', 'treatment', 'timeid_relevance'], dropna=False).apply(
                lambda x : pd.Series({
                    'value' : np.ma.filled(np.ma.average(np.ma.masked_invalid(x['value']), weights=x['weight']), fill_value=np.nan)
                })
            ).reset_index()
            aggregated3['value'] = aggregated3['value'] * aggregated3['timeid_relevance']

            # Calculate error on the training set.
            aggregated3_diff = aggregated3[aggregated3['timeid'].isin(self.timeid_train)].sort_values(['outcome', 'timeid', 'treatment'], ascending=True).groupby(['outcome', 'timeid']).agg({
                'value' : lambda x: x.iloc[0] - x.iloc[1]
            }).reset_index(level=['outcome', 'timeid'])
            
            aggregated3_diff_normalized = pd.merge(aggregated3_diff, amplitude, on='outcome', how='left')
            aggregated3_diff_normalized['value'] = aggregated3_diff_normalized['value'] / aggregated3_diff_normalized['amplitude']

            aggregated3_mean_outcome = aggregated3_diff_normalized.groupby(['outcome'], dropna=False).agg({
                'value' : lambda x: ((x ** 2).mean()) ** 0.5
            }).reset_index()
            error_train = aggregated3_mean_outcome['value'].to_numpy()

            # Calculate error on the validation set. This is crucial for early stopping and preventing overfitting.
            aggregated3_diff = aggregated3[aggregated3['timeid'].isin(self.timeid_valid)].sort_values(['outcome', 'timeid', 'treatment'], ascending=True).groupby(['outcome', 'timeid']).agg({
                'value' : lambda x: x.iloc[0] - x.iloc[1]
            }).reset_index(level=['outcome', 'timeid'])

            aggregated3_diff_normalized_valid = pd.merge(aggregated3_diff, amplitude, on='outcome', how='left')
            aggregated3_diff_normalized_valid['value'] = aggregated3_diff_normalized_valid['value'] / aggregated3_diff_normalized_valid['amplitude']

            aggregated3_mean_outcome = aggregated3_diff_normalized_valid.groupby(['outcome'], dropna=False).agg({
                'value' : lambda x: ((x ** 2).mean()) ** 0.5
            }).reset_index()
            error_valid = aggregated3_mean_outcome['value'].to_numpy()

            # Calculate error for the entire pre-treatment period.
            aggregated3_diff = aggregated3[aggregated3['timeid'].isin(self.timeid_train + self.timeid_valid)].sort_values(['outcome', 'timeid', 'treatment'], ascending=True).groupby(['outcome', 'timeid']).agg({
                'value' : lambda x: x.iloc[0] - x.iloc[1]
            }).reset_index(level=['outcome', 'timeid'])

            aggregated3_diff_normalized_pre = pd.merge(aggregated3_diff, amplitude, on='outcome', how='left')
            aggregated3_diff_normalized_pre['value'] = aggregated3_diff_normalized_pre['value'] / aggregated3_diff_normalized_pre['amplitude']

            aggregated3_mean_outcome = aggregated3_diff_normalized_pre.groupby(['outcome'], dropna=False).agg({
                'value' : lambda x: ((x ** 2).mean()) ** 0.5
            }).reset_index()
            error_pre_intervention = aggregated3_mean_outcome['value'].to_numpy()

            # Calculate error for the entire post-treatment period.
            aggregated3_diff = aggregated3[aggregated3['timeid'].isin(self.timeid_post_intervention)].sort_values(['outcome', 'timeid', 'treatment'], ascending=True).groupby(['outcome', 'timeid']).agg({
                'value' : lambda x: x.iloc[0] - x.iloc[1]
            }).reset_index(level=['outcome', 'timeid'])

            aggregated3_diff_normalized_post = pd.merge(aggregated3_diff, amplitude, on='outcome', how='left')
            aggregated3_diff_normalized_post['value'] = aggregated3_diff_normalized_post['value'] / aggregated3_diff_normalized_post['amplitude']

            aggregated3_mean_outcome = aggregated3_diff_normalized_post.groupby(['outcome'], dropna=False).agg({
                'value' : lambda x: ((x ** 2).mean()) ** 0.5
            }).reset_index()
            error_post_intervention = aggregated3_mean_outcome['value'].to_numpy()

            if (self.config['save_solution_period_error'] == "pre_intervention" and check_per_outcome_limits(error_pre_intervention, max_error)):
                np.random.seed(self.state['attempt'] + seed) 
                stat_power = estimate_scm_power_mean_shift(
                    aggregated3_diff_normalized_pre,
                    aggregated3_diff_normalized_post,
                    effect_sizes_sd_per_outcome=effect_sizes_sd_per_outcome,
                    n_sim=1000
                )
            elif (self.config['save_solution_period_error'] == "validation_folder" and check_per_outcome_limits(error_valid, max_error)):
                np.random.seed(self.state['attempt'] + seed) 
                stat_power = estimate_scm_power_mean_shift(
                    aggregated3_diff_normalized_valid,
                    aggregated3_diff_normalized_post,
                    effect_sizes_sd_per_outcome=effect_sizes_sd_per_outcome,
                    n_sim=1000
                )
            else:
                stat_power = np.full(len(outcomes), float('inf'))

            return error_train, error_valid, error_pre_intervention, error_post_intervention, stat_power

        def eval_metric_ipw(self, preds):
            """
            This function implements the ATT/IPW-based ranking.
            It converts the XGBoost predictions (propensity scores) into weights and then selects N donors.
            """            
            if any(p == 1 for p in preds):
                error_valid = np.full(len(outcomes), float('inf'))
                error_pre_intervention = np.full(len(outcomes), float('inf'))
                stat_power = np.full(len(outcomes), float('inf'))
                return error_valid, error_pre_intervention, stat_power
            
            # Calculate Inverse Probability Weights (IPW). The formula ps / (1 - ps) is used to rank control units.
            ipw = pd.DataFrame({
                'id': self.dataset_train['id'],
                'treatment': self.dataset_train['treatment'],
                'weight': np.where(dtrain.get_label() == 1, 1.0, preds / (1 - preds))
            })

            # Normalize weights for the control group to sum to 1.
            sum_weight_tratamento = ipw.groupby('treatment').agg({
                'weight' : 'sum'
            }).reset_index()
            sum_weight_tratamento_0 = sum_weight_tratamento[sum_weight_tratamento['treatment'] == 0]['weight'].values[0]
            ipw['weight'] = np.where(ipw['treatment'] == 0, ipw['weight'] / sum_weight_tratamento_0, ipw['weight'])

            min_weight = ipw[ipw['treatment'] == 0]['weight'].min()
            min_weight_observations = ipw[(ipw['treatment'] == 0) & (ipw['weight'] > min_weight)]
            num_units_bigger_min_weight = min_weight_observations.shape[0]

            if num_units_bigger_min_weight > maximum_num_units_on_attipw_support or num_units_bigger_min_weight == 0:
                error_valid = np.full(len(outcomes), float('inf'))
                error_pre_intervention = np.full(len(outcomes), float('inf'))
                stat_power = np.full(len(outcomes), float('inf'))
                return error_valid, error_pre_intervention, stat_power
            
            ipw = pd.concat([ipw[ipw['treatment'] != 0], ipw[ipw['treatment'] == 0].sort_values(['weight', 'id'], ascending=[False, True])[0:num_units_bigger_min_weight]], ignore_index=True)

            full_data = self.full_data.copy()
            depara = pd.merge(ipw, self.from_to, on='id', how='inner')
            depara.drop('treatment', axis=1, inplace=True)
            datat = pd.merge(full_data, depara, on='unitid', how='left').reset_index()
            datat = datat[datat['weight'].notna()]

            # Select non multicollinearity control units
            donor_unitids = datat[datat['treatment'] == 0]['unitid'].unique()
            
            if len(donor_unitids) < minimum_donor_selection:
                error_valid = np.full(len(outcomes), float('inf'))
                error_pre_intervention = np.full(len(outcomes), float('inf'))
                stat_power = np.full(len(outcomes), float('inf'))
                return error_valid, error_pre_intervention, stat_power

            # Uses cache to speed up the process
            combination_tuple = tuple(sorted(donor_unitids))
            if combination_tuple in self.state['estimated_solutions']:
                (error_valid, error_pre_intervention, stat_power) = self.state['estimated_solutions'][combination_tuple]
            else:
                data2 = datat[datat['unitid'].isin([treatment_unitid] + list(donor_unitids))].copy()
                donor_df = data2[(data2['treatment'] == 0) & (data2['timeid'].isin(self.timeid_train))].copy()
                pivot_donor = donor_df.pivot(index='timeid', columns=['unitid', 'outcome'], values='value')
                selected_donors, gram_cond_train = build_low_cond_set_greedy_robust(
                    X=pivot_donor.values,                
                    gram_thresholds=self.gram_thresholds_per_outcome_train,
                    M=len(outcomes),
                    seed=self.state['attempt']+seed
                )
                
                if len(selected_donors) < minimum_donor_selection:
                    error_valid = np.full(len(outcomes), float('inf'))
                    error_pre_intervention = np.full(len(outcomes), float('inf'))
                    stat_power = np.full(len(outcomes), float('inf'))
                    return error_valid, error_pre_intervention, stat_power
                
                current_limit_max_weight = self.config['maximum_control_unit_weight_train']
                if current_limit_max_weight is None:
                    current_limit_max_weight = w_max_bound(
                        tpre=self.ts_pre_length,
                        k_controls=len(selected_donors)
                    )
                
                current_limit_min_weight = self.config['minimum_control_unit_weight_train']
                if current_limit_min_weight is None:
                    snr_effective = np.min(self.snr_train)  # worst-case outcome SNR
                    current_limit_min_weight = w_min_bound(
                        tpre=self.ts_train_length,
                        snr=snr_effective,
                        k_controls=len(selected_donors)
                    )
                
                # Uses cache to speed up the process
                combination_tuple2 = tuple(sorted(donor_unitids[selected_donors]))
                if combination_tuple2 in self.state['estimated_solutions']:
                    (error_valid, error_pre_intervention, stat_power) = self.state['estimated_solutions'][combination_tuple2]
                else:
                    data2 = data2[(data2['treatment'] != 0) | (data2['unitid'].isin(donor_unitids[selected_donors]))]

                    # Estimate weights via traditional SCM
                    df = data2.sort_values(by=['treatment', 'timeid', 'unitid', 'outcome']).reset_index(drop=True)
                    df = df[df['timeid'].isin(self.timeid_train)]
                    treated_outcome_df = df[df['treatment'] == 1].copy()
                    treated_outcome_pivot = treated_outcome_df.pivot_table(
                        index='timeid',
                        columns='outcome',
                        values='value'
                    )
                    treated_outcome_data = treated_outcome_pivot.to_numpy()
                    control_outcome_df = df[df['treatment'] == 0].copy()
                    control_outcome_pivot = control_outcome_df.pivot_table(
                        index=['timeid', 'unitid'],
                        columns='outcome',
                        values='value'
                    )
                    n_periods = control_outcome_df['timeid'].nunique()
                    n_control_units = control_outcome_df['unitid'].nunique()
                    n_outcomes = control_outcome_df['outcome'].nunique()
                    control_outcome_data = control_outcome_pivot.to_numpy().reshape(n_periods, n_control_units, n_outcomes)
                    optimal_weights = fast_synthetic_control_fitting(
                        treated_pre_intervention = treated_outcome_data,
                        control_pre_intervention = control_outcome_data
                    )
                    control_unit_ids = control_outcome_df['unitid'].unique()
                    
                    # Uses cache to speed up the process
                    combination_tuple3 = tuple(sorted(control_unit_ids))
                    if combination_tuple3 in self.state['estimated_solutions']:
                        (error_valid, error_pre_intervention, stat_power) = self.state['estimated_solutions'][combination_tuple3]
                    else:                    
                        weight_mapping = dict(zip(control_unit_ids, optimal_weights))
                        filtered_weights = {unit: weight for unit, weight in weight_mapping.items() if weight > current_limit_min_weight}
                        k_controls = len(filtered_weights.keys())
                        current_combination_tuple = tuple(sorted(filtered_weights.keys()))
                        data2['weight'] = data2['unitid'].map(filtered_weights)
                        data2['weight'] = np.where(data2['treatment'] == 1, 1, data2['weight'])
                        data2 = data2[data2['weight'].notna()]

                        # Re-normalize weights after selecting the top N donors to sum to 1.
                        recalculate_control_weight = data2[['id','treatment','weight']].drop_duplicates().copy()
                        sum_weight_tratamento = recalculate_control_weight.groupby('treatment').agg({
                            'weight' : 'sum'
                        }).reset_index()
                        sum_weight_tratamento_0 = sum_weight_tratamento[sum_weight_tratamento['treatment'] == 0]['weight'].values[0]
                        data2['weight'] = np.where(data2['treatment'] == 0, data2['weight'] / sum_weight_tratamento_0, data2['weight'])
                        current_maximum_control_unit_weight_train = data2[data2['treatment'] == 0]['weight'].max()

                        current_save_solution_error = self.config['save_solution_maximum_error']
                        if current_save_solution_error is None:
                            current_save_solution_error = [nrmse_upper_bound(snr=self.snr_pre[m],tpre=self.ts_pre_length,k_controls=k_controls) for m in range(len(outcomes))]

                        # Evaluate the performance of this candidate donor pool.
                        data = self.pre_intervention_scaling(data=data2, train_period=self.timeid_train, valid_period=self.timeid_valid)
                        error_train, error_valid, error_pre_intervention, error_post_intervention, stat_power = self.evaluate_outcomes_metric(data=data, max_error=current_save_solution_error)

                        gram_cond_per_outcome_pre = calculate_gram_cond_by_shape(df = data[(data['treatment'] == 0) & (data['timeid'].isin(self.timeid_train + self.timeid_valid))])

                        if (current_combination_tuple and len(current_combination_tuple) >= minimum_donor_selection and
                            current_maximum_control_unit_weight_train < current_limit_max_weight and
                            check_per_outcome_limits(gram_cond_per_outcome_pre, self.gram_thresholds_per_outcome_pre) and
                            ((save_solution_period_error == "pre_intervention" and check_per_outcome_limits(error_pre_intervention, current_save_solution_error)) or
                            (save_solution_period_error == "validation_folder" and check_per_outcome_limits(error_valid, current_save_solution_error))) and
                            current_combination_tuple not in self.state['estimated_solutions']):

                            # Save the donor units, weights and performance metrics of this viable solution.
                            data = data[data['treatment'] == 0]
                            data['valor_m_weight'] = data['value'] * data['weight']
                            data['cycle'] = cycle
                            data['trial'] = self.state['current_trial'].number
                            data['solution_id'] = self.state['solution_id']
                            data['num_units_on_attipw_support_train'] = num_units_bigger_min_weight
                            data['gram_cond_train'] = round(gram_cond_train, 1)
                            data['max_weight_train'] = round(current_maximum_control_unit_weight_train, 2)
                            data['error_train'] = round(np.max(error_train), 3)
                            data['error_valid'] = round(np.max(error_valid), 3)
                            data['error_pre_intervention'] = round(np.max(error_pre_intervention), 3)
                            data['error_post_intervention'] = round(np.max(error_post_intervention), 3)
                            data['stat_power'] = round(np.min(stat_power), 4)
                            data['gram_cond_pre'] = round(np.max(gram_cond_per_outcome_pre), 1)
                            columns = ['outcome', 'timeid', 'value', 'treatment', 'weight', 'unitid', 'valor_m_weight', 'id', 'cycle', 'trial', 'solution_id', 'num_units_on_attipw_support_train', 'gram_cond_train', 'max_weight_train', 'error_train', 'error_valid', 'error_pre_intervention', 'error_post_intervention', 'stat_power', 'gram_cond_pre']
                            data[columns].to_csv(scm_donor_selection_candidate_units_data_file_path, mode='a', header=False, index=False)

                            self.state['solution_id'] += 1

                        # Save in cache the current found solution and its simplified version
                        if current_combination_tuple and current_combination_tuple not in self.state['estimated_solutions']:
                            self.state['estimated_solutions'][current_combination_tuple] = (error_valid, error_pre_intervention, stat_power)

            self.state['attempt'] += 1

            return error_valid, error_pre_intervention, stat_power

    # MAIN TRAINING LOOP
    # This outer loop performs Cross-Temporal Validation.
    cycle = 0    
    
    # Init Mutable State Context
    state_context = {
        'attempt': 0,
        'solution_id': 0,
        'estimated_solutions': dict(),
        'current_trial': None
    }
    
    # Init Configuration Context
    config_context = {
        'maximum_control_unit_weight_train': maximum_control_unit_weight_train,
        'minimum_control_unit_weight_train': minimum_control_unit_weight_train,
        'save_solution_maximum_error': save_solution_maximum_error,
        'save_solution_period_error': save_solution_period_error
    }

    timeid_train_indexes = [timeid_pre_intervention.index(x) + 1 for x in temporal_cross_search_splits]
    for timeid_train_index in timeid_train_indexes:
        # sampler = optuna.samplers.RandomSampler(seed=cycle+seed)
        sampler = optuna.samplers.TPESampler(seed=cycle+seed)
        
        print(f'train: {timeid_pre_intervention[0:timeid_train_index]}')
        print(f'valid: {timeid_pre_intervention[(timeid_train_index):len(timeid_pre_intervention)]}')
        cycle = cycle + 1
        print(f'Cycle: {cycle}')

        # Create a dataset object for the current train/validation split.
        dataset = Dataset(
            data = all_units,
            timeid_train = timeid_pre_intervention[0:timeid_train_index],
            timeid_valid = timeid_pre_intervention[(timeid_train_index):len(timeid_pre_intervention)],
            timeid_post_intervention = timeid_post_intervention,
            state_context = state_context,
            config_context = config_context
        )
        
        # Calculate scale_pos_weight to handle class imbalance (one treated unit vs. many controls).
        neg_count = (dataset.dataset_train['treatment'] == 0).sum()
        pos_count = (dataset.dataset_train['treatment'] == 1).sum()
        calculated_scale_pos_weight = neg_count / pos_count

        # Prepare the data for XGBoost.
        dtrain = xgb.DMatrix(
            data=dataset.dataset_train.drop(['unitid', 'treatment'], axis=1),
            label=dataset.dataset_train[['treatment']]
        )

        def causal_fitness(preds, dtrain):
            """
            Custom metric for XGBoost evaluation. This is the heart of the innermost loop.
            It calculates the "Causal Fitness" score and updates the best scores found so far.
            dtrain is not used. The incapsulation is violated to make it performant.
            """
            global current_error_valid, current_error_pre_intervention, current_stat_power

            error_valid, error_pre_intervention, stat_power = dataset.eval_metric_ipw(preds)
            unique_error = np.max(error_valid)
            if np.max(current_error_valid) > unique_error:
                current_error_valid = error_valid
                current_error_pre_intervention = error_pre_intervention
                current_stat_power = stat_power

            return 'causal_fitness', unique_error

        class training_callback(xgb.callback.TrainingCallback):
            """A callback to reset the error at the beginning of each training run."""
            def before_training(self, model):
                global current_error_valid, current_error_pre_intervention, current_stat_power
                current_error_valid = np.full(len(outcomes), float('inf'))
                current_error_pre_intervention = np.full(len(outcomes), float('inf'))
                current_stat_power = np.full(len(outcomes), float('inf'))

                return model

        # The middle loop: Optuna's hyperparameter search.
        def objective(trial):
            """
            The objective function for Optuna. It defines the search space for hyperparameters
            and runs the XGBoost training with the custom metric and early stopping.
            """
            state_context['current_trial'] = trial

            # Define the hyperparameter search space for Optuna.
            params = {
                'seed': state_context['attempt'] + seed,
                'tree_method': 'hist',
                'objective': 'binary:logistic',
                'disable_default_eval_metric': True,

                # --- Core Tree Parameters ---
                'max_depth': trial.suggest_int('max_depth', 1, 5),

                'learning_rate': trial.suggest_float('learning_rate', 0.01, 2.0, log=True),   # Learning rate
                'gamma': trial.suggest_float('gamma', 1e-8, 5.0, log=True),                   # Min loss reduction
                'lambda': trial.suggest_float('lambda', 1e-8, 10.0, log=True),                # L2 reg
                'alpha': trial.suggest_float('alpha', 1e-8, 10.0, log=True),                  # L1 reg
                'grow_policy': 'depthwise',

                # --- Subsampling Parameters ---
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
                'colsample_bylevel': trial.suggest_float('colsample_bytree', 0.4, 1.0),
                'colsample_bynode': trial.suggest_float('colsample_bytree', 0.4, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),

                # --- Mitiga o Desbalanceamento de Classes ---
                'scale_pos_weight': trial.suggest_float('scale_pos_weight', calculated_scale_pos_weight * 0.5, calculated_scale_pos_weight * 2.0)
            }

            # --- Run Cross-Validation with Pruning ---
            num_boost_round_max = 1000 # Max rounds if no early stopping/pruning

            try:
                # The innermost loop: Iterative evaluation and early stopping.
                # XGBoost is trained one tree at a time, and the custom metric is evaluated at each step.
                results = xgb.train(
                    params=params,
                    dtrain=dtrain,
                    maximize=False,
                    num_boost_round=num_boost_round_max,
                    evals=[(dtrain, 'placeholder_dataset')],
                    custom_metric=causal_fitness,
                    callbacks=[
                        training_callback(),
                        # Early stopping prevents overfitting by stopping when the validation score no longer improves.
                        xgb.callback.EarlyStopping(rounds=1, metric_name='causal_fitness', data_name='placeholder_dataset')               
                    ],
                    verbose_eval=False
                )

                # Store best iteration in trial attributes
                trial.set_user_attr("best_iteration", results.best_iteration)

            except optuna.TrialPruned:
                # If Optuna pruned the trial via the callback
                raise optuna.TrialPruned()
            except Exception as e:
                # Handle other potential errors during xgb.train
                print(f"An error occurred during xgb.train for trial {trial.number}: {e}")
                return float('inf')

            if (optuna_optimization_target == "pre_intervention"):
                return np.max(current_error_pre_intervention)
            elif(optuna_optimization_target == "validation_folder"):
                return np.max(current_error_valid)

            return float('inf')

        # --- 4. Run the Optuna Study ---
        directions=['minimize']

        study = optuna.create_study(
            directions=directions,
            sampler=sampler
        )

        # Run the Optuna study to find the best set of hyperparameters.
        study.optimize(objective, n_trials=optuna_number_trials, timeout=optuna_timeout_cycle, show_progress_bar=True)
