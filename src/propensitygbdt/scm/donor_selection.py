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
import random
from scipy.optimize import minimize
from scipy.linalg import qr
from scipy.linalg import svd
from scipy.signal import medfilt
import sys
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
def calculate_gram_cond_by_shape(df: pd.DataFrame) -> float:
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
            results.append({'outcome': outcome, 'gram_cond': cond_number})
    return pd.DataFrame(results)['gram_cond'].max()

"""
Estimates the latent rank r of the donor matrix via the Eigenvalue Ratio (ER) method,
robust to noise. Standardizes each series to mean 0, variance 1 before estimation to
focus on correlations. The ER method detects the largest gap in sorted singular values,
identifying the point where signal drops to noise level.

Parameters:
-----------
X : array-like, shape (T, K*M)
    Donor time series matrix (pre-intervention).
n_outcomes : int
    Number of outcomes (M) per donor.
kmax : int, optional
    Maximum rank to consider (default: min(T, total_cols) // 2).

Returns:
--------
r_hat : int
    Estimated rank r.
"""
def estimate_rank_robust(X, n_outcomes, kmax=None):
    T, total_cols = X.shape
    J = total_cols // n_outcomes
    if total_cols != J * n_outcomes:
        raise ValueError("Columns must match J * n_outcomes.")
    
    X_std = np.zeros_like(X)
    for k in range(n_outcomes):
        start = k * J
        end = (k + 1) * J
        sub_X = X[:, start:end]
        global_mean = np.mean(sub_X)
        global_std = np.std(sub_X) if np.std(sub_X) > 0 else 1.0
        sub_X = (sub_X - global_mean) / global_std  # Group scale
        # Optional column standardize
        for col in range(J):
            series = sub_X[:, col]
            mean = np.mean(series)
            std = np.std(series) if np.std(series) > 0 else 1.0
            sub_X[:, col] = (series - mean) / std
        X_std[:, start:end] = sub_X
    
    # Proceed with SVD and ER as in original
    _, S, _ = np.linalg.svd(X_std, full_matrices=False)
    m = len(S)
    if kmax is None:
        kmax = m // 2
    kmax = min(kmax, m - 1)
    if kmax < 1:
        return 0
    ratios = (S[:kmax]**2) / (S[1:kmax+1]**2)
    return np.argmax(ratios) + 1

"""
Estimates a robust coefficient of variation (CV) for the treated unit's pre-treatment time series,
handling multiple outcomes. This measure quantifies the relative variability (noise level) in the
data using a robust approach to inform adaptive buffers in donor selection.

The function detrends each outcome series using a median filter to remove low-frequency trends,
computes residuals, and then calculates a robust CV as the scaled Median Absolute Deviation (MAD)
normalized by the median absolute residual. The scaling factor 1.4826 ensures consistency with the
standard deviation under Gaussian assumptions. Results are averaged across outcomes for a single
scalar estimate.

Fixed: Calculates Noise / Signal.
Signal = Median of the Trend.
Noise = MAD of the Residuals.

Parameters:
treatment_ts : array-like, shape (T_pre, n_outcomes). Pre-treatment time series for the treated
    unit (one column per outcome).
n_outcomes : int. Number of outcomes (must match the second dimension of treatment_ts).
window_fraction : float, optional. Fraction of T_pre to determine the adaptive median filter window
    size (default: 0.1). The window is clamped to [3, 101] for stability.

Returns:
robust_cv : float. The average robust CV across all outcomes, used as a noise proxy (e.g.,
    in max_donors calculation to add buffers for high-noise data).
"""
def estimate_robust_cv(treatment_ts, n_outcomes, window_fraction=0.1):
    """

    """
    treatment_ts = np.atleast_2d(treatment_ts)
    T_pre, M = treatment_ts.shape
    
    robust_cvs = []
    # Ensure window is odd and at least 3
    window = int(window_fraction * T_pre)
    if window % 2 == 0: window += 1
    window = max(3, window)
    
    for m in range(n_outcomes):
        ts = treatment_ts[:, m]
        
        # 1. Extract Trend (Signal)
        trend = medfilt(ts, kernel_size=window)
        
        # 2. Extract Noise
        residuals = ts - trend
        
        # 3. Calculate Robust Sigma (Noise Level)
        # 1.4826 converts MAD to Sigma for Gaussian distribution
        median_resid = np.median(residuals)
        mad = np.median(np.abs(residuals - median_resid))
        sigma_robust = 1.4826 * mad
        
        # 4. Calculate Signal Magnitude
        # Use median of absolute TREND, not residuals
        signal_mag = np.median(np.abs(trend))
        
        if signal_mag > 1e-9:
            robust_cvs.append(sigma_robust / signal_mag)
        else:
            robust_cvs.append(np.inf)
    
    return np.mean(robust_cvs)

"""
Builds a low-condition-number donor set via greedy forward selection (scale-invariant robust version).
Iteratively adds the donor minimizing the prospective max Gram cond (across outcomes)
on normalized differenced series, using QR orthogonalization for stable cond estimation.
Differencing removes levels; per-donor standardization of differences removes scales, focusing purely on shape.
Stops if threshold exceeded or no improvement.

Parameters:
-----------
X : array-like, shape (T, K*M)
    Donor time series matrix (pre-intervention).
n_outcomes : int
    Number of outcomes (M) per donor.
gram_threshold : float, default=100.0
    Stop if max Gram cond >= this.
max_donors : int, optional
    Cap on selected donors.
seed : int, optional
    Seed for reproducible shuffling of initial donor order. If None, use natural order.

Returns:
--------
selected_indices : list of int
    Sorted donor indices.
final_max_cond : float
    Max Gram cond of the final set.
"""
def build_low_cond_set_greedy_robust(treatment_ts, X, n_outcomes, gram_threshold=100.0, max_donors=None, seed=None):
    if seed is not None:
        np.random.seed(seed)
    if max_donors is None:
        robust_cv = estimate_robust_cv(treatment_ts, n_outcomes)
        estimated_rank = estimate_rank_robust(X, n_outcomes)
        base_need = estimated_rank + 1
        noise_buffer = (robust_cv ** 2) * estimated_rank
        max_capacity = np.clip(np.sqrt(len(treatment_ts)) * max(1.0, 1.5 / np.log(base_need)), a_min=2 * estimated_rank + 1, a_max=50)  # Damps if high rank
        max_donors = np.clip(base_need + noise_buffer, a_min=base_need, a_max=max_capacity)
    
    T, total_cols = X.shape
    if total_cols % n_outcomes != 0:
        raise ValueError(f"Total columns in X ({total_cols}) must be divisible by n_outcomes ({n_outcomes}).")
    K = total_cols // n_outcomes
    if T < 2:
        raise ValueError(f"Insufficient time periods (T={T} < 2) for differencing.")
   
    selected_indices = set()
    available_indices = set(range(K))
    # Seeded shuffle for reproducibility (if seed provided)
    if seed is not None:
        available_list = list(available_indices)
        np.random.shuffle(available_list)
        available_indices = set(available_list)
    final_max_cond = 0.0
   
    while available_indices:
        best_idx = None
        best_cond = float('inf')
       
        for idx in list(available_indices):
            temp_selected = selected_indices | {idx}
            max_gram_cond = 0.0
            for m in range(n_outcomes):
                # Extract columns for this outcome from temp donors
                donor_cols_m = [X[:, j * n_outcomes + m] for j in sorted(temp_selected)]
                if not donor_cols_m:
                    continue
                outcome_X = np.column_stack(donor_cols_m)  # (T, |temp|)
                diff_outcome = np.diff(outcome_X, axis=0)  # (T-1, |temp|): removes levels, focuses on changes
                if diff_outcome.shape[1] > 0 and diff_outcome.shape[0] > 0:
                    # Standardize differences per donor: remove scales, focus purely on shape
                    # (mean of diffs is already ~0; divide by std for unit variance)
                    stds = np.std(diff_outcome, axis=0)
                    # Handle zero std (constant series after diff)
                    stds[stds == 0] = 1.0  # Treat as unit variance to avoid div-by-zero
                    diff_normalized = diff_outcome / stds[None, :]  # (T-1, |temp|)
                    
                    # Robust cond via QR orthogonalization
                    _, R, _ = qr(diff_normalized, mode='economic', pivoting=True)
                    if R.size > 0:
                        cond_r = np.linalg.cond(R)
                        cond_m = cond_r ** 2
                    else:
                        cond_m = 1.0
                    max_gram_cond = max(max_gram_cond, cond_m)
           
            if max_gram_cond < best_cond:
                best_cond = max_gram_cond
                best_idx = idx
       
        if best_idx is None or best_cond >= gram_threshold:
            break
       
        selected_indices.add(best_idx)
        available_indices.remove(best_idx)
        final_max_cond = best_cond  # Update to current best
       
        if max_donors and len(selected_indices) >= max_donors:
            break
   
    return sorted(list(selected_indices)), final_max_cond

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
Calculates a single "Impact Score" to identify models with no effect.
The score combines the p-value with a penalty for how far the
observed ratio deviates from the ideal of 1.0.

Calculates the p-value across multiple outcomes by comparing the
post-intervention to pre-intervention RMSPE ratio using a block bootstrap.

The p-value represents the probability that the observed increase in RMSPE
could have occurred by chance if the underlying process had not changed.

Args:
    data_pre (pd.DataFrame): DataFrame with pre-intervention data.
                                Must contain 'outcome', 'timeid', and 'value' columns.
    data_post (pd.DataFrame): DataFrame with post-intervention data.
                                Must contain 'outcome', 'timeid', and 'value' columns.
    n_bootstraps (int): The number of bootstrap samples to generate.
    alpha (float): A tuning parameter controlling the severity of the
                                penalty for ratios not equal to 1. Higher alpha
                                means a stricter penalty.

Returns:
    float: A score between 0 and 1. Higher is better (stronger
                                evidence of no effect).
"""
def block_bootstrap_rmspe_ratio_vectorized(data_pre, data_post, n_bootstraps=1000, alpha=2.0):
    all_observed_ratio = []
    all_p_values = []
    unique_outcomes = data_pre['outcome'].unique()

    for outcome in unique_outcomes:
        pre_values = data_pre[data_pre['outcome'] == outcome]['value'].to_numpy()
        post_values = data_post[data_post['outcome'] == outcome]['value'].to_numpy()

        if len(pre_values) == 0: continue
        
        # 1. Calculate the single observed RMSPE ratio (same as before)
        observed_pre_rmspe = np.sqrt(np.mean(pre_values**2))
        observed_post_rmspe = np.sqrt(np.mean(post_values**2))
        observed_ratio = observed_post_rmspe / observed_pre_rmspe if observed_pre_rmspe != 0 else np.inf

        # 2. Setup for vectorization
        combined_values = np.concatenate([pre_values, post_values])
        n_combined = len(combined_values)
        n_pre = len(pre_values)

        block_length = math.ceil(n_combined**(1/3))
        if block_length < 1: block_length = 1
        elif len(post_values) < 3: block_length = 1
        
        num_blocks_to_draw = math.ceil(n_combined / block_length)

        # 3. Generate ALL random starting indices for ALL bootstraps at once
        # This creates a 2D array of shape: (n_bootstraps, num_blocks_to_draw)
        start_indices = np.random.randint(
            0, n_combined - block_length + 1, 
            size=(n_bootstraps, num_blocks_to_draw)
        )

        # 4. Create ALL bootstrapped series at once using advanced indexing
        # This is the most complex step, but it's extremely fast.
        # It builds a giant index matrix to pull values from the original series.
        block_offsets = np.arange(block_length)
        # Broadcasting creates a 3D index matrix: (n_bootstraps, num_blocks, block_length)
        indices_3d = start_indices[:, :, np.newaxis] + block_offsets
        # Flatten the blocks and grab the values
        bootstrapped_series_full = combined_values[indices_3d].reshape(n_bootstraps, -1)
        # Trim to the correct length
        bootstrapped_series = bootstrapped_series_full[:, :n_combined]

        # 5. Calculate RMSPE ratios for ALL bootstraps in a vectorized way
        boot_pre_matrix = bootstrapped_series[:, :n_pre]
        boot_post_matrix = bootstrapped_series[:, n_pre:]

        boot_pre_rmspes = np.sqrt(np.mean(boot_pre_matrix**2, axis=1))
        boot_post_rmspes = np.sqrt(np.mean(boot_post_matrix**2, axis=1))
        
        # Handle division by zero for all bootstraps at once
        bootstrap_ratios = np.full(n_bootstraps, np.inf)
        valid_mask = boot_pre_rmspes != 0
        bootstrap_ratios[valid_mask] = boot_post_rmspes[valid_mask] / boot_pre_rmspes[valid_mask]
        
        # 6. Calculate the p-value
        p_value = np.mean(bootstrap_ratios >= observed_ratio)
        all_observed_ratio.append(observed_ratio)
        all_p_values.append(p_value)

    # The core of the penalty is the absolute distance from the ideal ratio of 1.0
    ratio_penalty = np.exp(-alpha * abs(np.min(all_observed_ratio) - 1))
    
    # The final score is one minus the p-value modulated by the ratio penalty
    score = 1 - np.min(all_p_values) * ratio_penalty

    return score

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
maximum_gram_cond_train : float, default=100.0
    Maximum allowable condition number for the Gram matrix of selected donors to ensure 
    linear independence (mitigates multicollinearity).
minimum_donor_selection : int, default=3
    Minimum number of donor units required to form a valid synthetic control.
maximum_control_unit_weight_train : float, default=0.5
    Constraint to ensure no single donor dominates the synthetic control (max weight < 0.5).
synthetic_control_bias_removal_period : Literal, default='pre_intervention'
    Strategy for centering/scaling control units relative to the treated unit (e.g., based on the full pre-period).
function_aggregate_outcomes_error : Literal, default='mean'
    Metric to aggregate errors across multiple outcomes ('mean' or 'max').
save_solution_period_error : Literal, default='pre_intervention'
    Determines which period's error is checked against `save_solution_maximum_error` to decide if a solution is saved.
save_solution_maximum_error : float, default=None
    The maximum allowable RMSPE (normalized) for a candidate solution to be saved to disk.
maximum_gram_cond_pre : float, Defaults to 100.0. The maximum allowable value for the Gram matrix condition number.
    This is used as a threshold to detect multicollinearity among control units; solutions exceeding it are flagged.
alpha_exponential_decay : float, default=0.00
    Decay factor for time-relevance weights; higher values give more weight to recent time periods.
optuna_optimization_target : Literal, default='pre_intervention'
    The error metric Optuna attempts to minimize ('pre_intervention' or 'validation_folder').
optuna_number_trials : int, default=1000
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
type_function_aggregate_outcomes_error = Literal['mean', 'max']
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
    maximum_gram_cond_train: float = 100.0,
    minimum_donor_selection: int = 3,
    maximum_control_unit_weight_train: float = 0.5,
    synthetic_control_bias_removal_period: type_synthetic_control_bias_removal_period = 'pre_intervention',
    function_aggregate_outcomes_error: type_function_aggregate_outcomes_error = 'mean',
    save_solution_period_error: type_save_solution_period_error = 'pre_intervention',
    save_solution_maximum_error: float = None,
    maximum_gram_cond_pre: float = 100.0,
    alpha_exponential_decay: float = 0.00,
    optuna_optimization_target: type_optuna_optimization_target = 'pre_intervention',
    optuna_number_trials: int = 1000,
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

    if not isinstance(maximum_gram_cond_train, float) or maximum_gram_cond_train <= 0:
        raise ValueError(f"maximum_gram_cond_train must be a positive real, got {maximum_gram_cond_train}")

    if not isinstance(minimum_donor_selection, int) or minimum_donor_selection <= 0:
        raise ValueError(f"minimum_donor_selection must be a positive integer, got {minimum_donor_selection}")

    if not isinstance(maximum_control_unit_weight_train, float) or maximum_control_unit_weight_train <= 0:
        raise ValueError(f"maximum_control_unit_weight_train must be a positive real, got {maximum_control_unit_weight_train}")
    
    valid_options = get_args(type_synthetic_control_bias_removal_period)
    if synthetic_control_bias_removal_period not in valid_options:
        raise ValueError(f"Invalid mode: '{synthetic_control_bias_removal_period}'. Expected one of: {valid_options}")
        
    valid_options = get_args(type_function_aggregate_outcomes_error)
    if function_aggregate_outcomes_error not in valid_options:
        raise ValueError(f"Invalid mode: '{function_aggregate_outcomes_error}'. Expected one of: {valid_options}")

    valid_options = get_args(type_save_solution_period_error)
    if save_solution_period_error not in valid_options:
        raise ValueError(f"Invalid mode: '{save_solution_period_error}'. Expected one of: {valid_options}")
    
    if save_solution_maximum_error is not None and not isinstance(save_solution_maximum_error, float) and save_solution_maximum_error <= 0:
        raise ValueError(f"save_solution_maximum_error must be a positive real, got {save_solution_maximum_error}")
        
    if not isinstance(maximum_gram_cond_pre, float) or maximum_gram_cond_pre <= 0:
        raise ValueError(f"maximum_gram_cond_pre must be a positive real, got {maximum_gram_cond_pre}")
        
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

    # ==============================================================================
    # START CHANGE: Noise-Adaptive NRMSE Threshold (Proposal 1)
    # ==============================================================================
    # 1. Estimate Intrinsic Noise (Sigma_Hat) using MAD of first differences
    #    Formula: median(|diff - median(diff)|) * 1.4826 (Robust estimator of sigma)
    def calculate_robust_noise(x):
        vals = x.values
        if len(vals) < 2: return 0.0
        diff = np.diff(vals)
        mad = np.median(np.abs(diff - np.median(diff)))
        return mad * 1.4826

    if save_solution_maximum_error is None:
        noise_estimates = all_units[
            (all_units['treatment'] == 1) & 
            (all_units['pre_intervention'] == 1)
        ].groupby('outcome')['value'].apply(calculate_robust_noise).reset_index(name='sigma_hat')

        # 2. Merge with amplitude to calculate the Noise Ratio (eta)
        adaptive_df = pd.merge(amplitude, noise_estimates, on='outcome')
        
        # 3. Calculate Adaptive Threshold (tau) per outcome
        #    Formula: tau = sqrt(0.10^2 + (2 * eta)^2)
        #    This allows a base error of 0.10 plus a tolerance for 2x the intrinsic noise.
        adaptive_df['eta'] = adaptive_df['sigma_hat'] / adaptive_df['amplitude']
        adaptive_df['tau'] = np.sqrt(0.10**2 + (2 * adaptive_df['eta'])**2)

        # 4. Aggregate to a single scalar threshold based on the user's config
        if function_aggregate_outcomes_error == 'mean':
            adaptive_threshold = adaptive_df['tau'].mean()
        elif function_aggregate_outcomes_error == 'max':
            adaptive_threshold = adaptive_df['tau'].max()
        else:
            adaptive_threshold = adaptive_df['tau'].mean() # Default fallback

        print("\n--- Adaptive NRMSE Threshold Calculation ---")
        print(adaptive_df[['outcome', 'amplitude', 'sigma_hat', 'eta', 'tau']])
        print(f"User Configured Threshold: {save_solution_maximum_error}")
        print(f"Calculated Adaptive Threshold: {adaptive_threshold}")
        print("--------------------------------------------\n")

        # 5. Overwrite the threshold variable used in the optimization loop
        save_solution_maximum_error = adaptive_threshold

    all_units.drop('pre_intervention', axis=1, inplace=True)

    def pre_intervention_scaling(data, train_period, valid_period):
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
    scm_donor_selection_candidate_units_data['impact_score'] = None
    scm_donor_selection_candidate_units_data['gram_cond_pre'] = None
    scm_donor_selection_candidate_units_data.to_csv(scm_donor_selection_candidate_units_data_file_path, mode='w', header=True, index=False)

    class Dataset:
        """
        This class handles the data for the training and evaluation of the model.
        It's responsible for creating the training set, evaluating the IPW-based metric,
        and calculating the outcome metrics for the synthetic control.
        """
        def __init__(self, data, timeid_train, timeid_valid, timeid_post_intervention):
            self.timeid_train = timeid_train
            self.timeid_valid = timeid_valid
            self.timeid_post_intervention = timeid_post_intervention

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

        def evaluate_outcomes_metric(self, data):
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
            
            if function_aggregate_outcomes_error == "mean":
                error_train = aggregated3_mean_outcome['value'].mean()
            elif function_aggregate_outcomes_error == "max":
                error_train = aggregated3_mean_outcome['value'].max()

            # Calculate error on the validation set. This is crucial for early stopping and preventing overfitting.
            aggregated3_diff = aggregated3[aggregated3['timeid'].isin(self.timeid_valid)].sort_values(['outcome', 'timeid', 'treatment'], ascending=True).groupby(['outcome', 'timeid']).agg({
                'value' : lambda x: x.iloc[0] - x.iloc[1]
            }).reset_index(level=['outcome', 'timeid'])

            aggregated3_diff_normalized_valid = pd.merge(aggregated3_diff, amplitude, on='outcome', how='left')
            aggregated3_diff_normalized_valid['value'] = aggregated3_diff_normalized_valid['value'] / aggregated3_diff_normalized_valid['amplitude']

            aggregated3_mean_outcome = aggregated3_diff_normalized_valid.groupby(['outcome'], dropna=False).agg({
                'value' : lambda x: ((x ** 2).mean()) ** 0.5
            }).reset_index()
            
            if function_aggregate_outcomes_error == "mean":
                error_valid = aggregated3_mean_outcome['value'].mean()
            elif function_aggregate_outcomes_error == "max":
                error_valid = aggregated3_mean_outcome['value'].max()

            # Calculate error for the entire pre-treatment period.
            aggregated3_diff = aggregated3[aggregated3['timeid'].isin(self.timeid_train + self.timeid_valid)].sort_values(['outcome', 'timeid', 'treatment'], ascending=True).groupby(['outcome', 'timeid']).agg({
                'value' : lambda x: x.iloc[0] - x.iloc[1]
            }).reset_index(level=['outcome', 'timeid'])

            aggregated3_diff_normalized_pre = pd.merge(aggregated3_diff, amplitude, on='outcome', how='left')
            aggregated3_diff_normalized_pre['value'] = aggregated3_diff_normalized_pre['value'] / aggregated3_diff_normalized_pre['amplitude']

            aggregated3_mean_outcome = aggregated3_diff_normalized_pre.groupby(['outcome'], dropna=False).agg({
                'value' : lambda x: ((x ** 2).mean()) ** 0.5
            }).reset_index()
            
            if function_aggregate_outcomes_error == "mean":
                error_pre_intervention = aggregated3_mean_outcome['value'].mean()
            elif function_aggregate_outcomes_error == "max":
                error_pre_intervention = aggregated3_mean_outcome['value'].max()

            # Calculate error for the entire post-treatment period.
            aggregated3_diff = aggregated3[aggregated3['timeid'].isin(self.timeid_post_intervention)].sort_values(['outcome', 'timeid', 'treatment'], ascending=True).groupby(['outcome', 'timeid']).agg({
                'value' : lambda x: x.iloc[0] - x.iloc[1]
            }).reset_index(level=['outcome', 'timeid'])

            aggregated3_diff_normalized_post = pd.merge(aggregated3_diff, amplitude, on='outcome', how='left')
            aggregated3_diff_normalized_post['value'] = aggregated3_diff_normalized_post['value'] / aggregated3_diff_normalized_post['amplitude']

            aggregated3_mean_outcome = aggregated3_diff_normalized_post.groupby(['outcome'], dropna=False).agg({
                'value' : lambda x: ((x ** 2).mean()) ** 0.5
            }).reset_index()
            
            if function_aggregate_outcomes_error == "mean":
                error_post_intervention = aggregated3_mean_outcome['value'].mean()
            elif function_aggregate_outcomes_error == "max":
                error_post_intervention = aggregated3_mean_outcome['value'].max()

            if (save_solution_period_error == "pre_intervention" and error_pre_intervention < save_solution_maximum_error):
                np.random.seed(attempt + seed) 
                impact_score = block_bootstrap_rmspe_ratio_vectorized(
                    aggregated3_diff_normalized_pre,
                    aggregated3_diff_normalized_post,
                    n_bootstraps=1000
                )
            elif (save_solution_period_error == "validation_folder" and error_valid < save_solution_maximum_error):
                np.random.seed(attempt + seed) 
                impact_score = block_bootstrap_rmspe_ratio_vectorized(
                    aggregated3_diff_normalized_valid,
                    aggregated3_diff_normalized_post,
                    n_bootstraps=1000
                )
            else:
                impact_score = float('inf')

            return error_train, error_valid, error_pre_intervention, error_post_intervention, impact_score

        def eval_metric_ipw(self, preds):
            """
            This function implements the ATT/IPW-based ranking.
            It converts the XGBoost predictions (propensity scores) into weights and then selects N donors.
            """
            global attempt, solution_id, estimated_solutions
            
            if any(p == 1 for p in preds):
                error_valid = float('inf')
                error_pre_intervention = float('inf')
                impact_score = float('inf')
                return error_valid, error_pre_intervention, impact_score
            
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
                error_valid = float('inf')
                error_pre_intervention = float('inf')
                impact_score = float('inf')
                return error_valid, error_pre_intervention, impact_score
            
            ipw = pd.concat([ipw[ipw['treatment'] != 0], ipw[ipw['treatment'] == 0].sort_values(['weight', 'id'], ascending=[False, True])[0:num_units_bigger_min_weight]], ignore_index=True)

            full_data = self.full_data.copy()
            depara = pd.merge(ipw, self.from_to, on='id', how='inner')
            depara.drop('treatment', axis=1, inplace=True)
            datat = pd.merge(full_data, depara, on='unitid', how='left').reset_index()
            datat = datat[datat['weight'].notna()]

            # Select non multicollinearity control units
            donor_unitids = datat[datat['treatment'] == 0]['unitid'].unique()
            
            if len(donor_unitids) < minimum_donor_selection:
                error_valid = float('inf')
                error_pre_intervention = float('inf')
                impact_score = float('inf')
                return error_valid, error_pre_intervention, impact_score

            # Uses cache to speed up the process
            combination_tuple = tuple(sorted(donor_unitids))
            if combination_tuple in estimated_solutions:
                (error_valid, error_pre_intervention, impact_score) = estimated_solutions[combination_tuple]
            else:
                data2 = datat[datat['unitid'].isin([treatment_unitid] + list(donor_unitids))].copy()
                treatment_df = data2[(data2['treatment'] == 1) & (data2['timeid'].isin(self.timeid_train))]
                pivot_treatment = treatment_df.pivot(index='timeid', columns=['outcome'], values='value')
                donor_df = data2[(data2['treatment'] == 0) & (data2['timeid'].isin(self.timeid_train))].copy()
                pivot_donor = donor_df.pivot(index='timeid', columns=['unitid', 'outcome'], values='value')
                selected_donors, gram_cond_train = build_low_cond_set_greedy_robust(
                    treatment_ts=pivot_treatment.values,
                    X=pivot_donor.values,
                    n_outcomes=len(outcomes),
                    gram_threshold=maximum_gram_cond_train,
                    seed=attempt+seed
                )
                
                if len(selected_donors) < minimum_donor_selection:
                    error_valid = float('inf')
                    error_pre_intervention = float('inf')
                    impact_score = float('inf')
                    return error_valid, error_pre_intervention, impact_score
                
                # Uses cache to speed up the process
                combination_tuple2 = tuple(sorted(donor_unitids[selected_donors]))
                if combination_tuple2 in estimated_solutions:
                    (error_valid, error_pre_intervention, impact_score) = estimated_solutions[combination_tuple2]
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
                    if combination_tuple3 in estimated_solutions:
                        (error_valid, error_pre_intervention, impact_score) = estimated_solutions[combination_tuple3]
                    else:                    
                        weight_mapping = dict(zip(control_unit_ids, optimal_weights))
                        filtered_weights = {unit: weight for unit, weight in weight_mapping.items() if weight > 0.1}
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

                        # Evaluate the performance of this candidate donor pool.
                        data = pre_intervention_scaling(data=data2, train_period=self.timeid_train, valid_period=self.timeid_valid)
                        error_train, error_valid, error_pre_intervention, error_post_intervention, impact_score = self.evaluate_outcomes_metric(data=data)
                        
                        gram_cond_max = calculate_gram_cond_by_shape(df = data[(data['treatment'] == 0) & (data['timeid'].isin(self.timeid_train + self.timeid_valid))])
                        if gram_cond_max == float('inf'):
                            gram_cond_max = 9999999.0
                            
                        if (current_combination_tuple and len(current_combination_tuple) >= minimum_donor_selection and
                            current_maximum_control_unit_weight_train < maximum_control_unit_weight_train and
                            gram_cond_max < maximum_gram_cond_pre and
                            ((save_solution_period_error == "pre_intervention" and error_pre_intervention < save_solution_maximum_error) or
                            (save_solution_period_error == "validation_folder" and error_valid < save_solution_maximum_error)) and
                            current_combination_tuple not in estimated_solutions):

                            # Save the donor units, weights and performance metrics of this viable solution.
                            data = data[data['treatment'] == 0]
                            data['valor_m_weight'] = data['value'] * data['weight']
                            data['cycle'] = cycle
                            data['trial'] = pruning_trial.number
                            data['solution_id'] = solution_id
                            data['num_units_on_attipw_support_train'] = num_units_bigger_min_weight
                            data['gram_cond_train'] = round(gram_cond_train, 1)
                            data['max_weight_train'] = round(current_maximum_control_unit_weight_train, 2)
                            data['error_train'] = round(error_train, 3)
                            data['error_valid'] = round(error_valid, 3)
                            data['error_pre_intervention'] = round(error_pre_intervention, 3)
                            data['error_post_intervention'] = round(error_post_intervention, 3)
                            data['impact_score'] = round(impact_score, 4)
                            data['gram_cond_pre'] = round(gram_cond_max, 1)
                            columns = ['outcome', 'timeid', 'value', 'treatment', 'weight', 'unitid', 'valor_m_weight', 'id', 'cycle', 'trial', 'solution_id', 'num_units_on_attipw_support_train', 'gram_cond_train', 'max_weight_train', 'error_train', 'error_valid', 'error_pre_intervention', 'error_post_intervention', 'impact_score', 'gram_cond_pre']
                            data[columns].to_csv(scm_donor_selection_candidate_units_data_file_path, mode='a', header=False, index=False)

                            solution_id = solution_id + 1

                        # Save in cache the current found solution and its simplified version
                        if current_combination_tuple and current_combination_tuple not in estimated_solutions:
                            estimated_solutions[current_combination_tuple] = (error_valid, error_pre_intervention, impact_score)

            attempt = attempt + 1

            return error_valid, error_pre_intervention, impact_score

    # MAIN TRAINING LOOP
    # This outer loop performs Cross-Temporal Validation.
    cycle = 0
    global attempt, solution_id, estimated_solutions
    attempt = 0
    solution_id = 0
    # EXECUTION CACHE
    estimated_solutions = dict()

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
            timeid_post_intervention = timeid_post_intervention
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
            global current_error_valid, current_error_pre_intervention, current_impact_score

            error_valid, error_pre_intervention, impact_score = dataset.eval_metric_ipw(preds)

            if current_error_valid > error_valid:
                current_error_valid = error_valid
                current_error_pre_intervention = error_pre_intervention
                current_impact_score = impact_score

            return 'causal_fitness', error_valid

        class training_callback(xgb.callback.TrainingCallback):
            """A callback to reset the error at the beginning of each training run."""
            def before_training(self, model):
                global current_error_valid, current_error_pre_intervention, current_impact_score
                current_error_valid = float('inf')
                current_error_pre_intervention = float('inf')
                current_impact_score = float('inf')

                return model

        # The middle loop: Optuna's hyperparameter search.
        def objective(trial):
            """
            The objective function for Optuna. It defines the search space for hyperparameters
            and runs the XGBoost training with the custom metric and early stopping.
            """
            global pruning_trial
            pruning_trial = trial

            # Define the hyperparameter search space for Optuna.
            params = {
                'seed': attempt + seed,
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
                return current_error_pre_intervention
            elif(optuna_optimization_target == "validation_folder"):
                return current_error_valid

            return -1.0

        # --- 4. Run the Optuna Study ---
        directions=['minimize']

        study = optuna.create_study(
            directions=directions,
            sampler=sampler
        )

        # Run the Optuna study to find the best set of hyperparameters.
        study.optimize(objective, n_trials=optuna_number_trials, timeout=optuna_timeout_cycle, show_progress_bar=True)
