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
from scipy import linalg as la
from scipy.optimize import minimize
import sys
import xgboost as xgb
xgb.set_config(verbosity=0)
import optuna
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

"""
Filters the maximum number of non-multicollinear donors using greedy forward selection.

This function is fully invariant to the level (mean) and scale (variance) of the
donor data, focusing only on the shape of the time series to detect collinearity.

It uses two checks:
1.  **Pairwise Pearson Correlation:** Ensures no two selected donors are too similarly shaped.
2.  **Gram Matrix Condition Number:** Ensures the entire set of selected donors is not
    collectively multicollinear.

Parameters:
-----------
X : array-like, shape (T, K*M)
    Donor time series matrix. For multiple outcomes (M), columns must be concatenated per donor.
n_outcomes : int
    The number of outcome variables (M) per donor, required to interpret the shape of X.
collinear_threshold : float, default=0.85
    Maximum absolute Pearson correlation allowed between any two selected donors.
max_cond : float, default=500.0
    Maximum condition number of the correlation matrix for the selected donors.
seed : int, optional
    Seed for the random number generator to ensure a reproducible selection order.

Returns:
--------
selected_indices : list of int
    Indices (from 0 to K-1) of the selected subset of donors.
gram_cond : float
    Condition number of the final selected donor matrix (inf if empty, 1.0 for a single donor).
selected_size : int
    The number of selected donors.
"""
def filter_donors_by_collinearity(X, n_outcomes, collinear_threshold=0.85, max_cond=100.0, seed=None):

    if seed is not None:
        np.random.seed(seed)

    T, total_cols = X.shape

    if total_cols % n_outcomes != 0:
        raise ValueError(f"Total columns in X ({total_cols}) must be divisible by n_outcomes ({n_outcomes}).")
    K = total_cols // n_outcomes
    
    # Reshape X so each column is the full time series for one donor (across all outcomes)
    X_reshaped = X.reshape(T * n_outcomes, K)

    # --- Fully Invariant Pairwise Check ---
    # Compute the donor-donor Pearson correlation matrix.
    # This is invariant to both level and scale.
    # np.corrcoef expects variables as rows, so we pass the transpose of our reshaped X.
    corr_matrix = np.corrcoef(X_reshaped.T)
    np.fill_diagonal(corr_matrix, 0) # Exclude self-correlation from checks

    # Create a randomized order for evaluating donors to avoid bias
    order = list(range(K))
    np.random.shuffle(order)

    # --- Greedy selection loop ---
    selected = []
    for idx in order:
        # Check 1: Pairwise collinearity using Pearson correlation
        if any(abs(corr_matrix[idx, sel]) >= collinear_threshold for sel in selected):
            continue

        # Check 2: Overall multicollinearity via condition number
        temp_indices = selected + [idx]
        if len(temp_indices) > 1:
            temp_X = X_reshaped[:, temp_indices]
            # Standardize columns for a scale-invariant condition number calculation
            temp_X_std = (temp_X - temp_X.mean(axis=0)) / (temp_X.std(axis=0) + 1e-9)
            gram_temp = temp_X_std.T @ temp_X_std
            if np.linalg.cond(gram_temp) > max_cond:
                continue

        selected.append(idx)

    # --- Compute final metrics for the selected set ---
    selected_size = len(selected)
    if selected_size <= 1:
        gram_cond = 1.0 if selected_size == 1 else np.inf
    else:
        final_X = X_reshaped[:, selected]
        final_X_std = (final_X - final_X.mean(axis=0)) / (final_X.std(axis=0) + 1e-9)
        gram_final = final_X_std.T @ final_X_std
        gram_cond = np.linalg.cond(gram_final)

    return selected, gram_cond

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

Returns:
    np.ndarray: A 1D array of the optimal weights for the control units.
"""
def fast_synthetic_control_fitting(treated_pre_intervention, control_pre_intervention):
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

    # Bounds: weights must be between 0 and 1.
    bounds = tuple((0, 1) for _ in range(num_control_units))

    # Initial guess for the weights (equal weighting).
    initial_weights = np.ones(num_control_units) / num_control_units

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
    """
    A fully vectorized implementation of the block bootstrap for calculating p-values.
    """
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

This function executes a multi-loop search to find the optimal donor pool for Synthetic Control Methods.

This function uses a combination of cross-temporal validation and Optuna-driven
hyperparameter optimization of an XGBoost model to identify a set of control units
(donors) that best match the treated unit's pre-intervention trends. The core of
the method is an ATT/IPW-based ranking system derived from the model's propensity scores.

The function does not return a value but saves the results of viable donor pool
candidates to CSV files in the specified `workspace_folder`.

Parameters
----------
all_units : pd.DataFrame
    A DataFrame in long format containing all unit-time observations.
yname : str
    The name of the column that contains the names of the outcome variables.
unitname : str
    The name of the column that contains the unique identifiers for each unit.
tname : str
    The name of the column that contains the time period identifiers (e.g., year).
value : str
    The name of the column that contains the numerical values for the outcomes.
treatment : str
    The name of the column that indicates treatment status (1 for treated, 0 for control).
pre_intervention : str
    The name of the column indicating the pre-intervention period (1 if pre, 0 if post).
temporal_cross_search : list of str
    A list of time periods within the pre-intervention phase to use as split points for
    cross-temporal validation. Each period marks the end of a training set.
workspace_folder : str
    The path to a directory where output files (candidate donors and performance) will be saved.
seed : int, optional
    Random seed for reproducibility of the optimization process. Defaults to 111.
maximum_num_units_on_support_first_filter : int, optional
    The maximum number of units allowed in the on-support group during the first pruning step,
    used to penalize trivial solutions. Defaults to 50.
maximum_error_pre_intervention : float, optional
    The maximum acceptable error (e.g., RMSE or MAE) on the pre-treatment outcomes for a
    candidate donor pool to be saved. Defaults to 0.15.
include_impact_score_in_optuna_objective : bool, optional
    Flag to include or not the impact score in the Optuna's search criteria. Defaults to False.
number_optuna_trials : int, optional
    The number of hyperparameter optimization trials to run for each cross-temporal fold.
    Defaults to 300.
timeout_optuna_cycle : int, optional
    The maximum time in seconds for a single Optuna optimization cycle (one cross-temporal fold).
    Defaults to 900.
"""
def search(
    all_units,
    yname,
    unitname,
    tname,
    value,
    treatment,
    pre_intervention,
    temporal_cross_search,
    workspace_folder,
    seed = 111,
    maximum_num_units_on_support_first_filter = 50,
    maximum_error_pre_intervention = 0.15,
    include_impact_score_in_optuna_objective = False,
    number_optuna_trials = 1000,
    timeout_optuna_cycle = 900
):
    # RENAME AND INFORM ERRORS
    if tname in all_units.columns:
        all_units.rename(columns={tname: 'timeid'}, inplace=True)
    else:
        print(f"The '{tname}' column does not exist in the dataframe.")
        sys.exit()

    if unitname in all_units.columns:
        all_units.rename(columns={unitname: 'unitid'}, inplace=True)
    else:
        print(f"The '{unitname}' column does not exist in the dataframe.")
        sys.exit()

    if yname in all_units.columns:
        all_units.rename(columns={yname: 'outcome'}, inplace=True)
    else:
        print(f"The '{yname}' column does not exist in the dataframe.")
        sys.exit()

    if value in all_units.columns:
        all_units.rename(columns={value: 'value'}, inplace=True)
    else:
        print(f"The '{value}' column does not exist in the dataframe.")
        sys.exit()

    if treatment in all_units.columns:
        all_units.rename(columns={treatment: 'treatment'}, inplace=True)
    else:
        print(f"The '{treatment}' column does not exist in the dataframe.")
        sys.exit()

    if pre_intervention in all_units.columns:
        all_units.rename(columns={pre_intervention: 'pre_intervention'}, inplace=True)
    else:
        print(f"The '{pre_intervention}' column does not exist in the dataframe.")
        sys.exit()

    # CHECK pre_intervention COULUMN AGAINST temporal_cross_search
    # temporal_cross_search MUST BE A SUBSET OF pre_intervention
    if not set(temporal_cross_search).issubset(set(all_units[all_units['pre_intervention'] == 1]['timeid'].unique().tolist())):
        print("ERROR: temporal_cross_search MUST BE A SUBSET OF pre_intervention")
        print(f"temporal_cross_search: {sorted(temporal_cross_search)}")
        print(f"pre_intervention: {sorted(all_units[all_units['pre_intervention'] == 1]['timeid'].unique().tolist())}")
        sys.exit()

    # CHECH DIRECTORY EXISTS AND CREATE IT IF NOT
    if not os.path.exists(workspace_folder):
        os.makedirs(workspace_folder)
        print(f"Created workspace folder: {workspace_folder}")
   
    file_path = resources.files('propensitygbdt.data').joinpath('scm_donor_selection_candidate_performance.xlsx')
    try:
        with resources.as_file(file_path) as source_path:
            shutil.copy(source_path, workspace_folder)
    except (FileNotFoundError, ModuleNotFoundError):
        print("Error: Could not find the source file or the 'propensitygbdt.data' package.")
        print("Please ensure the package is correctly installed.")

    file_path = resources.files('propensitygbdt.data').joinpath('scm_donor_selection_candidate_units_data.xlsx')
    try:
        with resources.as_file(file_path) as source_path:
            shutil.copy(source_path, workspace_folder)
    except (FileNotFoundError, ModuleNotFoundError):
        print("Error: Could not find the source file or the 'propensitygbdt.data' package.")
        print("Please ensure the package is correctly installed.")

    all_units = all_units[['timeid', 'pre_intervention', 'unitid', 'treatment', 'outcome', 'value']]

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

    treatment_unitid = all_units[all_units['treatment'] == 1]['unitid'].iloc[0]
    outcomes = all_units['outcome'].sort_values().unique().tolist()
    # timeids = all_units['timeid'].sort_values().unique().tolist()
    timeid_pre_intervention = all_units[all_units['pre_intervention'] == 1]['timeid'].sort_values().unique().tolist()
    timeid_post_intervention = all_units[all_units['pre_intervention'] == 0]['timeid'].sort_values().unique().tolist()

    hyperparameter_search_extra_criteria = []  
    if include_impact_score_in_optuna_objective:
        hyperparameter_search_extra_criteria.append('post_intervention_period')

    scm_donor_selection_candidate_units_data_file_path = workspace_folder + 'scm_donor_selection_candidate_units_data.csv'
    if os.path.exists(scm_donor_selection_candidate_units_data_file_path):
        os.remove(scm_donor_selection_candidate_units_data_file_path)

    scm_donor_selection_candidate_performance_file_path = workspace_folder + 'scm_donor_selection_candidate_performance.csv'
    if os.path.exists(scm_donor_selection_candidate_performance_file_path):
        os.remove(scm_donor_selection_candidate_performance_file_path)

    all_units.sort_values(by=['unitid', 'timeid', 'outcome'], inplace=True)
    all_units['weight'] = 1.0

    amplitude = all_units[(all_units['treatment'] == 1) & all_units['pre_intervention'] == 1].groupby(['outcome']).apply(
        lambda x : pd.Series({
            'amplitude': 1 if x['value'].max() == x['value'].min() else x['value'].max() - x['value'].min()
        })
    ).reset_index()

    all_units.drop('pre_intervention', axis=1, inplace=True)

    def pre_intervention_scaling(data, period):
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
        standardization = aggregated[aggregated['timeid'].isin(period)].groupby(['outcome', 'treatment'], dropna=False).agg(
            avg = ('value', lambda x: np.mean(x)),
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

    # Initialize CSV files for storing performance results and the donor candidates data with headers.
    scm_donor_selection_candidate_units_data = treatment[['outcome', 'timeid', 'value', 'treatment', 'weight', 'unitid']].copy()
    scm_donor_selection_candidate_units_data['valor_m_weight'] = scm_donor_selection_candidate_units_data['value'] * scm_donor_selection_candidate_units_data['weight']
    scm_donor_selection_candidate_units_data['id'] = None
    scm_donor_selection_candidate_units_data['cycle'] = None
    scm_donor_selection_candidate_units_data['trial'] = None
    scm_donor_selection_candidate_units_data['solution_id'] = None
    scm_donor_selection_candidate_units_data['error_train'] = None
    scm_donor_selection_candidate_units_data['error_valid'] = None
    scm_donor_selection_candidate_units_data['error_pre_intervention'] = None
    scm_donor_selection_candidate_units_data['error_post_intervention'] = None
    scm_donor_selection_candidate_units_data['impact_score'] = None    
    scm_donor_selection_candidate_units_data['num_units_bigger_min_weight'] = None
    scm_donor_selection_candidate_units_data.to_csv(scm_donor_selection_candidate_units_data_file_path, mode='w', header=True, index=False)

    # Initialize CSV files for storing performance results with headers.
    scm_donor_selection_candidate_performance = treatment[['outcome', 'timeid', 'treatment', 'value']].copy()
    scm_donor_selection_candidate_performance['qtty'] = 1
    scm_donor_selection_candidate_performance['cycle'] = None
    scm_donor_selection_candidate_performance['trial'] = None
    scm_donor_selection_candidate_performance['solution_id'] = None
    scm_donor_selection_candidate_performance['error_train'] = None
    scm_donor_selection_candidate_performance['error_valid'] = None
    scm_donor_selection_candidate_performance['error_pre_intervention'] = None
    scm_donor_selection_candidate_performance['error_post_intervention'] = None
    scm_donor_selection_candidate_performance['impact_score'] = None    
    scm_donor_selection_candidate_performance['num_units_bigger_min_weight'] = None
    scm_donor_selection_candidate_performance.to_csv(scm_donor_selection_candidate_performance_file_path, mode='w', header=True, index=False)

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
            data = pre_intervention_scaling(data=data, period=self.timeid_train + self.timeid_valid)
            aggregated3 = data.groupby(['outcome', 'timeid', 'treatment'], dropna=False).apply(
                lambda x : pd.Series({
                    'value' : np.ma.filled(np.ma.average(np.ma.masked_invalid(x['value']), weights=x['weight']), fill_value=np.nan)
                })
            ).reset_index()

            # Calculate error on the training set.
            aggregated3_diff = aggregated3[aggregated3['timeid'].isin(self.timeid_train)].sort_values(['outcome', 'timeid', 'treatment'], ascending=True).groupby(['outcome', 'timeid']).agg({
                'value' : lambda x: x.iloc[0] - x.iloc[1]
            }).reset_index(level=['outcome', 'timeid'])
            
            aggregated3_diff_normalized = pd.merge(aggregated3_diff, amplitude, on='outcome', how='left')
            aggregated3_diff_normalized['value'] = aggregated3_diff_normalized['value'] / aggregated3_diff_normalized['amplitude']

            error_train = ((aggregated3_diff_normalized['value'] ** 2).mean()) ** 0.5

            # Calculate error on the validation set. This is crucial for early stopping and preventing overfitting.
            aggregated3_diff = aggregated3[aggregated3['timeid'].isin(self.timeid_valid)].sort_values(['outcome', 'timeid', 'treatment'], ascending=True).groupby(['outcome', 'timeid']).agg({
                'value' : lambda x: x.iloc[0] - x.iloc[1]
            }).reset_index(level=['outcome', 'timeid'])

            aggregated3_diff_normalized = pd.merge(aggregated3_diff, amplitude, on='outcome', how='left')
            aggregated3_diff_normalized['value'] = aggregated3_diff_normalized['value'] / aggregated3_diff_normalized['amplitude']

            error_valid = ((aggregated3_diff_normalized['value'] ** 2).mean()) ** 0.5

            # Calculate error for the entire pre-treatment period.
            aggregated3_diff = aggregated3[aggregated3['timeid'].isin(self.timeid_train + self.timeid_valid)].sort_values(['outcome', 'timeid', 'treatment'], ascending=True).groupby(['outcome', 'timeid']).agg({
                'value' : lambda x: x.iloc[0] - x.iloc[1]
            }).reset_index(level=['outcome', 'timeid'])

            aggregated3_diff_normalized_pre = pd.merge(aggregated3_diff, amplitude, on='outcome', how='left')
            aggregated3_diff_normalized_pre['value'] = aggregated3_diff_normalized_pre['value'] / aggregated3_diff_normalized_pre['amplitude']

            error_pre_intervention = ((aggregated3_diff_normalized_pre['value'] ** 2).mean()) ** 0.5

            # Calculate error for the entire post-treatment period.
            # --- Diagnostic tool, not a core component, and it is disabled by default ---
            # To validate against bias from using post-treatment data, run this script twice:
            #  - Run A (Unbiased): Without `error_post_intervention` in the Optuna objective.
            #  - Run B (Biased):   With `error_post_intervention` included.
            # --- How to Interpret the Results ---
            # * If both runs agree on the donor pool -> The result is likely robust.
            # * If both runs find good fits but disagree -> Bias is likely. Trust Run A.
            # * If Run A fails but Run B succeeds -> Run B's result is suspect. Rerun Run A with more Optuna trials.
            # * If both runs fail -> This indicates a more fundamental problem, such as no suitable donors in the pool
            # or an insufficient Optuna trials, reduce the number of selected outcomes and rerun.
            aggregated3_diff = aggregated3[aggregated3['timeid'].isin(self.timeid_post_intervention)].sort_values(['outcome', 'timeid', 'treatment'], ascending=True).groupby(['outcome', 'timeid']).agg({
                'value' : lambda x: x.iloc[0] - x.iloc[1]
            }).reset_index(level=['outcome', 'timeid'])

            aggregated3_diff_normalized_post = pd.merge(aggregated3_diff, amplitude, on='outcome', how='left')
            aggregated3_diff_normalized_post['value'] = aggregated3_diff_normalized_post['value'] / aggregated3_diff_normalized_post['amplitude']

            error_post_intervention = ((aggregated3_diff_normalized_post['value'] ** 2).mean()) ** 0.5

            if error_pre_intervention < maximum_error_pre_intervention:
                np.random.seed(attempt + seed) 
                impact_score = block_bootstrap_rmspe_ratio_vectorized(
                    aggregated3_diff_normalized_pre,
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

            if num_units_bigger_min_weight > maximum_num_units_on_support_first_filter or num_units_bigger_min_weight == 0:
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
            data2 = datat[datat['unitid'].isin([treatment_unitid] + list(donor_unitids))].copy()
            donor_df = data2[data2['treatment'] == 0].copy()
            pivot_donor = donor_df.pivot(index='timeid', columns=['unitid', 'outcome'], values='value')
            column_order = pd.MultiIndex.from_product([donor_unitids, outcomes], names=['unitid', 'outcome'])
            pivot_donor = pivot_donor.reindex(columns=column_order, fill_value=0)  # Fill NAs if sparse
            selected_donors, gram_cond = filter_donors_by_collinearity(X=pivot_donor.values, n_outcomes=len(outcomes), collinear_threshold=0.85, max_cond=100.0, seed=(attempt + seed))
            # CHECK IF selected_donors ARE IN CACHE AND SKIP IF FOUND
            combination_tuple = tuple(sorted(donor_unitids[selected_donors]))
            if combination_tuple in estimated_solutions:
                (error_valid, error_pre_intervention, impact_score) = estimated_solutions[combination_tuple]
            else:
                data2 = data2[(data2['treatment'] != 0) | (data2['unitid'].isin(donor_unitids[selected_donors]))]

                # Estimate weights via traditional SCM
                df = data2.sort_values(by=['treatment', 'timeid', 'unitid', 'outcome']).reset_index(drop=True)
                df = df[df['timeid'].isin(self.timeid_train + self.timeid_valid)]
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
                weight_mapping = dict(zip(control_unit_ids, optimal_weights))
                filtered_weights = {unit: weight for unit, weight in weight_mapping.items() if weight > 0.001}
                current_combination_tuple = tuple(sorted(filtered_weights.keys()))
                data2['weight'] = data2['unitid'].map(filtered_weights)
                data2['weight'] = np.where(data2['treatment'] == 1, 1, data2['weight'])
                data2 = data2[data2['weight'].notna()]
                current_ipw = data2[['id','treatment','weight']].drop_duplicates().copy()

                # Re-normalize weights after selecting the top N donors to sum to 1.
                sum_weight_tratamento = current_ipw.groupby('treatment').agg({
                    'weight' : 'sum'
                }).reset_index()
                sum_weight_tratamento_0 = sum_weight_tratamento[sum_weight_tratamento['treatment'] == 0]['weight'].values[0]
                current_ipw['weight'] = np.where(current_ipw['treatment'] == 0, current_ipw['weight'] / sum_weight_tratamento_0, current_ipw['weight'])
                current_ipw.drop('treatment', axis=1, inplace=True)

                # Evaluate the performance of this candidate donor pool.
                error_train, error_valid, error_pre_intervention, error_post_intervention, impact_score = self.evaluate_outcomes_metric(data=data2)

                if error_pre_intervention < maximum_error_pre_intervention and gram_cond <= 100 and current_combination_tuple and current_combination_tuple not in estimated_solutions:
                    current_ipw.drop
                    temp, data = dataset.full_data_treatment_control_scaling(current_ipw)
                    # Save the donor units, weights and performance metrics of this viable solution.
                    data = data[data['treatment'] == 0]
                    data['valor_m_weight'] = data['value'] * data['weight']
                    data['cycle'] = cycle
                    data['trial'] = pruning_trial.number
                    data['solution_id'] = solution_id
                    data['error_train'] = error_train
                    data['error_valid'] = error_valid
                    data['error_pre_intervention'] = error_pre_intervention
                    data['error_post_intervention'] = error_post_intervention
                    data['impact_score'] = impact_score
                    data['num_units_bigger_min_weight'] = num_units_bigger_min_weight
                    columns = ['outcome', 'timeid', 'value', 'treatment', 'weight', 'unitid', 'valor_m_weight', 'id', 'cycle', 'trial', 'solution_id', 'error_train', 'error_valid', 'error_pre_intervention', 'error_post_intervention', 'impact_score', 'num_units_bigger_min_weight']
                    data[columns].to_csv(scm_donor_selection_candidate_units_data_file_path, mode='a', header=False, index=False)

                    # Save the weights and performance metrics of this viable solution.
                    temp = temp[temp['treatment'] == 0]
                    temp['cycle'] = cycle
                    temp['trial'] = pruning_trial.number
                    temp['solution_id'] = solution_id
                    temp['error_train'] = error_train
                    temp['error_valid'] = error_valid
                    temp['error_pre_intervention'] = error_pre_intervention
                    temp['error_post_intervention'] = error_post_intervention
                    temp['impact_score'] = impact_score                        
                    temp['num_units_bigger_min_weight'] = num_units_bigger_min_weight
                    temp.to_csv(scm_donor_selection_candidate_performance_file_path, mode='a', header=False, index=False)

                    solution_id = solution_id + 1

                # Save in cache the current found solution and its simplified version
                if current_combination_tuple and current_combination_tuple not in estimated_solutions:
                    estimated_solutions[current_combination_tuple] = (error_valid, error_pre_intervention, impact_score)

            attempt = attempt + 1

            return error_valid, error_pre_intervention, impact_score

        def full_data_treatment_control_scaling(self, weight):
            """Prepares the full dataset with the final weights for saving if it is good enough."""
            temp = pd.merge(weight, self.from_to, on='id', how='inner')
            temp = pd.merge(self.full_data, temp, on='unitid', how='left').reset_index()
            data = pre_intervention_scaling(data=temp, period=self.timeid_train + self.timeid_valid)
            aggregated3 = data.groupby(['outcome', 'timeid', 'treatment'], dropna=False).apply(
                lambda x : pd.Series({
                    'value' : np.ma.filled(np.ma.average(np.ma.masked_invalid(x['value']), weights=x['weight']), fill_value=np.nan),
                    'qtty' : np.ma.filled(np.ma.count(np.ma.masked_invalid(x['value'])), fill_value=np.nan)
                })
            ).reset_index()
            return aggregated3, data

    # MAIN TRAINING LOOP
    # This outer loop performs Cross-Temporal Validation.
    cycle = 0
    global attempt, solution_id, estimated_solutions
    attempt = 0
    solution_id = 0
    # EXECUTION CACHE
    estimated_solutions = dict()

    timeid_train_indexes = [timeid_pre_intervention.index(x) + 1 for x in temporal_cross_search]
    for timeid_train_index in timeid_train_indexes:
        if include_impact_score_in_optuna_objective:
            sampler = optuna.samplers.NSGAIIISampler(seed=cycle+seed)
        else:
            sampler = optuna.samplers.RandomSampler(seed=cycle+seed)
        
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
                if 'post_intervention_period' in hyperparameter_search_extra_criteria:
                    return float('inf'), float('inf')
                else:
                    return float('inf')
            
            # The article describes a multi-objective optimization problem. 
            # Optuna is configured to minimize, pre-treatment error, and post-intervention error (optional for diagnose).
            if 'post_intervention_period' in hyperparameter_search_extra_criteria:
                return current_error_pre_intervention, current_impact_score
            else:
                return current_error_pre_intervention

        # --- 4. Run the Optuna Study ---
        if 'post_intervention_period' in hyperparameter_search_extra_criteria:
            directions=['minimize', 'minimize']
        else:
            directions=['minimize']

        study = optuna.create_study(
            directions=directions,
            sampler=sampler
        )

        # Run the Optuna study to find the best set of hyperparameters.
        study.optimize(objective, n_trials=number_optuna_trials, timeout=timeout_optuna_cycle, show_progress_bar=True)
