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
1.  **Donor Selection:** A carefully tuned XGBoost model is used to identify a
    small, causally valid donor units. This is achieved through a
    sophisticated multi-loop search process that leverages ATT/IPW-based
    ranking and cross-temporal validation.
2.  **Optimization:** [THIS SCRIPT] The selected donors are then used in a
    Bayesian SCM algorithm to estimate the effect and its uncertainty
    associated with the synthetic control.

This implementation aims to properly estimate the uncertainty associated with
the synthetic control for causal inference.

.. _Medium Article:
    https://medium.com/@frederico.nogueira/a-new-lens-for-donor-selection-att-ipw-based-ranking-198b9d30bc69

"""
import math
import os
import pandas as pd
import numpy as np
import cmdstanpy
from scipy.linalg import qr
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from plotnine import *
import arviz as az
from importlib import resources
import shutil
from datetime import datetime
import warnings
# Suppress warnings to keep the output clean.
warnings.filterwarnings("ignore")

# Define a function to create a list of matrices, one for each outcome.
# Each matrix will have time periods as rows and control units as columns.
def get_control_matrices_by_outcome(df, time_range, outcomes, unitids):
    matrices = []
    if not time_range:
        return [np.array([[]])] * N_outcomes
    subset = df[df['timeid'].isin(time_range)]
    if subset.empty:
        return [np.array([[]])] * N_outcomes
        
    for outcome in outcomes:
        outcome_df = subset[subset['outcome'] == outcome]
        # Pivot to get a timeid x unitid matrix for the current outcome.
        pivoted = outcome_df.pivot_table(index='timeid', columns='unitid', values='value')[unitids]
        matrices.append(pivoted.to_numpy())
    return matrices

"""
Classifies the structural uncertainty band into 5 types:
1. Tight (Ideal): Shape is irrelevant.
2. Funnel (Good): Starts wide, narrows before intervention.
3. Trumpet (Bad): Starts narrow, widens before intervention.
4. Homogenous (Too Wide): Too width.
5. Breather (Noisy): Oscillates with no clear trend.
"""
def classify_structural_uncertainty_shape(df, outcome_col='outcome'):     
    results = []
    
    # Get list of outcomes
    outcomes = df[outcome_col].unique()
    
    for outcome in outcomes:
        subset = df[df[outcome_col] == outcome].copy()
        
        # Sort by time to ensure correlation works
        subset = subset.sort_values('timeid')
        
        # Calculate the Width
        subset['band_width'] = subset['q97_5'] - subset['q2_5']
        widths = subset['band_width'].values
        time_steps = np.arange(len(widths))
        
        amp_signal = np.max(subset['mean']) - np.min(subset['mean'])
        if amp_signal == 0: amp_signal = 1e-6
        
        max_width = np.max(widths)
        relative_width = max_width / amp_signal
        
        # --- METRICS ---
        corr, _ = pearsonr(time_steps, widths)
        cv = np.std(widths) / np.mean(widths) if np.mean(widths) > 0 else 0
        
        # --- CLASSIFICATION LOGIC ---
        shape_class = "Unknown"
        quality_score = 0 
        
        # Thresholds
        THINNESS_THRESHOLD = 0.35
        CORR_THRESHOLD = 0.4
        CV_THRESHOLD = 0.4      
        
        # 1. NEW CRITERIA: THINNESS CHECK
        if relative_width < THINNESS_THRESHOLD:
            shape_class = "Tight (Ideal)"
            quality_score = 10            
        elif corr < -CORR_THRESHOLD:
            shape_class = "Funnel (Good)"
            quality_score = 9 
        elif corr > CORR_THRESHOLD:
            shape_class = "Trumpet (Risky)"
            quality_score = 1 
        elif cv > CV_THRESHOLD:
            shape_class = "Breather (Noisy)"
            quality_score = 3
        else:
            shape_class = "Homogenous (Too Wide)"
            quality_score = 2

        results.append({
            'outcome': outcome,
            'shape': shape_class,
            'correlation': round(corr, 3),
            'volatility_cv': round(cv, 3),
            'max_width': round(max_width, 4),
            'relative_width': round(relative_width, 4),
            'quality_score': quality_score
        })
        
    return pd.DataFrame(results).sort_values('quality_score', ascending=False)

"""
Performs a comprehensive Bayesian Synthetic Control Method (BSCM) analysis.

This function automates the entire BSCM pipeline. It begins by loading and
preprocessing the data, then fits a Bayesian model using Stan to estimate the
Average Treatment Effect on the Treated (ATT) across multiple outcomes.
Finally, it generates and saves detailed post-processing summaries and
visualizations of the results.

Args:
    timeid_previous_intervention (str): The identifier for the final time
                                        period before the intervention starts. This
                                        value marks the end of the pre-treatment window.
    workspace_folder (str): The file path to the root directory that contains the
                            input data file ('scm_donor_selection_candidate_units_data.csv')
                            and will store all output files (CSVs, plots).
    solution_id (int, optional): The specific ID for a pre-selected group of control
                                 units (the donor pool). If set to None, the function
                                 will automatically choose the solution that demonstrates the
                                 best balance between pre-intervention fit and impact score.
                                 Defaults to None.
    period_effect_format (str, optional): A format string used to display numeric
                                          results in the plot annotations, such as the
                                          average treatment effect. Defaults to '{:.2f}'.
    seed (int, optional): A random seed to ensure the reproducibility of the
                          MCMC sampling and any other stochastic processes.
                          Defaults to 222.
    nrmse_terminal_over_train (float, optional): Tolerance for overfitting on the last time
                      point befor the intervention. This check is needed because when 
                      assumes low rank hidden trends and we don't require a perfect fit.
                      Defaults to 2.0.
    length_post_intervention_period (int, optional): Defines the length of the post
                      intervention period. Defaults to None (All available).
"""
def estimate(
    timeid_previous_intervention: str,
    workspace_folder: str,
    solution_id: int = None,
    period_effect_format: str = '{:.2f}',
    seed: int = 222,
    nrmse_terminal_over_train: float = 2.5,
    length_post_intervention_period: int = None
):
    # Set the random seed for NumPy to ensure reproducibility of its random operations.
    np.random.seed(seed)
    
    global parallel_trends_violated

    # Remove the last char from workspace_folder if it is /
    if workspace_folder.endswith('/') or workspace_folder.endswith('\\'):
        workspace_folder = workspace_folder[:-1]
        
    if length_post_intervention_period is not None and not isinstance(length_post_intervention_period, int) and length_post_intervention_period <= 0:
        raise ValueError(f"length_post_intervention_period must be a positive int, got {length_post_intervention_period}")

    # Check if CmdStanPy is installed and can find the CmdStan command-line interface.
    # This is a crucial dependency for running the Bayesian models.
    try:
        cmdstan_path = cmdstanpy.cmdstan_path()
        print("CmdStanPy is installed and has located a CmdStan installation.")
        print(f"CmdStan path: {cmdstan_path}")
        print(f"CmdStan version: {cmdstanpy.cmdstan_version()}")
    except ValueError:
        print("CmdStanPy is not installed.")
        print("Installing CmdStan with compiler toolchain (one-time, ~5-10 min)...")
        cmdstanpy.install_cmdstan(compiler=True)
        print(f"Installation complete. Add directory {cmdstan_path}\\RTools40\\mingw64\\bin to PATH.")
        print("Then restart session for update PATH.")
        # Note: If in a script/IDE, restart and rerun to ensure PATH is updated

    # 2. DATA
    # -----------------------------------------------------------------------------
    # Load the dataset from a CSV file. A try-except block handles the case where the file is not found.
    # Data types for ID columns are explicitly set to string to avoid unintended numeric conversions.
    try:
        dataset_raw = pd.read_csv(f"{workspace_folder}/scm_donor_selection_candidate_units_data.csv", dtype={'outcome': str, 'timeid': str, 'unitid': str})
        if solution_id is None:
            # Select the 'solution_id' with the minimum distance to the origin (0,0) in the error-impact space.
            solutions_id = dataset_raw[['solution_id']].drop_duplicates()['solution_id']
            solutions_id = solutions_id[solutions_id.notna()].astype(int).to_list()
            filtered_unitid = False
        else:
            # Filter the dataset to include only the treated unit and the control units from the specified 'solution_id'.
            solutions_id = []
            solutions_id_unitids = {}
            for item in solution_id:
                solutions_id.append(item["solution_id"])
                if "unitids" in item:
                    solutions_id_unitids[item["solution_id"]] = item["unitids"]
                    filtered_unitid = True
                else:
                    filtered_unitid = False
            
    except FileNotFoundError:
        print("Error: 'scm_donor_selection_candidate_units_data.csv' not found.")
        exit()

    temp_folder = f"temp_{datetime.now().strftime("%Y%m%d_%H%M%S")}"
    
    # Check file already exists in destination
    if not os.path.exists(f"{workspace_folder}/bscm.stan"):
        # Specify the path to the Stan model file.
        # Resolve path to the source code in the package's data folder
        file_path = resources.files('propensitygbdt.data').joinpath('bscm.stan')

        # Copy the Stan model file to the workspace folder (writable)
        try:
            with resources.as_file(file_path) as source_path:
                shutil.copy(source_path, workspace_folder)
            print(f"Copied {file_path} to {workspace_folder}/bscm.stan")
        except (FileNotFoundError, ModuleNotFoundError) as e:
            print(f"Error: Could not find or copy the source file: {e}")
            print("Please ensure the package is correctly installed with package_data including 'data/*.stan'.")
            raise

    # Verify toolchain accessibility (debug: check if mingw32-make is in PATH)
    make_in_path = shutil.which('mingw32-make') is not None
    print(f"mingw32-make available in PATH: {make_in_path}")
    if not make_in_path:
        raise RuntimeError(f"Add directory {cmdstan_path}\\RTools40\\mingw64\\bin to PATH.")

    # Load the model (compiles to workspace_folder/bscm.exe, then caches)
    stan_file_path = f"{workspace_folder}/bscm.stan"
    print(f"Compiling/loading model from: {stan_file_path}")
    model = cmdstanpy.CmdStanModel(stan_file=stan_file_path)

    summary_data_collection = []
    unacceptable_solutions = []
    weight_solutions_list = []
    for solution_id in solutions_id:
        parallel_trends_violated = False
        dataset = dataset_raw.copy()
        solution_has_problem = ""
        dataset = dataset[(dataset['treatment'] == 1) | (dataset['solution_id'] == solution_id)]
        if filtered_unitid:
            dataset = dataset[(dataset['treatment'] == 1) | (dataset['unitid'].isin(solutions_id_unitids[solution_id]))]
            unitids = sorted(solutions_id_unitids[solution_id])
        else:
            # Get sorted lists of unique control unit IDs and outcome names.
            unitids = sorted(dataset[dataset['treatment'] == 0]['unitid'].unique())
        print(f"Using unitids: {unitids}")

        # Keep only the essential columns for the analysis.
        dataset = dataset[["outcome", "timeid", "value", "treatment", "unitid"]]
        
        if not set([timeid_previous_intervention]).issubset(set(dataset['timeid'].unique().tolist())):
            print(f"timeid_previous_intervention: {timeid_previous_intervention}")
            print(f"pre_intervention: {sorted(dataset['timeid'].unique().tolist())}")
            raise ValueError("ERROR: timeid_previous_intervention MUST BE A SUBSET OF pre_intervention")

        # Ensure that the analysis only uses time periods where data is available for all outcomes.
        # This avoids issues with missing data in the model.
        timeid_counts = dataset.groupby(['timeid', 'outcome']).size().reset_index(name='N')
        timeid_outcome_counts = timeid_counts.groupby('timeid').size().reset_index(name='N_outcomes')
        max_outcomes = timeid_outcome_counts['N_outcomes'].max()
        valid_timeids = timeid_outcome_counts[timeid_outcome_counts['N_outcomes'] == max_outcomes]['timeid']
        dataset = dataset[dataset['timeid'].isin(valid_timeids)]

        # Convert the 'timeid' column to a categorical type and then to integer codes.
        # Stan requires integer indices, so this mapping is necessary. We add 1 because Stan uses 1-based indexing.
        dataset['timeid'] = pd.Categorical(dataset['timeid'])
        mapping_timeid = dataset['timeid'].cat.categories
        dataset['timeid'] = dataset['timeid'].cat.codes + 1  # Stan uses 1-based indexing

        # Define the start, base (end of pre-treatment), and end time periods using the integer codes.
        timeid_inicio = 1
        timeid_base = np.where(mapping_timeid == timeid_previous_intervention)[0][0] + 1
        if length_post_intervention_period is None:
            timeid_fim = dataset['timeid'].max()
        elif len(mapping_timeid) >= timeid_base + length_post_intervention_period:
            timeid_fim = timeid_base + length_post_intervention_period
        else:
            raise ValueError(f"length_post_intervention_period maust be equal or smaller than the available time points after the pre_intervention")
            
        mapping_timeid = mapping_timeid[:timeid_fim]
        
        dataset = dataset[dataset['timeid'].isin(range(0, len(mapping_timeid) + 1))]

        # Define a validation period using the last 20% of the pre-treatment data with minimum of 2. This period is used for model validation (e.g., calculating NRMSE).
        val_period_size = max(2, int(0.2 * timeid_base))

        # Standardize the 'value' for each outcome (e.g., indicator) independently using only the pre-intervention period.
        # This is done by subtracting the mean and dividing by the standard deviation, calculated per outcome.
        # This puts all outcomes on a similar scale, which can improve model stability and performance.
        treated_train_df = dataset[dataset['timeid'] <= timeid_base - val_period_size]
        std_avg_dt = treated_train_df.groupby('outcome')['value'].agg(['mean', 'std']).reset_index()
        std_avg_dt.rename(columns={'mean': 'avg_value', 'std': 'std_value'}, inplace=True)
        std_avg_dt['std_value'] = std_avg_dt['std_value'].clip(lower=1e-10)  # Avoid div0

        # Apply to full dataset
        dataset = pd.merge(dataset, std_avg_dt, on='outcome', how='left')
        dataset['value'] = (dataset['value'] - dataset['avg_value']) / dataset['std_value']
        dataset.drop(columns=['avg_value', 'std_value'], inplace=True)

        # Separate the control unit data.
        control_df = dataset[dataset['treatment'] == 0]
        outcomes = sorted(dataset['outcome'].unique())
        N_outcomes = len(outcomes)

        # 3. PREPARE DATA FOR STAN
        # -----------------------------------------------------------------------------
        # Define a helper function to pivot the data from long format to wide format.
        # The wide format will have outcomes as columns and timeid/unitid as rows.
        def pivot_and_extract(df, time_range):
            if not time_range:
                return pd.DataFrame()
            subset = df[df['timeid'].isin(time_range)]
            if subset.empty:
                return pd.DataFrame()
            # The pivoted table is then stripped of its ID columns to create a pure numeric matrix for Stan.
            return subset.pivot_table(index=['timeid', 'unitid'], columns='outcome', values='value').reset_index().drop(columns=['timeid', 'unitid'])

        # Separate the dataset into treated and control unit data.
        treated_df = dataset[dataset['treatment'] == 1]

        # Create the data matrices for the treated unit for the post-treatment period.
        Y_treated_post = pivot_and_extract(treated_df, range(timeid_base + 1, timeid_fim + 1))
        N_post = len(Y_treated_post)

        # Create data matrices for the treated unit for the pre-treatment training period and the validation period.
        Y_treated_train = pivot_and_extract(treated_df, range(timeid_inicio, timeid_base - val_period_size + 1))
        Y_treated_val = pivot_and_extract(treated_df, range(timeid_base - val_period_size + 1, timeid_base + 1))

        # Create the control data matrices for the trainning, validation, and post-treatment periods.
        Y_control_train = get_control_matrices_by_outcome(control_df, range(timeid_inicio, timeid_base - val_period_size + 1), outcomes, unitids)
        Y_control_val = get_control_matrices_by_outcome(control_df, range(timeid_base - val_period_size + 1, timeid_base + 1), outcomes, unitids)
        Y_control_post = get_control_matrices_by_outcome(control_df, range(timeid_base + 1, timeid_fim + 1), outcomes, unitids)

        # Get counts of control units, trainning periods, and validation periods.
        N_controls = len(unitids)
        N_train = len(Y_treated_train)
        N_val = len(Y_treated_val)

        # Assemble all the data into a dictionary, which is the required format for input to a Stan model via CmdStanPy.
        # This dictionary contains all the data, dimensions, and hyperparameters for the model.
        stan_data = {
        'N_train': N_train,
        'N_val': N_val,
        'N_post': N_post,
        'N_controls': N_controls,
        'N_outcomes': N_outcomes,
        'Y_treated_train': Y_treated_train.to_numpy(),
        'Y_treated_val': Y_treated_val.to_numpy(),
        'Y_treated_post': Y_treated_post.to_numpy(),
        'Y_control_train': Y_control_train,
        'Y_control_val': Y_control_val,
        'Y_control_post': Y_control_post,
        'dirichlet_alpha': np.repeat(1.0, N_controls), # Hyperparameter for the Dirichlet prior on weights (uninformative).
        'tau_nrmse_prior': np.repeat(0.1, N_outcomes), # Hyperparameter for the prior on the NRMSE noise term.
        'noise_floor_fraction': 0.01
        }

        # 4. FIT THE STAN MODEL
        # -----------------------------------------------------------------------------
        # Fit the model using MCMC sampling.
        # 'data' is the input data dictionary.
        # 'seed' ensures reproducibility of the sampling process.
        # 'chains' specifies the number of independent MCMC chains to run.
        # 'parallel_chains' runs the chains in parallel to speed up computation.
        # 'iter_warmup' is the number of "burn-in" iterations to discard.
        # 'iter_sampling' is the number of posterior samples to keep from each chain.
        fit = model.sample(
            data=stan_data,
            seed=seed,
            chains=4,
            parallel_chains=4,
            iter_warmup=1000,
            iter_sampling=2000,
            refresh=500
        )

        os.makedirs(f"{workspace_folder}/{temp_folder}/solution_id_{solution_id}", exist_ok=True)

        # Print a summary of the R-hat diagnostic statistic. R-hat values close to 1.0
        # indicate that the MCMC chains have converged to the same posterior distribution.
        # fit_dt = fit.summary()

        # Convert the CmdStanPy fit object to an ArviZ InferenceData object.
        # EXPLICITLY specify the log_likelihood variable name so ArviZ can find it.
        inference_data = az.from_cmdstanpy(fit, log_likelihood="log_lik")
        
        fit_dt = az.summary(inference_data, var_names=["w", "sigma", "tau_nrmse", "hhi_weight"])
        
        print(fit_dt['r_hat'].describe())
        fit_dt.to_csv(f"{workspace_folder}/{temp_folder}/solution_id_{solution_id}/all_parameters_summary.csv", index=True)

        r_hat = fit_dt['r_hat'].mean()
        if 0.99 > r_hat or r_hat > 1.01:
            solution_has_problem = solution_has_problem + 'D'
            unacceptable_solutions.append(solution_id)

        # 1. Calculate LOO using ArviZ
        # pointwise=True allows us to inspect specific data points that might be outliers
        loo_results = az.loo(inference_data, pointwise=True) 

        # 2. Save LOO Summary statistics
        # Converts the LOO object to a series/dataframe for saving
        loo_summary_df = pd.DataFrame({
            'elpd_loo': [loo_results.elpd_loo],
            'p_loo': [loo_results.p_loo],
            'loo_ic': [loo_results.loo_i.values.sum() * -2], # Information Criterion scale
            'se': [loo_results.se],
            'warning': [loo_results.warning]
        })
        loo_summary_df.to_csv(f"{workspace_folder}/{temp_folder}/solution_id_{solution_id}/loo_summary_stats.csv", index=False)

        # 3. Save Pointwise diagnostics (Pareto k values AND loo_i contribution)
        # We must extract the values and create a Pandas DataFrame manually
        # because ArviZ returns xarray objects.
        pointwise_df = pd.DataFrame({
            'loo_i': loo_results.loo_i.values,     # The pointwise predictive density
            'pareto_k': loo_results.pareto_k.values # The diagnostic shape parameter
        })
        pointwise_df.to_csv(f"{workspace_folder}/{temp_folder}/solution_id_{solution_id}/loo_pointwise_diagnostics.csv", index_label="observation_index")

        # 4. Generate and Save Pareto k Diagnostic Plot
        # k > 0.7 indicates the model is overfitting to that specific data point
        plt.figure(figsize=(10, 6))
        az.plot_khat(loo_results)
        plt.title(f"PSIS-LOO Pareto k Diagnostics\n(Values > 0.7 indicate influential outliers)")
        plt.tight_layout()
        plt.savefig(f"{workspace_folder}/{temp_folder}/solution_id_{solution_id}/loo_pareto_k_plot.png")
        
        # 5. POST-PROCESSING AND VISUALIZATION
        # -----------------------------------------------------------------------------
        
        # EXTRACT SAMPLES AS NUMPY ARRAYS (Fast, no Regex, no Melt)
        # stan_vars is a dict where keys are variable names and values are ndarrays 
        # Shape: (N_draws, Dim1, Dim2, ...)
        stan_vars = fit.stan_variables()
        
        # Save full draws if strictly necessary (this is I/O bound, slows things down if kept)
        # If you don't need the raw debug CSV, comment the next line out for more speed.
        fit.save_csvfiles(dir=f"{workspace_folder}/{temp_folder}/solution_id_{solution_id}")

        # --- VALIDATION CHECKS (Vectorized) ---
        # Check NRMSE Terminal vs Train ratio
        # struc_nrmse_* shapes are (N_draws, N_outcomes)
        term_q975 = np.quantile(stan_vars['struc_nrmse_terminal'], 0.975, axis=0)
        train_q975 = np.quantile(stan_vars['struc_nrmse_train'], 0.975, axis=0)
        
        # Check if any outcome violates the NRMSE threshold
        ratio = term_q975 / train_q975
        if np.any(ratio > nrmse_terminal_over_train):
            solution_has_problem += "L"

        if solution_has_problem != "":
            os.rename(f"{workspace_folder}/{temp_folder}/solution_id_{solution_id}", 
                      f"{workspace_folder}/{temp_folder}/{solution_has_problem}_solution_id_{solution_id}")
            unacceptable_solutions.append(solution_id)
        else:
            # --- PREPARE DATA FOR PLOTTING (Vectorized) ---
            
            # Helper to process time-series variables (concatenating Train, Val, Post)
            def process_time_series(var_train, var_val, var_post):
                # Concatenate along time axis (axis 1) -> (Draws, Total_Time, Outcomes)
                arr = np.concatenate([stan_vars[var_train], stan_vars[var_val], stan_vars[var_post]], axis=1)
                # Compute stats: (3, Total_Time, Outcomes) -> Indices: 0=2.5%, 1=Mean, 2=97.5%
                stats = np.percentile(arr, [2.5, 50, 97.5], axis=0) 
                # Replace 50% percentile with strict Mean to match original logic
                stats[1] = np.mean(arr, axis=0)
                return stats

            # 1. Structural (Absolute)
            stats_struct = process_time_series('y_synth_train_scaled', 'y_synth_val_scaled', 'y_synth_post_scaled')
            
            # 2. Predictive (Absolute)
            stats_pred = process_time_series('predictive_train', 'predictive_val', 'predictive_post')
            
            # 3. Effects (Relative)
            # Combine residuals (pre) and effects (post)
            # residuals shape: (Draws, T_pre, Outcomes)
            # effect_post shape: (Draws, T_post, Outcomes)
            eff_arr = np.concatenate([
                stan_vars['residuals_train'], 
                stan_vars['residuals_val'], 
                stan_vars['effect_post']
            ], axis=1)
            
            stats_eff_time = np.percentile(eff_arr, [2.5, 50, 97.5], axis=0)
            stats_eff_time[1] = np.mean(eff_arr, axis=0)
            
            # Calculate Period Averages (Pre vs Post)
            # Pre: indices 0 to timeid_base-1
            # Post: indices timeid_base to end
            pre_arr = eff_arr[:, :timeid_base, :] # (Draws, T_pre, Outcomes)
            post_arr = eff_arr[:, timeid_base:, :] # (Draws, T_post, Outcomes)
            
            # Means per draw per period
            pre_means = np.mean(pre_arr, axis=1) # (Draws, Outcomes)
            post_means = np.mean(post_arr, axis=1) # (Draws, Outcomes)
            
            # Quantiles of the means
            stats_pre_period = np.array([np.percentile(pre_means, 2.5, axis=0), np.mean(pre_means, axis=0), np.percentile(pre_means, 97.5, axis=0)])
            stats_post_period = np.array([np.percentile(post_means, 2.5, axis=0), np.mean(post_means, axis=0), np.percentile(post_means, 97.5, axis=0)])

            # Container lists
            absolute_structural_timeid_list = []
            absolute_predictive_timeid_list = []
            att_period_list = []
            att_timeid_list = []

            # --- PLOTTING LOOP ---
            # Now we loop over outcomes, but the heavy math is already done.
            # We just slice the pre-calculated arrays.
            
            # Prepare Treated Data for plotting
            treated_df['timeid_mapped'] = treated_df['timeid'].apply(lambda x: mapping_timeid.tolist().index(x) + 1 if x in mapping_timeid else -1)
            
            # Convert integer timeids back to original string labels for merging
            # treated_df currently has int codes (1..N). mapping_timeid has the labels.
            treated_df = treated_df.copy()
            treated_df['timeid'] = treated_df['timeid'].apply(
                lambda x: mapping_timeid[x-1] if isinstance(x, (int, np.integer)) and 0 < x <= len(mapping_timeid) else np.nan
            )
            
            # Ensure proper types for merging
            treated_df['timeid'] = treated_df['timeid'].astype(str)
            
            for idx, current_indicator in enumerate(outcomes):
                os.makedirs(f"{workspace_folder}/{temp_folder}/solution_id_{solution_id}/{current_indicator}", exist_ok=True)

                # Scaling factors
                indicator_info = std_avg_dt[std_avg_dt['outcome'] == current_indicator]
                avg_val = indicator_info['avg_value'].iloc[0]
                std_val = indicator_info['std_value'].iloc[0]

                # --- 1. PREPARE DATAFRAMES FOR THIS OUTCOME ---
                
                # A. Structural
                # Slice the numpy array: (3, Total_Time, Outcomes) -> (3, Total_Time)
                curr_struct = stats_struct[:, :, idx] 
                # Unscale
                curr_struct_unscaled = curr_struct * std_val + avg_val
                
                df_struct = pd.DataFrame({
                    'timeid_idx': range(1, curr_struct.shape[1] + 1),
                    'q2_5': curr_struct_unscaled[0],
                    'mean': curr_struct_unscaled[1],
                    'q97_5': curr_struct_unscaled[2]
                })
                # Map back to real timeid strings
                df_struct['timeid'] = df_struct['timeid_idx'].apply(lambda x: mapping_timeid[x-1] if x <= len(mapping_timeid) else np.nan)
                df_struct['period'] = np.where(df_struct['timeid_idx'] > timeid_base, "Post", "Pre")
                
                # Merge observed treated data
                subset_treated = treated_df[treated_df['outcome'] == current_indicator]
                # Re-scale treated value to original scale
                # Note: 'value' in treated_df is currently scaled. 
                # We need raw value or unscale it. The code previously unscaled it in the loop.
                subset_treated_plot = subset_treated.copy()
                subset_treated_plot['treatment'] = subset_treated_plot['value'] * std_val + avg_val
                
                df_struct = pd.merge(df_struct, subset_treated_plot[['timeid', 'treatment']], on='timeid', how='left')
                
                # B. Predictive
                curr_pred = stats_pred[:, :, idx]
                curr_pred_unscaled = curr_pred * std_val + avg_val
                df_pred = pd.DataFrame({
                    'timeid_idx': range(1, curr_pred.shape[1] + 1),
                    'q2_5': curr_pred_unscaled[0],
                    'mean': curr_pred_unscaled[1],
                    'q97_5': curr_pred_unscaled[2]
                })
                df_pred['timeid'] = df_pred['timeid_idx'].apply(lambda x: mapping_timeid[x-1] if x <= len(mapping_timeid) else np.nan)
                df_pred['period'] = np.where(df_pred['timeid_idx'] > timeid_base, "Post", "Pre")
                df_pred = pd.merge(df_pred, subset_treated_plot[['timeid', 'treatment']], on='timeid', how='left')

                # C. ATT (Effects) - Relative
                curr_eff = stats_eff_time[:, :, idx] * std_val # Only multiply std (relative)
                
                df_eff = pd.DataFrame({
                    'timeid_idx': range(1, curr_eff.shape[1] + 1),
                    'q2_5': curr_eff[0],
                    'effect': curr_eff[1],
                    'q97_5': curr_eff[2]
                })
                df_eff['timeid'] = df_eff['timeid_idx'].apply(lambda x: mapping_timeid[x-1] if x <= len(mapping_timeid) else np.nan)
                df_eff['period'] = np.where(df_eff['timeid_idx'] > timeid_base, "Post", "Pre")

                # D. Period Stats
                # Pre
                df_period_pre = pd.DataFrame({
                    'period': ['Pre'],
                    'q2_5': [stats_pre_period[0, idx] * std_val],
                    'mean': [stats_pre_period[1, idx] * std_val],
                    'q97_5': [stats_pre_period[2, idx] * std_val]
                })
                # Post
                df_period_post = pd.DataFrame({
                    'period': ['Post'],
                    'q2_5': [stats_post_period[0, idx] * std_val],
                    'mean': [stats_post_period[1, idx] * std_val],
                    'q97_5': [stats_post_period[2, idx] * std_val]
                })
                df_period = pd.concat([df_period_pre, df_period_post])
                
                # Collect for final CSVs
                df_struct['outcome'] = current_indicator
                absolute_structural_timeid_list.append(df_struct)
                
                df_pred['outcome'] = current_indicator
                absolute_predictive_timeid_list.append(df_pred)
                
                df_eff['outcome'] = current_indicator
                att_timeid_list.append(df_eff) # Note: Column name is 'effect' not 'mean' here
                
                df_period['outcome'] = current_indicator
                att_period_list.append(df_period)

                # --- PLOTTING ---
                # (Logic remains similar, just using the new DataFrames)
                
                # Helper for tick marks
                num_unique_timeids = len(mapping_timeid)
                length_desired_breaks = math.ceil(num_unique_timeids / 40.0)
                time_breaks = mapping_timeid[::length_desired_breaks]

                # 1. Plot Structural
                p = (
                    ggplot(data=df_struct) +
                    geom_ribbon(aes(x="timeid", ymin="q2_5", ymax="q97_5", group=1, fill="'Uncertainty'"), alpha=0.2) +
                    scale_fill_manual(values={"Uncertainty": "gray"}) +
                    geom_vline(xintercept=timeid_base, linetype="dashed", color="red") + 
                    geom_line(aes(x="timeid", y="mean", group=1, color="'Synthetic Control'")) +
                    geom_line(aes(x="timeid", y="treatment", group=1, color="'Treatment'")) +
                    scale_color_manual(values={"Mean": "black", "Treatment": "blue"}) +
                    theme_classic() +
                    ggtitle(f"{current_indicator} - Low-Rank Trends - Structural View") +
                    ylab("Absolute Value") +
                    theme(axis_text_x=element_text(angle=45, hjust=1), legend_position="top", legend_title=element_blank(), plot_title=element_text(hjust=0)) +
                    scale_x_discrete(breaks=list(time_breaks), name="timeid")
                )
                p.save(f"{workspace_folder}/{temp_folder}/solution_id_{solution_id}/{current_indicator}/absolute structural view.png", width=14, height=6, dpi=300)

                # 2. Plot Predictive
                p = (
                    ggplot(data=df_pred) +
                    geom_ribbon(aes(x="timeid", ymin="q2_5", ymax="q97_5", group=1, fill="'Uncertainty'"), alpha=0.2) +
                    scale_fill_manual(values={"Uncertainty": "gray"}) +
                    geom_vline(xintercept=timeid_base, linetype="dashed", color="red") + 
                    geom_line(aes(x="timeid", y="mean", group=1, color="'Synthetic Control'")) +
                    geom_line(aes(x="timeid", y="treatment", group=1, color="'Treatment'")) +
                    scale_color_manual(values={"Mean": "black", "Treatment": "blue"}) +
                    theme_classic() +
                    ggtitle(f"{current_indicator} - Strict Parallel Trends - Predictive View") +
                    ylab("Absolute Value") +
                    theme(axis_text_x=element_text(angle=45, hjust=1), legend_position="top", legend_title=element_blank(), plot_title=element_text(hjust=0)) +
                    scale_x_discrete(breaks=list(time_breaks), name="timeid")
                )
                p.save(f"{workspace_folder}/{temp_folder}/solution_id_{solution_id}/{current_indicator}/absolute predictive view.png", width=14, height=6, dpi=300)

                # 3. Plot Effect (ATT)
                # Annotation Logic
                def format_annotation(df, p_name, fmt):
                    global parallel_trends_violated
                    row = df[df['period'] == p_name].iloc[0]
                    mn, low, upp = row['mean'], row['q2_5'], row['q97_5']
                    sig = "" if low < 0.0 < upp else "*"
                    if p_name == "Pre" and sig != "": parallel_trends_violated = True
                    prefix = "Parallel Trends: " if p_name == "Pre" else "ATT: "
                    return f"{prefix}{fmt.format(mn)}{sig} [{fmt.format(low)}, {fmt.format(upp)}]"

                x_pre_annotation = (timeid_base + 1) / 2
                x_post_annotation = (timeid_base + 1) + ((len(mapping_timeid) - (timeid_base + 1)) / 2)

                p = (
                    ggplot(df_eff, aes(x='factor(timeid)', y='effect', color='factor(period)')) +
                    geom_vline(xintercept=timeid_base, linetype="dashed", color="red") + 
                    geom_hline(yintercept=0.0, linetype="dashed", color="black") + 
                    geom_point(size=1.5) + 
                    geom_errorbar(aes(ymin='q2_5', ymax='q97_5'), width=0.2) + 
                    scale_color_manual(name=None, values={"Pre": "#e87d72", "Post": "#56bcc2"}, breaks=['Pre', 'Post']) +
                    ggtitle(current_indicator) +
                    theme_classic() +
                    scale_x_discrete(breaks=list(time_breaks), name="timeid") +
                    theme(axis_text_x=element_text(angle=45, hjust=1), legend_position="top", legend_title=element_blank(), plot_title=element_text(hjust=0)) +
                    annotate("text", x=x_pre_annotation, y=df_eff['q97_5'].max(),
                            label=format_annotation(df_period, "Pre", period_effect_format), color="black", size=10, fontweight='bold') +
                    annotate("text", x=x_post_annotation, y=df_eff['q97_5'].max(),
                            label=format_annotation(df_period, "Post", period_effect_format), color="black", size=10, fontweight='bold') +
                    ylab("Effect Magnitude")
                )
                p.save(f"{workspace_folder}/{temp_folder}/solution_id_{solution_id}/{current_indicator}/effect.png", width=14, height=6, dpi=300)

                # 4. Forest Plots (ArviZ)
                # We use the original 'fit' object for this as ArviZ handles it well enough, 
                # but we constrain coords to just the current index to save rendering time.
                # Note: plot_forest is somewhat slow. If strict speed is needed, this can be skipped or optimized.
                az.plot_forest(
                    fit,
                    var_names=["nrmse_train", "nrmse_val_minus_two", "nrmse_terminal"],
                    combined=True,
                    figsize=(25, 10),
                    hdi_prob=0.95,
                    coords={"nrmse_train_dim_0": [idx], "nrmse_val_minus_two_dim_0": [idx], "nrmse_terminal_dim_0": [idx]}
                )
                plt.suptitle("Strict Parallel Trends - Predictive NRMSE", fontsize=16)
                plt.tight_layout()
                plt.savefig(f"{workspace_folder}/{temp_folder}/solution_id_{solution_id}/{current_indicator}/predictive_nrmse.png")
                plt.close()

                az.plot_forest(
                    fit,
                    var_names=["struc_nrmse_train", "struc_nrmse_val_minus_two", "struc_nrmse_terminal"],
                    combined=True,
                    figsize=(25, 10),
                    hdi_prob=0.95,
                    coords={"struc_nrmse_train_dim_0": [idx], "struc_nrmse_val_minus_two_dim_0": [idx], "struc_nrmse_terminal_dim_0": [idx]}
                )
                plt.suptitle("Low-Rank Trends - Structural NRMSE", fontsize=16)
                plt.tight_layout()
                plt.savefig(f"{workspace_folder}/{temp_folder}/solution_id_{solution_id}/{current_indicator}/structural_nrmse.png")
                plt.close()

            # --- CONSOLIDATE AND SAVE ---
            absolute_structural_timeid_df = pd.concat(absolute_structural_timeid_list)
            shape_report = classify_structural_uncertainty_shape(absolute_structural_timeid_df)
            shape_report.to_csv(f"{workspace_folder}/{temp_folder}/solution_id_{solution_id}/structural_uncertainty_shape_report.csv", index=False)                
            
            absolute_predictive_timeid_df = pd.concat(absolute_predictive_timeid_list)
            att_period_df = pd.concat(att_period_list)
            att_timeid_df = pd.concat(att_timeid_list)
            
            absolute_structural_timeid_df.to_csv(f"{workspace_folder}/{temp_folder}/solution_id_{solution_id}/absolute_structural_timeid_dt.csv", index=False)
            absolute_predictive_timeid_df.to_csv(f"{workspace_folder}/{temp_folder}/solution_id_{solution_id}/absolute_predictive_timeid_dt.csv", index=False)
            att_period_df.to_csv(f"{workspace_folder}/{temp_folder}/solution_id_{solution_id}/att_period_dt.csv", index=False)
            att_timeid_df.to_csv(f"{workspace_folder}/{temp_folder}/solution_id_{solution_id}/att_timeid_dt.csv", index=False)

            # --- ARVIZ PLOTS (Weights) ---
            inference_data = az.from_cmdstanpy(fit)

            # --- Create a customized density plot for key parameters ---
            # Define a color map for the MCMC chains.
            cmap = plt.get_cmap("Blues")
            legend_colors = [cmap(x) for x in np.linspace(0.9, 0.4, 4)]

            # Use ArviZ to create a grid of density plots for the weights (w), sigma, and tau parameters.
            # This plot is excellent for visually checking chain convergence and the shape of posterior distributions.
            axes = az.plot_density(
                [inference_data.sel(chain=[0]), inference_data.sel(chain=[1]),
                inference_data.sel(chain=[2]), inference_data.sel(chain=[3])],
                var_names=["sigma", "tau_nrmse", "hhi_weight"],
                shade=0.0,
                hdi_prob=0.95,
                point_estimate="mean",
                colors=legend_colors,
                figsize=(14, 6)
            )
            # Remove the default legends generated by ArviZ for each subplot.
            for ax in axes.flatten():
                if ax.get_legend() is not None:
                    ax.get_legend().remove()

            # Customize the plot aesthetics.
                                                        
            fig = axes.flatten()[0].get_figure()
            fig.set_constrained_layout(True) 

            # Create user-friendly titles for each subplot.
            plot_titles = [name for name in az.summary(inference_data).index.tolist() if name.startswith(("sigma", "tau_nrmse"))]
            new_s_names = {f"sigma[{index}]": f"sigma[{name}]" for index, name in enumerate(outcomes)}
            new_t_names = {f"tau_nrmse[{index}]": f"tau_nrmse[{name}]" for index, name in enumerate(outcomes)}
            new_names = new_s_names | new_t_names
            plot_titles = [new_names.get(name, name) for name in plot_titles]

            for i, ax in enumerate(axes.flatten()):
                if i < len(plot_titles):
                    ax.set_title(plot_titles[i], fontsize=12)
                    ax.tick_params(axis='x', labelsize=12)

            # Create a single, custom legend for the entire figure.
            legend_handles = [Line2D([0], [0], color=c, lw=2, label=str(i+1)) for i, c in enumerate(legend_colors)]
                                                    
            axes.flatten()[2].legend(
                handles=legend_handles,
                title="Chain",
                loc='upper right',
                bbox_to_anchor=(1.8, 1.0), # Position legend outside the plot area.
                frameon=False,
                fontsize=12,
                title_fontsize=12
            )
            plt.savefig(f"{workspace_folder}/{temp_folder}/solution_id_{solution_id}/unitid_weight_individual_distribution_sigma.png")
            plt.close()

            # Forest Plot (Weights)
            az.plot_forest(
                fit, var_names="w", combined=True, figsize=(14, max(6, N_controls * 0.3)),
                hdi_prob=0.95, labeller=az.labels.MapLabeller(coord_map={"w_dim_0": {i: n for i, n in enumerate(unitids)}})
            )
            plt.suptitle("Posterior Distribution of Control Unit Weights (w)", fontsize=16)
            plt.savefig(f"{workspace_folder}/{temp_folder}/solution_id_{solution_id}/unitid_weight_simultanous_distribution.png")
            plt.close('all')

            # --- FILE RENAMING LOGIC ---
            if solution_has_problem == "":
                if parallel_trends_violated:
                    os.rename(f"{workspace_folder}/{temp_folder}/solution_id_{solution_id}", f"{workspace_folder}/{temp_folder}/P_solution_id_{solution_id}")
                    unacceptable_solutions.append(solution_id)
                else:
                    # Calculate NRMSE Score (Vectorized)
                    # Uval + abs(Uval - Utrain)
                    u_val = np.quantile(stan_vars['struc_nrmse_val_minus_two'], 0.975, axis=0) # shape (N_outcomes,)
                    u_train = np.quantile(stan_vars['struc_nrmse_train'], 0.975, axis=0)
                    scores = u_val + np.abs(u_val - u_train)
                    predictive_reliability_score = np.max(scores)
                    
                    hhi_mean = np.mean(stan_vars['hhi_weight'])
                    
                    os.rename(f"{workspace_folder}/{temp_folder}/solution_id_{solution_id}", 
                              f"{workspace_folder}/{temp_folder}/R{predictive_reliability_score:1.3f}_S{shape_report['relative_width'].max():1.3f}_H{hhi_mean:.2f}_K{max(loo_results.pareto_k.values):.2f}_solution_id_{solution_id}")
            else:
                prefix = f"{solution_has_problem}P" if parallel_trends_violated else solution_has_problem
                os.rename(f"{workspace_folder}/{temp_folder}/solution_id_{solution_id}", 
                          f"{workspace_folder}/{temp_folder}/{prefix}_solution_id_{solution_id}")

            # Collect Weights for final table
            # Compute mean weights directly from numpy array
            w_means = np.mean(stan_vars['w'], axis=0) # Shape (N_controls,)
            temp_weight_solutions = pd.DataFrame({
                'unitid': unitids,
                'value': w_means
            })
            
            weights_dict = temp_weight_solutions.set_index('unitid')['value'].to_dict()
            
            temp_weight_solutions['Unrmse'] = predictive_reliability_score
            temp_weight_solutions['solution_id'] = solution_id

            weight_solutions_list.append(temp_weight_solutions)

            # 1. Metrics
            nrmse_score = predictive_reliability_score # Calculated previously in your code
            synth_score = shape_report['relative_width'].max()

            # 2. Effects (Formatting helper)
            def format_effect_string(mean, lower, upper):
                # Check significance (0 not in interval)
                sig = "*" if (lower > 0 or upper < 0) else ""
                return f"{mean:1.2f}{sig} [{lower:1.2f} {upper:1.2f}]"

            effect_rows = []
            
            # Group outcomes by category if they have prefixes (e.g., "Limited service - average wage")
            # Assuming outcome names in data are like "Limited service - average weekly wage"
            for outcome_name in outcomes:
                # Parse Category and Metric from outcome name if possible, else use defaults
                parts = outcome_name.split(' - ')
                if len(parts) >= 2:
                    category, metric_name = parts[0], parts[1]
                else:
                    category, metric_name = outcome_name, "ATT"

                # A. Post-Intervention Average
                # Filter att_period_df calculated previously
                post_data = att_period_df[(att_period_df['outcome'] == outcome_name) & (att_period_df['period'] == 'Post')].iloc[0]
                val_post = format_effect_string(post_data['mean'], post_data['q2_5'], post_data['q97_5'])
                
                effect_rows.append({
                    'Category': category, 'Item': metric_name, 
                    'Description': 'ATT post-intervention', 'Value': val_post
                })

                # B. Spot Check Date (e.g., 2016Q1)
                # Filter att_timeid_df calculated previously
                # We need to map the string 'spot_check_timeid' to the integer/mapped ID used in att_timeid_df
                
                # Check if the specific date exists in the results
                spot_check_timeid = att_timeid_df['timeid'].iloc[-1]
                spot_data = att_timeid_df[(att_timeid_df['outcome'] == outcome_name) & (att_timeid_df['timeid'] == spot_check_timeid)]
                
                if not spot_data.empty:
                    row = spot_data.iloc[0]
                    val_spot = format_effect_string(row['effect'], row['q2_5'], row['q97_5'])
                    effect_rows.append({
                        'Category': category, 'Item': metric_name, 
                        'Description': f'ATT {spot_check_timeid}', 'Value': val_spot
                    })
                else:
                    # Fallback if date not found
                    effect_rows.append({
                        'Category': category, 'Item': metric_name, 
                        'Description': f'ATT {spot_check_timeid}', 'Value': "N/A"
                    })

            # Store everything
            summary_data_collection.append({
                'solution_id': solution_id,
                'metrics': {
                    'NRMSE': nrmse_score,
                    'Synth': synth_score,
                    'HHI': hhi_mean
                },
                'effects': effect_rows,
                'weights': weights_dict
            })

    if weight_solutions_list:
        weight_solutions = pd.concat(weight_solutions_list, ignore_index=True)
    else:
        weight_solutions = pd.DataFrame()

    if weight_solutions.shape[0] > 0:
        # Define the rows structure as per the image
        final_rows = []

        # 1. Header Metrics Rows
        metric_defs = [
            ('NRMSE', 'Predictive Upper Bound', 'Uval + abs(Uval - Utrain)', 'NRMSE', '{:.3f}'),
            ('Synth', 'Structural Alignment', 'Max Relative Width', 'Synth', '{:.3f}'),
            ('HHI', 'Weight Concentration', 'Mean', 'HHI', '{:.3f}')
        ]

        # Get valid solution IDs (those that weren't filtered out by unacceptable_solutions if applicable, 
        # though the image shows all provided IDs). Let's use all collected data.
        
        # Sort collected data by solution_id to match columns or custom sort
        # The image implies no specific sort, but we usually sort by ID or NRMSE. 
        # Let's keep input order.
        
        # Extract unique Solution IDs for columns
        sol_ids = [d['solution_id'] for d in summary_data_collection]

        # --- Section 1: Standard Metrics ---
        for cat, met, sub, key, fmt in metric_defs:
            row = {'Category': cat, 'Item': met, 'Description': sub}
            for data in summary_data_collection:
                val = data['metrics'][key]
                row[data['solution_id']] = fmt.format(val)
            final_rows.append(row)

        # --- Section 2: Effects (ATT) ---
        # We need to extract the structure from the first solution to know the rows
        if summary_data_collection:
            first_sol_effects = summary_data_collection[0]['effects']
            for eff_def in first_sol_effects:
                row = {
                    'Category': eff_def['Category'], 
                    'Item': eff_def['Item'], 
                    'Description': eff_def['Description']
                }
                # Fill values for all solutions
                for data in summary_data_collection:
                    # Find matching effect for this solution
                    match = next((x for x in data['effects'] 
                                  if x['Category'] == eff_def['Category'] 
                                  and x['Item'] == eff_def['Item']
                                  and x['Description'] == eff_def['Description']), None)
                    row[data['solution_id']] = match['Value'] if match else ""
                final_rows.append(row)

        # --- Section 3: Weights ---
        # Get all unique unitids across all solutions
        all_unitids = sorted(list(set().union(*[d['weights'].keys() for d in summary_data_collection])))
        
        for uid in all_unitids:
            row = {'Category': 'unitid', 'Item': uid, 'Description': ''}
            has_val = False
            for data in summary_data_collection:
                w_val = data['weights'].get(uid, 0.0)
                # Only show weight if > threshold (e.g. 0.01) or per image logic (blanks for very small?)
                # Image shows empty cells. Let's assume threshold of 0.01
                if w_val >= 0.01:
                    row[data['solution_id']] = "{:.2f}".format(w_val)
                    has_val = True
                else:
                    row[data['solution_id']] = ""
            
            # Only add row if at least one solution uses this unit (optional, but cleaner)
            if has_val:
                final_rows.append(row)

        # Create DataFrame
        final_df = pd.DataFrame(final_rows)

        # Reorder columns: Category, Metric, Description, then Solution IDs
        cols = ['Category', 'Item', 'Description'] + sol_ids
        final_df = final_df[cols]
        
        row_index_to_sort_by = 0
        ascending_order = True
        fixed_cols = final_df.columns[:3].tolist()
        sortable_cols_df = final_df.iloc[[row_index_to_sort_by], 3:]
        sorted_columns = sortable_cols_df.sort_values(
            by=row_index_to_sort_by, 
            axis=1, 
            ascending=ascending_order
        ).columns.tolist()
        new_column_order = fixed_cols + sorted_columns
        final_df = final_df[new_column_order]
        
        # Save
        final_csv_path = f"{workspace_folder}/{temp_folder}/final_summary_table.csv"
        final_df.to_csv(final_csv_path, index=False)
