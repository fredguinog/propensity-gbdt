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
Classifies the structural uncertainty band into 4 types:
1. Funnel (Good): Starts wide, narrows before intervention.
2. Trumpet (Bad): Starts narrow, widens before intervention.
3. Homogenous (Neutral): Roughly constant width.
4. Breather (Noisy): Oscillates with no clear trend.
"""
def classify_structural_uncertainty_shape(df, outcome_col='outcome'):    
    results = []
        
    # Get list of outcomes
    outcomes = df[outcome_col].unique()
    
    for outcome in outcomes:
        subset = df[df[outcome_col] == outcome].copy()
        
        # Sort by time to ensure correlation works
        subset = subset.sort_values('timeid')
        
        # 2. Calculate the Width of the Gray Band
        # Width = Upper Bound (q97_5) - Lower Bound (q2_5)
        subset['band_width'] = subset['q97_5'] - subset['q2_5']
        
        widths = subset['band_width'].values
        time_steps = np.arange(len(widths))
        
        # 3. Calculate Metrics
        
        # A. Trend (Pearson Correlation)
        # Is the width consistently growing or shrinking over time?
        corr, _ = pearsonr(time_steps, widths)
        
        # B. Volatility (Coefficient of Variation)
        # How much does the width fluctuate relative to its size?
        # CV = Std_Dev / Mean
        cv = np.std(widths) / np.mean(widths)
        
        # 4. Classification Logic (Thresholds can be tuned)
        shape_class = "Unknown"
        quality_score = 0 # Simple score to help ranking (Higher is better)
        
        # Thresholds
        CORR_THRESHOLD = 0.5  # Strong enough trend to be called Funnel/Trumpet
        CV_THRESHOLD = 0.5    # High enough volatility to be called Breather
        
        if corr < -CORR_THRESHOLD:
            shape_class = "Funnel (Ideal)"
            # Good because uncertainty decreases near intervention
            quality_score = 10 
        elif corr > CORR_THRESHOLD:
            shape_class = "Trumpet (Risky)"
            # Bad because uncertainty is highest right when we need precision
            quality_score = 1 
        else:
            # If no strong trend, check volatility
            if cv > CV_THRESHOLD:
                shape_class = "Breather (Noisy)"
                quality_score = 2
            else:
                shape_class = "Homogenous (Stable)"
                quality_score = 7

        results.append({
            'outcome': outcome,
            'shape': shape_class,
            'correlation': round(corr, 3),
            'volatility_cv': round(cv, 3),
            'avg_width': round(np.mean(widths), 4),
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
    maximum_gram_cond (float, optional): The maximum allowable value for the Gram
                                         matrix condition number. This is used as a
                                         threshold to detect multicollinearity among
                                         control units; solutions exceeding it are flagged.
                                         Defaults to 100.0.
    maximum_mean_gini_weights (float, optional): The maximum permissible mean gini of the weight distribution
                                          for the synthetic control. This helps prevent the model
                                          from relying too heavily on one control unit.
                                          Defaults to 0.6.
"""
def estimate(
    timeid_previous_intervention,
    workspace_folder,
    solution_id=None,
    period_effect_format='{:.2f}',
    seed=222,
    maximum_gram_cond=100.0,
    maximum_mean_gini_weights=0.6
):
    # Set the random seed for NumPy to ensure reproducibility of its random operations.
    np.random.seed(seed)
    
    global parallel_trends_violated

    # Remove the last char from workspace_folder if it is /
    if workspace_folder.endswith('/') or workspace_folder.endswith('\\'):
        workspace_folder = workspace_folder[:-1]

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

    unacceptable_solutions = []
    weight_solutions = pd.DataFrame()
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

        impact_score = dataset[dataset['treatment'] == 0]['impact_score'].unique()[0]

        # Keep only the essential columns for the analysis.
        dataset = dataset[["outcome", "timeid", "value", "treatment", "unitid"]]

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
        timeid_fim = dataset['timeid'].max()

        # Define a "test" period using the last 20% of the pre-treatment data with minimum of 2. This period is used for model validation (e.g., calculating NRMSE).
        test_period_size = max(2, int(0.2 * timeid_base))

        # Standardize the 'value' for each outcome (e.g., indicator) independently using only the pre-intervention period.
        # This is done by subtracting the mean and dividing by the standard deviation, calculated per outcome.
        # This puts all outcomes on a similar scale, which can improve model stability and performance.
        treated_pre_df = dataset[dataset['timeid'] <= timeid_base - test_period_size]
        std_avg_dt = treated_pre_df.groupby('outcome')['value'].agg(['mean', 'std']).reset_index()
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

        gram_cond_max = calculate_gram_cond_by_shape(df = dataset[(dataset['treatment'] == 0) & (dataset['timeid'] <= timeid_base - test_period_size)])
        if maximum_gram_cond < gram_cond_max:
            unacceptable_solutions.append(solution_id)
            os.rename(f"{workspace_folder}/{temp_folder}/solution_id_{solution_id}", f"{workspace_folder}/{temp_folder}/M_solution_id_{solution_id}")
        else:
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

            # Create data matrices for the treated unit for the pre-treatment training period and the test period.
            Y_treated_pre = pivot_and_extract(treated_df, range(timeid_inicio, timeid_base - test_period_size + 1))
            Y_treated_test = pivot_and_extract(treated_df, range(timeid_base - test_period_size + 1, timeid_base + 1))

            # Create the control data matrices for the pre-treatment, test, and post-treatment periods.
            Y_control_pre = get_control_matrices_by_outcome(control_df, range(timeid_inicio, timeid_base - test_period_size + 1), outcomes, unitids)
            Y_control_test = get_control_matrices_by_outcome(control_df, range(timeid_base - test_period_size + 1, timeid_base + 1), outcomes, unitids)
            Y_control_post = get_control_matrices_by_outcome(control_df, range(timeid_base + 1, timeid_fim + 1), outcomes, unitids)

            # Get counts of control units, pre-treatment periods, and test periods.
            N_controls = len(unitids)
            N_pre = len(Y_treated_pre)
            N_test = len(Y_treated_test)

            # Assemble all the data into a dictionary, which is the required format for input to a Stan model via CmdStanPy.
            # This dictionary contains all the data, dimensions, and hyperparameters for the model.
            stan_data = {
            'N_pre': N_pre,
            'N_test': N_test,
            'N_post': N_post,
            'N_controls': N_controls,
            'N_outcomes': N_outcomes,
            'Y_treated_pre': Y_treated_pre.to_numpy(),
            'Y_treated_test': Y_treated_test.to_numpy(),
            'Y_treated_post': Y_treated_post.to_numpy(),
            'Y_control_pre': Y_control_pre,
            'Y_control_test': Y_control_test,
            'Y_control_post': Y_control_post,
            'dirichlet_alpha': np.repeat(1.0, N_controls), # Hyperparameter for the Dirichlet prior on weights (uninformative).
            'tau_nrmse_prior': np.repeat(0.1, N_outcomes) # Hyperparameter for the prior on the NRMSE noise term.
            }

            # 4. FIT THE STAN MODEL
            # -----------------------------------------------------------------------------
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

            os.makedirs(f"{workspace_folder}/{temp_folder}/solution_id_{solution_id}", exist_ok=True)

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

            # Print a summary of the R-hat diagnostic statistic. R-hat values close to 1.0
            # indicate that the MCMC chains have converged to the same posterior distribution.
            print(fit.summary()['R_hat'].describe())
            fit.summary().to_csv(f"{workspace_folder}/{temp_folder}/solution_id_{solution_id}/all_parameters_summary.csv", index=True)

            r_hat = fit.summary()['R_hat'].mean()
            if 0.99 > r_hat or r_hat > 1.01:
                solution_has_problem = solution_has_problem + 'D'
                unacceptable_solutions.append(solution_id)

            # Convert the CmdStanPy fit object to an ArviZ InferenceData object.
            # EXPLICITLY specify the log_likelihood variable name so ArviZ can find it.
            inference_data = az.from_cmdstanpy(fit, log_likelihood="log_lik")
            
            # 1. Calculate LOO using ArviZ
            # pointwise=True allows us to inspect specific data points that might be outliers
            loo_results = az.loo(inference_data, pointwise=True)
            
            if (loo_results.warning):
                solution_has_problem = solution_has_problem + 'O'  

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
            plt.close() # Close to free memory
            
            # 5. POST-PROCESSING AND VISUALIZATION
            # -----------------------------------------------------------------------------
            # Extract the MCMC draws (posterior samples) into a pandas DataFrame for easier manipulation.
            draws_df = fit.draws_pd()
            
            chains_w_df = draws_df.melt(
                id_vars=['chain__', 'iter__'],
                value_vars=[col for col in draws_df.columns if any(col.startswith(v) for v in "w")],
                var_name='variable'
            )
            chains_w_df.rename(columns={'chain__': 'chain', 'iter__': 'iteration'}, inplace=True)

            # The variable names from Stan (e.g., "effect_post[1,2]") contain index information.
            # This code extracts the variable name and its indices into separate columns.
            extracted_info = chains_w_df['variable'].str.extract(r'(\w+)\[(\d+)\]')
            chains_w_df['var_name'] = extracted_info[0]
            chains_w_df['index_1'] = pd.to_numeric(extracted_info[1]) # Corresponds to unitid

            # Map the first index back to the unitid name.
            chains_w_df['unitid'] = chains_w_df['index_1'].apply(lambda x: unitids[x-1])

            temp_weight_solutions = chains_w_df.groupby('unitid').agg({
                'value' : 'mean'
            }).reset_index()
            temp_weight_solutions['impact_score'] = impact_score
            temp_weight_solutions['solution_id'] = solution_id

            weight_solutions = pd.concat([weight_solutions, temp_weight_solutions])

            if (draws_df['gini_weight'].mean() > maximum_mean_gini_weights):
                solution_has_problem = solution_has_problem + 'W'         
                
            if solution_has_problem != "":
                os.rename(f"{workspace_folder}/{temp_folder}/solution_id_{solution_id}", f"{workspace_folder}/{temp_folder}/{solution_has_problem}_solution_id_{solution_id}")
                unacceptable_solutions.append(solution_id)
            else:
                # Filter for variables related to the treatment effect (residuals and effects) and reshape the data.
                # The 'melt' function converts the DataFrame from wide format (one column per parameter instance)
                # to long format, which is more convenient for grouping and plotting.
                vars_to_extract = ["residuals_pre", "residuals_test", "effect_post",
                                   "y_synth_pre_scaled", "y_synth_test_scaled", "y_synth_post_scaled",
                                   "predictive_pre", "predictive_test", "predictive_post"]
                chains_effect_df = draws_df.melt(
                    id_vars=['chain__', 'iter__'],
                    value_vars=[col for col in draws_df.columns if any(col.startswith(v) for v in vars_to_extract)],
                    var_name='variable'
                )
                chains_effect_df.rename(columns={'chain__': 'chain', 'iter__': 'iteration'}, inplace=True)

                # The variable names from Stan (e.g., "effect_post[1,2]") contain index information.
                # This code extracts the variable name and its indices into separate columns.
                extracted_info = chains_effect_df['variable'].str.extract(r'(\w+)\[(\d+),(\d+)\]')
                chains_effect_df['var_name'] = extracted_info[0]
                chains_effect_df['index_1'] = pd.to_numeric(extracted_info[1]) # Corresponds to time
                chains_effect_df['index_2'] = pd.to_numeric(extracted_info[2]) # Corresponds to outcome

                # Define a function to reconstruct the original 'timeid' from the Stan indices.
                # This requires knowing the structure of the pre, test, and post periods.
                def get_timeid(row):
                    if row['var_name'] == 'residuals_pre' or row['var_name'] == 'y_synth_pre_scaled' or row['var_name'] == 'predictive_pre':
                        return row['index_1'] + timeid_inicio - 1
                    elif row['var_name'] == 'residuals_test' or row['var_name'] == 'y_synth_test_scaled' or row['var_name'] == 'predictive_test':
                        return row['index_1'] + timeid_inicio + N_pre - 1
                    elif row['var_name'] == 'effect_post' or row['var_name'] == 'y_synth_post_scaled' or row['var_name'] == 'predictive_post':
                        return row['index_1'] + timeid_base
                    return -1

                # Apply the function to create a 'timeid' column with the original time labels.
                chains_effect_df['timeid'] = chains_effect_df.apply(get_timeid, axis=1)
                chains_effect_df['timeid'] = chains_effect_df['timeid'].apply(lambda x: mapping_timeid[x-1] if x > 0 and pd.notna(x) else -1)
                chains_effect_df['timeid'] = np.where(chains_effect_df['timeid'] == -1, np.nan, chains_effect_df['timeid'])
                # Map the second index back to the outcome name.
                chains_effect_df['outcome'] = chains_effect_df['index_2'].apply(lambda x: outcomes[x-1])
                chains_effect_df['type'] = np.where((chains_effect_df['var_name'] == 'residuals_pre') | (chains_effect_df['var_name'] == 'residuals_test') | (chains_effect_df['var_name'] == 'effect_post'), 'relative',
                                                    np.where((chains_effect_df['var_name'] == 'y_synth_pre_scaled') | (chains_effect_df['var_name'] == 'y_synth_test_scaled') | (chains_effect_df['var_name'] == 'y_synth_post_scaled'), 'absolute_structural',
                                                             np.where((chains_effect_df['var_name'] == 'predictive_pre') | (chains_effect_df['var_name'] == 'predictive_test') | (chains_effect_df['var_name'] == 'predictive_post'), 'absolute_predictive', 'error')))
                # Drop the now redundant columns.
                chains_effect_df.drop(columns=['variable', 'var_name', 'index_1', 'index_2'], inplace=True)

                # Define a function to create formatted text annotations for the plots.
                # This text will summarize the average effect for the pre and post periods,
                # including a 95% credible interval and a significance star.
                def format_annotation(df, p, period_effect_format):
                    global parallel_trends_violated
                    
                    row = df[df['period'] == p].iloc[0]
                    mean_value = row['mean']
                    lower_value = row['q2_5']
                    upper_value = row['q97_5']
                    # A '*' indicates significance if the 95% credible interval does not contain zero.
                    significative = "" if lower_value < 0.0 < upper_value else "*"
                    
                    if p == "Pre" and significative != "":
                        parallel_trends_violated = True
                    
                    prefix = "Parallel Trends: " if p == "Pre" else "ATT: "
                    
                    return (f"{prefix}"
                            f"{period_effect_format.format(mean_value)} "
                            f"[{period_effect_format.format(lower_value)}, "
                            f"{period_effect_format.format(upper_value)}]"
                            f"{significative}")


                # Initialize lists to store summary statistics from each outcome's plot.
                absolute_structural_timeid_list = []
                absolute_predictive_timeid_list = []
                att_period_list = []
                att_timeid_list = []

                chains_relative_effect_df = chains_effect_df[chains_effect_df['type'] == 'relative']
                chains_relative_effect_df.drop(['type'], axis=1, inplace=True)

                treated_df['timeid'] = treated_df['timeid'].apply(lambda x: mapping_timeid[x-1] if x > 0 and pd.notna(x) else -1)

                chains_absolute_structural_effect_df = chains_effect_df[chains_effect_df['type'] == 'absolute_structural']
                chains_absolute_structural_effect_df.drop(['type'], axis=1, inplace=True)

                chains_absolute_predictive_effect_df = chains_effect_df[chains_effect_df['type'] == 'absolute_predictive']
                chains_absolute_predictive_effect_df.drop(['type'], axis=1, inplace=True)

                # Loop through each outcome to generate and save individual plots.
                for idx, current_indicator in enumerate(outcomes):
                    # Create a directory for the current outcome to save its plots, if it doesn't already exist.
                    os.makedirs(f"{workspace_folder}/{temp_folder}/solution_id_{solution_id}/{current_indicator}", exist_ok=True)

                    # Retrieve the mean and standard deviation for the current outcome to un-standardize the results.
                    indicator_info = std_avg_dt[std_avg_dt['outcome'] == current_indicator]
                    avg_value = indicator_info['avg_value'].iloc[0]
                    std_value = indicator_info['std_value'].iloc[0]

                    treated_subset = treated_df[treated_df['outcome'] == current_indicator].copy()
                    treated_subset['value'] = treated_subset['value'] * std_value + avg_value
                    treated_subset = treated_subset[['timeid', 'value']]
                    treated_subset.columns = ['timeid', 'treatment']
                    
                    # ABSOLUTE STRUCTURAL
                    # Filter the time series data for the current outcome.
                    absolute_structural_subset = chains_absolute_structural_effect_df[chains_absolute_structural_effect_df['outcome'] == current_indicator].copy()
                    # Create a 'period' column to distinguish between pre- and post-treatment time points.
                    absolute_structural_subset['period'] = np.where(absolute_structural_subset['timeid'].astype(str) > str(mapping_timeid[timeid_base-1]), "Post", "Pre")
                    # Rescale the 'value' (the effect) back to the original units by multiplying by the standard deviation.
                    # The mean is not added back because we are looking at effects (differences), where the mean cancels out.
                    absolute_structural_subset['value'] = absolute_structural_subset['value'] * std_value + avg_value

                    absolute_structural_timeid = absolute_structural_subset.groupby(['timeid', 'period'])['value'].agg(
                        q2_5=lambda x: x.quantile(0.025),
                        mean='mean',
                        q97_5=lambda x: x.quantile(0.975)
                    ).reset_index()

                    absolute_structural_timeid = pd.merge(absolute_structural_timeid, treated_subset, on=['timeid'], how='left')

                    # Determine the tick marks for the time axis to avoid overcrowding.
                    num_unique_timeids = len(chains_absolute_structural_effect_df['timeid'].unique())
                    length_desired_breaks = math.ceil(num_unique_timeids / 40.0)
                    time_breaks = mapping_timeid[::length_desired_breaks]

                    p = (
                        ggplot(data=absolute_structural_timeid) +
                        geom_ribbon(aes(x="timeid", ymin="q2_5", ymax="q97_5", group=1, fill="'Uncertainty'"), alpha=0.2) +
                        scale_fill_manual(values={"Uncertainty": "gray"}) +
                        geom_vline(xintercept=timeid_base, linetype="dashed", color="red") + # Vertical line at the intervention point.
                        geom_line(aes(x="timeid", y="mean", group=1, color="'Synthetic Control'")) +
                        geom_line(aes(x="timeid", y="treatment", group=1, color="'Treatment'")) +
                        scale_color_manual(values={"Mean": "black", "Treatment": "blue"}) +
                        theme_classic() +
                        ggtitle(f"{current_indicator} - Low-Rank Trends - Structural View") +
                        ylab("Absolute Value") +
                        theme(axis_text_x=element_text(angle=45, hjust=1), legend_position="top", legend_title=element_blank(), plot_title=element_text(hjust=0)) +
                        scale_x_discrete(breaks=list(time_breaks), name="timeid")
                    )
                    # Save the plot to a file.
                    p.save(f"{workspace_folder}/{temp_folder}/solution_id_{solution_id}/{current_indicator}/absolute structural view.png", width=14, height=6, dpi=300)

                    # ABSOLUTE PREDICTIVE
                    # Filter the time series data for the current outcome.
                    absolute_predictive_subset = chains_absolute_predictive_effect_df[chains_absolute_predictive_effect_df['outcome'] == current_indicator].copy()
                    # Create a 'period' column to distinguish between pre- and post-treatment time points.
                    absolute_predictive_subset['period'] = np.where(absolute_predictive_subset['timeid'].astype(str) > str(mapping_timeid[timeid_base-1]), "Post", "Pre")
                    # Rescale the 'value' (the effect) back to the original units by multiplying by the standard deviation.
                    # The mean is not added back because we are looking at effects (differences), where the mean cancels out.
                    absolute_predictive_subset['value'] = absolute_predictive_subset['value'] * std_value + avg_value

                    absolute_predictive_timeid = absolute_predictive_subset.groupby(['timeid', 'period'])['value'].agg(
                        q2_5=lambda x: x.quantile(0.025),
                        mean='mean',
                        q97_5=lambda x: x.quantile(0.975)
                    ).reset_index()

                    absolute_predictive_timeid = pd.merge(absolute_predictive_timeid, treated_subset, on=['timeid'], how='left')

                    # Determine the tick marks for the time axis to avoid overcrowding.
                    num_unique_timeids = len(chains_absolute_predictive_effect_df['timeid'].unique())
                    length_desired_breaks = math.ceil(num_unique_timeids / 40.0)
                    time_breaks = mapping_timeid[::length_desired_breaks]

                    p = (
                        ggplot(data=absolute_predictive_timeid) +
                        geom_ribbon(aes(x="timeid", ymin="q2_5", ymax="q97_5", group=1, fill="'Uncertainty'"), alpha=0.2) +
                        scale_fill_manual(values={"Uncertainty": "gray"}) +
                        geom_vline(xintercept=timeid_base, linetype="dashed", color="red") + # Vertical line at the intervention point.
                        geom_line(aes(x="timeid", y="mean", group=1, color="'Synthetic Control'")) +
                        geom_line(aes(x="timeid", y="treatment", group=1, color="'Treatment'")) +
                        scale_color_manual(values={"Mean": "black", "Treatment": "blue"}) +
                        theme_classic() +
                        ggtitle(f"{current_indicator} - Strict Parallel Trends - Predictive View") +
                        ylab("Absolute Value") +
                        theme(axis_text_x=element_text(angle=45, hjust=1), legend_position="top", legend_title=element_blank(), plot_title=element_text(hjust=0)) +
                        scale_x_discrete(breaks=list(time_breaks), name="timeid")
                    )
                    # Save the plot to a file.
                    p.save(f"{workspace_folder}/{temp_folder}/solution_id_{solution_id}/{current_indicator}/absolute predictive view.png", width=14, height=6, dpi=300)

                    # RELATIVE 
                    # Filter the effects data for the current outcome.
                    effect_subset = chains_relative_effect_df[chains_relative_effect_df['outcome'] == current_indicator].copy()
                    # Create a 'period' column to distinguish between pre- and post-treatment time points.
                    effect_subset['period'] = np.where(effect_subset['timeid'].astype(str) > str(mapping_timeid[timeid_base-1]), "Post", "Pre")
                    # Rescale the 'value' (the effect) back to the original units by multiplying by the standard deviation.
                    # The mean is not added back because we are looking at effects (differences), where the mean cancels out.
                    effect_subset['value'] *= std_value
                    
                    # Calculate the average effect per period for each MCMC iteration.
                    att_chain_period = effect_subset.groupby(['iteration', 'chain', 'period'])['value'].agg(
                        mean='mean'
                    ).reset_index()

                    # Summarize the posterior distribution of the average period effects to get the mean and 95% credible interval.
                    att_period = att_chain_period.groupby('period')['mean'].agg(
                        q2_5=lambda x: x.quantile(0.025),
                        mean='mean',
                        q97_5=lambda x: x.quantile(0.975)
                    ).reset_index()

                    # Summarize the posterior distribution of the effect for each individual time point.
                    att_timeid = effect_subset.groupby(['timeid', 'period'])['value'].agg(
                        q2_5=lambda x: x.quantile(0.025),
                        effect='mean',
                        q97_5=lambda x: x.quantile(0.975)
                    ).reset_index()

                    # Determine the tick marks for the time axis to avoid overcrowding.
                    num_unique_timeids = len(chains_relative_effect_df['timeid'].unique())
                    length_desired_breaks = math.ceil(num_unique_timeids / 40.0)
                    time_breaks = mapping_timeid[::length_desired_breaks]

                    # Calculate the x-coordinates for placing the summary annotations on the plot.
                    x_pre_annotation = (timeid_base + 1) / 2
                    post_period_start_index = timeid_base + 1
                    post_period_length = len(mapping_timeid) - post_period_start_index
                    x_post_annotation = post_period_start_index + (post_period_length / 2)
                    
                    # Create the main effect plot using the plotnine (ggplot) library.
                    p = (
                        ggplot(att_timeid, aes(x='factor(timeid)', y='effect', color='factor(period)')) +
                        geom_vline(xintercept=timeid_base, linetype="dashed", color="red") + # Vertical line at the intervention point.
                        geom_hline(yintercept=0.0, linetype="dashed", color="black") + # Horizontal line at zero effect.
                        geom_point(size=1.5) + # Points for the mean effect at each time point.
                        geom_errorbar(aes(ymin='q2_5', ymax='q97_5'), width=0.2) + # Error bars for the 95% credible interval.
                        scale_color_manual(name=None, values={"Pre": "#e87d72", "Post": "#56bcc2"}, breaks=['Pre', 'Post']) +
                        ggtitle(current_indicator) +
                        theme_classic() +
                        scale_x_discrete(breaks=list(time_breaks), name="timeid") +
                        theme(axis_text_x=element_text(angle=45, hjust=1), legend_position="top", legend_title=element_blank(), plot_title=element_text(hjust=0)) +
                        # Add the summary text annotations to the plot.
                        annotate("text", x=x_pre_annotation, y=att_timeid['q97_5'].max(),
                                label=format_annotation(att_period, "Pre", period_effect_format), color="black", size=10, fontweight='bold') +
                        annotate("text", x=x_post_annotation, y=att_timeid['q97_5'].max(),
                                label=format_annotation(att_period, "Post", period_effect_format), color="black", size=10, fontweight='bold') +
                        ylab("Effect Magnitude")
                    )
                    # Save the plot to a file.
                    p.save(f"{workspace_folder}/{temp_folder}/solution_id_{solution_id}/{current_indicator}/effect.png", width=14, height=6, dpi=300)

                    # Assumption 3' (Strict Parallel Trends)
                    # Generate and save forest plots for the Predictive NRMSE (Normalized Root Mean Square Error) model fit statistics using ArviZ.
                    # These plots help diagnose how well the synthetic control complies with the strict parallel trends assumption.
                    az.plot_forest(
                        fit,
                        var_names=["nrmse_pre", "nrmse_test", "nrmse_post", "nrmse_pre_test"],
                        combined=True,
                        figsize=(14, 6),
                        hdi_prob=0.95,
                        coords={ # 'coords' is used to select only the NRMSE for the current outcome (indexed by 'idx').
                            "nrmse_pre_dim_0": [idx],
                            "nrmse_test_dim_0": [idx],
                            "nrmse_post_dim_0": [idx],
                            "nrmse_pre_test_dim_0": [idx]
                        }
                    )
                    plt.suptitle("Strict Parallel Trends - Predictive RMSE (Normalized Root Mean Square Error)", fontsize=16, y=0.97)
                    plt.savefig(f"{workspace_folder}/{temp_folder}/solution_id_{solution_id}/{current_indicator}/predictive_nrmse.png")

                    # Assumption 3 (Low-Rank Trends)
                    # Generate and save forest plots for the Structural NRMSE (Normalized Root Mean Square Error) model fit statistics using ArviZ.
                    # These plots help diagnose how well the synthetic control complies with the low-rank trends assumption.
                    az.plot_forest(
                        fit,
                        var_names=["struc_nrmse_pre", "struc_nrmse_test", "struc_nrmse_post", "struc_nrmse_pre_test"],
                        combined=True,
                        figsize=(20, 6),
                        hdi_prob=0.95,
                        coords={ # 'coords' is used to select only the NRMSE for the current outcome (indexed by 'idx').
                            "struc_nrmse_pre_dim_0": [idx],
                            "struc_nrmse_test_dim_0": [idx],
                            "struc_nrmse_post_dim_0": [idx],
                            "struc_nrmse_pre_test_dim_0": [idx]
                        }
                    )
                    plt.suptitle("Low-Rank Trends - Structural RMSE (Normalized Root Mean Square Error)", fontsize=16, y=0.97)
                    plt.savefig(f"{workspace_folder}/{temp_folder}/solution_id_{solution_id}/{current_indicator}/structural_nrmse.png")

                    # Store the summary results for the current outcome in the lists.
                    absolute_structural_timeid['outcome'] = current_indicator
                    absolute_structural_timeid_list.append(absolute_structural_timeid)

                    absolute_predictive_timeid['outcome'] = current_indicator
                    absolute_predictive_timeid_list.append(absolute_predictive_timeid)

                    att_period['outcome'] = current_indicator
                    att_period_list.append(att_period)

                    att_timeid['outcome'] = current_indicator
                    att_timeid_list.append(att_timeid)

                # Combine the summary results from all outcomes into single DataFrames.
                absolute_structural_timeid_df = pd.concat(absolute_structural_timeid_list)
                
                shape_report = classify_structural_uncertainty_shape(absolute_structural_timeid_df)

                if shape_report['quality_score'].min() < 5:
                    solution_has_problem = solution_has_problem + 'S'
                    unacceptable_solutions.append(solution_id)

                # Save the report
                shape_report.to_csv(f"{workspace_folder}/{temp_folder}/solution_id_{solution_id}/structural_uncertainty_shape_report.csv", index=False)                
                
                absolute_predictive_timeid_df = pd.concat(absolute_predictive_timeid_list)
                att_period_df = pd.concat(att_period_list)
                att_timeid_df = pd.concat(att_timeid_list)
                # Save the summarized results and the full set of MCMC draws to CSV files for further analysis.
                absolute_structural_timeid_df.to_csv(f"{workspace_folder}/{temp_folder}/solution_id_{solution_id}/absolute_structural_timeid_dt.csv", index=False)
                absolute_predictive_timeid_df.to_csv(f"{workspace_folder}/{temp_folder}/solution_id_{solution_id}/absolute_predictive_timeid_dt.csv", index=False)
                att_period_df.to_csv(f"{workspace_folder}/{temp_folder}/solution_id_{solution_id}/att_period_dt.csv", index=False)
                att_timeid_df.to_csv(f"{workspace_folder}/{temp_folder}/solution_id_{solution_id}/att_timeid_dt.csv", index=False)
                draws_df.to_csv(f"{workspace_folder}/{temp_folder}/solution_id_{solution_id}/chains_all_parameters.csv", index=False)

                # Convert the CmdStanPy fit object to an ArviZ InferenceData object for advanced plotting.
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
                    var_names=["w", "sigma", "tau_nrmse", "gini_weight"],
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
                plot_titles = [name for name in az.summary(inference_data).index.tolist() if name.startswith(("w", "sigma", "tau_nrmse"))]
                new_w_names = {f"w[{index}]": f"w[{name}]" for index, name in enumerate(unitids)}
                new_s_names = {f"sigma[{index}]": f"sigma[{name}]" for index, name in enumerate(outcomes)}
                new_t_names = {f"tau_nrmse[{index}]": f"tau_nrmse[{name}]" for index, name in enumerate(outcomes)}
                new_names = new_w_names | new_s_names | new_t_names
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

                # --- Create a forest plot for the donor weights (w) ---
                # This plot shows the posterior distribution (mean and 95% HDI) for each control unit's weight.
                # It clearly visualizes which control units are most important for constructing the synthetic control.
                az.plot_forest(
                    fit,
                    var_names="w",
                    combined=True,
                    figsize=(14, max(6, N_controls * 0.3)), # Adjust height based on number of controls.
                    hdi_prob=0.95,
                    labeller=az.labels.MapLabeller(coord_map={"w_dim_0": {index: name for index, name in enumerate(unitids)}})
                )
                plt.suptitle("Posterior Distribution of Control Unit Weights (w)", fontsize=16, y=0.97)
                plt.savefig(f"{workspace_folder}/{temp_folder}/solution_id_{solution_id}/unitid_weight_simultanous_distribution.png")
                
                if solution_has_problem == "":
                    if parallel_trends_violated:
                        os.rename(f"{workspace_folder}/{temp_folder}/solution_id_{solution_id}", f"{workspace_folder}/{temp_folder}/P_solution_id_{solution_id}")
                        unacceptable_solutions.append(solution_id)
                    else:
                        os.rename(f"{workspace_folder}/{temp_folder}/solution_id_{solution_id}", f"{workspace_folder}/{temp_folder}/B{loo_results.elpd_loo:3.1f}_W{draws_df['gini_weight'].mean():.2f}_H{draws_df['hhi_weight'].mean():.2f}_solution_id_{solution_id}")
                else:
                    if parallel_trends_violated:
                        os.rename(f"{workspace_folder}/{temp_folder}/solution_id_{solution_id}", f"{workspace_folder}/{temp_folder}/{solution_has_problem}P_solution_id_{solution_id}")
                        unacceptable_solutions.append(solution_id)
                    else:
                        os.rename(f"{workspace_folder}/{temp_folder}/solution_id_{solution_id}", f"{workspace_folder}/{temp_folder}/{solution_has_problem}_solution_id_{solution_id}")

    complemento = list(set(solutions_id) - set(np.unique(unacceptable_solutions)))

    weight_solutions_filtred = weight_solutions[weight_solutions['solution_id'].isin(complemento)]

    # 2. Create the 'N' and 'sid' columns
    weight_solutions_filtred['N'] = weight_solutions_filtred['value'].round(2).astype(str)
    weight_solutions_filtred['impact_score_solution_id'] = weight_solutions_filtred['impact_score'].astype(str) + '_' + weight_solutions_filtred['solution_id'].astype(int).astype(str)

    # 3. Reshape the data using pivot_table and fill missing values
    result_df = weight_solutions_filtred.pivot_table(
        index='unitid',
        columns='impact_score_solution_id',
        values='N',
        aggfunc='first',  # Use 'first' as we expect one value per cell
        fill_value=""
    ).reset_index()

    print(result_df)
    result_df.to_csv(f"{workspace_folder}/{temp_folder}/weight_solutions.csv", mode='w', header=True, index=False)
