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
2.  **Optimization:** The selected donors is then used in a traditional
    SCM algorithm to determine the final weights for the synthetic control.

This implementation aims to transform the donor selection process from a
subjective "dark art" into a more scientific and trustworthy methodology for
causal inference.

.. _Medium Article:
    https://medium.com/@frederico.nogueira/a-new-lens-for-donor-selection-att-ipw-based-ranking-198b9d30bc69

#######################################################################################################

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
    The name of the column that contains the names of the outcome variables and covariates.
unitname : str
    The name of the column that contains the unique identifiers for each unit.
tname : str
    The name of the column that contains the time period identifiers (e.g., year).
value : str
    The name of the column that contains the numerical values for the outcomes/covariates.
treatment : str
    The name of the column that indicates treatment status (1 for treated, 0 for control).
pre_intervention : str
    The name of the column indicating the pre-intervention period (1 if pre, 0 if post).
temporal_cross_search : list of str
    A list of time periods within the pre-intervention phase to use as split points for
    cross-temporal validation. Each period marks the end of a training set.
workspace_folder : str
    The path to a directory where output files (candidate donors and performance) will be saved.
tname_covariate : str, optional
    A special value in the `tname` column used to identify rows that are time-invariant covariates.
    Defaults to None.
seed : int, optional
    Random seed for reproducibility of the optimization process. Defaults to 111.
maximum_num_units_on_support_first_filter : int, optional
    The maximum number of units allowed in the on-support group during the first pruning step,
    used to penalize trivial solutions. Defaults to 50.
maximum_error_pre_intervention : float, optional
    The maximum acceptable error (e.g., RMSE or MAE) on the pre-treatment outcomes for a
    candidate donor pool to be saved. Defaults to 0.15.
maximum_error_covariates : float, optional
    The maximum acceptable error on the covariates for a candidate donor pool to be saved.
    Defaults to 0.15.
proportion_pre_intervention_period_outcomes_donor : int, optional
    Used to calculate the upper limit for the donor pool size. The limit is determined by
    (num_pre_intervention_periods * num_outcomes) / this_value. Defaults to 10.
inferior_limit_maximum_donor_pool_size : int, optional
    The minimum number of donors to be selected in any candidate pool. Defaults to 2.
on_support_first_filter : {'max_weight', 'maximum_num_units_on_support_first_filter', 'bigger_than_min_weight'}, optional
    The primary strategy for identifying the initial set of "on-support" donors from the
    IPW-ranked list. Defaults to 'max_weight'.
on_support_second_filter : {'randomN', 'all'}, optional
    The secondary strategy to prune the on-support set down to the final donor pool size.
    'randomN' randomly samples N units, while 'all' takes the top N deterministically.
    Defaults to 'randomN'.
include_error_post_intervention_in_optuna_objective : bool, optional
    Flag to include or not the post-intervention error in the Optuna's search criteria. Defaults to False.
number_optuna_trials : int, optional
    The number of hyperparameter optimization trials to run for each cross-temporal fold.
    Defaults to 300.
timeout_optuna_cycle : int, optional
    The maximum time in seconds for a single Optuna optimization cycle (one cross-temporal fold).
    Defaults to 900.
time_serie_covariate_metric : {'rmse', 'mae'}, optional
    The metric used to evaluate the fit on time-series outcomes and covariates.
    Defaults to 'rmse'.
"""
from importlib import resources
import numpy as np
import pandas as pd

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
    tname_covariate = None,
    seed = 111,
    maximum_num_units_on_support_first_filter = 50,
    maximum_error_pre_intervention = 0.15,
    maximum_error_covariates = 0.15,
    proportion_pre_intervention_period_outcomes_donor = 10,
    inferior_limit_maximum_donor_pool_size = 2,
    on_support_first_filter = 'max_weight',
    on_support_second_filter = 'randomN',
    include_error_post_intervention_in_optuna_objective = False,
    number_optuna_trials = 300,
    timeout_optuna_cycle = 900,
    time_serie_covariate_metric = 'rmse'
):
    # RENAME AND INFORM ERRORS
    if tname in all_units.columns:
        all_units.rename(columns={tname: 'timeid'}, inplace=True)
    else:
        print(f"The '{tname}' column does not exist in the dataframe.")
        import sys
        sys.exit()

    if unitname in all_units.columns:
        all_units.rename(columns={unitname: 'unitid'}, inplace=True)
    else:
        print(f"The '{unitname}' column does not exist in the dataframe.")
        import sys
        sys.exit()
    
    if yname in all_units.columns:
        all_units.rename(columns={yname: 'variable'}, inplace=True)
    else:
        print(f"The '{yname}' column does not exist in the dataframe.")
        import sys
        sys.exit()

    if value in all_units.columns:
        all_units.rename(columns={value: 'value'}, inplace=True)
    else:
        print(f"The '{value}' column does not exist in the dataframe.")
        import sys
        sys.exit()
    
    if treatment in all_units.columns:
        all_units.rename(columns={treatment: 'treatment'}, inplace=True)
    else:
        print(f"The '{treatment}' column does not exist in the dataframe.")
        import sys
        sys.exit()
    
    if pre_intervention in all_units.columns:
        all_units.rename(columns={pre_intervention: 'pre_intervention'}, inplace=True)
    else:
        print(f"The '{pre_intervention}' column does not exist in the dataframe.")
        import sys
        sys.exit()

    # CHECK pre_intervention COULUMN AGAINST temporal_cross_search
    # temporal_cross_search MUST BE A SUBSET OF pre_intervention
    if not set(temporal_cross_search).issubset(set(all_units[all_units['pre_intervention'] == 1]['timeid'].unique().tolist())):
        print("ERROR: temporal_cross_search MUST BE A SUBSET OF pre_intervention")
        print(f"temporal_cross_search: {sorted(temporal_cross_search)}")
        print(f"pre_intervention: {sorted(all_units[all_units['pre_intervention'] == 1]['timeid'].unique().tolist())}")
        import sys
        sys.exit()

    # CHECH DIRECTORY EXISTS AND CREATE IT IF NOT
    import os
    if not os.path.exists(workspace_folder):
        os.makedirs(workspace_folder)
        print(f"Created workspace folder: {workspace_folder}")

    from importlib import resources
    import shutil
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

    # CHECK tname_covariate COULUMN AGAINST all_units['timeid']
    # WHEN tname_covariate IS NOT NONE, MUST BE A VALUE FROM all_units['timeid']
    if tname_covariate is not None and tname_covariate not in all_units['timeid'].unique().tolist():
        print(f"ERROR: tname_covariate '{tname_covariate}' MUST BE A VALUE FROM all_units['timeid']")
        print(f"all_units['timeid']: {sorted(all_units['timeid'].unique().tolist())}")
        import sys
        sys.exit()

    # CHECK IF THE on_support_first_filter HAS ONLY THE VALUES: 'max_weight', 'maximum_num_units_on_support_first_filter' AND/OR 'bigger_than_min_weight'
    if on_support_first_filter not in ['max_weight', 'maximum_num_units_on_support_first_filter', 'bigger_than_min_weight']:
        print("ERROR: on_support_first_filter MUST HAVE ONLY THE VALUES: 'max_weight', 'maximum_num_units_on_support_first_filter' AND/OR 'bigger_than_min_weight'")
        print(f"on_support_first_filter: {on_support_first_filter}")
        import sys
        sys.exit()

    # CHECK IF THE on_support_second_filter HAS ONLY THE VALUES: 'randomN' AND/OR 'bigger_than_min_weight'
    if on_support_second_filter not in ['randomN', 'all']:
        print("ERROR: on_support_second_filter MUST HAVE ONLY THE VALUES: 'randomN' AND/OR 'all'")
        print(f"on_support_second_filter: {on_support_second_filter}")
        import sys
        sys.exit()

    treatment_unitid = all_units[all_units['treatment'] == 1]['unitid'].iloc[0]
    outcomes = all_units[all_units['timeid'] != tname_covariate]['variable'].sort_values().unique().tolist()
    covariates = all_units[all_units['timeid'] == tname_covariate]['variable'].sort_values().unique().tolist()
    # timeids = all_units['timeid'].sort_values().unique().tolist()
    timeid_pre_intervention = all_units[(all_units['pre_intervention'] == 1) & (all_units['timeid'] != tname_covariate)]['timeid'].sort_values().unique().tolist()
    num_pre_intervention_periods_per_outcome = all_units[(all_units['pre_intervention'] == 1) & (all_units['timeid'] != tname_covariate)].groupby('variable').agg({'timeid': 'nunique' }).reset_index()
    # timeid_post_intervention = all_units[all_units['pre_intervention'] == 0]['timeid'].sort_values().unique().tolist()

    hyperparameter_search_extra_criteria = []
    if len(covariates) > 0:
        hyperparameter_search_extra_criteria.append('covariate')
        
    if include_error_post_intervention_in_optuna_objective:
        hyperparameter_search_extra_criteria.append('post_intervention_period')

    import os
    import sys
    import xgboost as xgb
    xgb.set_config(verbosity=0)
    import optuna
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=DeprecationWarning)

    scm_donor_selection_candidate_units_data_file_path = workspace_folder + 'scm_donor_selection_candidate_units_data.csv'
    if os.path.exists(scm_donor_selection_candidate_units_data_file_path):
        os.remove(scm_donor_selection_candidate_units_data_file_path)

    scm_donor_selection_candidate_performance_file_path = workspace_folder + 'scm_donor_selection_candidate_performance.csv'
    if os.path.exists(scm_donor_selection_candidate_performance_file_path):
        os.remove(scm_donor_selection_candidate_performance_file_path)

    superior_limit_maximum_donor_pool_size = int(num_pre_intervention_periods_per_outcome['timeid'].sum() / proportion_pre_intervention_period_outcomes_donor)
    if superior_limit_maximum_donor_pool_size < 2:
        superior_limit_maximum_donor_pool_size = 2

    timeid_post_intervention = all_units[all_units['pre_intervention'] == 0]['timeid'].sort_values().unique().tolist()

    all_units.sort_values(by=['unitid', 'timeid', 'variable'], inplace=True)
    all_units['weight'] = 1.0

    amplitude = all_units[(all_units['treatment'] == 1) & (all_units['pre_intervention'] == 1) & (all_units['timeid'] != tname_covariate)].groupby(['variable']).apply(
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
        # data = data.copy()
        data = data[data['weight'].notna()]
        # Generate the times series created by the weighted average of the units in the dataframe in the defined period 
        aggregated = data.groupby(['variable', 'timeid', 'treatment'], dropna=False).apply(
            lambda x : pd.Series({
                'value' : np.ma.filled(np.ma.average(np.ma.masked_invalid(x['value']), weights=x['weight']), fill_value=np.nan)
            })
        ).reset_index()

        # Calculate the average and standard deviation of the time series in the given period
        standardization = aggregated[aggregated['timeid'].isin(period)].groupby(['variable', 'treatment'], dropna=False).agg(
            avg = ('value', lambda x: np.mean(x)),
            std = ('value', lambda x: np.std(x, ddof=1))
        ).reset_index()
        standardization['std'] = np.where(standardization['std'].isna(), 1.0, standardization['std'])

        # Standardize the control times series created by the weighted average of the control unit in the dataframe in the defined period,
        # ignoring the values related to the covariates which are timeless
        data = pd.merge(data, standardization, on=['variable', 'treatment'], how='left')
        data['value'] = np.where((data['treatment'] == 0) & (~data['timeid'].isin([tname_covariate])), (data['value'] - data['avg']) / data['std'], data['value'])
        data.drop(['avg', 'std'], axis=1, inplace=True)

        # Apply the treatment's time series mean and standard deviation to the standardized control units.
        # Positioning them in level and amplitude guaranteeing the only difference between them is the time series' shape.
        data = pd.merge(data, standardization[standardization['treatment'] == 1][['variable', 'avg', 'std']], on=['variable'], how='left')
        data['value'] = np.where((data['treatment'] == 0) & (~data['timeid'].isin([tname_covariate])), (data['value'] * data['std']) + data['avg'], data['value'])
        data.drop(['avg', 'std'], axis=1, inplace=True)

        return data

    treatment = all_units[all_units['treatment'] == 1]
    treatment.loc[:, 'unitid'] = treatment_unitid

    # Initialize CSV files for storing performance results and the donor candidates data with headers.
    scm_donor_selection_candidate_units_data = treatment[['variable', 'timeid', 'value', 'treatment', 'weight', 'unitid']].copy()
    scm_donor_selection_candidate_units_data['valor_m_weight'] = scm_donor_selection_candidate_units_data['value'] * scm_donor_selection_candidate_units_data['weight']
    scm_donor_selection_candidate_units_data['id'] = None
    scm_donor_selection_candidate_units_data['cycle'] = None
    scm_donor_selection_candidate_units_data['trial'] = None
    scm_donor_selection_candidate_units_data['error_train'] = None
    scm_donor_selection_candidate_units_data['error_valid'] = None
    scm_donor_selection_candidate_units_data['error_pre_intervention'] = None
    scm_donor_selection_candidate_units_data['error_post_intervention'] = None
    scm_donor_selection_candidate_units_data['error_covariates'] = None
    scm_donor_selection_candidate_units_data['num_units_bigger_min_weight'] = None
    scm_donor_selection_candidate_units_data['num_units_max_weight'] = None
    scm_donor_selection_candidate_units_data.to_csv(scm_donor_selection_candidate_units_data_file_path, mode='w', header=True, index=False)

    # Initialize CSV files for storing performance results with headers.
    scm_donor_selection_candidate_performance = treatment[['variable', 'timeid', 'treatment', 'value']].copy()
    scm_donor_selection_candidate_performance['qtty'] = 1
    scm_donor_selection_candidate_performance['cycle'] = None
    scm_donor_selection_candidate_performance['trial'] = None
    scm_donor_selection_candidate_performance['error_train'] = None
    scm_donor_selection_candidate_performance['error_valid'] = None
    scm_donor_selection_candidate_performance['error_pre_intervention'] = None
    scm_donor_selection_candidate_performance['error_post_intervention'] = None
    scm_donor_selection_candidate_performance['error_covariates'] = None
    scm_donor_selection_candidate_performance['num_units_bigger_min_weight'] = None
    scm_donor_selection_candidate_performance['num_units_max_weight'] = None
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
            dataset_train = data[data['timeid'].isin(timeid_train + [tname_covariate])].pivot(
                index=['unitid', 'treatment', 'weight'],
                columns=['variable', 'timeid'],
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

        def evaluate_outcomes_metric(self, metric, weights):
            """
            Evaluates the performance of the synthetic control based on the provided weights.
            The article describes this as evaluating "Causal Fitness", which includes pre-intervention
            outcome balance, covariate balance, and post-intervention "null" balance.
            """
            full_data = self.full_data.copy()
            depara = pd.merge(weights, self.from_to, on='id', how='inner')
            full_data_depara = pd.merge(full_data, depara, on='unitid', how='left').reset_index()
            data = pre_intervention_scaling(data=full_data_depara, period=self.timeid_train + self.timeid_valid + [tname_covariate])
            aggregated3 = data.groupby(['variable', 'timeid', 'treatment'], dropna=False).apply(
                lambda x : pd.Series({
                    'value' : np.ma.filled(np.ma.average(np.ma.masked_invalid(x['value']), weights=x['weight']), fill_value=np.nan)
                })
            ).reset_index()

            # Calculate error on the training set.
            aggregated3_diff = aggregated3[aggregated3['timeid'].isin(self.timeid_train)].sort_values(['variable', 'timeid', 'treatment'], ascending=True).groupby(['variable', 'timeid']).agg({
                'value' : lambda x: x.iloc[0] - x.iloc[1]
            }).reset_index(level=['variable', 'timeid'])
            
            aggregated3_diff_normalized = pd.merge(aggregated3_diff, amplitude, on='variable', how='left')
            aggregated3_diff_normalized['value'] = aggregated3_diff_normalized['value'] / aggregated3_diff_normalized['amplitude']

            if metric == 'mae':
                error_train = aggregated3_diff_normalized['value'].abs().mean()
            elif metric == 'rmse':
                error_train = ((aggregated3_diff_normalized['value'] ** 2).mean()) ** 0.5

            # Calculate error on the validation set. This is crucial for early stopping and preventing overfitting.
            aggregated3_diff = aggregated3[aggregated3['timeid'].isin(self.timeid_valid)].sort_values(['variable', 'timeid', 'treatment'], ascending=True).groupby(['variable', 'timeid']).agg({
                'value' : lambda x: x.iloc[0] - x.iloc[1]
            }).reset_index(level=['variable', 'timeid'])

            aggregated3_diff_normalized = pd.merge(aggregated3_diff, amplitude, on='variable', how='left')
            aggregated3_diff_normalized['value'] = aggregated3_diff_normalized['value'] / aggregated3_diff_normalized['amplitude']

            if metric == 'mae':
                error_valid = aggregated3_diff_normalized['value'].abs().mean()
            elif metric == 'rmse':
                error_valid = ((aggregated3_diff_normalized['value'] ** 2).mean()) ** 0.5

            # Calculate error for the entire pre-treatment period.
            aggregated3_diff = aggregated3[aggregated3['timeid'].isin(self.timeid_train + self.timeid_valid)].sort_values(['variable', 'timeid', 'treatment'], ascending=True).groupby(['variable', 'timeid']).agg({
                'value' : lambda x: x.iloc[0] - x.iloc[1]
            }).reset_index(level=['variable', 'timeid'])

            aggregated3_diff_normalized = pd.merge(aggregated3_diff, amplitude, on='variable', how='left')
            aggregated3_diff_normalized['value'] = aggregated3_diff_normalized['value'] / aggregated3_diff_normalized['amplitude']

            if metric == 'mae':
                error_pre_intervention = aggregated3_diff_normalized['value'].abs().mean()
            elif metric == 'rmse':
                error_pre_intervention = ((aggregated3_diff_normalized['value'] ** 2).mean()) ** 0.5

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
            # or an insufficient Optuna trials, reduce the number of selected outcomes and covariates and rerun.
            aggregated3_diff = aggregated3[aggregated3['timeid'].isin(self.timeid_post_intervention)].sort_values(['variable', 'timeid', 'treatment'], ascending=True).groupby(['variable', 'timeid']).agg({
                'value' : lambda x: x.iloc[0] - x.iloc[1]
            }).reset_index(level=['variable', 'timeid'])

            aggregated3_diff_normalized = pd.merge(aggregated3_diff, amplitude, on='variable', how='left')
            aggregated3_diff_normalized['value'] = aggregated3_diff_normalized['value'] / aggregated3_diff_normalized['amplitude']

            if metric == 'mae':
                error_post_intervention = aggregated3_diff_normalized['value'].abs().mean()
            elif metric == 'rmse':
                error_post_intervention = ((aggregated3_diff_normalized['value'] ** 2).mean()) ** 0.5

            # Evaluate covariate balance.
            aggregated3_diff = aggregated3[aggregated3['timeid'].isin([tname_covariate])].sort_values(['variable', 'timeid', 'treatment'], ascending=True).groupby(['variable', 'timeid']).agg({
                'value' : lambda x: x.iloc[0] - x.iloc[1]
            }).reset_index(level=['variable', 'timeid'])

            if metric == 'mae':
                error_covariates = aggregated3_diff['value'].abs().mean()
            elif metric == 'rmse':
                error_covariates = ((aggregated3_diff['value'] ** 2).mean()) ** 0.5

            return error_train, error_valid, error_pre_intervention, error_post_intervention, error_covariates

        def eval_metric_ipw(self, preds, metric):
            """
            This function implements the ATT/IPW-based ranking.
            It converts the XGBoost predictions (propensity scores) into weights and then selects N donors.
            """
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

            max_weight = ipw[ipw['treatment'] == 0]['weight'].max()
            max_weight_observations = ipw[(ipw['treatment'] == 0) & (ipw['weight'] == max_weight)]
            num_units_max_weight = max_weight_observations.shape[0]

            maximum_donor_pool_size = pruning_trial.params['maximum_donor_pool_size']

            # ===================================================================================
            # FINAL DONOR POOL PRUNING
            #
            # The following block implements various strategies for the final, crucial step of
            # pruning the ranked list of all potential donors into a small, final donor pool.
            # The specific strategy is controlled by the `on_support_first_filter` and
            # `on_support_second_filter` configuration variables.
            # ===================================================================================

            # --- STRATEGY 1: From max-weighted units, randomly sample N. ---
            if on_support_first_filter == 'max_weight' and on_support_second_filter == 'randomN':
                # PENALTY: To encourage donor diversity, heavily penalize solutions where an
                # excessive number of units share the same maximum weight. This prevents the model
                # from converging on trivial solutions where many donors are equally "best".
                if num_units_max_weight > maximum_num_units_on_support_first_filter:
                    error_train = float('inf')
                    error_valid = float('inf')
                    error_pre_intervention = float('inf')
                    error_post_intervention = float('inf')
                    error_covariates = float('inf')
                    return error_train, error_valid, error_pre_intervention, error_post_intervention, error_covariates, ipw, num_units_bigger_min_weight, num_units_max_weight
                
                # If the number of units sharing the max weight is larger than the desired pool size,
                # randomly sample from this top tier to select the final donors.
                if max_weight_observations.shape[0] > maximum_donor_pool_size:
                    ipw = pd.concat([ipw[ipw['treatment'] != 0], max_weight_observations.sort_values(['id'], ascending=[True]).sample(n=maximum_donor_pool_size, random_state=attempt + seed)], ignore_index=True)
                # Otherwise, if the pool of max-weighted units is smaller than the desired size,
                # take all of them deterministically.
                else:
                    temp = min(maximum_donor_pool_size, num_units_max_weight)
                    ipw = pd.concat([ipw[ipw['treatment'] != 0], ipw[ipw['treatment'] == 0].sort_values(['weight', 'id'], ascending=[False, True])[0:temp]], ignore_index=True)

            # --- STRATEGY 2: From max-weighted units, take all (top N deterministically). ---
            elif on_support_first_filter == 'max_weight' and on_support_second_filter == 'all':
                # This is a deterministic version of the above. It selects the top N units from the
                # max-weighted pool, sorted by weight and unit ID to ensure reproducibility.
                temp = min(maximum_donor_pool_size, num_units_max_weight)
                ipw = pd.concat([ipw[ipw['treatment'] != 0], ipw[ipw['treatment'] == 0].sort_values(['weight', 'id'], ascending=[False, True])[0:temp]], ignore_index=True)

            # --- STRATEGY 3: From all ranked units, randomly sample N. ---
            elif on_support_first_filter == 'maximum_num_units_on_support_first_filter' and on_support_second_filter == 'randomN':
                # This strategy ignores tiers and simply performs a random sample of N donors
                # from the entire ranked list of potential controls.
                ipw = pd.concat([ipw[ipw['treatment'] != 0], ipw.sort_values(['id'], ascending=[True]).sample(n=maximum_donor_pool_size, random_state=attempt + seed)], ignore_index=True)

            # --- STRATEGY 4: From all ranked units, take all (top N deterministically). ---
            elif on_support_first_filter == 'maximum_num_units_on_support_first_filter'and on_support_second_filter == 'all':
                # This is the most straightforward strategy: deterministically select the top N
                # highest-weighted donors from the entire control group.
                ipw = pd.concat([ipw[ipw['treatment'] != 0], ipw[ipw['treatment'] == 0].sort_values(['weight', 'id'], ascending=[False, True])[0:maximum_num_units_on_support_first_filter]], ignore_index=True)

            # --- STRATEGY 5: From units with non-minimal weight, randomly sample N. ---
            elif on_support_first_filter == 'bigger_than_min_weight' and on_support_second_filter == 'randomN':
                # PENALTY: Similar to the max-weight penalty, this discourages solutions where
                # nearly all units are considered "on support" (i.e., not the absolute minimum weight).
                if num_units_bigger_min_weight > maximum_num_units_on_support_first_filter:
                    error_train = float('inf')
                    error_valid = float('inf')
                    error_pre_intervention = float('inf')
                    error_post_intervention = float('inf')
                    error_covariates = float('inf')
                    return error_train, error_valid, error_pre_intervention, error_post_intervention, error_covariates, ipw, num_units_bigger_min_weight, num_units_max_weight

                # If the number of on-support units exceeds the desired pool size,
                # randomly sample from them.
                if min_weight_observations.shape[0] > maximum_donor_pool_size:
                    ipw = pd.concat([ipw[ipw['treatment'] != 0], min_weight_observations.sort_values(['id'], ascending=[True]).sample(n=maximum_donor_pool_size, random_state=attempt + seed)], ignore_index=True)
                # Otherwise, take all on-support units deterministically.
                else:
                    temp = min(maximum_donor_pool_size, num_units_bigger_min_weight)
                    ipw = pd.concat([ipw[ipw['treatment'] != 0], ipw[ipw['treatment'] == 0].sort_values(['weight', 'id'], ascending=[False, True])[0:temp]], ignore_index=True)

            # --- STRATEGY 6: From units with non-minimal weight, take all. ---
            elif on_support_first_filter == 'bigger_than_min_weight' and on_support_second_filter == 'all':
                # This strategy intends to deterministically select all units that are not
                # assigned the absolute minimum weight.
                ipw = pd.concat([ipw[ipw['treatment'] != 0], ipw[ipw['treatment'] == 0].sort_values(['weight', 'id'], ascending=[False, True])[0:num_units_bigger_min_weight]], ignore_index=True)
            # --- WRONG OPTION ---
            else:
                print("WRONG ON-SUPPORT STRATEGY OPTION!")
                sys.exit()

            # Re-normalize weights after selecting the top N donors to sum to 1.
            sum_weight_tratamento = ipw.groupby('treatment').agg({
                'weight' : 'sum'
            }).reset_index()
            sum_weight_tratamento_0 = sum_weight_tratamento[sum_weight_tratamento['treatment'] == 0]['weight'].values[0]
            ipw['weight'] = np.where(ipw['treatment'] == 0, ipw['weight'] / sum_weight_tratamento_0, ipw['weight'])

            ipw.drop('treatment', axis=1, inplace=True)
            
            # Evaluate the performance of this candidate donor pool.
            error_train, error_valid, error_pre_intervention, error_post_intervention, error_covariates = self.evaluate_outcomes_metric(metric=metric, weights=ipw)

            return error_train, error_valid, error_pre_intervention, error_post_intervention, error_covariates, ipw, num_units_bigger_min_weight, num_units_max_weight

        def full_data_treatment_control_scaling(self, weight):
            """Prepares the full dataset with the final weights for saving if it is good enough."""
            temp = pd.merge(weight, self.from_to, on='id', how='inner')
            temp = pd.merge(self.full_data, temp, on='unitid', how='left').reset_index()
            data = pre_intervention_scaling(data=temp, period=self.timeid_train + self.timeid_valid + [tname_covariate])
            aggregated3 = data.groupby(['variable', 'timeid', 'treatment'], dropna=False).apply(
                lambda x : pd.Series({
                    'value' : np.ma.filled(np.ma.average(np.ma.masked_invalid(x['value']), weights=x['weight']), fill_value=np.nan),
                    'qtty' : np.ma.filled(np.ma.count(np.ma.masked_invalid(x['value'])), fill_value=np.nan)
                })
            ).reset_index()
            return aggregated3, data

    # MAIN TRAINING LOOP
    # This outer loop performs Cross-Temporal Validation.
    cycle = 0
    global attempt
    attempt = 0
    timeid_train_indexes = [timeid_pre_intervention.index(x) + 1 for x in temporal_cross_search]
    for timeid_train_index in timeid_train_indexes:
        sampler = optuna.samplers.NSGAIIISampler(seed=seed)
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
            global current_error_train, current_error_valid, current_error_pre_intervention, current_error_post_intervention, current_error_covariates, current_weights, current_preds, current_num_units_bigger_min_weight, current_num_units_max_weight

            error_train, error_valid, error_pre_intervention, error_post_intervention, error_covariates, weights, num_units_bigger_min_weight, num_units_max_weight = dataset.eval_metric_ipw(preds, metric=time_serie_covariate_metric)

            if current_error_valid > error_valid:
                current_weights = weights
                current_error_train = error_train
                current_error_valid = error_valid
                current_error_pre_intervention = error_pre_intervention
                current_error_post_intervention = error_post_intervention
                current_error_covariates = error_covariates
                current_preds = preds
                current_num_units_bigger_min_weight = num_units_bigger_min_weight
                current_num_units_max_weight = num_units_max_weight

            return 'causal_fitness', error_valid

        class training_callback(xgb.callback.TrainingCallback):
            """A callback to reset the error at the beginning of each training run."""
            def before_training(self, model):
                global current_error_valid, attempt
                current_error_valid = float('inf')
                attempt = attempt + 1

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
                'scale_pos_weight': trial.suggest_float('scale_pos_weight', calculated_scale_pos_weight * 0.5, calculated_scale_pos_weight * 2.0),

                'maximum_donor_pool_size' : trial.suggest_int('maximum_donor_pool_size', inferior_limit_maximum_donor_pool_size, superior_limit_maximum_donor_pool_size),
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

                if current_error_valid != float('inf'):
                    temp, data = dataset.full_data_treatment_control_scaling(current_weights)
                    # Only save results that meet the pre-treatment error threshold.

                    if 'covariate' in hyperparameter_search_extra_criteria:
                        condition_save = current_error_pre_intervention < maximum_error_pre_intervention and len(covariates) > 0 and current_error_covariates < maximum_error_covariates
                    else:
                        condition_save = current_error_pre_intervention < maximum_error_pre_intervention

                    if condition_save:
                        # Save the donor units, weights and performance metrics of this viable solution.
                        data = data[data['treatment'] == 0]
                        data['valor_m_weight'] = data['value'] * data['weight']
                        data['cycle'] = cycle
                        data['trial'] = trial.number
                        data['error_train'] = current_error_train
                        data['error_valid'] = current_error_valid
                        data['error_pre_intervention'] = current_error_pre_intervention
                        data['error_post_intervention'] = current_error_post_intervention
                        data['error_covariates'] = current_error_covariates
                        data['num_units_bigger_min_weight'] = current_num_units_bigger_min_weight
                        data['num_units_max_weight'] = current_num_units_max_weight
                        columns = ['variable', 'timeid', 'value', 'treatment', 'weight', 'unitid', 'valor_m_weight', 'id', 'cycle', 'trial', 'error_train', 'error_valid', 'error_pre_intervention', 'error_post_intervention', 'error_covariates', 'num_units_bigger_min_weight', 'num_units_max_weight']
                        data[columns].to_csv(scm_donor_selection_candidate_units_data_file_path, mode='a', header=False, index=False)

                        # Save the weights and performance metrics of this viable solution.
                        temp = temp[temp['treatment'] == 0]
                        temp['cycle'] = cycle
                        temp['trial'] = trial.number
                        temp['error_train'] = current_error_train
                        temp['error_valid'] = current_error_valid
                        temp['error_pre_intervention'] = current_error_pre_intervention
                        temp['error_post_intervention'] = current_error_post_intervention
                        temp['error_covariates'] = current_error_covariates
                        temp['num_units_bigger_min_weight'] = current_num_units_bigger_min_weight
                        temp['num_units_max_weight'] = current_num_units_max_weight
                        temp.to_csv(scm_donor_selection_candidate_performance_file_path, mode='a', header=False, index=False)
                else:
                    if 'covariate' in hyperparameter_search_extra_criteria and 'post_intervention_period' in hyperparameter_search_extra_criteria:
                        return float('inf'), float('inf'), float('inf')
                    elif 'covariate' in hyperparameter_search_extra_criteria:
                        return float('inf'), float('inf')
                    elif 'post_intervention_period' in hyperparameter_search_extra_criteria:
                        return float('inf'), float('inf')
                    else:
                        return float('inf')

                # Store best iteration in trial attributes
                trial.set_user_attr("best_iteration", results.best_iteration)

            except optuna.TrialPruned:
                # If Optuna pruned the trial via the callback
                raise optuna.TrialPruned()
            except Exception as e:
                # Handle other potential errors during xgb.train
                print(f"An error occurred during xgb.train for trial {trial.number}: {e}")
                if 'covariate' in hyperparameter_search_extra_criteria and 'post_intervention_period' in hyperparameter_search_extra_criteria:
                    return float('inf'), float('inf'), float('inf')
                elif 'covariate' in hyperparameter_search_extra_criteria:
                    return float('inf'), float('inf')
                elif 'post_intervention_period' in hyperparameter_search_extra_criteria:
                    return float('inf'), float('inf')
                else:
                    return float('inf')
            
            # The article describes a multi-objective optimization problem. 
            # Optuna is configured to minimize, pre-treatment error, covariate error, and post-intervention error (optional for diagnose).
            if 'covariate' in hyperparameter_search_extra_criteria and 'post_intervention_period' in hyperparameter_search_extra_criteria:
                return current_error_pre_intervention, current_error_covariates, current_error_post_intervention
            elif 'covariate' in hyperparameter_search_extra_criteria:
                return current_error_pre_intervention, current_error_covariates
            elif 'post_intervention_period' in hyperparameter_search_extra_criteria:
                return current_error_pre_intervention, current_error_post_intervention
            else:
                return current_error_pre_intervention

        # --- 4. Run the Optuna Study ---
        if 'covariate' in hyperparameter_search_extra_criteria and 'post_intervention_period' in hyperparameter_search_extra_criteria:
            directions=['minimize', 'minimize', 'minimize']
        elif 'covariate' in hyperparameter_search_extra_criteria:
            directions=['minimize', 'minimize']
        elif 'post_intervention_period' in hyperparameter_search_extra_criteria:
            directions=['minimize', 'minimize']
        else:
            directions=['minimize']

        study = optuna.create_study(
            directions=directions,
            sampler=sampler
        )

        # Run the Optuna study to find the best set of hyperparameters.
        study.optimize(objective, n_trials=number_optuna_trials, timeout=timeout_optuna_cycle, show_progress_bar=True)
