// -*- coding: utf-8 -*-

// Copyright (c) 2025 Frederico Guilherme Nogueira (frederico.nogueira@gmail.com)
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// -----------------------------------------------------------------------------
// Bayesian Synthetic Control Model with Multiple Outcomes and Custom Standardization
//
// Author: Frederico Guilherme Nogueira
// Date:   June 26, 2025
//
// Description:
// This Stan model estimates a SINGLE set of weights `w` for a synthetic control
// that provides the best joint fit across MULTIPLE outcome variables.
//
// Key Feature:
// The model implements a special standardization procedure for each outcome. For each
// MCMC draw of the weights `w`, the resulting pre-treatment synthetic control series
// for each outcome is standardized and rescaled using the statistics of the
// corresponding treated unit's pre-treatment series before the likelihood is evaluated.
// -----------------------------------------------------------------------------
functions {
  // Optimized vectorized Normalized Gini coefficient (0 to 1 scale)
  real normalized_gini_simplex(vector y) {
    int n = num_elements(y);
    vector[n] y_sorted;
    vector[n] idx;
    
    if (n == 0) reject("Gini requires n > 0");
    if (n == 1) return 0.0; // A single unit has 100% weight, but variance is 0. Conventionally 0 or 1, but 0 is safer for penalty logic.
    
    y_sorted = sort_asc(y);
    for (i in 1:n) idx[i] = i;
    
    real sum_iy = dot_product(idx, y_sorted);
    
    return (2 * sum_iy - (n + 1.0)) / (n - 1.0);
  }
  
  // Normalized Herfindahl-Hirschman Index (HHI)
  // Scale: 0.0 (Perfectly Diversified) to 1.0 (Single Unit Concentration)
  real normalized_hhi_simplex(vector w) {
    int n = num_elements(w);
    
    // Safety check for single donor case
    if (n <= 1) return 1.0; 
    
    // HHI is simply the sum of squared weights.
    // In vector notation, this is the dot product of w with itself.
    real raw_hhi = dot_product(w, w);
    
    // Calculate theoretical minimum HHI (1/N)
    real min_hhi = 1.0 / n;
    
    // Normalize: (Val - Min) / (Max - Min)
    // Max is always 1.0 for a simplex
    return (raw_hhi - min_hhi) / (1.0 - min_hhi);
  }
}

data {
  // --- Dimensions ---
  int<lower=1> N_pre;       // Number of pre-treatment train time periods
  int<lower=1> N_test;      // Number of pre-treatment test time periods
  int<lower=0> N_post;      // Number of post-treatment time periods
  int<lower=1> N_controls;  // Number of control units in the donor pool
  int<lower=1> N_outcomes;  // Number of outcome variables to fit simultaneously

  // --- Pre-treatment Train Data ---
  matrix[N_pre, N_outcomes] Y_treated_pre;     // Outcomes for the treated unit
  array[N_outcomes] matrix[N_pre, N_controls] Y_control_pre; // Control unit outcomes for each outcome variable

  // --- Pre-treatment Test Data ---
  matrix[N_test, N_outcomes] Y_treated_test;     // Test outcomes for the treated unit
  array[N_outcomes] matrix[N_test, N_controls] Y_control_test; // Test control unit outcomes for each outcome variable

  // --- Post-treatment Data ---
  matrix[N_post, N_outcomes] Y_treated_post;    // Outcomes for the treated unit
  array[N_outcomes] matrix[N_post, N_controls] Y_control_post; // Control unit outcomes for each outcome variable
  
  // --- Priors and Constraints ---
  vector<lower=0>[N_controls] dirichlet_alpha; // Hyperparameter for Dirichlet prior on weights. Sparsity requires a value smaller than 1
  vector<lower=0>[N_outcomes] tau_nrmse_prior;
  
  // The fraction of treated volatility to use as the noise floor (e.g., 0.1)
  real<lower=0> noise_floor_fraction; 
}

transformed data {
  // Pre-calculate statistics for each outcome of the treated unit.
  // This is efficient as it only needs to be done once.
  vector[N_outcomes] mean_Y_treated_pre;
  vector[N_outcomes] sd_Y_treated_pre;
  vector[N_outcomes] nu; // The statistical noise floor per outcome
  
  for (k in 1:N_outcomes) {
	mean_Y_treated_pre[k] = mean(Y_treated_test[, k]);
    sd_Y_treated_pre[k] = sd(Y_treated_pre[, k]);
	
    // Calculate nu based on treated unit volatility
    nu[k] = sd_Y_treated_pre[k] * noise_floor_fraction;
  }
  
  // Create a time index vector for trend calculation
  vector[N_pre] time_index_pre = linspaced_vector(N_pre, 1, N_pre);
}

parameters {
  // The weights for the control units. A simplex is a vector of positive
  // numbers that sum to 1. This is the single, shared core parameter.
  simplex[N_controls] w;

  // The standard deviation of the model's error term for EACH outcome.
  // This captures how well the synthetic control can match the treated unit for each outcome.
  vector<lower=0>[N_outcomes] sigma;
  
  vector<lower=0>[N_outcomes] tau_nrmse;  // Hierarchical prior for NRMSE per outcome
}

transformed parameters {
  // This block is executed for every MCMC sample of the parameters `w` and `sigma`.
  
  // This matrix will hold the final, rescaled synthetic control for each outcome.
  matrix[N_pre, N_outcomes] y_synth_pre_scaled;
  matrix[N_test, N_outcomes] y_synth_test_scaled;
  matrix[N_post, N_outcomes] y_synth_post_scaled;
  vector[N_outcomes] amplitude;
  
  for (k in 1:N_outcomes) {
    // 1. Create the pre-treatment synthetic control for the current outcome `k` using the shared weights `w`.
    vector[N_pre] y_synth_pre_k = Y_control_pre[k] * w;
    vector[N_test] y_synth_test_k = Y_control_test[k] * w;
    vector[N_post] y_synth_post_k = Y_control_post[k] * w;
    
    // 2. Standardize this synthetic control using its OWN mean and sd.
	real mean_Y_synth_pre_k = mean(y_synth_test_k);
    real sd_Y_synth_pre_k = sd(y_synth_pre_k);
    vector[N_pre] y_synth_pre_std_k;
    vector[N_test] y_synth_test_std_k;
    vector[N_post] y_synth_post_std_k;

    // --- Noise Floor Implementation (INA Fix) ---
    // We use the statistical noise floor `nu` calculated in transformed data.
    // Denominator = sqrt(sd_synth^2 + nu^2)    
    real denominator = sqrt(square(sd_Y_synth_pre_k) + square(nu[k]));
    
    y_synth_pre_std_k = (y_synth_pre_k - mean_Y_synth_pre_k) / denominator;
    y_synth_test_std_k = (y_synth_test_k - mean_Y_synth_pre_k) / denominator;
    y_synth_post_std_k = (y_synth_post_k - mean_Y_synth_pre_k) / denominator;
    
    // 3. Rescale the standardized synthetic control to the scale of the TREATED unit's outcome.
    y_synth_pre_scaled[, k] = (y_synth_pre_std_k * sd_Y_treated_pre[k]) + mean_Y_treated_pre[k];
    y_synth_test_scaled[, k] = (y_synth_test_std_k * sd_Y_treated_pre[k]) + mean_Y_treated_pre[k];
    y_synth_post_scaled[, k] = (y_synth_post_std_k * sd_Y_treated_pre[k]) + mean_Y_treated_pre[k];
    
    amplitude[k] = max(Y_treated_pre[, k]) - min(Y_treated_pre[, k]);
  }
}

model {
  // --- Priors ---
  // A Dirichlet prior is the natural choice for the shared weights `w`.
  w ~ dirichlet(dirichlet_alpha);

  // A weakly informative prior on the error term for each outcome.
  sigma ~ cauchy(0, 2.5);
  
  tau_nrmse ~ cauchy(0, tau_nrmse_prior);

  // --- Likelihood and Constraints (applied per outcome) ---
  for (k in 1:N_outcomes) {
    // --- Likelihood ---
    // The model's likelihood for outcome k is defined on the treatment scale.
    // It states that the pre-treatment outcome of the treated unit is normally
    // distributed around the rescaled pre-treatment synthetic control.
    Y_treated_pre[, k] ~ normal(y_synth_pre_scaled[, k], sigma[k]);
  
    // --- CONSTRAINTS / PENALTIES ---
    // These terms penalize the log-posterior if the residuals for this outcome have bad properties.
    
    // 1. Calculate pre-treatment residuals (on the original, un-standardized scale)
    vector[N_pre] residuals_pre = Y_treated_pre[, k] - y_synth_pre_scaled[, k];
    vector[N_test] residuals_test = Y_treated_test[, k] - y_synth_test_scaled[, k];

    // 2. Calculate diagnostic correlations and nrmse
    real nrmse_pre = sqrt(mean(square(residuals_pre / amplitude[k])));
    real nrmse_test = sqrt(mean(square(residuals_test / amplitude[k])));

    // 3. Add soft constraints by specifying tight priors on these nrmse
    nrmse_pre ~ normal(0, tau_nrmse[k]) T[0,];
    nrmse_test ~ normal(0, tau_nrmse[k]) T[0,];
  }
}

generated quantities {
  // This block generates quantities of interest for each outcome using the posterior draws.
  
  // We flatten the log-likelihoods into a vector of size (N_pre * N_outcomes)
  vector[N_pre * N_outcomes] log_lik;
  int idx = 1;
  
  // Generate draws from the posterior predictive distribution for the effect of each outcome.
  // This incorporates the estimated model noise `sigma` into our prediction. 
  // Predictive residuals
  matrix[N_pre, N_outcomes] predictive_pre;
  matrix[N_test, N_outcomes] predictive_test;
  matrix[N_post, N_outcomes] predictive_post;
  matrix[N_pre, N_outcomes] residuals_pre;
  matrix[N_test, N_outcomes] residuals_test;
  matrix[N_post, N_outcomes] effect_post;
  vector[N_outcomes] nrmse_pre;
  vector[N_outcomes] nrmse_test;
  vector[N_outcomes] nrmse_post;
  real gini_weight = normalized_gini_simplex(w);
  real hhi_weight = normalized_hhi_simplex(w);
  
  // Structural for raw trends
  matrix[N_pre, N_outcomes] struc_residuals_pre;
  matrix[N_test, N_outcomes] struc_residuals_test;
  matrix[N_post, N_outcomes] struc_effect_post;
  vector[N_outcomes] struc_nrmse_pre;
  vector[N_outcomes] struc_nrmse_test;
  vector[N_outcomes] struc_nrmse_post;
  
  for (k in 1:N_outcomes) {
    for (t in 1:N_pre) {
		
	  // Calculate the log-probability of the data point given the parameters
	  log_lik[idx] = normal_lpdf(Y_treated_pre[t, k] | y_synth_pre_scaled[t, k], sigma[k]);
	  idx += 1;
		
	  predictive_pre[t, k] = normal_rng(y_synth_pre_scaled[t, k], sigma[k]);
      residuals_pre[t, k] = Y_treated_pre[t, k] - predictive_pre[t, k];
	  struc_residuals_pre[t, k] = Y_treated_pre[t, k] - y_synth_pre_scaled[t, k];
    }
    for (t in 1:N_test) {
	  predictive_test[t, k] = normal_rng(y_synth_test_scaled[t, k], sigma[k]);
      residuals_test[t, k] = Y_treated_test[t, k] - predictive_test[t, k];
	  struc_residuals_test[t, k] = Y_treated_test[t, k] - y_synth_test_scaled[t, k];
    }
    for (t in 1:N_post) {
	  predictive_post[t, k] = normal_rng(y_synth_post_scaled[t, k], sigma[k]);
      effect_post[t, k] = Y_treated_post[t, k] - predictive_post[t, k];
	  struc_effect_post[t, k] = Y_treated_post[t, k] - y_synth_post_scaled[t, k];
    }

    nrmse_pre[k] = sqrt(mean(square(residuals_pre[, k] / amplitude[k])));
    nrmse_test[k] = sqrt(mean(square(residuals_test[, k] / amplitude[k])));
    nrmse_post[k] = sqrt(mean(square(effect_post[, k] / amplitude[k])));

    struc_nrmse_pre[k] = sqrt(mean(square(struc_residuals_pre[, k] / amplitude[k])));
    struc_nrmse_test[k] = sqrt(mean(square(struc_residuals_test[, k] / amplitude[k])));
    struc_nrmse_post[k] = sqrt(mean(square(struc_effect_post[, k] / amplitude[k])));
  }
  
  vector[N_outcomes] nrmse_pre_test = (nrmse_pre + nrmse_test) / 2;
  vector[N_outcomes] struc_nrmse_pre_test = (struc_nrmse_pre + struc_nrmse_test) / 2;

}
