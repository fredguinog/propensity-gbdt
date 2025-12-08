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
  int<lower=1> N_train;       // Number of pre-treatment train time periods
  int<lower=1> N_val;      // Number of pre-treatment validation time periods
  int<lower=0> N_post;      // Number of post-treatment time periods
  int<lower=1> N_controls;  // Number of control units in the donor pool
  int<lower=1> N_outcomes;  // Number of outcome variables to fit simultaneously

  // --- Pre-treatment Train Data ---
  matrix[N_train, N_outcomes] Y_treated_train;     // Outcomes for the treated unit
  array[N_outcomes] matrix[N_train, N_controls] Y_control_train; // Control unit outcomes for each outcome variable

  // --- Pre-treatment Validation Data ---
  matrix[N_val, N_outcomes] Y_treated_val;     // Validation outcomes for the treated unit
  array[N_outcomes] matrix[N_val, N_controls] Y_control_val; // Validation control unit outcomes for each outcome variable

  // --- Post-treatment Data ---
  matrix[N_post, N_outcomes] Y_treated_post;    // Outcomes for the treated unit
  array[N_outcomes] matrix[N_post, N_controls] Y_control_post; // Control unit outcomes for each outcome variable
  
  // --- Priors and Constraints ---
  vector<lower=0>[N_controls] dirichlet_alpha; // Hyperparameter for Dirichlet prior on weights. Sparsity requires a value smaller than 1
  vector<lower=0>[N_outcomes] tau_nrmse_prior;
  
  // The fraction of treated volatility to use as the noise floor (e.g., 0.01)
  real<lower=0> noise_floor_fraction; 
}

transformed data {
  // Pre-calculate statistics for each outcome of the treated unit.
  // This is efficient as it only needs to be done once.
  vector[N_outcomes] mean_Y_treated_train;
  vector[N_outcomes] sd_Y_treated_train;
  vector[N_outcomes] nu; // The statistical noise floor per outcome
  real nu_T = 1.0 / log(N_train + N_val);
  // defensive clamp to avoid extreme values for tiny Tpre:
  if (nu_T < 1e-6) nu_T = 1e-6;
  if (nu_T > 0.5) nu_T = 0.5; // optional upper limit to avoid pure-standardization for tiny samples
  
  for (k in 1:N_outcomes) {
	vector[N_train + N_val] Y_treated_pre = append_row(Y_treated_train[, k], Y_treated_val[, k]);
	mean_Y_treated_train[k] = mean(Y_treated_pre);
    sd_Y_treated_train[k] = sd(Y_treated_pre);
	
    // Calculate nu based on treated unit volatility
    nu[k] = sd_Y_treated_train[k] * noise_floor_fraction;
  }
  
  // Create a time index vector for trend calculation
  vector[N_train] time_index_train = linspaced_vector(N_train, 1, N_train);
}

parameters {
  // The weights for the control units. A simplex is a vector of positive
  // numbers that sum to 1. This is the single, shared core parameter.
  simplex[N_controls] w;

  // The standard deviation of the model's error term for EACH outcome.
  // This captures how well the synthetic control can match the treated unit for each outcome.
  vector<lower=1e-6>[N_outcomes] sigma;
  
  vector<lower=0>[N_outcomes] tau_nrmse;  // Hierarchical prior for NRMSE per outcome
}

transformed parameters {
  // This block is executed for every MCMC sample of the parameters `w` and `sigma`.
  
  // This matrix will hold the final, rescaled synthetic control for each outcome.
  matrix[N_train, N_outcomes] y_synth_train_scaled;
  matrix[N_val, N_outcomes] y_synth_val_scaled;
  matrix[N_post, N_outcomes] y_synth_post_scaled;
  vector[N_outcomes] amplitude;
  
  for (k in 1:N_outcomes) {
    // 1. raw synthetic controls (pre-treatment train/val/post)
    vector[N_train] y_synth_train_k = Y_control_train[k] * w;
    vector[N_val]   y_synth_val_k   = Y_control_val[k]   * w;
    vector[N_post]  y_synth_post_k  = Y_control_post[k]  * w;

    // 2. compute means and pooled sd for synth pre-period (train+val)
    int N1 = N_train;
    int N2 = N_val;
    int Ntot = N1 + N2;
    real m1 = mean(y_synth_train_k);
    real m2 = mean(y_synth_val_k);
    // pooled variance (numerically stable)
    real ss1 = (N1 - 1) * variance(y_synth_train_k);
    real ss2 = (N2 - 1) * variance(y_synth_val_k);
    real ss_total = ss1 + ss2 + (N1 * N2 * square(m1 - m2)) / (Ntot);
    real mean_Y_synth_train_k = (N1 * m1 + N2 * m2) / (Ntot);
    real sd_Y_synth_train_k = sqrt( ss_total / (Ntot - 1) );

    // 3. per-outcome noise floor (keep your original logic; treat as epsilon_k)
    real epsilon_k = nu[k]; // nu[k] computed in transformed data as sd_treated * noise_floor_fraction
    // defensive lower bound
    if (epsilon_k < 1e-8) epsilon_k = 1e-8;

    // 4. Standardized synthetic control (shape), using variance floor inside denominator
    real denom_std = sqrt(square(sd_Y_synth_train_k) + square(epsilon_k)); 
    vector[N_train] y_synth_train_std_k = (y_synth_train_k - mean_Y_synth_train_k) / denom_std;
    vector[N_val]   y_synth_val_std_k   = (y_synth_val_k   - mean_Y_synth_train_k) / denom_std;
    vector[N_post]  y_synth_post_std_k  = (y_synth_post_k  - mean_Y_synth_train_k) / denom_std;

    // 5. Rescale standardized synth to the treated unit's scale (treated mean & sd)
    real mu_treated = mean_Y_treated_train[k];
    real sd_treated = sd_Y_treated_train[k];
    vector[N_train] std_rescaled_train = y_synth_train_std_k * sd_treated + mu_treated;
    vector[N_val]   std_rescaled_val   = y_synth_val_std_k   * sd_treated + mu_treated;
    vector[N_post]  std_rescaled_post  = y_synth_post_std_k  * sd_treated + mu_treated;

    // 6. Raw rescaled: shift synthetic series to treated mean (no variance normalization)
    // This aligns scales by mean only and leaves variance of donors untouched.
    vector[N_train] raw_rescaled_train = (y_synth_train_k - mean_Y_synth_train_k) + mu_treated;
    vector[N_val]   raw_rescaled_val   = (y_synth_val_k   - mean_Y_synth_train_k) + mu_treated;
    vector[N_post]  raw_rescaled_post  = (y_synth_post_k  - mean_Y_synth_train_k) + mu_treated;

    // 7. Final mixture: (1 - nu_T) * raw + nu_T * standardized-rescaled
    y_synth_train_scaled[, k] = (1.0 - nu_T) * raw_rescaled_train + nu_T * std_rescaled_train;
    y_synth_val_scaled[, k]   = (1.0 - nu_T) * raw_rescaled_val   + nu_T * std_rescaled_val;
    y_synth_post_scaled[, k]  = (1.0 - nu_T) * raw_rescaled_post  + nu_T * std_rescaled_post;
    
    amplitude[k] = max(Y_treated_train[, k]) - min(Y_treated_train[, k]);
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
    Y_treated_train[, k] ~ normal(y_synth_train_scaled[, k], sigma[k]);
  
    // --- CONSTRAINTS / PENALTIES ---
    // These terms penalize the log-posterior if the residuals for this outcome have bad properties.
    
    // 1. Calculate pre-treatment residuals (on the original, un-standardized scale)
    vector[N_train] residuals_train = Y_treated_train[, k] - y_synth_train_scaled[, k];
    vector[N_val] residuals_val = Y_treated_val[, k] - y_synth_val_scaled[, k];

    // 2. Calculate diagnostic correlations and nrmse
    real nrmse_train = sqrt(mean(square(residuals_train / amplitude[k])));
    real nrmse_val_minus_one = sqrt(mean(square(residuals_val[1:(N_val-1)] / amplitude[k])));
	real nrmse_terminal = abs(residuals_val[N_val] / amplitude[k]);
	
    // 3. Add soft constraints by specifying tight priors on these nrmse
    nrmse_train ~ normal(0, tau_nrmse[k]) T[0,];
    nrmse_val_minus_one ~ normal(0, tau_nrmse[k]) T[0,];
	nrmse_terminal ~ normal(0, tau_nrmse[k]) T[0,];
  }
}

generated quantities {
  // This block generates quantities of interest for each outcome using the posterior draws.
  
  // We flatten the log-likelihoods into a vector of size (N_train * N_outcomes)
  vector[N_train * N_outcomes] log_lik;
  int idx = 1;
  
  // Generate draws from the posterior predictive distribution for the effect of each outcome.
  // This incorporates the estimated model noise `sigma` into our prediction. 
  // Predictive residuals
  matrix[N_train, N_outcomes] predictive_train;
  matrix[N_val, N_outcomes] predictive_val;
  matrix[N_post, N_outcomes] predictive_post;
  matrix[N_train, N_outcomes] residuals_train;
  matrix[N_val, N_outcomes] residuals_val;
  matrix[N_post, N_outcomes] effect_post;
  vector[N_outcomes] nrmse_train;
  vector[N_outcomes] nrmse_val_minus_one;
  vector[N_outcomes] nrmse_terminal;
  vector[N_outcomes] nrmse_post;
  real gini_weight = normalized_gini_simplex(w);
  real hhi_weight = normalized_hhi_simplex(w);
  
  // Structural for raw trends
  matrix[N_train, N_outcomes] struc_residuals_train;
  matrix[N_val, N_outcomes] struc_residuals_val;
  matrix[N_post, N_outcomes] struc_effect_post;
  vector[N_outcomes] struc_nrmse_train;
  vector[N_outcomes] struc_nrmse_val_minus_one;
  vector[N_outcomes] struc_nrmse_terminal;
  vector[N_outcomes] struc_nrmse_post;
  
  for (k in 1:N_outcomes) {
    for (t in 1:N_train) {
		
	  // Calculate the log-probability of the data point given the parameters
	  log_lik[idx] = normal_lpdf(Y_treated_train[t, k] | y_synth_train_scaled[t, k], sigma[k]);
	  idx += 1;
		
	  predictive_train[t, k] = normal_rng(y_synth_train_scaled[t, k], sigma[k]);
      residuals_train[t, k] = Y_treated_train[t, k] - predictive_train[t, k];
	  struc_residuals_train[t, k] = Y_treated_train[t, k] - y_synth_train_scaled[t, k];
    }
    for (t in 1:N_val) {
	  predictive_val[t, k] = normal_rng(y_synth_val_scaled[t, k], sigma[k]);
      residuals_val[t, k] = Y_treated_val[t, k] - predictive_val[t, k];
	  struc_residuals_val[t, k] = Y_treated_val[t, k] - y_synth_val_scaled[t, k];
    }
    for (t in 1:N_post) {
	  predictive_post[t, k] = normal_rng(y_synth_post_scaled[t, k], sigma[k]);
      effect_post[t, k] = Y_treated_post[t, k] - predictive_post[t, k];
	  struc_effect_post[t, k] = Y_treated_post[t, k] - y_synth_post_scaled[t, k];
    }

    nrmse_train[k] = sqrt(mean(square(residuals_train[, k] / amplitude[k])));
    nrmse_val_minus_one[k] = sqrt(mean(square(residuals_val[1:(N_val-1), k] / amplitude[k])));
	nrmse_terminal[k] = abs(residuals_val[N_val, k]) / amplitude[k];
    nrmse_post[k] = sqrt(mean(square(effect_post[, k] / amplitude[k])));

    struc_nrmse_train[k] = sqrt(mean(square(struc_residuals_train[, k] / amplitude[k])));
    struc_nrmse_val_minus_one[k] = sqrt(mean(square(struc_residuals_val[1:(N_val-1), k] / amplitude[k])));
    struc_nrmse_terminal[k] = abs(struc_residuals_val[N_val, k]) / amplitude[k];
	struc_nrmse_post[k] = sqrt(mean(square(struc_effect_post[, k] / amplitude[k])));
  }
}
