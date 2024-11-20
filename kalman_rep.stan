



// Data block: specify observations and parameters for the Kalman filter.
data {
  int<lower=1> N;           // Number of time steps
  int<lower=1> T;           // Number of competitors
  vector[N] y;              // Observations (e.g., race times)
  real<lower=0> obs_sigma;  // Observation noise standard deviation
  real<lower=0> mu_0;        // Initial time prior
  real<lower=0> sigma_0;     // std of start time prior
}

// Parameters block: define latent states and model parameters.
parameters {
  vector[N] latent_state;                      // Latent states (fitness levels across time)
  real<lower=-1, upper=1> s_mu;                // state drift
  real<lower=0> s_cov;                         // state covariance
}

// Model block: define the state-space model and likelihood.
model {
  
  // Initial state priors
  latent_state[1] ~ normal(mu_0, sigma_0);
  
  // State transition model (Gaussian random walk)
  for (t in 2:N) {
    latent_state[t] ~ normal(latent_state[t-1] + s_mu , s_cov);
  }

  // Observation model
  for (t in 1:N) {
    y[t] ~ normal(latent_state[t], obs_sigma);
  }
}

// Generated quantities block (optional): predictive checks, etc.
generated quantities {
  vector[N] y_pred; // Predicted observations
  for (t in 1:N) {
    y_pred[t] = normal_rng(latent_state[t], obs_sigma);
  }
}