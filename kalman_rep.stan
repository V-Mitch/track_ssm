



// Data block: specify observations and parameters for the Kalman filter.
data {
  int<lower=1> N;           // Number of time steps
  int<lower=1> T;           // Number of competitors
  vector[N] y;              // Observations (e.g., race times)
  real<lower=0> obs_sigma;  // Observation noise standard deviation
  real<lower=0> trans_sigma; // Transition noise standard deviation
}

// Parameters block: define latent states and model parameters.
parameters {
  vector[N] latent_state;   // Latent states (fitness levels across time)
  real mu_0;                // Initial state mean
  real<lower=0> sigma_0;    // Initial state standard deviation
}

// Model block: define the state-space model and likelihood.
model {
  // Initial state prior
  latent_state[1] ~ normal(mu_0, sigma_0);
  
  // State transition model (Gaussian random walk)
  for (t in 2:N) {
    latent_state[t] ~ normal(latent_state[t-1], trans_sigma);
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