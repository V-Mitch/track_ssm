import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

def build_tfp_lg_ssm(num_timesteps: int, params: dict):
  
  obs_coeff = params.get('obs_coeff', 1)
  s_mu = params.get('s_mu', -0.02)
  s_cov = params.get('s_cov', 0.05)
  obs_mu = params.get('obs_mu', 1)
  init_s_mu = params.get('init_s_mu', 10)
  init_s_cov = params.get('init_s_cov', 0.05)
  obs_cov = params.get('obs_cov', 0.05)
  
  transition_matrix = [[1.]]
  observation_matrix = [[obs_coeff]]
  transition_noise = tfd.MultivariateNormalDiag(
      loc = [tf.convert_to_tensor(s_mu)],
      scale_diag= [tf.convert_to_tensor(s_cov)])
  observation_noise = tfd.MultivariateNormalDiag(
      loc = [obs_mu],
      scale_diag = [obs_cov])
  initial_state_prior = tfd.MultivariateNormalDiag(
      loc =[init_s_mu], 
      scale_diag = [init_s_cov])
  model = tfd.LinearGaussianStateSpaceModel(
    num_timesteps=num_timesteps,
    transition_matrix=transition_matrix,
    transition_noise=transition_noise,
    observation_matrix=observation_matrix,
    observation_noise=observation_noise,
    initial_state_prior=initial_state_prior)
  
  return model

def build_linear_gaussian_jdc(num_timesteps, params):
    def model():
        # Extract parameters with defaults
        obs_coeff = params.get('obs_coeff', 1.00)
        s_mu = params.get('s_mu', -0.02)
        s_cov = params.get('s_cov', 0.05)
        obs_mu = params.get('obs_mu', 1.00)
        init_s_mu = params.get('init_s_mu', 10)
        init_s_cov = params.get('init_s_cov', 0.05)
        obs_cov = params.get('obs_cov', 0.05)

        # Define matrices and noise distributions
        transition_matrix = tf.constant([[1.]])  # Transition matrix
        observation_matrix = tf.constant([[obs_coeff]])  # Observation matrix

        # Initial state prior (assumed Gaussian)
        state_t = yield tfd.MultivariateNormalDiag(
            loc=[init_s_mu], 
            scale_diag=[init_s_cov]
        )

        for t in range(num_timesteps):  # Use the num_timesteps from outer scope
            # Transition model: Linear transition with Gaussian noise
            state_t = yield tfd.MultivariateNormalDiag(
                loc=tf.linalg.matvec(transition_matrix, state_t),
                scale_diag=[s_cov]  # Constant transition noise scale
            )
            
            # Observation model: Linear emission with Gaussian noise
            obs_t = yield tfd.MultivariateNormalDiag(
                loc=tf.linalg.matvec(observation_matrix, state_t) + obs_mu,
                scale_diag=[obs_cov]  # Constant observation noise scale
            )

    return tfd.JointDistributionCoroutine(model)

def forward_filter_nl(non_linear_ssm, params, observations):
    """ Perform filtering for the non-linear state-space model. """
    num_timesteps = observations.shape[0]
    
    obs_coeff = params['obs_coeff']
    obs_mu = params['obs_mu']
    obs_cov = params['obs_cov']
    s_mu = params['s_mu']
    s_cov = params['s_cov']
    init_s_cov = params['init_s_cov']
    init_s_mu = params['init_s_mu']
    
    # Initial state prior
    sample_output = non_linear_ssm.sample()  # Sample from the model
    current_state_mean = tf.expand_dims(tf.convert_to_tensor(init_s_mu), axis = 0) # Accessing var0 directly
    current_state_cov = tf.expand_dims(tf.convert_to_tensor(init_s_cov), axis = 0)  # Set initial covariance, adjust as necessary

    filtered_means = []
    filtered_covs = []
    predicted_means = []
    predicted_covs = []
    observation_means = []
    observation_covs = []
    likelihoods = []
    
    for t in range(num_timesteps):
        # Predict step
        predicted_state_mean = current_state_mean  # Non-linear transition function
        
        # Assume transition noise as some constant for now
        transition_noise = tf.convert_to_tensor(s_cov)  
        predicted_state_cov =  transition_noise  # Add transition noise
        
        # Observation prediction (non-linear emission)
        predicted_obs_mean = obs_coeff * predicted_state_mean + obs_mu
        observation_noise = tf.convert_to_tensor(obs_cov)  # Assume some constant observation noise
        predicted_obs_cov = observation_noise  # Add observation noise
        
        # Store predictions
        predicted_means.append(tf.convert_to_tensor(predicted_state_mean, dtype=tf.float32))
        predicted_covs.append(tf.convert_to_tensor(predicted_state_cov, dtype=tf.float32))
        observation_means.append(tf.convert_to_tensor(predicted_obs_mean, dtype=tf.float32))
        observation_covs.append(tf.convert_to_tensor(predicted_obs_cov, dtype=tf.float32))
        
        # Observation update (using observations)
        innovation = observations[t] - predicted_obs_mean
        kalman_gain = kalman_gain = predicted_state_cov / (predicted_state_cov + observation_noise)
        
        # Update current state mean and covariance
        current_state_mean = predicted_state_mean + kalman_gain * innovation
        current_state_cov = (1 - kalman_gain) * predicted_state_cov
        
        # Store filtered estimates
        filtered_means.append(tf.convert_to_tensor(current_state_mean, dtype=tf.float32))
        filtered_covs.append(tf.convert_to_tensor(current_state_cov, dtype=tf.float32))
        
        # Calculate likelihood
        # Assuming Gaussian likelihood: P(observation | prediction) = N(observation; predicted_obs_mean, predicted_obs_cov)
        # Compute the likelihood
        likelihood = -0.5 * (tf.math.log(2 * np.pi * predicted_obs_cov) + 
                              tf.square(innovation) / predicted_obs_cov)
        likelihoods.append(likelihood)
    
    # Convert lists to tensors and check their shapes
    filtered_means_tensor = tf.stack(filtered_means)
    filtered_covs_tensor = tf.stack(filtered_covs)
    predicted_means_tensor = tf.stack(predicted_means)
    predicted_covs_tensor = tf.stack(predicted_covs)
    observation_means_tensor = tf.stack(observation_means)
    observation_covs_tensor = tf.stack(observation_covs)
    likelihoods_tensor = tf.stack(likelihoods)

    return (likelihoods_tensor, 
            filtered_means_tensor, 
            filtered_covs_tensor,
            predicted_means_tensor, 
            predicted_covs_tensor, 
            observation_means_tensor,
            observation_covs_tensor)
    
# Example forward filtering call
# observations = tf.random.normal([num_timesteps, 1])  # Mock observation data
# dict_nl = forward_filter_nl(model, obs_true)
