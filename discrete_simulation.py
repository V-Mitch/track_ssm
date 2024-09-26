import functools
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import util as tfp_util 
import sys

# alias
tfd = tfp.distributions

n_plots = 7
n_sim = 20
n_steps = 100

class SyntheticDataSetGenerator(object):
    """ Linear transition and emission functions """
    def __init__(self, 
                 s_dim:int,
                 s_drift:float,
                 init_s_mu:float,
                 init_s_cov:float,
                 trans_cov:float,
                 obs_dim:int,
                 obs_drift:float,               
                 obs_cov:float,
                 obs_coeff:float,
                 obs_dist:str):
      
        self.s_dim = s_dim
        self.s_drift = s_drift
        self.init_s_mu = init_s_mu
        self.init_s_cov = init_s_cov
        self.trans_cov = trans_cov    
        self.obs_dim = obs_dim        
        self.obs_drift = obs_drift
        self.obs_cov = obs_cov   
        self.obs_coeff = obs_coeff
        self.obs_dist = obs_dist  # Store the observation distribution type
    
    def transition(self, s):
        return s + self.s_drift
    
    def emission(self, s):
        """ Emission function depends on the observation distribution """
        if self.obs_dist == "Normal":
            return self.obs_coeff * s + self.obs_drift  # Linear emission for Normal
        elif self.obs_dist == "Poisson":
            # Exponential to ensure the rate is positive for Poisson
            return np.exp(self.obs_coeff * s + self.obs_drift)
    
    def sample_normal(self, mu, cov):
        """ Sampling from a normal distribution """
        return mu + np.random.randn(*mu.shape) * np.sqrt(cov)
    
    def sample_poisson(self, rate):
        """ Sampling from a Poisson distribution """
        return np.random.poisson(rate)
    
    def __call__(self, num_samples:int, num_steps:int):    
        all_s = []
        
        # create the first s
        s_prev_mu = np.ones((num_samples,1,self.s_dim)) * self.init_s_mu    
        s_prev = self.sample_normal(s_prev_mu, self.init_s_cov)
        
        all_s.append(s_prev)
        
        # now we generate s for subsequent time steps
        for t in range(num_steps - 1):
            # apply the transition function
            s_mu = self.transition(s_prev)
            s_prev = self.sample_normal(s_mu, self.trans_cov)
            all_s.append(s_prev)
    
        s_true = np.concatenate(all_s, axis=1)
    
        # generating the obs from these latent variables
        obs_emitted = self.emission(s_true)

        # obs_emitted acts as our mu for Normal or rate for Poisson
        if self.obs_dist == "Normal":
            obs_true = self.sample_normal(obs_emitted, self.obs_cov)
        elif self.obs_dist == "Poisson":
            obs_true = self.sample_poisson(obs_emitted)
          
        return s_true, obs_true
    
# The Dataset Generator takes parameters
# so that we can customize the synthetic data
#
# for now tested only with below configuration

syn_dataset = SyntheticDataSetGenerator(
      s_dim = 1,    
      s_drift = 0.,
      init_s_mu = 0,    
      init_s_cov = 1.,
      trans_cov = 0.25, 
      obs_dim = 1,
      obs_drift = 0.,
      obs_cov = 2.,
      obs_coeff= 0.5,
      obs_dist = "Poisson")
      
s_true, obs_true = syn_dataset(n_sim, n_steps)
s_true.shape, obs_true.shape

# a helper function to plot both latent variable s
# and observation obs

def make_plots(
    sequences,
    x, 
    s,     
    extra_plots=[]):

  num_plots = len(sequences)

  fig, ax = plt.subplots(len(sequences), 2, figsize=(16, 8))
  fig.subplots_adjust(hspace=.5)

  def plot_x():
    for i in range(num_plots):
      d =  x[sequences[i]]
      x_axis = np.arange(len(d))
      ax[i][1].plot(x_axis, d, label='true obs')
      ax[i][1].legend(loc='best')

  def plot_s():
    for i in range(num_plots):
      d = s[sequences[i]]
      x_axis = np.arange(len(d))
      ax[i][0].plot(x_axis, d, label='true latent')  
      ax[i][0].legend(loc='best')

  plot_x()
  plot_s()
  
  # call the user supplied plot
  # functions (if any)
  for e in extra_plots:
    e(sequences, ax)

  left_side = ax[0,0]
  left_side.set_title('Latent Space')
  left_side.legend(loc='upper center', bbox_to_anchor=(0.5, 2.2),ncol=3, frameon=True)
  ax[0][0].legend(loc='best')

  right_side = ax[0,1]
  right_side.set_title('Observations')
  right_side.legend(loc='upper center', bbox_to_anchor=(0.5, 2.2),ncol=3, frameon=True)
  ax[0][1].legend(loc='best')

  plt.show(fig)
  

# specialized for training plots
make_train_plots = functools.partial(make_plots, x=obs_true, s=s_true)
  
# just get a 3 random sequences to plot
np.random.seed(3)
sequences = np.random.permutation(n_sim)[:n_plots]

make_train_plots(sequences=sequences)
      
      
def build_tfp_lg_ssm(syn_dataset: SyntheticDataSetGenerator,  num_timesteps: int):  

  transition_matrix = [[1.]]
  observation_matrix = [[syn_dataset.obs_coeff]]

  transition_noise = tfd.MultivariateNormalDiag(
      loc = [syn_dataset.s_drift],
      scale_diag= [syn_dataset.trans_cov])

  observation_noise = tfd.MultivariateNormalDiag(
      loc = [syn_dataset.obs_dim],
      scale_diag = [syn_dataset.obs_cov])

  initial_state_prior = tfd.MultivariateNormalDiag(
      loc =[syn_dataset.init_s_mu], 
      scale_diag = [syn_dataset.init_s_cov])

  model = tfd.LinearGaussianStateSpaceModel(
    num_timesteps=num_timesteps,
    transition_matrix=transition_matrix,
    transition_noise=transition_noise,
    observation_matrix=observation_matrix,
    observation_noise=observation_noise,
    initial_state_prior=initial_state_prior)
  
  return model

class CustomPoissonNoise(tfd.Poisson):
    def __init__(self, rate, validate_args=False, allow_nan_stats=True, name="CustomPoissonNoise"):
        # Call the base class constructor first
        super(CustomPoissonNoise, self).__init__(
            rate=rate,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            name=name
        )
        # Store the parameters for later access
        self._parameters = dict(rate=rate)

    def _parameter_properties(self, num_classes=None):
        return dict(
            rate=tfp.util.ParameterProperties(event_ndims=0)
        )

    def mean(self):
        return super(CustomPoissonNoise, self).mean()  # Call the base class's mean method

    def variance(self):
        return super(CustomPoissonNoise, self).variance()  # Call the base class's variance method
      
    def covariance(self):
        # Covariance for a Poisson distribution is equivalent to its variance
        return tf.linalg.diag(self.variance())  # Return a diagonal matrix of the variance
      
def build_custom_poisson_ssm(syn_dataset: SyntheticDataSetGenerator, num_timesteps: int):  
    # Define the transition matrix and noise
    transition_matrix = [[1.0]]  # Simple linear state transition
    observation_matrix = [[syn_dataset.obs_coeff]]  # Controls how latent state affects emissions

    # Transition noise is Gaussian
    transition_noise = tfd.MultivariateNormalDiag(
        loc=[syn_dataset.s_drift],
        scale_diag=[syn_dataset.trans_cov]
    )

    # Initial state prior is Gaussian
    initial_state_prior = tfd.MultivariateNormalDiag(
        loc=[syn_dataset.init_s_mu], 
        scale_diag=[syn_dataset.init_s_cov]
    )
    
    # Instead of using deterministic noise, use the custom Poisson noise
    observation_noise = CustomPoissonNoise(rate=tf.ones([1]))  # Placeholder rate

    # Build a custom state-space model with Poisson emissions
    class PoissonStateSpaceModel(tfd.LinearGaussianStateSpaceModel):
        def observation_distribution(self, latent_state):
            """
            Override the observation distribution with Poisson instead of Gaussian.
            The rate (lambda) for Poisson is the exponential of the linear transformation of the state.
            """
            rate = tf.exp(tf.linalg.matmul(latent_state, observation_matrix) + syn_dataset.obs_drift)
            return tfd.Poisson(rate=rate)

    # Instantiate the custom model
    model = PoissonStateSpaceModel(
        num_timesteps=num_timesteps,
        transition_matrix=transition_matrix,
        transition_noise=transition_noise,
        observation_matrix=observation_matrix,  # Technically not used for the Poisson observation but for the latent process
        observation_noise=observation_noise,  # Not used, as we override it with Poisson
        initial_state_prior=initial_state_prior
    )
    
    return model


# model = build_tfp_lg_ssm(syn_dataset)
model_poisson = build_custom_poisson_ssm(syn_dataset, num_timesteps = n_steps)

# Make a tensorflow dataset
X_train = tf.data.Dataset.from_tensors(obs_true)

# batch it up
x = next(iter(X_train.batch(batch_size=n_sim).map(lambda x : tf.cast(x, dtype=tf.float32))))[0]
print(x.shape)

# _, filtered_means, filtered_covs, predicted_means, predicted_covs, _, _ = model.forward_filter(x)
_, filtered_means_poisson, _, predicted_means_poisson, _, observation_means, _ = model_poisson.forward_filter(x)

predicted_means.shape, filtered_means.shape  # predicted_means gives the t+1 prediction

# def plot_tfp_kalman_s(sequences, ax):
#   for i in range(len(sequences)):
#     d =  filtered_means[sequences[i]]
#     x_axis = np.arange(len(d))
#     ax[i][0].plot(x_axis, d, label='s_kf_estimate',color='orange')
#     ax[i][0].legend(loc='best')
#     # Now plot the corresponding predictions of observations based on the filtered means
#     obs_predictions = syn_dataset.emission(d)  # Apply the emission function to the filtered means
#     ax[i][1].plot(x_axis, obs_predictions, label='obs_kf_pred', linestyle='--',color='#33FF57')
#     ax[i][1].legend(loc='best')
    
def plot_tfp_poisson_s(sequences, ax):
  for i in range(len(sequences)):
    d = 2*np.log(filtered_means_poisson[sequences[i]])
    x_axis = np.arange(len(d))
    ax[i][0].plot(x_axis, d, label='s_ss_estimate',color='orange')
    ax[i][0].legend(loc='best')
    # Now plot the corresponding predictions of observations based on the filtered means
    obs_predictions = syn_dataset.emission(d)  # Apply the emission function to the filtered means
    ax[i][1].plot(x_axis, obs_predictions, label='obs_ss_pred', linestyle='--',color='#33FF57')
    ax[i][1].legend(loc='best')
    
# change seed to other seqeuences
np.random.seed(3)
sequences = np.random.permutation(n_sim)[:n_plots]
make_train_plots(
    sequences=sequences, 
    extra_plots=[plot_tfp_poisson_s])

