
import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)
from datetime import datetime
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
tfd = tfp.distributions

df_raw = pd.read_csv('men 100m from_2022-01-01 to_2024-09-30.csv')

def convert_dates(date):
    try:
        return pd.to_datetime(date, format='%d %b %Y')
    except ValueError:
        return pd.to_datetime(date, format='%Y', errors='coerce')  
      
df_raw['DOB'] = df_raw['DOB'].apply(convert_dates)
df_raw['Date'] = df_raw['Date'].apply(convert_dates)

# Only interested in the line-up of the 2024 olympics race
competitors = ['Noah LYLES', 'Kishane THOMPSON', 'Fred KERLEY', 
              'Akani SIMBINE', 'Oblique SEVILLE', 
              'Lamont Marcell JACOBS', 'Letsile TEBOGO', 'Kenneth BEDNAREK']
date_max = pd.to_datetime('2024-08-03')

df_raw = df_raw[df_raw['Date'] <= date_max]
df_raw = df_raw[df_raw['Competitor'].isin(competitors)]

def df_by_athlete(data):
  data_sorted = data.sort_values(by=['Competitor', 'Date'])
  competitor_dfs = {name: group for name, group in data_sorted.groupby('Competitor')}
  return(competitor_dfs)
  
personal_df_list = df_by_athlete(df_raw)

params = {
    'obs_coeff': 1,
    's_drift': -0.02,
    'trans_cov': 0.05,
    'obs_dim': 0,
    'init_s_mu': 10,
    'init_s_cov': 0.05,
    'obs_cov': 0.05  
}

def build_tfp_lg_ssm(num_timesteps: int, params: dict):
  
  obs_coeff = params.get('obs_coeff', 1)
  s_drift = params.get('s_drift', -0.02)
  trans_cov = params.get('trans_cov', 0.05)
  obs_dim = params.get('obs_dim', 1)
  obs_cov = params.get('obs_cov', 0.15)
  init_s_mu = params.get('init_s_mu', 10)
  init_s_cov = params.get('init_s_cov', 0.05)
  obs_cov = params.get('obs_cov', 0.05)
  
  transition_matrix = [[1.]]
  observation_matrix = [[obs_coeff]]
  transition_noise = tfd.MultivariateNormalDiag(
      loc = [s_drift],
      scale_diag= [trans_cov])
  observation_noise = tfd.MultivariateNormalDiag(
      loc = [obs_dim],
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

n_steps = personal_df_list['Noah LYLES'].shape[0]
model = build_tfp_lg_ssm(params = params, num_timesteps = n_steps)
obs_true = np.array(personal_df_list['Noah LYLES']['Mark'])[:,np.newaxis]

# Make a tensorflow dataset
X_train = tf.data.Dataset.from_tensors(obs_true)
# batch it up
x = next(iter(X_train.batch(batch_size=100).map(lambda x : tf.cast(x, dtype=tf.float32))))[0]
print(x.shape)
_, filtered_means, filtered_covs, predicted_means, predicted_covs, observation_means, _ = model.forward_filter(x)

def emission_function(filtered_means, obs_coeff, obs_dim):
    # Apply the linear emission function
    return obs_coeff * filtered_means + obs_dim
  
def plot_tfp_kalman_s(sequences, ax, obs_true, filtered_means, filtered_covs, 
    obs_coeff, predicted_means, observation_means, obs_dim):
    x_axis = np.arange(len(filtered_means))

    # Calculate predicted observations
    # obs_predictions = emission_function(observation_means.flatten(), obs_coeff, obs_dim)
    obs_predictions = observation_means.numpy().flatten()

    # Calculate prediction intervals
    lower_bound = obs_predictions - 1.96 * np.sqrt(filtered_covs.flatten())  # 95% prediction interval lower bound
    upper_bound = obs_predictions + 1.96 * np.sqrt(filtered_covs.flatten())  # 95% prediction interval upper bound

    # Plot the filtered state estimates
    ax[0].plot(x_axis, filtered_means.flatten(), label='s_kf_estimate', color='orange')
    ax[0].legend(loc='best')
    ax[0].set_title('Filtered State Estimates')

    # Plot the predicted observations
    ax[1].plot(x_axis, obs_predictions, label='obs_kf_pred', linestyle='--', color='#33FF57')
    ax[1].plot(x_axis, obs_true.flatten(), label='obs_true', color='blue')  # Plot true observations for comparison
    
    # Shade the prediction intervals
    ax[1].fill_between(x_axis, lower_bound, upper_bound, color='lightgreen', alpha=0.5, label='Prediction Interval')
    
    ax[1].legend(loc='best')
    ax[1].set_title('Predicted Observations vs True Observations')
    
    
# Create a figure with side-by-side plots
fig, ax = plt.subplots(1, 2, figsize=(12, 6))  # 1 row, 2 columns

# Call the plot function
plot_tfp_kalman_s(sequences=None, ax=ax, obs_true=obs_true, 
                   predicted_means=predicted_means.numpy(),
                   filtered_means=filtered_means.numpy(),
                   filtered_covs=filtered_covs.numpy(), 
                   observation_means=observation_means,
                   obs_coeff=params.get('obs_coeff', 1), 
                   obs_dim=params.get('obs_dim', 1))

plt.tight_layout()
plt.show()

(some of the old plotting code, there was also the make_plot)
def plot_tfp_kalman_s(sequences, ax):
    d =  filtered_means
    x_axis = np.arange(len(d))
    ax[i][0].plot(x_axis, d, label='s_kf_estimate',color='orange')
    ax[i][0].legend(loc='best')
    # Now plot the corresponding predictions of observations based on the filtered means
    # obs_predictions = syn_dataset.emission(d)  # Apply the emission function to the filtered means
    # ax[i][1].plot(x_axis, obs_predictions, label='obs_kf_pred', linestyle='--',color='#33FF57')
    ax[i][1].legend(loc='best')



