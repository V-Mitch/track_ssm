reticulate::use_virtualenv("~/.virtualenvs/r-reticulate", required = TRUE)
import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)
from datetime import datetime
import tensorflow_probability as tfp
tfd = tfp.distributions

df_raw = pd.read_csv('men 100m from_2024-01-01 to_2024-09-26.csv')

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
    'obs_dim': 1,
    'obs_cov': 0.15,
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

n_steps = personal_df_list['Letsile TEBOGO'].shape[0]
model = build_tfp_lg_ssm(params = params, num_timesteps = n_steps)
obs_true = np.array(personal_df_list['Letsile TEBOGO']['Mark'])[:,np.newaxis]

# Make a tensorflow dataset
X_train = tf.data.Dataset.from_tensors(obs_true)
# batch it up
x = next(iter(X_train.batch(batch_size=n_sim).map(lambda x : tf.cast(x, dtype=tf.float32))))[0]
print(x.shape)
_, filtered_means, filtered_covs, predicted_means, predicted_covs, observation_means, _ = model.forward_filter(x)

def plot_tfp_kalman_s(sequences, ax):
    d =  filtered_means
    x_axis = np.arange(len(d))
    ax[i][0].plot(x_axis, d, label='s_kf_estimate',color='orange')
    ax[i][0].legend(loc='best')
    # Now plot the corresponding predictions of observations based on the filtered means
    # obs_predictions = syn_dataset.emission(d)  # Apply the emission function to the filtered means
    # ax[i][1].plot(x_axis, obs_predictions, label='obs_kf_pred', linestyle='--',color='#33FF57')
    ax[i][1].legend(loc='best')



