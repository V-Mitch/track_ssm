

import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)
from datetime import datetime
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
import matplotlib.pyplot as plt
from optimize_transmission_noise import *
from plot_tfp_kalman_s import *
from build_ssm import *
from putils import *
from add_wind_mark import *

df_raw = pd.read_csv('men 100m from_2020-01-01 to_2024-09-30.csv')

def convert_dates(date):
    try:
        return pd.to_datetime(date, format='%d %b %Y')
    except ValueError:
        return pd.to_datetime(date, format='%Y', errors='coerce')  
      
df_raw['DOB'] = df_raw['DOB'].apply(convert_dates)
df_raw['Date'] = df_raw['Date'].apply(convert_dates)

# Analysis      
avg_std = df_raw.groupby(['Competitor', df_raw['Date'].dt.year])['Mark'].std().mean().round(4)
mean_std = df_raw.groupby(['Competitor', df_raw['Date'].dt.year])['Mark'].mean().mean().round(4)      
min_seas = df_raw.groupby(['Competitor', df_raw['Date'].dt.year])['Mark'].min()          
max_seas = df_raw.groupby(['Competitor', df_raw['Date'].dt.year])['Mark'].max() 
seas_improvements = max_seas - min_seas
improvement_avg = (max_seas - min_seas).mean().round(4)  
improvement_avg = df_raw.groupby(['Competitor'])['Mark'].diff().mean()
      

# Only interested in the line-up of the 2024 olympics race
competitors = ['Noah LYLES', 'Kishane THOMPSON', 'Fred KERLEY', 
              'Akani SIMBINE', 'Oblique SEVILLE', 
              'Lamont Marcell JACOBS', 'Letsile TEBOGO', 'Kenneth BEDNAREK']
date_max = pd.to_datetime('2024-08-03')

df_spec = df_raw[df_raw['Date'] <= date_max]
df_spec = df_spec[df_spec['Competitor'].isin(competitors)]

df_spec = add_wind_mark(df_spec)

def df_by_athlete(data):
  data_sorted = data.sort_values(by=['Competitor', 'Date'])
  competitor_dfs = {name: group for name, group in data_sorted.groupby('Competitor')}
  return(competitor_dfs)
  
personal_df_list = df_by_athlete(df_spec)

params = {
    'obs_coeff': 1.00,
    's_mu': -float(improvement_avg),
    # 's_mu': 0.00,
    's_cov': 0.01 ** 2,
    'init_s_mu': float(mean_std),
    'init_s_cov': 0.05 ** 2,
    'obs_mu': 0.00,
    'obs_cov': float(avg_std) ** 2
}

  
results = {}

# Create a figure with side-by-side plots
fig, ax = plt.subplots(1, 2, figsize=(12, 6))  # 1 row, 2 columns

obs_true = np.array(personal_df_list['Letsile TEBOGO']['Mark_wm'])[:, np.newaxis]

# Make a tensorflow dataset
X_train = tf.data.Dataset.from_tensors(obs_true)
x = next(iter(X_train.batch(batch_size=100).map(lambda x: tf.cast(x, dtype=tf.float32))))[0]
model = build_tfp_lg_ssm(len(obs_true), params)
model_lgssm = build_linear_gaussian_jdc(len(obs_true), params)

L, filtered_means, filtered_covs, predicted_means, predicted_covs, observation_means, observation_covs = \
  model.forward_filter(x, final_step_only=False)
#   
# L, filtered_means, filtered_covs, predicted_means, predicted_covs, observation_means, observation_covs = \
#   forward_filter_lgssm(model_lgssm, params, x)

L, filtered_means, filtered_covs, predicted_means, predicted_covs, observation_means, observation_covs = \
  forward_filter_lgssm_mv( params, x)

# Call the plot function
plot_single_kalman_s(sequences=None, ax=ax, obs_true=obs_true,
                   predicted_means=predicted_means.numpy(),
                   filtered_means=filtered_means.numpy(),
                   filtered_covs=filtered_covs.numpy(),
                   observation_means=observation_means,
                   observation_covs=observation_covs,
                   obs_coeff=params.get('obs_coeff', 1),
                   obs_mu=params.get('obs_mu', 1))

plt.tight_layout()
plt.savefig('single_path_kalman_plot.png')
plt.show()


# Assuming `personal_df_list` is a dictionary with competitors' data
num_competitors = len(personal_df_list)
num_rows = num_competitors
num_cols = 2  # 8 competitors, so we will have 8 columns

# Create a figure with a 2x8 grid of subplots
fig, ax = plt.subplots(num_rows, num_cols, figsize=(8, 16), dpi=300)  # 2 columns, 8 rows

for i, competitor in enumerate(personal_df_list):
    n_steps = personal_df_list[competitor].shape[0]
    model = build_tfp_lg_ssm(params=params, num_timesteps=n_steps)
    obs_true = np.array(personal_df_list[competitor]['Mark_wm'])[:, np.newaxis]
    obs_alter = np.array(personal_df_list[competitor]['Mark'])[:, np.newaxis]
    
    # Make a tensorflow dataset
    X_train = tf.data.Dataset.from_tensors(obs_true)
    x = next(iter(X_train.batch(batch_size=100).map(lambda x: tf.cast(x, dtype=tf.float32))))[0]
    print(f"{competitor}: {x.shape}")
    opt_params = optimize_transmission_noise(x = x, params = params, ssm = forward_filter_lgssm_mv)
    
    # opt_params = optimize_transmission_noise(params = params,
    # build_function = forward_filter_lgssm_mv,
    # x = x)
    
    L, filtered_means, filtered_covs, predicted_means, predicted_covs, observation_means, observation_covs = model.forward_filter(x, final_step_only=False)
    
    ax_row = ax[i, :]

    plot_tfp_kalman_s2(ax[i, :], obs_true, obs_alter, filtered_means, observation_means, observation_covs, 
                      opt_params, competitor, n_steps=n_steps)

plt.tight_layout()
plt.savefig('competitor_kalman_plots_2.png')  # Save the combined plot
plt.show()

