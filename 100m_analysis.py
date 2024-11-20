

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

# from cmdstanpy import install_cmdstan
# install_cmdstan(overwrite=True)

from cmdstanpy import CmdStanModel


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
    's_cov': 0.125997,
    'init_s_mu': float(mean_std),
    'init_s_cov': 0.05,
    'obs_mu': 0.00,
    'obs_cov': float(avg_std) 
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
  forward_filter_lgssm_mv( x, params)

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
    
    L, filtered_means, filtered_covs, predicted_means, predicted_covs, observation_means, observation_covs = \
    forward_filter_lgssm_mv( x, opt_params)
    
    ax_row = ax[i, :]

    plot_tfp_kalman_s2(ax[i, :], obs_true, obs_alter, filtered_means, observation_means, observation_covs, 
                      opt_params, competitor, n_steps=n_steps)

plt.tight_layout()
plt.savefig('competitor_kalman_plots_2.png')  # Save the combined plot
plt.show()




# Initialize data
np.random.seed(43)
N = len(obs_true)  # Number of time steps
T = 1    # Number of competitors (if multiple, modify the model accordingly)
obs_sigma = float(avg_std) # Observation noise
y = obs_true.flatten()  # Observations
mu_0 = float(mean_std)
sigma_0 = 0.05

# Prepare data for Stan
stan_data = {
    "N": N,                      # Number of time steps
    "T": T,                      # Number of competitors
    "y": y.tolist(),             # Observations
    "obs_sigma": obs_sigma,      # Observation noise
    "mu_0": mu_0,   # initial time point
    "sigma_0": sigma_0,  # std of start point
}

stan_model_file = "kalman_rep.stan"
model = CmdStanModel(stan_file=stan_model_file)

# Fit the model
fit = model.sample(data=stan_data, chains=4, iter_sampling=1500, iter_warmup=500)
print(fit.summary())

# Extract latent states
latent_state_samples = fit.stan_variable("latent_state")

# Calculate posterior mean and credible intervals
s_mu_samples = fit.stan_variable("s_mu")
filtered_s_mu = np.mean(latent_state_samples, axis=0)
s_cov = np.mean(fit.stan_variable("s_cov"))
one_step_ahead_samples = latent_state_samples[:, :-1] + s_mu
# obs_prediction = np.mean(one_step_ahead_samples, axis=0)
num_samples, N = latent_state_samples.shape
t1_samples = np.random.normal(mu_0 + s_mu_samples, sigma_0, size=num_samples)
lagged_samples = latent_state_samples[:, :-1] 
one_step_ahead_samples = lagged_samples + s_mu_samples[:, None] 
all_one_step_ahead_samples = np.hstack([t1_samples[:, None], one_step_ahead_samples]) 
# Step 2: Compute posterior mean and credible intervals
posterior_mean = np.mean(all_one_step_ahead_samples, axis=0)  # Shape: [N]
lower_credible = np.percentile(all_one_step_ahead_samples, 2.5, axis=0)  # Shape: [N]
upper_credible = np.percentile(all_one_step_ahead_samples, 97.5, axis=0)  # Shape: [N]

filtered_covs = np.var(one_step_ahead_samples, axis=0)

# Create a figure with side-by-side plots
fig, ax = plt.subplots(1, 2, figsize=(12, 6))  # 1 row, 2 columns

# Call your plotting function with the processed data
plot_single_stan_outcome(
    sequences=None,
    ax=ax,
    lower_credible = lower_credible, 
    upper_credible = upper_credible,
    obs_true=y,  # Observed data (replace with actual observed data array)
    filtered_means=filtered_s_mu,
    observation_means=posterior_mean,
)

plt.tight_layout()
plt.show()

# Plot the results
# import matplotlib.pyplot as plt

# plt.figure(figsize=(10, 6))
# plt.plot(y, label="Observed Data", color="black")
# plt.plot(posterior_mean, label="Posterior Mean of Latent State", color="blue")
# plt.fill_between(range(N), lower_credible, upper_credible, color="blue", alpha=0.2, label="95% Credible Interval")
# plt.legend()
# plt.xlabel("Time")
# plt.ylabel("Value")
# plt.title("Latent State Estimation with Bayesian MCMC")
# plt.show()


