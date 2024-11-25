
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


# Create a figure with a 2x8 grid of subplots
fig, ax = plt.subplots(num_rows, num_cols, figsize=(8, 16), dpi=300)  # 2 columns, 8 rows

for i, competitor in enumerate(personal_df_list):

    n_steps = personal_df_list[competitor].shape[0]
    obs_true = np.array(personal_df_list[competitor]['Mark_wm'])[:, np.newaxis]
    obs_alter = np.array(personal_df_list[competitor]['Mark'])[:, np.newaxis]
    
    # Initialize data defaults
    np.random.seed(43)
    N = len(obs_true)  # Number of time steps
    T = 1    # Number of competitors (if multiple, modify the model accordingly)
    obs_sigma = float(avg_std) # Observation noise
    y = obs_true.flatten()  # Observations
    y_alt = obs_alter.flatten()
    mu_0 = float(mean_std)
    sigma_0 = 0.05
    
    # Prepare data for Stan
    stan_data = {
          "N": N,                      # Number of time steps
          "T": T,                      # Number of competitors
          "y": y.tolist(),               # Observations
          "obs_sigma": obs_sigma,      # Observation noise
          "mu_0": mu_0,   # initial time point
          "sigma_0": sigma_0,  # std of start point
    }
    
    # Choice of the stan model
    stan_model_file = "kalman_rep.stan"
    model = CmdStanModel(stan_file=stan_model_file)
    # Run
    fit = model.sample(data=stan_data, chains=4, iter_sampling=1500, iter_warmup=500)
    # Extract latent states
    latent_state_samples = fit.stan_variable("latent_state")
    # Calculate posterior mean and credible intervals
    s_mu_samples = fit.stan_variable("s_mu")
    filtered_s_mu = np.mean(latent_state_samples, axis=0)
    s_cov = np.mean(fit.stan_variable("s_cov"))
    one_step_ahead_samples = latent_state_samples[:, :-1] + np.mean(s_mu_samples)
    next_race_sample = latent_state_samples[:, -1] + np.mean(s_mu_samples)
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
    lower_bound_next = np.percentile(next_race_sample, 2.5, axis=0)
    upper_bound_next =  np.percentile(next_race_sample, 97.5, axis=0)
    predicted_next = np.mean(next_race_sample, axis=0)
    
    # # Make a tensorflow dataset
    # X_train = tf.data.Dataset.from_tensors(obs_true)
    # x = next(iter(X_train.batch(batch_size=100).map(lambda x: tf.cast(x, dtype=tf.float32))))[0]
    # print(f"{competitor}: {x.shape}")
    # opt_params = optimize_transmission_noise(x = x, params = params, ssm = forward_filter_lgssm_mv)
    # 
    # L, filtered_means, filtered_covs, predicted_means, predicted_covs, observation_means, observation_covs = \
    # forward_filter_lgssm_mv( x, opt_params)
    
    ax_row = ax[i, :]

    # Call the plotting function with the processed data
    plot_multi_stan_outcome(
        sequences=None,
        ax=ax_row,
        competitor=competitor,
        lower_credible = lower_credible, 
        upper_credible = upper_credible,
        lower_credible_next = lower_bound_next,
        upper_credible_next = upper_bound_next,
        predicted_next = predicted_next, 
        obs_true=y,  # Observed data (replace with actual observed data array)
        obs_alter=y_alt,
        filtered_means=filtered_s_mu,
        observation_means=posterior_mean,
        fitted_mcmc = fit
    )

plt.tight_layout()
plt.savefig('competitor_kalman_plots_3.png')  # Save the combined plot
plt.show()




