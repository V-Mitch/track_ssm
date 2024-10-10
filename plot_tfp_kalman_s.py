import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

def plot_tfp_kalman_s(ax_row, obs_true, filtered_means, observation_means, observation_covs, 
                      params, competitor, n_steps):
    s_mu = round(params.get('s_mu'), 4)
    s_cov = round(params.get('s_cov'), 4)
    
    x_axis = np.arange(len(filtered_means))

    # Calculate prediction intervals for existing observations
    lower_bound = observation_means.numpy().flatten() - 1.96 * np.sqrt(observation_covs.numpy().flatten())
    upper_bound = observation_means.numpy().flatten() + 1.96 * np.sqrt(observation_covs.numpy().flatten())

    # Plot the filtered state estimates
    ax_row[0].plot(x_axis, filtered_means.numpy().flatten(), label='fit_level', color='orange')
    ax_row[0].legend(loc='lower left', ncol=1)
    ax_row[0].set_title(f'{competitor} - Filtered State Estimates')
    ax_row[0].set_ylim(bottom=9.2)  # Adjust y-axis as needed

    # Add fitness_progression_avg, fitness_variability, and number_of_races as text annotations on the left graph
    textstr = (f'fit_prog_avg: {s_mu:.4f}\n'
               f'fit_var: {s_cov:.4f}\n'
               f'n_races: {n_steps}')
               
    ax_row[0].text(0.95, 0.05, textstr, transform=ax_row[0].transAxes, fontsize=10,
                   verticalalignment='bottom', horizontalalignment='right',
                   bbox=dict(facecolor='white', edgecolor = 'lightgray', alpha=0.5))

    # Plot the predicted observations
    ax_row[1].plot(x_axis, observation_means.numpy().flatten(), label='obs_pred', linestyle='--', color='#33FF57')
    ax_row[1].plot(x_axis, obs_true.flatten(), label='Race Result', color='blue')  # Plot true observations for comparison
    
    # Shade the prediction intervals for existing observations
    ax_row[1].fill_between(x_axis, lower_bound, upper_bound, 
                           color='lightgreen', alpha=0.5, label='Prediction Interval')
    
    # Get the last filtered state and covariance for the next observation prediction
    last_filtered_mean = observation_means.numpy()[-1]  # Last filtered mean
    last_filtered_cov = observation_covs.numpy()[-1]  # Last filtered covariance

    # Predict the next observation (t + 1)
    predicted_next_mean = last_filtered_mean * params.get('obs_coeff', 1) + params.get('obs_mu', 0)

    # Calculate the standard deviation for prediction interval
    predicted_next_std = np.sqrt(last_filtered_cov * params.get('obs_coeff', 1)**2)

    # Calculate prediction interval bounds for the next observation
    lower_bound_next = (predicted_next_mean - 1.96 * predicted_next_std).flatten()
    upper_bound_next = (predicted_next_mean + 1.96 * predicted_next_std).flatten()
    x_values_next = np.array([len(filtered_means), len(filtered_means) + 1])  # Shape (2,)
    y_lower_values = np.array([lower_bound_next, lower_bound_next]).flatten()  # Shape (2,)
    y_upper_values = np.array([upper_bound_next, upper_bound_next]).flatten()  # Shape (2,)
    predicted_display_mean = np.array([predicted_next_mean, predicted_next_mean])
    predicted_next_mean_display = round(float(predicted_display_mean[0,0]),4)

    # Extend the prediction interval for the next observation
    ax_row[1].fill_between(x_values_next, 
                            y_lower_values, 
                            y_upper_values,
                            color='lightcoral', alpha=0.5, label='Next Obs Prediction Interval')
    ax_row[1].plot(x_values_next, predicted_display_mean, label=f'Next Race Prediction: {predicted_next_mean_display}', linestyle='--', color='coral')
    
    # Plot the predicted next observation as a continuation
    # ax_row[1].plot(len(filtered_means), predicted_next_mean, 'o', color='lightcoral', label='Predicted Next Obs')

    ax_row[1].legend(loc='best', fontsize=6, ncol=2)
    ax_row[1].set_title(f'{competitor} - Predicted vs True Observations')
    ax_row[1].set_ylim(bottom=9.2)  # Ensure y-axis is the same for comparison

def plot_single_kalman_s(sequences, ax, obs_true, filtered_means, filtered_covs,
    obs_coeff, predicted_means, observation_means, observation_covs, obs_mu):
    x_axis = np.arange(len(filtered_means))

    # Calculate predicted observations
    obs_predictions = observation_means.numpy().flatten()

    # Calculate prediction intervals
    lower_bound = obs_predictions - 1.96 * np.sqrt(observation_covs.numpy().flatten())  # 95% prediction interval lower bound
    upper_bound = obs_predictions + 1.96 * np.sqrt(observation_covs.numpy().flatten())  # 95% prediction interval upper bound

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
