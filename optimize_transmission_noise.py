# Optimization

import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)
from datetime import datetime
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt

def optimize_transmission_noise(params: dict, build_function: callable, x: tf.Tensor,
initial_lr = 0.02, num_epochs = 80, decay_factor = 0.1, decay_epoch = 50):

  # Initial parameters dictionary
  initial_params = params
  
  # The tensorflow variables
  s_mu = tf.Variable(initial_params['s_mu'], dtype=tf.float32)
  s_cov = tf.Variable(initial_params['s_cov'], dtype=tf.float32)
  
  optimizer = tf.optimizers.Adam(learning_rate=0.01)
  
  # Optimization loop
  for epoch in range(num_epochs):
      if epoch > decay_epoch:
          lr = initial_lr * (decay_factor ** ((epoch - decay_epoch) // 10))
      else:
          lr = initial_lr
      with tf.GradientTape(persistent=True) as tape:
          params = {
              'obs_coeff': initial_params['obs_coeff'],
              's_mu': s_mu,
              's_cov': s_cov,
              'obs_mu': initial_params['obs_mu'],
              'init_s_mu': initial_params['init_s_mu'],
              'init_s_cov': initial_params['init_s_cov'],
              'obs_cov': initial_params['obs_cov']  
          }
          model = build_tfp_lg_ssm(num_timesteps=x.shape[0], params=params)
          L, _, _, _, _, _, _ = model.forward_filter(x, final_step_only=True)
          loss = -L
          
      grads = tape.gradient(loss, [s_mu, s_cov])
      
      optimizer.apply_gradients(zip(grads, [s_mu, s_cov]))
  
      if epoch % 10 == 0:  # Print every 10 epochs
          print(f'Epoch {epoch}, NLL: {loss.numpy()}, s_mu: {s_mu.numpy()}, s_cov: {s_cov.numpy()}')
          
      final_params = {
          'obs_coeff': initial_params['obs_coeff'],
          's_mu': s_mu.numpy(),
          's_cov': s_cov.numpy(),
          'obs_mu': initial_params['obs_mu'],
          'init_s_mu': initial_params['init_s_mu'],
          'init_s_cov': initial_params['init_s_cov'],
          'obs_cov': initial_params['obs_cov']  
      }
          
  return(final_params)
