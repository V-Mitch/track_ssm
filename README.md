**Application of State-Space model for 8 athletes of the 100m sprint**

# Goal:

Apply and understand the State-Space model approach with Bayesian Filtering. The example case is the 100m time progression of the 8 athletes who made it to the Paris 100m Olympic final.

I use several resources for running the project denoted in [the references section](#References)

# Main Depiction

![](https://github.com/V-Mitch/track_ssm/blob/master/depiction_1.png)
  
$s_t$ is the hidden state(s) 
$y_t$ is the observation we have: the time of the sprinter
$t$ is the race number $t$ at which the competitor competed
  
# Model 1:
  
## Assumptions Model 1
- First, I assume the state-space model is a Kalman Filter with linear emission and transmission functions. So this means that the $w_t$ and $v_t$ follow the normal distribution.
- Next, I fill in the covariance of the state noise and measurement noise with very rough approximations.
- 
  
![](https://github.com/V-Mitch/track_ssm/blob/master/plot_1.png)

Naive Assumptions:  
$`\sqrt{3x-1}+(1+x)^2`$

# References 