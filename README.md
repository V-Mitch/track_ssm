**Application of State-Space model for 8 athletes of the 100m sprint**

# Goal:

Apply and understand the State-Space model approach with Bayesian Filtering. The example case is the 100m time progression of the 8 athletes who made it to the Paris 100m Olympic final.

I use several resources for running the project denoted in [the references section](#References)

# Main Depiction

![](https://github.com/V-Mitch/track_ssm/blob/master/depiction_1.png)
  
$s_t$ is the hidden state(s) that be interpreted as the fitness level of the sprinter  
$y_t$ is the observation we have: the 100m time of the sprinter  
$t$ is the race number $t$ at which the competitor competed  
  
# Model 1:
  
## Assumptions Model 1
- First, I assume the state-space model is a Kalman Filter with linear emission and transmission functions. So this means that the $w_t$ and $v_t$ follow the normal distribution.  
- Next, I fill in the variance of measurement noise, $v_t$ with a rough approximation.  
- The tranmission noise or $w_t$ has it's mean and variance optimized with the tensorflow gradient function. (Automatic Differentiation) in such a way that minimizes the loss.

Specifically, using the notation:
  
$s_t = f(s_{t-1},w_t) = s_{t-1} + w_t$  
where:  
$w_t \sim \mathcal{N}(\mu_{w}, \sigma_{w})$
  
$s_0 = \hat{\mu}_{clockings}$  
Set as the average clocking for all athletes in all races in the database since 2020. This only includes elite athletes who have run a time below 10.30. It is universal and is not athlete-specific.
$\mu_w$ and $\sigma_w$ are optimized. 
  
$y_t = h(s_t, v_t) = \beta * s_t + v_t$
where:  
$w_t \sim \mathcal{N}(\mu_{v}, \sigma_{v})$
  
$\beta = 1$  
This means that we can interpret the unobserved fitness level as the clocking he is capable of. 
$\mu_v = 0$  
This implicitly translates to: "on average, the athlete performs at his fitness level".  
$\sigma_v = \hat{\sigma}_{clockings}$   
This is naively fixing the randomness during any race as the average per-season deviation of the athletes in the database.  
  

![](https://github.com/V-Mitch/track_ssm/blob/master/competitor_kalman_plots.png)

# References 
Inspiration for code:
https://colab.research.google.com/drive/1TdVykmUdLp8Qzr5-4XnG1Seov6HhSgPc?usp=sharing  
Main resource for theory:
Hagiwara, Junichiro. *Time Series Analysis for the State-Space Model with R/Stan. Springer EBooks*, Springer Nature, 1 Jan. 2021.  