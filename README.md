**Application of State-Space model for 8 athletes of the 100m sprint**

# Goal:

Apply and understand the State-Space model approach with Bayesian Filtering. The example case is the 100m time progression of the 8 athletes who made it to the Paris 100m Olympic final.

-   Model the progression path of each of the 8 athletes over their past 4 seasons in the 100m.

<!-- -->

-   Make a prediction for the 100m final that took place at the Paris Olympics 2024 on August 4th using only data from before that day.

Models 1 to 3 are focused on understanding and implementation. From Model 4 onward, more emphasis will be placed on assumptions and exploratory data analysis.

I use several resources for running the project denoted in [the references section](#References)

![Fabrizio Bensch/Reuters](https://github.com/V-Mitch/track_ssm/blob/master/assets/start_100m.jpg)

(Photo: Fabrizio Bensch/Reuters)

### Table of Contents

0.  [Depiction of the State-Space Model](#Depiction-of-State-Space-Model)
1.  [Model 1: Kalman Filter & Tensorflow](#Model-1)
2.  [Model 2: Wind Adjustment](#Model-2)
3.  [Model 3: Changing the State Optimizer to MCMC with Stan](#Model-3)
4.  *Work In Progress*
5.  [Comparing Results](#Comparing-Results)

# Depiction of State-Space Model {#depiction-of-state-space-model}

$s_t$ is the hidden state(s) that be interpreted as the fitness level of the sprinter $y_t$ is the observation we have: the 100m time of the sprinter $t$ is the race number $t$ at which the competitor competed

![](https://github.com/V-Mitch/track_ssm/blob/master/assets/depiction_1.png)

# Model 1: 

*Kalman Filter & Tensorflow*

## Assumptions of Model 1

-   First, I assume the state-space model is a Kalman Filter with linear emission and transmission functions. So this means that the $w_t$ and $v_t$ follow the normal distribution.
-   Next, I fill in the variance of measurement noise, $v_t$ with a rough approximation.
-   The transmission noise or $w_t$ has it's mean and variance optimized with the tensorflow gradient function. (Automatic Differentiation) in such a way that minimizes the loss.
-   Set $s_0$, the initial state as the average clocking for all athletes in all races in the database since 2020. This only includes elite athletes who have run a time below 10.30. It is universal and is not athlete-specific.
-   $\mu_w$ and $\sigma_w$ are optimized in such a way that minimizes the loss function. In this case, it is the negative log-likelihood function for the entire path of a single athlete.
-   Fixing the parameter, $\beta = 1$ means that we can interpret the *unobserved fitness level* as the 100m clocking the athlete is capable of.
-   $\sigma_v$"Naively fixing the randomness that can occur during any race as the average per-season deviation of the athletes in the database."

Specifically, using standard notation:

$s_t = f(s_{t-1},w_t) = s_{t-1} + w_t$

$w_t \sim \mathcal{N}(\mu_{w}, \sigma_{w})$

$s_0 = \hat{\mu}_{clockings}$\
$\mu_w$ and $\sigma_w$ are optimized.

$y_t = h(s_t, v_t) = \beta * s_t + v_t$\
$v_t \sim \mathcal{N}(\mu_{v}, \sigma_{v})$

$\beta = 1$\
$\mu_v = 0$\
$\sigma_v = \hat{\sigma}_{clockings}$

## Results of Model 1

| Name | State Variable |   | Name | Observation Variable |
|---------------|:-------------:|:-------------:|:-------------:|:-------------:|
| ${\color{orange}{fit\ level}}$ | $s_t$ |  | ${\color{blue}{Race\ Result}}$ | $y_t$ |
| fit_prog_avg | $\mu_{w}$ |  | obs_pred | $p(y_t \| s_{t-1})$ |
| fit_var | $\sigma_{w}$ |  | Next Race Prediction | $p(y_T \| s_{T-1})$ |

![](https://github.com/V-Mitch/track_ssm/blob/master/assets/competitor_kalman_plots_1.png)

## Commentary of Model 1

Many of the assumptions do not reflect realism in this first iteration of the state-space model. Hopefully, it provides a good starting point for improvement.

In order of importance the following most important points are not yet accounted for correctly. These initial observations are from general knowledge of the sport:

-   The runners do not exert full effort in races for qualifications or where the stakes are lower to avoid injury and preserve strength for later races or stages.
-   Earlier in a season, runners are slower than later in a season even if their general progression is on an uptrend. - As the fitness level improves, it becomes increasingly difficult to make incremental improvements. Thus, the transmission function of the fitness level state is not linear and does not have normally distributed "noise".
-   From my observation and experience it seems as though the distribution of performances is likely negatively skewed and not normally distributed. It is more likely for a blunder to happen every so often whereas a "perfect" race is only marginally better than the average performance.
-   Wind is not accounted for. At each race, it is known that a wind blowing against a runner will slow them down whereas a wind blowing behind them will speed them up.

# Model 2: 

*Wind Adjustment*

## Assumptions of Model 2

-   I stick to the same assumptions as model 1 but I change the $y_t$ variable from being the race result to a race result "modified for wind". Essentially, I decided to borrow the results of the *Moinat* paper that determines that the *change* in a 100m time due to wind can be determined by the following equation:

$$ \Delta P = P - 0.0049 w + 0.009459 P w - 0.0042w^2 $$

## Results of Model 2

| Name | State Variable |   | Name | Observation Variable |
|---------------|:-------------:|:-------------:|:-------------:|:-------------:|
| ${\color{orange}{fit\ level}}$ | $s_t$ |  | ${\color{red}{Race\ Result}}$ | not used |
| fit_prog_avg | $\mu_{w}$ |  | obs_pred | $p(y_t \| s_{t-1})$ |
| fit_var | $\sigma_{w}$ |  | Next Race Prediction | $p(y_T \| s_{T-1})$ |
|  |  |  | ${\color{blue}{Wind-Corrected Result}}$ | $y_t$ |

![](https://github.com/V-Mitch/track_ssm/blob/master/assets/competitor_kalman_plots_2.png)

## Commentary of Model 2

Changing the observations to wind-adjusted performances has generally lead to (a worsening) increase in the loss function across the athletes. Nonetheless, I will keep it as is for now since it enables me to compare the times of different athletes without worrying about a wind advantage that may have been more favorable for some. The difference appears negligible qualitatively.

# Model 3: 

*Changing the State Optimizer and framework to MCMC with Stan*

## Assumptions of Model 3

When working on the first 2 projects, I first used the [LinearGaussianStateSpaceModel](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/LinearGaussianStateSpaceModel) tensorflow function that was pre-built and easier to deploy. In order to: 1.) Ensure that the behavior of the function was correct and 2.) Future adaptations to tackle the challenge of relaxing assumptions and providing a more realistic model, 3.) Improve my understanding of the State-Space model, I built a re-engineered function "[forward_filter_lgssm_mv](https://github.com/V-Mitch/track_ssm/blob/master/build_ssm.py)". Despite successfully re-creating the Tensorflow function and exposing the specific closed-form formulas of the Kalman Filter, I realized I needed more flexibility. I figured that switching to an MCMC framework (that also alows for optimization of $\mu_w$ and $\sigma_w$) would mainly help with the following:

-   Experimenting seamlessly with different non-gaussian distributions for emission and transition functions.
-   No need to explicitly discover the closed-form solutions like in the Kalman Filter.
-   The State transition could be non-linear.
-   Uncertainty quantification in the Bayesian sense (interpretation of credible intervals).

## Results of Model 3

| Name | State Variable |   | Name | Observation Variable |
|---------------|:-------------:|:-------------:|:-------------:|:-------------:|
| ${\color{orange}{fit\ level}}$ | $s_t$ |  | ${\color{red}{Race\ Result}}$ | not used |
| fit_prog_avg | $\mu_{w}$ |  | obs_pred | $p(y_t \| s_{t-1})$ |
| fit_var | $\sigma_{w}$ |  | Next Race Prediction | $p(y_T \| s_{T-1})$ |
|  |  |  | ${\color{blue}{Wind-Corrected Result}}$ | $y_t$ |

![](https://github.com/V-Mitch/track_ssm/blob/master/assets/competitor_kalman_plots_3.png)

# Comparing Results

## Overall Path Error

| Runner   | MSE\*100 | Model 1 | Model 2 | Model 3 |
|----------|----------|---------|---------|---------|
| LYLES    |          | 0.95    | 0.88    | 0.50    |
| THOMPSON |          | 1.28    | 1.18    | 0.93    |
| KERLEY   |          | 1.01    | 0.88    | 0.53    |
| SIMBINE  |          | 2.58    | 1.49    | 1.00    |
| JACOBS   |          | 0.92    | 0.74    | 0.44    |
| TEBOGO   |          | 3.73    | 1.82    | 1.46    |
| BEDNAREK |          | 0.95    | 1.10    | 0.75    |
| SEVILLE  |          | 1.76    | 0.69    | 0.39    |

## Race day Prediction

| Runner   | Time  | Model 1 | Model 2 | Model 3 |
|----------|-------|---------|---------|---------|
| LYLES    | 9.784 | 9.79    | 9.81    | 9.90    |
| THOMPSON | 9.789 | 9.83    | 9.84    | 9.98    |
| KERLEY   | 9.81  | 9.90    | 9.91    | 9.97    |
| SIMBINE  | 9.82  | 9.90    | 9.89    | 9.94    |
| JACOBS   | 9.85  | 9.92    | 9.95    | 10.02   |
| TEBOGO   | 9.86  | 9.94    | 9.95    | 9.99    |
| BEDNAREK | 9.88  | 9.91    | 9.94    | 9.95    |
| SEVILLE  | 9.91  | 9.82    | 9.83    | 9.92    |

![](https://github.com/V-Mitch/track_ssm/blob/master/assets/finish_100m.jpg)

# References {#references}

*Inspiration for code*: <https://colab.research.google.com/drive/1TdVykmUdLp8Qzr5-4XnG1Seov6HhSgPc?usp=sharing>

*Theory*:

-   Hagiwara, Junichiro. *Time Series Analysis for the State-Space Model with R/Stan. Springer EBooks*, Springer Nature, 1 Jan. 2021.

<!-- -->

-   Moinat M, Fabius O, Emanuel KS. *Data-driven quantification of the effect of wind on athletics performance.* Eur J Sport Sci. 2018 Oct;18(9):1185-1190. doi: 10.1080/17461391.2018.1480062. Epub 2018 Jun 11. PMID: 29890896.
