
# LIBRARIES ########################

library(lightgbm)
library(data.table)
library(ggplot2)
library(scales)

# cleanup
rm(list = ls())
gc()

# FUNCTIONS ########################

split_gain <- function(gradient_l, hessian_l, gradient_r, hessian_r, reg_lambda, reg_gamma){
  
  return((
    (gradient_l^2 / (hessian_l+reg_lambda)) + 
      (gradient_r^2 / (hessian_r+reg_lambda)) - 
      ((gradient_l + gradient_r)^2 / (hessian_l+hessian_r+reg_lambda))
  ) - 
    reg_gamma) 
}

generate_claim_counts <- function(dt, var_impact){
  dt <- merge.data.table(x = dt, y = var_impact, by = c("var1", "var2"))
  random_pois <- as.numeric(lapply(dt[,lambda], function(x){rpois(n = 1, lambda = x)}))
  dt[, target := random_pois]
  return(dt)
}

run_model <- function(learning_rate, num_iterations, 
                      min_sum_hessian,  
                      dtrain_data, dtrain_label){
  
  dtrain <- lgb.Dataset(
    data = dtrain_data,
    label = dtrain_label,
    categorical_feature = c(1,2))
  
  param <- list(
    objective = "poisson",
    num_iterations = num_iterations, 
    learning_rate = learning_rate,
    min_sum_hessian = min_sum_hessian)
  
  lgb_model <- lgb.train(
    params = param, 
    data = dtrain,
    boosting = "gbdt",
    verbose = 1)
  
  return(lgb_model)
}

# DATA ################################

set.seed(100) 

# choose one, data 1 or data 2

# data 1: only 1 split
data_curr <- data.table(
  var1 = sample(c(0,1), prob = c(0.75, 0.25), size = 1000, replace = TRUE),
  var2 = 0
)

# data 2: two variables
data_curr <- data.table(
  var1 = sample(c(0,1), prob = c(0.6, 0.4), size = 1000, replace = TRUE),
  var2 = sample(c(0,1), prob = c(0.8, 0.2), size = 1000, replace = TRUE)
)

var_impact <- data.table(
  var1 = c(0,1,0,1),
  var2 = c(0,0,1,1),
  lambda = c(0.3, 0.7, 1.3, 1.9)
)

data_curr <- generate_claim_counts(data_curr, var_impact)

# MODEL #################################

dtrain_data <- as.matrix(data_curr[,.(var1, var2)])
dtrain_label <- as.matrix(data_curr[,.(target)])

rm(lgb_model)
lgb_model <- run_model(
  learning_rate = 0.3, num_iterations = 100, 
  min_sum_hessian = 276.75,
  dtrain_data = dtrain_data, dtrain_label = dtrain_label)

data_curr[,predict := predict(lgb_model,dtrain_data)]
data_curr[,predict_raw := predict(lgb_model,dtrain_data, rawscore = TRUE)]

data_curr[,.(.N, mean_target = mean(target),predict = predict[1], 
             predict_raw = predict_raw[1]), keyby = .(var1, var2)]
#    var1 var2   N mean_target   predict predict_raw
# 1:    0    0 457   0.2735230 0.2735230  -1.2963696
# 2:    0    1 117   1.2051282 1.2051282   0.1865859
# 3:    1    0 340   0.6823529 0.6823529  -0.3822083
# 4:    1    1  86   2.0813953 2.0813953   0.7330385

data_curr[,mean(target)] 
# 0.677

data_curr[,.(.N, sum(target)), keyby = .(var1, var2)]
#    var1 var2   N  V2
# 1:    0    0 457 125
# 2:    0    1 117 141
# 3:    1    0 340 232
# 4:    1    1  86 179



# THEORY ######################

# according to 
# https://github.com/microsoft/LightGBM/blob/4b1b412452218c5be5ac0f238454ec9309036798/src/objective/regression_objective.hpp

# poisson gradient: exp(raw_score) - label
# poisson hessian: exp(raw_score + max_delta_step)

tree_chart <- lgb.model.dt.tree(lgb_model)
tree_chart[tree_index < 3,]

# RECALCULATE RESULTS ################################

# main question here: can we set min_sum_hessian in a way that one split does not happen? 

## case 1) - data 2, learning rate 0.3, max_delta_step 0.6, min_sum_hessian 0 --------------------

# starting main internal value is simply the overall expected value, 0.677: 
exp(-0.3900840061)

# first split: by var 2
gradient_l <- 797 * exp(-0.3900840061) - 357
hessian_l <- 797 * exp(-0.3900840061 + 0.7) 
(-gradient_l / hessian_l) * 0.3 + -0.3900840061 # -0.4404915


gradient_r <- 203 * exp(-0.3900840061) - 320
hessian_r <- 203 * exp(-0.3900840061 + 0.7) 
(-gradient_r / hessian_r) * 0.3 + -0.3900840061 # -0.1921787

# gain of the fist split: 

split_gain(gradient_l = gradient_l, hessian_l = hessian_l,
           gradient_r = gradient_r, hessian_r = hessian_r, 
           reg_lambda = 0, reg_gamma = 0) # 151.1141, checks out


# now, on to the leaves,

# first split: 797, where var2 = 0

gradient_l <- 457 * exp(-0.3900840061) - 125
hessian_l <- 457 * exp(-0.3900840061 + 0.7) 
(-gradient_l / hessian_l) * 0.3 + -0.3900840061 # -0.4788702

# OK, interestingly, it doesn't use the internal value from the previous split, uses the overall
# so the first split's internal value does not really matter...

gradient_r <- 340 * exp(-0.3900840061) - 232
hessian_r <- 340 * exp(-0.3900840061 + 0.7) 
(-gradient_r / hessian_r) * 0.3 + -0.3900840061 # -0.3889061

# does the gain match? 

split_gain(gradient_l = gradient_l, hessian_l = hessian_l,
           gradient_r = gradient_r, hessian_r = hessian_r, 
           reg_lambda = 0, reg_gamma = 0) # 23.90163

# now the other split, where var2 = 1, 203 in total

gradient_l <- 117 * exp(-0.3900840061) - 141
hessian_l <- 117 * exp(-0.3900840061 + 0.7) 
(-gradient_l / hessian_l) * 0.3 + -0.3900840061 # -0.273868

gradient_r <- 86 * exp(-0.3900840061) - 179
hessian_r <- 86 * exp(-0.3900840061 + 0.7) 
(-gradient_r / hessian_r) * 0.3 + -0.3900840061 # -0.0810432


#    var1 var2   N  V2
# 1:    0    0 457 125
# 2:    0    1 117 141
# 3:    1    0 340 232
# 4:    1    1  86 179


# let's check the second tree for the same leaves
# at this point, the starting predictions changed from the original -0.3900840061

gradient_l <- 457 * exp(-0.4788702) - 125
hessian_l <- 457 * exp(-0.4788702 + 0.7) 
(-gradient_l / hessian_l) * 0.3  # -0.08319775

gradient_r <- 340 * exp(-0.3889061) - 232
hessian_r <- 340 * exp(-0.3889061 + 0.7) 
(-gradient_r / hessian_r) * 0.3  # -0.001001166

# and one more check: on the second tree, can we replicate the first internal_value? 
# the trick is that there will be different prediction values by the other variable

gradient_l <- 457 * exp(-4.882079e-01) + 340 * exp(-3.887822e-01) - 357
hessian_l <- 457 * exp(-4.882079e-01 + 0.6) + 340 * exp(-3.887822e-01 + 0.6)
(-gradient_l / hessian_l) * 0.3  # -0.04960785


# MIN SUM HESSIAN ANALYSED #####################

# min_sum_hessian formula for Poisson: exp(score + 0.7)

# theory: if we set min_sum_hessian to 120, the var2 = 1 route won't be changed
# running model again
# yes, it doesn't split it in the first tree, var2 = 1 gets a -0.192178698 combined
# however, after that, it does get split in the second tree
# recalculating the results in the second tree

gradient_l <- 117 * exp(-0.192178698) - 141
hessian_l <- 117 * exp(-0.192178698 + 0.7) 
(-gradient_l / hessian_l) * 0.3 # 0.06860017

gradient_r <- 86 * exp(-0.192178698) - 179
hessian_r <- 86 * exp(-0.192178698 + 0.7) 
(-gradient_r / hessian_r) * 0.3 # 0.2268028

# if we increase the min_sum_hessian to 145, there won't be a split in the second tree either
# yes, it starts splitting the tree at the 3rd level
# re-calculate splits at level 3

gradient_l <- 117 * exp(-0.192178698 + 0.1356219900) - 141
hessian_l <- 117 * exp(-0.192178698 + 0.1356219900 + 0.7) 
(-gradient_l / hessian_l) * 0.3 # 0.04100561

gradient_r <- 86 * exp(-0.192178698 + 0.1356219900) - 179
hessian_r <- 86 * exp(-0.192178698 + 0.1356219900 + 0.7) 
(-gradient_r / hessian_r) * 0.3 # 0.1791439

# all right, and if we set the hessian high enough
# in the branch with 86 observations
# rough estimation of a min_sum_hessian that should not let the var2 = 1 branch to split: 

86 * (141 + 179) / (117 + 86) * 2.01  # 271.133

# let's round up, 275 min_sum_hessian should be large enough for that branch not to be split
# the hessian of the first var2 split's smaler branch is 276.752

# no, apparently, in this example, life finds a way

# CHART ######################################

# trying to get a proper visualisation 

# if we have an actual value of 1, how does the log-likelihood, gradient, hessian look? 

# likelihood of Poisson: 
# pred ^ actu * exp(-pred) / factorial(actu)
# log-likelihood: 
# actu * log(pred) - pred - log(factorial(actu))
# in raw score terms, this is the same as
# actu * raw_score - exp(raw_score) - log(factorial(actu))

# ! that is important, derivative is by the actual raw prediction
# and we are taking the derivative of the loss function, not the likelihood

# gradient by raw score: exp(raw_score) - actu
# hessian: exp(raw_score)

# from LightGBM documentation: 
# *  loss = exp(f) - label * f
# *  grad = exp(f) - label
# *  hess = exp(f)

# try to plot this whole thing

# plot 1 - Poisson probability by actual - we won't use this

rm(plot_tbl)
plot_tbl <- data.table(actual = seq(0,10))
prediction <- 0.6
plot_tbl[,probability := prediction ^ actual * exp(-prediction) / factorial(actual)]
ggplot(data = plot_tbl, aes(x = actual, y = probability)) + geom_point() + 
  scale_x_continuous(breaks = pretty_breaks())

# plot 2 - same but as function of predictions for a specific actual

rm(plot_tbl)
plot_tbl <- data.table(prediction = seq(0.1,5,by = 0.01))
actual <- 3
plot_tbl[,probability := prediction ^ actual * exp(-prediction) / factorial(actual)]
ggplot(data = plot_tbl, aes(x = prediction, y = probability)) + geom_line() + 
  scale_x_continuous(breaks = pretty_breaks())
# also do log-likelihood while we are at it
plot_tbl[,loglikelihood:=log(probability)]
ggplot(data = plot_tbl, aes(x = prediction, y = loglikelihood)) + geom_line() + 
  scale_x_continuous(breaks = pretty_breaks())
# and the raw_prediction, which is log(prediction)
plot_tbl[,raw_prediction:=log(prediction)]
ggplot(data = plot_tbl, aes(x = raw_prediction, y = loglikelihood)) + geom_line() + 
  scale_x_continuous(breaks = pretty_breaks())
# now the loss
plot_tbl[,loss:=exp(raw_prediction) - actual * raw_prediction]
ggplot(data = plot_tbl, aes(x = raw_prediction, y = loss)) + geom_line() + 
  scale_x_continuous(breaks = pretty_breaks())
# add gradient and hessian
plot_tbl[,gradient := exp(raw_prediction) - actual]
plot_tbl[,hessian := exp(raw_prediction)]
ggplot(data = plot_tbl, aes(x = raw_prediction, y = gradient)) + geom_line() + 
  scale_x_continuous(breaks = pretty_breaks())
ggplot(data = plot_tbl, aes(x = prediction, y = gradient)) + geom_line() + 
  scale_x_continuous(breaks = pretty_breaks())
ggplot(data = plot_tbl, aes(x = raw_prediction, y = hessian)) + geom_line() + 
  scale_x_continuous(breaks = pretty_breaks())
ggplot(data = plot_tbl, aes(x = prediction, y = hessian)) + geom_line() + 
  scale_x_continuous(breaks = pretty_breaks())

# try to visualise the derivative at a given point, what if we predict 2? 

plot_tbl[prediction == 2]

derivative_line_x_middle <- plot_tbl[prediction == 2, raw_prediction]
derivative_line_y_middle <- plot_tbl[prediction == 2, loss]
derivative_line_slope <- plot_tbl[prediction == 2, gradient]

derivative_line_x_1 <- 0
derivative_line_x_2 <- 1
derivative_line_y_1 <- derivative_line_y_middle + 
  (derivative_line_x_1 - derivative_line_x_middle) * 
  derivative_line_slope
derivative_line_y_2 <- derivative_line_y_middle + 
  (derivative_line_x_2 - derivative_line_x_middle) * 
  derivative_line_slope

ggplot(data = plot_tbl[raw_prediction >= 0], aes(x = raw_prediction, y = loss)) + geom_line() + 
  geom_point(data = plot_tbl[prediction == 2]) + 
  geom_line(data = data.table(
    raw_prediction = c(derivative_line_x_1, derivative_line_x_2),
    loss = c(derivative_line_y_1, derivative_line_y_2))) +
  scale_x_continuous(breaks = pretty_breaks())

# repeating the same for the gradient, of which the derivative is the hessian
# so in the previous block, we change: 
# raw_prediction to gradient, loss to raw_prediction, and gradient to hessian

derivative_line_x_middle <- plot_tbl[prediction == 2, gradient]
derivative_line_y_middle <- plot_tbl[prediction == 2, raw_prediction]
derivative_line_slope <- plot_tbl[prediction == 2, hessian]

derivative_line_x_1 <- -2
derivative_line_x_2 <- 0
derivative_line_y_1 <- derivative_line_y_middle + 
  (derivative_line_x_1 - derivative_line_x_middle) * 
  derivative_line_slope
derivative_line_y_2 <- derivative_line_y_middle + 
  (derivative_line_x_2 - derivative_line_x_middle) * 
  derivative_line_slope

ggplot(data = plot_tbl[raw_prediction >= 0], aes(x = gradient, y = raw_prediction)) + geom_line() + 
  geom_point(data = plot_tbl[prediction == 2]) + 
  # geom_line(data = data.table(
  #   gradient = c(derivative_line_x_1, derivative_line_x_2),
  #   raw_prediction = c(derivative_line_y_1, derivative_line_y_2))) +
  scale_x_continuous(breaks = pretty_breaks())


# NOTES ###################

# still the best summary: 
# https://xgboost.readthedocs.io/en/stable/tutorials/model.html



