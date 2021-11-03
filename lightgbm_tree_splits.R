library(lightgbm)
library(data.table)

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

# DATA ################################

set.seed(100) 

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

generate_claim_counts <- function(dt, var_impact){
  dt <- merge.data.table(x = dt, y = var_impact, by = c("var1", "var2"))
  random_pois <- as.numeric(lapply(dt[,lambda], function(x){rpois(n = 1, lambda = x)}))
  dt[, target := random_pois]
  return(dt)
}

data_curr <- generate_claim_counts(data_curr, var_impact)

data_curr

# MODEL #################################

run_model <- function(learning_rate, num_iterations, min_sum_hessian, poisson_max_delta_step){
  
  dtrain_data <- as.matrix(data_curr[,.(var1, var2)])
  dtrain_label <- as.matrix(data_curr[,.(target)])
  
  dtrain <- lgb.Dataset(
    data = dtrain_data,
    label = dtrain_label,
    categorical_feature = c(1,2))
  
  param <- list(
    objective = "poisson",
    num_iterations = num_iterations, 
    learning_rate = learning_rate,
    min_sum_hessian = min_sum_hessian,
    poisson_max_delta_step = poisson_max_delta_step)
  
  lgb_model <- lgb.train(
    params = param, 
    data = dtrain,
    boosting = "gbdt",
    verbose = 1)
  
  return(lgb_model)
}

lgb_model <- run_model(learning_rate = 1, num_iterations = 100, min_sum_hessian = 0, poisson_max_delta_step = 0.6, )
data_curr[,predict := predict(lgb_model,dtrain_data)]
data_curr[,predict_raw := predict(lgb_model,dtrain_data, rawscore = TRUE)]
data_curr[,.(.N, mean_target = mean(target),predict = predict[1], predict_raw = predict_raw[1]), keyby = .(var1, var2)]

#    var1 var2   N mean_target   predict predict_raw
# 1:    0    0 457   0.2735230 0.2735230  -1.2963696
# 2:    0    1 117   1.2051282 1.2051282   0.1865859
# 3:    1    0 340   0.6823529 0.6823529  -0.3822083
# 4:    1    1  86   2.0813953 2.0813953   0.7330385

data_curr[,mean(target)] # 0.677

data_curr[,.(.N, sum(target)), keyby = .(var1, var2)]

#    var1 var2   N  V2
# 1:    0    0 457 125
# 2:    0    1 117 141
# 3:    1    0 340 232
# 4:    1    1  86 179

data_curr[,.(.N, sum(target)), keyby = .(var2)]

#    var2   N  V2
# 1:    0 797 357
# 2:    1 203 320

tree_chart <- lgb.model.dt.tree(lgb_model)
View(tree_chart)

# this is how the predictions are made: 
tree_chart[,exp(sum(leaf_value)), by = leaf_count] # doesn't work generally of course, you would have to check leaves individually


# THEORY ######################

# according to 
# https://github.com/microsoft/LightGBM/blob/4b1b412452218c5be5ac0f238454ec9309036798/src/objective/regression_objective.hpp

# poisson gradient: exp(raw_score) - label
# poisson hessian: exp(raw_score + max_delta_step)

# RECALCULATE RESULTS ##################

## case 1) - data 1, learning rate 1, max_delta_step 0.7 -----------------------

# the first internal value, -0.8486321, is the average: 
exp(-0.8486321)

# trying to replicate leaf value -9.437754e-01 on leaf counr 737
# maybe internal value is used at this point, scores are internal values, and then we add the calculated one...? 

gradient_l <- 737 * exp(-0.8486321) - 255
hessian_l <- 737 * exp(-0.8486321 + 0.7) # 831.336
-gradient_l / hessian_l + -0.8486321 # -0.9437754

# other branch: -5.820137e-01 on leaf count 263

gradient_r <- 263 * exp(-0.8486321) - 173
hessian_r <- 263 * exp(-0.8486321 + 0.7) # 226.6761 
-gradient_r / hessian_r + -0.8486321 # -0.5820137

# what about second round? 
gradient_l <- 737 * exp(-9.437754e-01) - 255
hessian_l <- 737 * exp(-9.437754e-01 + 0.7) # 831.336
-gradient_l / hessian_l # -0.0550728

gradient_r <- 263 * exp(-5.820137e-01) - 173
hessian_r <- 263 * exp(-5.820137e-01 + 0.7) # 295
-gradient_r / hessian_r # 0.08800224

# and what is the split gain? from th XGBoost documentation, long formula: 

reg_lambda <- 0
reg_gamma <- 0

(
  (gradient_l^2 / (hessian_l+reg_lambda)) + 
  (gradient_r^2 / (hessian_r+reg_lambda)) - 
  ((gradient_l + gradient_r)^2 / (hessian_l+hessian_r+reg_lambda))
  ) - 
  reg_gamma # 21.86343

# what about second level? yes, matches, you just use the same formula with different values

# does not match XGBoost formula, no divide by 2


## case 2) - data 1, learning_rate 0.5, max_delta_step 0.7 ----------------------

gradient_l <- 737 * exp(-0.8486321) - 255
hessian_l <- 737 * exp(-0.8486321 + 0.7) 
(-gradient_l / hessian_l) * 0.5 + -0.8486321 # -0.8962038

gradient_r <- 263 * exp(-0.8486321) - 173
hessian_r <- 263 * exp(-0.8486321 + 0.7) 
(-gradient_r / hessian_r) * 0.5 + -0.8486321 # -0.7153229

gradient_l <- 737 * exp(-8.962037e-01) - 255
hessian_l <- 737 * exp(-8.962037e-01 + 0.7) 
(-gradient_l / hessian_l) * 0.5  # -0.03779227

gradient_r <- 263 * exp(-7.153229e-01) - 173
hessian_r <- 263 * exp(-7.153229e-01 + 0.7) 
(-gradient_r / hessian_r) * 0.5 # 0.08568316


## case 3) - data 2, learning rate 0.3, max_delta_step 0.6 --------------------

# first, let's have a look at the internal_values

# main internal value is simply the overall expected value, 0.677: 
exp(-0.3900840061)

# first split: by var 1
# expected values there: 
data_curr[,log(mean(target)), by = var2]
#    var2         V1
# 1:    0 -0.8031189
# 2:    1  0.4551150

# these are not the values in the internal value, even if we add the -0.39

# another option is that they really are what the documentation says: the leaf value, if we ended here

gradient_l <- 797 * exp(-0.3900840061) - 357
hessian_l <- 797 * exp(-0.3900840061 + 0.6) 
(-gradient_l / hessian_l) * 0.3 + -0.3900840061 # -0.4457929

# yup, the first split internal value is really calculated as if it were a leaf

gradient_r <- 203 * exp(-0.3900840061) - 320
hessian_r <- 203 * exp(-0.3900840061 + 0.6) 
(-gradient_r / hessian_r) * 0.3 + -0.3900840061 # -0.1713648

# gain of the fist split: 

split_gain(gradient_l = gradient_l, hessian_l = hessian_l,
           gradient_r = gradient_r, hessian_r = hessian_r, 
           reg_lambda = 0, reg_gamma = 0) # 167.0069, checks out


# now, on to the leaves, can we simply put in the internal values for scores? 

#    var1 var2   N  V2
# 1:    0    0 457 125
# 2:    0    1 117 141
# 3:    1    0 340 232
# 4:    1    1  86 179

# first split: 797, where var2 = 0

gradient_l <- 457 * exp(-0.3900840061) - 125
hessian_l <- 457 * exp(-0.3900840061 + 0.6) 
(-gradient_l / hessian_l) * 0.3 + -0.3900840061 # -0.4882079

# OK, interestingly, it doesn't use the internal value from the previous split, uses the overall

gradient_r <- 340 * exp(-0.3900840061) - 232
hessian_r <- 340 * exp(-0.3900840061 + 0.6) 
(-gradient_r / hessian_r) * 0.3 + -0.3900840061 # -0.3887822

# does the gain match? 

split_gain(gradient_l = gradient_l, hessian_l = hessian_l,
           gradient_r = gradient_r, hessian_r = hessian_r, 
           reg_lambda = 0, reg_gamma = 0) # 26.41538

# let's check the second tree for the same leaves

gradient_l <- 457 * exp(-0.4882079) - 125
hessian_l <- 457 * exp(-0.4882079 + 0.6) 
(-gradient_l / hessian_l) * 0.3  # -0.09126574

gradient_r <- 340 * exp(-0.3887822) - 232
hessian_r <- 340 * exp(-0.3887822 + 0.6) 
(-gradient_r / hessian_r) * 0.3  # -0.001085924

# and one more check: on the second tree, can we replicate the first internal_value? 
# the trick is that there will be different prediction values by the other variable

gradient_l <- 457 * exp(-4.882079e-01) + 340 * exp(-3.887822e-01) - 357
hessian_l <- 457 * exp(-4.882079e-01 + 0.6) + 340 * exp(-3.887822e-01 + 0.6)
(-gradient_l / hessian_l) * 0.3  # -0.04960785


# NOTES #####################

# in XGBoost, max_delta_step is set to 0.7 by default in Poisson regression (used to safeguard optimization)
# https://xgboost.readthedocs.io/en/latest/parameter.html

# when we start training, there is this line: Start training from score -0.390084
# so I think it starts from the overall expected value (which is sensible)

# in LightGBM, there is this comment: the final max output of leaves is learning_rate * max_delta_step

# there is a separate Poisson max_delta_step, which also changes the results. 

# well this is annoying. poisson_max_delta_step and max_delta_step don't do the same thing