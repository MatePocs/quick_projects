
# GOAL ###########################

# I want to show how the L1 and L2 regularisation works in a decisiton tree model
# generate a synthetic data (claim frequency of course)
# my hope is that we will be able to pick up good relationships with regularisation
# do some nice plots, etc
# maybe do it by scenario - for example, in scenario 1, we have one categorical feature, but 
# two of them share the Poisson lambda
# or we have an unnecessary column, only random
# or we have a continuous variable, but only one split is important

# LIBRARIES ########################

library(lightgbm)
library(data.table)
library(ggplot2)
library(scales)

# cleanup
rm(list = ls())
gc()

# FUNCTIONS ##############################

split_gain <- function(gradient_l, hessian_l, gradient_r, hessian_r, reg_lambda, reg_gamma){
 
  # same as in min_sum_hessian file
  # interestingly enough, we did nt use the regularisation parameters there, but they are still in the functions
  return(
    ((gradient_l^2 / (hessian_l+reg_lambda)) + 
    (gradient_r^2 / (hessian_r+reg_lambda)) - 
    ((gradient_l + gradient_r)^2 / (hessian_l+hessian_r+reg_lambda))) - 
    reg_gamma
    ) 
}

generate_claim_counts <- function(dt, var_impact){
  # careful, different from other functions of the same name
  # assumes dt only have columns to merge onto var_impact
  dt <- merge.data.table(x = dt, y = var_impact, by = colnames(dt))
  random_pois <- as.numeric(lapply(dt[,lambda], function(x){rpois(n = 1, lambda = x)}))
  dt[, target := random_pois]
  return(dt)
}


# SCENARIO 1 ########################

# we have one categorical feature with 3 different values
# however, value 1 and 2 are supposed to generate the same value
# we don't want the small difference due to random stuff pop up 

## data  ---------------------------

set.seed(100)

data_curr <- data.table(
  var1 = sample(c(0,1,2), prob = c(0.5, 0.3, 0.2), size = 1000, replace = TRUE))

var_impact <- data.table(
  var1 = c(0,1,2),
  lambda = c(0.3, 0.7, 0.7))

data_curr <- generate_claim_counts(data_curr, var_impact)

## model ---------------------------------

# lambda_l1_to_check <- c(0, 1, 2, 5, 10, 20, 50, 100, 200)
lambda_l1_to_check <- seq(0,250,by=1)
# lambda_l1_to_check <- seq(0,1,by=0.01)

result_dt <- data.table()

get_lgbm_model <- function(data_input, lambda_l1_input){
  
  dtrain_data <- as.matrix(data_input[,.(var1)])
  dtrain_label <- as.matrix(data_input[,.(target)])
  dtrain <- lgb.Dataset(data = dtrain_data,label = dtrain_label,categorical_feature = c(1))
  
  parameters <- list(
    objective = "poisson",
    num_iterations = 1000, 
    learning_rate = 0.5, 
    lambda_l1 = lambda_l1_input)
  
  lgb_model <- lgb.train(
    params = parameters, 
    data = dtrain,
    boosting = "gbdt",
    verbose = -1)
  
  return(lgb_model)
}

for (curr_lambda in lambda_l1_to_check){
  
  cat(paste0('currently checking lambda l1: ', curr_lambda)); cat('\n')
  
  lgb_model <- get_lgbm_model(data_curr, curr_lambda)
  
  dtest_data <- as.matrix(data.table(var1 = c(0,1,2))) # we only need the 3 basic predictions
  curr_predictions <-  predict(lgb_model, dtest_data)
  
  curr_result_dt <- data.table(lambda_l1 = curr_lambda,  
                               value_0_predict = curr_predictions[1],
                               value_1_predict = curr_predictions[2],
                               value_2_predict = curr_predictions[3])
  # kind of a weird solution here
  
  result_dt <- rbind(result_dt, curr_result_dt)
}

## plot --------------------------

plot_line_shift <- 0.005
plot_dt <- melt.data.table(result_dt, id.vars = "lambda_l1")
# need to tweak it a bit so the plot line overlaps won't look weird
plot_dt[variable == "value_0_predict", value := value - plot_line_shift]
plot_dt[variable == "value_1_predict", value := value + plot_line_shift]
p <- ggplot(data = plot_dt, aes(x = lambda_l1, y = value)) + 
  geom_line(aes(colour = variable), size = 2)
p

## results analysis -----------------------


# remarks regarding the results: 
# unfortunately group 1 and 2 were not brought to their average
# there are really weird jumps
# in the end, everything is on the same level
# value 0 and value 1 get changed immediately, but value 2 does not

result_dt

# ground truth: 
data_curr[,.(lambda = lambda[1], number = .N, mean_target = mean(target), sum_target = sum(target)), keyby = .(var1)]
#    var1 lambda number mean_target sum_target
# 1:    0    0.3    484   0.3347107        162
# 2:    1    0.7    297   0.7306397        217
# 3:    2    0.7    219   0.6118721        134
data_curr[,mean(target)]
# 0.513
log(0.513)
# -0.6674794
217 + 134
# 351

# let's check a few concrete tress, see where they change
# at 36, they groups 1 and 2 are merged
# at 86, the 3 groups are almost the same
# and then suddenly there is a jump at 86 upwards
# they remain separate up until 126, and then at 127, the 3 are merged, and they don't change anymore

result_dt[125:150]

# value 1 and 2 become the same at lambda l1 = 36, and they won't change anymore

lgb_model_0 <- get_lgbm_model(data_curr, 0)
lgb_model_36 <- get_lgbm_model(data_curr, 36)
lgb_model_86 <- get_lgbm_model(data_curr, 86)
lgb_model_87 <- get_lgbm_model(data_curr, 87)
lgb_model_100 <- get_lgbm_model(data_curr, 100)
lgb_model_126 <- get_lgbm_model(data_curr, 126)
lgb_model_127 <- get_lgbm_model(data_curr, 127)

tree_0 <- lgb.model.dt.tree(lgb_model_0)
tree_36 <- lgb.model.dt.tree(lgb_model_36)
tree_86 <- lgb.model.dt.tree(lgb_model_86)
tree_87 <- lgb.model.dt.tree(lgb_model_87)
tree_100 <- lgb.model.dt.tree(lgb_model_100)
tree_126 <- lgb.model.dt.tree(lgb_model_126)
tree_127 <- lgb.model.dt.tree(lgb_model_127)

## question 1 - 87 vs 126 --------------------------------

# let's focus on a few at a time
# what is the difference between 87 and 126? 

tree_87
tree_126

result_dt[lambda_l1 %in% c(87, 126)]
#     lambda_l1 value_0_predict value_1_predict value_2_predict
# 1:        87           0.513       0.6485181       0.6485181
# 2:       126           0.513       0.6038833       0.6038833

0.513 * exp(0.2344141) # 0.6485181, the prediction from 87
0.513 * exp(0.1631051) # 0.6038833, the prediction from 126


# let's try to recalculate the leaf value for 87, 0.2344141

gradient_r <- 516 * 0.513 - 351
hessian_r <- 516 * 0.513 * exp(0.7)
(-gradient_r / (hessian_r + 87)) * 0.5


gradient_r
# -86.29199 - a number between 86 and 87. coincidence? i think not!
hessian_r

split_gain(gradient_l = 0, hessian_l = 1, gradient_r = gradient_r, hessian_r = hessian_r, 
           reg_lambda = 0, reg_gamma = 87)


# NOTES ###################

# still the best summary: 
# https://xgboost.readthedocs.io/en/stable/tutorials/model.html
