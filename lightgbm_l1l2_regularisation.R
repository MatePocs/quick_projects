
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

lambda_l1_to_check <- c(0, 1, 2, 5, 10, 20, 50, 100, 200)

lambda_l1_to_check <- seq(0,250,by=1)

result_dt <- data.table()

for (curr_lambda in lambda_l1_to_check){
  
  cat(paste0('currently checking lambda l1: ', curr_lambda)); cat('\n')
  
  dtrain_data <- as.matrix(data_curr[,.(var1)])
  dtrain_label <- as.matrix(data_curr[,.(target)])
  dtrain <- lgb.Dataset(data = dtrain_data,label = dtrain_label,categorical_feature = c(1))
  
  dtest_data <- as.matrix(data.table(var1 = c(0,1,2)))
  
  parameters <- list(
    objective = "poisson",
    num_iterations = 1000, 
    learning_rate = 0.5, 
    lambda_l1 = curr_lambda)
  
  lgb_model <- lgb.train(
    params = parameters, 
    data = dtrain,
    boosting = "gbdt",
    verbose = -1)
  
  curr_predictions <-  predict(lgb_model, dtest_data)
  
  curr_result_dt <- data.table(lambda_l1 = curr_lambda,  
                               value_0_predict = curr_predictions[1],
                               value_1_predict = curr_predictions[2],
                               value_2_predict = curr_predictions[3])
  # kind of a weird solution here
  
  result_dt <- rbind(result_dt, curr_result_dt)
  
}

result_dt

data_curr[,.(lambda = lambda[1], number = .N, target = mean(target)), keyby = .(var1)]

plot_dt <- melt.data.table(result_dt, id.vars = "lambda_l1")
plot_dt
p <- ggplot(data = plot_dt, aes(x = lambda_l1, y = value)) + 
  geom_line(aes(colour = variable, linetype = variable), size = 2, position = position_dodge(width = 3))
p


result_dt



