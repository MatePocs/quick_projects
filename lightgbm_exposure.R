library(lightgbm)
library(data.table)
library(ggplot2)

# cleanup

rm(list = ls())
gc()

set.seed(100) 

# DATA ################################

data_basic <- data_easy <- data.table(
  var1 = c(rep("A", times = 5000), rep("B", times = 5000)),
  var2 = rep(c(rep("C", times = 2500), rep("D", times = 2500)),2),
  expos = 1
)

data_easy <- data.table(
  var1 = c(rep("A", times = 5000), rep("B", times = 5000)),
  var2 = rep(c(rep("C", times = 2500), rep("D", times = 2500)),2),
  expos = rep(seq(0.1, 1, by = 0.1), times = 1000)
)

data_mod <- data.table(
  var1 = c(rep("A", times = 5000), rep("B", times = 5000)),
  var2 = rep(c(rep("C", times = 2500), rep("D", times = 2500)),2),
  expos = round(runif(10000),4)
)

var_impact <- data.table(
  var1 = c("A", "B", "A", "B"),
  var2 = c("C", "C", "D", "D"),
  lambda_base = c(0.3, 0.7, 1.3, 1.9)
)

generate_claim_counts <- function(dt, var_impact){
  dt <- merge.data.table(x = dt, y = var_impact, by = c("var1", "var2"))
  dt[, lambda := expos * lambda_base]
  random_pois <- as.numeric(lapply(dt[,lambda], function(x){rpois(n = 1, lambda = x)}))
  dt[, claim_count := random_pois]
  dt[, claim_count_adjusted := claim_count / expos]
  return(dt)
}

data_basic <- generate_claim_counts(data_basic, var_impact)
data_easy <- generate_claim_counts(data_easy, var_impact)
data_mod <- generate_claim_counts(data_mod, var_impact)

# checks - basic counts
data_basic[,.N, keyby = .(var1, var2, expos)]
data_easy[,.N, keyby = .(var1, var2, expos)]
data_mod[,.N, keyby = .(var1, var2, lambda_base, round(expos,1))]

# checks - Poisson
data_easy[,mean(claim_count), keyby = .(var1, var2, expos, lambda)]
mean(data_easy[,mean(claim_count), keyby = .(var1, var2, expos, lambda)][,V1/lambda])

data_mod[,.(.N, mean_claim_count = mean(claim_count)), keyby = .(var1, var2, round(lambda,1))]
data_mod[,.(sum(lambda), sum(claim_count))]
data_mod[var1 == "A" & var2 == "C" & round(lambda,1) == 0.2,.N, by = claim_count]
mean(data_mod[,mean(claim_count), keyby = .(var1, var2, round(lambda,1))][round != 0, V1 / round])

# not sure that this is correct, but this could be a sort of "realistic" data then
# the main point is whether solutions give the same result

# BASIC ###########################

# understanding tree structure without exposure

## base model -----------

data_curr <- data_basic

data_curr_recoded <- lgb.convert_with_rules(
  data = data_curr[,.(var1, var2)])$data

dtrain <- lgb.Dataset(
  data = as.matrix(data_curr_recoded),
  label = as.matrix(data_curr[,.(claim_count)]),
  categorical_feature = c(1,2))

param <- list(
  objective = "poisson",
  num_iterations = 1000, 
  learning_rate = 0.3)

lgb_model <- lgb.train(
  params = param, 
  data = dtrain, 
  verbose = -11)

temp_predict <- predict(lgb_model, as.matrix(data_curr_recoded), rawscore = TRUE)
data_basic[,predict_raw := temp_predict]
data_basic[,predict := exp(predict_raw)]
data_basic[,.N, keyby = .(lambda_base, predict)]

data_basic[,.(mean_claim_count = mean(claim_count)), keyby = .(lambda_base, predict)]

# OK, that checks out, it accurately predicts the mean_claim_count

## tree map analyis --------------

# this is what we want: 
data_basic[,.N, keyby = .(var1, var2, predict_raw)]

tree_chart <- lgb.model.dt.tree(lgb_model)
View(tree_chart)

tree_chart[split_index == 0,]
# this is only non-zero at the beginning

tree_chart[split_index == 1,]
# goes down, but not to 0

tree_chart[split_index == 2,]
# same

tree_chart[,.N, by = threshold]
tree_chart[,.N, by = decision_type]


tree_chart[,.N, keyby = .(split_index, split_feature,threshold, decision_type )]
tree_chart[split_index == 0,]

tree_chart[tree_index %in% c(995, 996)]

tree_chart[,sum(leaf_value), keyby = leaf_index]

tree_chart[,sum(internal_value), keyby = split_index]

tree_chart[,sum(leaf_value, na.rm = TRUE)]
sum(data_basic[,.N, keyby = .(var1, var2, predict_raw)][,predict_raw])

# ok, we can assume that it's OK and the predictions are indeed the sum of leaf_values

# SOLUTION 1 - init_score #################################

# we add the log(exposure) as init score
# model fitting starts from that score
# in the predictions, need to multiply with expos
# or add log(expos), if we run predict with rawscore = TRUE

solution_1_predict <- function(data_curr){

  data_curr_recoded <- lgb.convert_with_rules(
    data = data_curr[,.(var1, var2)])$data
  
  dtrain <- lgb.Dataset(
    data = as.matrix(data_curr_recoded),
    label = as.matrix(data_curr[,.(claim_count)]),
    init_score = as.matrix(data_curr[,.(log(expos))]),
    categorical_feature = c(1,2))
  
  param <- list(
    objective = "poisson",
    num_iterations = 100, 
    learning_rate = 0.5)
  
  lgb_model <- lgb.train(
    params = param, 
    data = dtrain, 
    verbose = -1)
  
  return(predict(lgb_model, as.matrix(data_curr_recoded)))
}

# add predictions
temp_predict <- solution_1_predict(data_easy)
data_easy[,sol_1_predict_raw := temp_predict]
data_easy[,sol_1_predict := sol_1_predict_raw * expos]
temp_predict <- solution_1_predict(data_mod)
data_mod[,sol_1_predict_raw := temp_predict]
data_mod[,sol_1_predict := sol_1_predict_raw * expos]
rm(temp_predict)


## checks ---------------------------------


data_easy[,.N, keyby = .(lambda, sol_1_predict)]
data_easy[,.N, keyby = .(lambda_base, sol_1_predict_raw)]
data_easy[,.N, keyby = .(lambda_base/ sol_1_predict_raw)]

data_mod[,.N, keyby = .(lambda_base, sol_1_predict_raw)]
data_mod[,.N, keyby = .(lambda_base/ sol_1_predict_raw)]


# claim count check

data_easy[,.(
  claim_count = sum(claim_count), 
  pred_claim_count = sum(sol_1_predict),
  theoretical_claim_count = sum(lambda))]

data_mod[,.(
  claim_count = sum(claim_count), 
  pred_claim_count = sum(sol_1_predict),
  theoretical_claim_count = sum(lambda))]


# check with maxmimum likelihood as well

data_easy[,sol_1_ll := dpois(x = claim_count, lambda = sol_1_predict, log = TRUE)]
data_easy[,base_ll := dpois(x = claim_count, lambda = lambda, log = TRUE)]
data_easy[,saturated_ll := dpois(x = claim_count, lambda = claim_count, log = TRUE)]
data_easy[,null_ll := dpois(x = claim_count, lambda = mean(claim_count), log = TRUE)]
data_easy[,.(sol_1_ll = sum(sol_1_ll), base_ll = sum(base_ll), 
             saturated_ll = sum(saturated_ll), null_ll = sum(null_ll), 
             deviance_explained = (sum(null_ll) - sum(sol_1_ll)) / 
               (sum(null_ll) - sum(saturated_ll)))]


data_mod[,sol_1_ll := dpois(x = claim_count, 
                            lambda = sol_1_predict, log = TRUE)]
data_mod[,base_ll := dpois(x = claim_count, 
                           lambda = lambda, log = TRUE)]
data_mod[,saturated_ll := dpois(x = claim_count, 
                            lambda = claim_count, log = TRUE)]
data_mod[,null_ll := dpois(x = claim_count, 
                            lambda = mean(claim_count), log = TRUE)]
data_mod[,.(sol_1_ll = sum(sol_1_ll), 
            base_ll = sum(base_ll), 
            saturated_ll = sum(saturated_ll), 
            null_ll = sum(null_ll))]
data_mod[,.(sol_1_ll = sum(sol_1_ll), 
            base_ll = sum(base_ll), 
            saturated_ll = sum(saturated_ll), 
            null_ll = sum(null_ll), 
             deviance_explained_sol_1 = (sum(null_ll) - sum(sol_1_ll)) / 
               (sum(null_ll) - sum(saturated_ll)),
            deviance_explained_base = (sum(null_ll) - sum(base_ll)) / 
              (sum(null_ll) - sum(saturated_ll)))]


# interesting, deviance explained is only about 30%


# SOLUTION 1B - same but with Tweedie loss ##################################

solution_1b_predict <- function(data_curr){
  
  data_curr_recoded <- lgb.convert_with_rules(
    data = data_curr[,.(var1, var2)])$data
  
  dtrain <- lgb.Dataset(
    data = as.matrix(data_curr_recoded),
    label = as.matrix(data_curr[,.(claim_count)]),
    init_score = as.matrix(data_curr[,.(log(expos))]),
    categorical_feature = c(1,2))
  
  param <- list(
    objective = "tweedie",
    tweedie_variance_power = 1,
    num_iterations = 100, 
    learning_rate = 0.5)
  
  lgb_model <- lgb.train(
    params = param, 
    data = dtrain, 
    verbose = -1)
  
  return(predict(lgb_model, as.matrix(data_curr_recoded)))
}

# add predictions
temp_predict <- solution_1b_predict(data_easy)
data_easy[,sol_1b_predict_raw := temp_predict]
data_easy[,sol_1b_predict := sol_1b_predict_raw * expos]
temp_predict <- solution_1b_predict(data_mod)
data_mod[,sol_1b_predict_raw := temp_predict]
data_mod[,sol_1b_predict := sol_1b_predict_raw * expos]
rm(temp_predict)

data_easy[round(sol_1_predict,4) != round(sol_1b_predict,4)]
data_mod[round(sol_1_predict,4) != round(sol_1b_predict,4)]

# same as above

# SOLUTION 2 - claims adjusted ####################


# we predict the adjusted claim counts and also weigh by exposures
# in prediction, just like in solution 1, need to multiply with exposure


solution_2_predict <- function(data_curr){
  
  data_curr_recoded <- lgb.convert_with_rules(
    data = data_curr[,.(var1, var2)])$data
  
  dtrain <- lgb.Dataset(
    data = as.matrix(data_curr_recoded),
    label = as.matrix(data_curr[,.(claim_count_adjusted)]),
    weight = as.matrix(data_curr[,.(expos)]),
    categorical_feature = c(1,2))
  
  param <- list(
    max_depth = 2,
    objective = "poisson",
    num_iterations = 100, 
    learning_rate = 0.5)
  
  lgb_model <- lgb.train(
    params = param, 
    data = dtrain, 
    verbose = -1)
  
  return(predict(lgb_model, as.matrix(data_curr_recoded)))
}

# add predictions
temp_predict <- solution_2_predict(data_easy)
data_easy[,sol_2_predict_raw := temp_predict]
data_easy[,sol_2_predict := sol_2_predict_raw * expos]
temp_predict <- solution_2_predict(data_mod)
data_mod[,sol_2_predict_raw := temp_predict]
data_mod[,sol_2_predict := sol_2_predict_raw * expos]
rm(temp_predict)

# same results
data_easy[round(sol_1_predict,4) != round(sol_2_predict,4)]
data_mod[round(sol_1_predict,4) != round(sol_2_predict,4)]

## matching gradient analysis -------------

# calculate an example 

test_expos = 0.3
test_claim = 2
test_claim_adjusted = test_claim / test_expos
test_score = 3 # should be the same at ALL possible scores I guess

# gradient formula: exp(score - label) * weight

(exp(test_score) - test_claim_adjusted) * test_expos

exp(test_score + log(test_expos)) - test_claim

rm(test_expos, test_claim, test_claim_adjusted, test_score)


# SOLUTION 3 - custom objective function #############################

# add the exposure manually to the objective function 
# every time we have a prediction, we add the exposure
# https://github.com/microsoft/LightGBM/blob/4b1b412452218c5be5ac0f238454ec9309036798/src/objective/regression_objective.hpp

my_poisson_w_exposure <- function(preds, dtrain){
  labels <- getinfo(dtrain, "label")
  preds <- matrix(preds, nrow = length(labels))
  preds_expos_adj <- preds + log(dtrain_expos)
  grad <- exp(preds_expos_adj) - labels
  hess <- exp(preds_expos_adj + 0.7)
  
  return(list(grad = grad, hess = hess))
}


solution_3_predict <- function(data_curr){
  
  data_curr_recoded <- lgb.convert_with_rules(
    data = data_curr[,.(var1, var2)])$data
  
  dtrain <- lgb.Dataset(
    data = as.matrix(data_curr_recoded),
    label = as.matrix(data_curr[,.(claim_count)]),
    categorical_feature = c(1,2))
  
  param <- list(
    max_depth = 2,
    objective = my_poisson_w_exposure,
    metric = "mae",
    num_iterations = 100, 
    learning_rate = 0.5)
  
  lgb_model <- lgb.train(
    params = param, 
    data = dtrain, 
    verbose = -1)
  
  return(predict(lgb_model, as.matrix(data_curr_recoded)))
}

# add predictions
dtrain_expos <- as.matrix(data_easy[,.(expos)])
temp_predict <- solution_3_predict(data_easy)
data_easy[,sol_3_predict_raw := temp_predict]
data_easy[,sol_3_predict := exp(sol_3_predict_raw) * expos]
dtrain_expos <- as.matrix(data_mod[,.(expos)])
temp_predict <- solution_3_predict(data_mod)
data_mod[,sol_3_predict_raw := temp_predict]
data_mod[,sol_3_predict := exp(sol_3_predict_raw) * expos]
rm(temp_predict)

# same results
data_easy[round(sol_1_predict,4) != round(sol_3_predict,4)]
data_mod[round(sol_1_predict,4) != round(sol_3_predict,4)]

# ANALYSIS ###############################

View(lgb.model.dt.tree(lgb_model))

