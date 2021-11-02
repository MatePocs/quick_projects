library(lightgbm)
library(data.table)
library(ggplot2)

# cleanup

rm(list = ls())
gc()

set.seed(100) 

# DATA ################################

data_curr <- data.table(
  var1 = c(rep("A", times = 5000), rep("B", times = 5000)),
  var2 = rep(c(rep("C", times = 2500), rep("D", times = 2500)),2)
)

var_impact <- data.table(
  var1 = c("A", "B", "A", "B"),
  var2 = c("C", "C", "D", "D"),
  lambda = c(0.3, 0.7, 1.3, 1.9)
)

generate_claim_counts <- function(dt, var_impact){
  dt <- merge.data.table(x = dt, y = var_impact, by = c("var1", "var2"))
  random_pois <- as.numeric(lapply(dt[,lambda], function(x){rpois(n = 1, lambda = x)}))
  dt[, claim_count := random_pois]
  return(dt)
}

data_curr <- generate_claim_counts(data_curr, var_impact)

# MODEL #################################

data_curr_recoded <- lgb.convert_with_rules(
  data = data_curr[,.(var1, var2)])$data

dtrain <- lgb.Dataset(
  data = as.matrix(data_curr_recoded),
  label = as.matrix(data_curr[,.(claim_count)]),
  categorical_feature = c(1,2))

param <- list(
  objective = "poisson",
  num_iterations = 1000, 
  learning_rate = 1)

lgb_model <- lgb.train(
  params = param, 
  data = dtrain, 
  verbose = -11)

data_curr[,claim_count_predict := predict(lgb_model, as.matrix(data_curr_recoded))]

data_curr[,.(mean(claim_count),claim_count_predict[1]), keyby = .(var1, var2)]


# TREE ##################

tree_chart <- lgb.model.dt.tree(lgb_model)
View(tree_chart)

# according to 
# https://github.com/microsoft/LightGBM/blob/4b1b412452218c5be5ac0f238454ec9309036798/src/objective/regression_objective.hpp

# poisson gradient: exp(raw_score) - label
# poisson hessian: exp(raw_score) + max_delta_step




