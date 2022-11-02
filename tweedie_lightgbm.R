
# GOAL ###########################

# I want to analyse Tweedie, the distribution we use to model total occurred claims
# specifically, my suspicion is that by enabling only one parameter as a hyperparameter, we are limiting the fit
# I will compare a traiditonal frequency - severity model against a compound one
# another question is how to compare a traiditonal model's performance with that of a Tweedie - which Tweedie should we use? 

# LIBRARIES ########################

library(lightgbm)
library(data.table)
library(ggplot2)

# cleanup
rm(list = ls())
gc()

# FUNCTIONS ##############################



# DATA ########################

# we have 2 categorical values, each with 2 possible values
# var1 defines number of claims
# var2 defines size of individual claims, when they occur
# ignore exposure for now

set.seed(100)

rownum = 1000000

dt <- data.table(
  var1 = sample(c(0,1), prob = c(0.6, 0.4), size = rownum, replace = TRUE), 
  var2 = sample(c(0,1), prob = c(0.7, 0.3), size = rownum, replace = TRUE))
dt[,ID := 1:.N]
dt[,var1:=as.factor(var1)]
dt[,var2:=as.factor(var2)]

# generate Poisson claim numbers for var1
# want to keep this realistic - under 10% chance 
lambda_parameters_dt <- data.table(
  var1 = c(0,1), 
  poisson_lambda = c(0.04, 0.1))
lambda_parameters_dt[,var1:=as.factor(var1)]
dt <- merge.data.table(x = dt, y = lambda_parameters_dt, by = "var1")
random_pois <- as.numeric(lapply(dt[,poisson_lambda], function(x){rpois(n = 1, lambda = x)}))
dt[, claim_count := random_pois]
# table of Poisson by var1
dt[,.N, keyby = .(var1, claim_count)]

# generate an individual claims 
claims_dt <- dt[claim_count > 0,]
claims_dt <- claims_dt[, .(claim_ID = seq(1, claim_count)), .(var1, var2, ID)]
claims_dt[,.N] == sum(dt[,claim_count]) # TRUE, OK
gamma_parameters_dt <- data.table(
  var2 = c(0,1), 
  k = c(2, 4),
  theta = c(1500, 2000))
gamma_parameters_dt[,var2:=as.factor(var2)]
claims_dt <- merge.data.table(x = claims_dt, y = gamma_parameters_dt, by = "var2")
random_gamma <- as.numeric(mapply(function(x,y){rgamma(n = 1, shape = x, rate = 1/y)}, 
                                  claims_dt[,k], claims_dt[,theta]))
claims_dt[,claim_amount := round(random_gamma,2)]
# plot individual claims by var
ggplot(data = claims_dt, aes(x = claim_amount, color = var2)) + geom_density()

# move back total claim amount to dt
total_claims_dt <- claims_dt[,.(total_claim_amount = sum(claim_amount)), by = ID]
dt <- merge.data.table(x = dt, y = total_claims_dt, by = "ID", all.x = TRUE)
dt[is.na(total_claim_amount), total_claim_amount:=0]
ggplot(data = dt, aes(x = total_claim_amount)) + geom_histogram()



# FREQ - SEV APPROACH ########################

# we fit a frequency model on the large dt, 
# and a severity model on the claims_dt - we only need the positive claims for this

## frequency model ----------------------

dtrain_data <- as.matrix(dt[,.(var1, var2)])
dtrain_label <- as.matrix(dt[,.(claim_count)])
dtrain <- lgb.Dataset(data = dtrain_data,label = dtrain_label,categorical_feature = c(1, 2))

freq_model <- lgb.train(
  learning_rate = 1, 
  num_iteration = 100,
  objective = "poisson",
  data = dtrain,
  boosting = "gbdt",
  verbose = -1)

dpred_data <- as.matrix(dt[,.(var1, var2)]) # same as dtrain, just for consistency
dt[,claim_count_pred := predict(freq_model, dpred_data)]


## severity model -------------------------

dtrain_data <- as.matrix(claims_dt[,.(var1, var2)])
dtrain_label <- as.matrix(claims_dt[,.(claim_amount)])
dtrain <- lgb.Dataset(data = dtrain_data,label = dtrain_label,categorical_feature = c(1, 2))

sev_model <- lgb.train(
  learning_rate = 1, 
  num_iteration = 100,
  objective = "gamma",
  data = dtrain,
  boosting = "gbdt",
  verbose = -1)

dpred_data <- as.matrix(dt[,.(var1, var2)]) # same as dtrain, just for consistency
dt[,claim_amount_pred := predict(sev_model, dpred_data)]

## combined -------------------

dt[,total_claim_amount_pred := claim_count_pred * claim_amount_pred]
dt[,.(sum(total_claim_amount), sum(total_claim_amount_pred))]

dt[,.(claim_count_pred = claim_count_pred[1], claim_count_actual = mean(claim_count), 
      claim_amount_pred = claim_amount_pred[1], 
      total_claim_amount_pred = total_claim_amount_pred[1], total_claim_amount_actual = mean(total_claim_amount) 
      ), keyby = .(var1, var2)]


# TWEEDIE ##########################

dtrain_data <- as.matrix(dt[,.(var1, var2)])
dtrain_label <- as.matrix(dt[,.(total_claim_amount)])
dtrain <- lgb.Dataset(data = dtrain_data,label = dtrain_label,categorical_feature = c(1, 2))

tweedie_model <- lgb.train(
  learning_rate = 1, 
  num_iteration = 100,
  objective = "tweedie",
  tweedie_variance_power = 1.8,
  data = dtrain,
  boosting = "gbdt",
  verbose = -1)

dpred_data <- as.matrix(dt[,.(var1, var2)]) # same as dtrain, just for consistency
dt[,total_claim_amount_pred_tweedie := predict(tweedie_model, dpred_data)]

dt[,.(total_claim_amount_pred_tweedie = total_claim_amount_pred_tweedie[1], 
      total_claim_amount_actual = mean(total_claim_amount)), keyby = .(var1, var2)]


# unfortunately this example seems to be too easy...