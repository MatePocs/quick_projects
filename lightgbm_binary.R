# Goal: 
# - demonstrate impact of learning rate
# - is there anything special about 1, can it go higher
# - relationship between learning_rate and num_rounds to get the same results

library(lightgbm)
library(data.table)

# cleanup
rm(list = ls())


# 1) ONE STEP PREDICTION #########################

# do a one-step prediction
# re-calculate the numbers
# to make sure we properly understand binary_logloss

## data -----------------------------

dt <- fread('./data/data_banknote_authentication.txt')
setnames(dt, old = c("V1", "V2", "V3", "V4", "V5"), new = c("variance", "skewness", "curtosis", "entropy", "label"))

dtrain <- lgb.Dataset(
  data = as.matrix(dt[,.(variance, skewness, curtosis, entropy)]), 
  label = as.matrix(dt[,.(label)]))

## model train ----------------------

valids = c(train = dtrain)

train_params <- list(
  num_leaves = 4,
  learning_rate = 0.3,
  num_rounds = 1,
  objective = "binary", 
  metric = c("binary_logloss", "binary_error")
)

bst <- lgb.train(
  data = dtrain,
  params = train_params, 
  valids = valids
)


lgb.get.eval.result(booster = bst, data_name = "train", eval_name = "binary_logloss")
lgb.get.eval.result(booster = bst, data_name = "train", eval_name = "binary_error")

dt[, predict_proba:=predict(bst, as.matrix(dt[,.(variance, skewness, curtosis, entropy)]))]
dt[, predict := ifelse(predict_proba < 0.5, 0,1)] # note: this assumes we put equal weights on labels
dt[,.N, keyby = .(predict, label)]


## recalculate error metrics ----------------------

# binary_error: 0.0845481
dt[,.N, keyby = .(predict, label)]
#    predict label   N
# 1:       0     0 715
# 2:       0     1  69
# 3:       1     0  47
# 4:       1     1 541
(69 + 47) / 1372

# binary_logloss: 0.5102119
dt[, binary_logloss := label * log(predict_proba) + (1-label) * log(1-predict_proba)]
-sum(dt[,binary_logloss])/1372


## recalculate predictions --------------------------

# we only have one step now, 4 different predict_proba

dt[,.N, keyby = predict_proba]
#    predict_proba   N
# 1:     0.3373846 679
# 2:     0.3702293 105
# 3:     0.5472543  37
# 4:     0.5905585 551

tree_chart <- lgb.model.dt.tree(bst)
View(tree_chart)

tree_chart[!is.na(leaf_value), .(1/(1+ exp(-leaf_value)),leaf_value, leaf_count)]

# btw, inverse of the sigmoid function is the logit function
1/(1+exp(-0.3662746)) # 0.5905585
log((0.5905585)/(1-0.5905585)) # 0.3662747






# NOTES ###############

# data: 
# https://archive.ics.uci.edu/ml/datasets/banknote+authentication#

# loss function details: 
# https://github.com/microsoft/LightGBM/blob/d517ba12f2e7862ac533908304dddbd770655d2b/src/objective/binary_objective.hpp

# good article on logloss
# https://towardsdatascience.com/understanding-binary-cross-entropy-log-loss-a-visual-explanation-a3ac6025181a

# sigmoid function: 
# https://en.wikipedia.org/wiki/Sigmoid_function
