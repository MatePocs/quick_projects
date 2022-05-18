# Goal: 
# - demonstrate impact of learning rate
# - is there anything special about 1, can it go higher
# - relationship between learning_rate and num_rounds to get the same results

library(lightgbm)
library(data.table)
library(ggplot2)


# cleanup
rm(list = ls())

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# do a two-step prediction
# re-calculate the numbers
# to make sure we properly understand binary_logloss
# with sigmoid parameter

## DATA ###################################

dt <- fread('./data/data_banknote_authentication.txt',
            col.names = c("variance", "skewness", "curtosis", "entropy", "label"))

dtrain_data <- as.matrix(dt[,.(variance, skewness, curtosis, entropy)])

dtrain <- lgb.Dataset(
  data = dtrain_data, 
  label = as.matrix(dt[,.(label)]))

# MODEL TRAIN ###############################

valids = c(train = dtrain)

train_params <- list(
  num_leaves = 4,
  learning_rate = 0.3,
  num_rounds = 2,
  objective = "binary", 
  metric = c("binary_logloss", "binary_error"),
  sigmoid = 0.7
)

bst <- lgb.train(
  data = dtrain,
  params = train_params, 
  valids = valids
)

dt[, predict_proba_nround1:=predict(bst, dtrain_data, num_iteration = 1)]
dt[, predict_rawscore_nround1:=predict(bst, dtrain_data, num_iteration = 1, rawscore = TRUE)]
dt[, predict_proba_nround2:=predict(bst, dtrain_data, num_iteration = 2)]
dt[, predict_rawscore_nround2:=predict(bst, dtrain_data, num_iteration = 2, rawscore = TRUE)]
dt[, predict_nround1 := ifelse(predict_proba_nround1 < 0.5, 0,1)]
dt[, predict_nround2 := ifelse(predict_proba_nround2 < 0.5, 0,1)] 

# RECALC ERROR METRICS ##################################

lgb.get.eval.result(booster = bst, data_name = "train", eval_name = "binary_logloss")
lgb.get.eval.result(booster = bst, data_name = "train", eval_name = "binary_error")

# binary_error: 0.07069971
dt[,.N, keyby = .(predict = predict_nround2, label)]
#    predict label   N
# 1:       0     0 734
# 2:       0     1  69
# 3:       1     0  28
# 4:       1     1 541
(69 + 28) / 1372

# binary_logloss: 0.3966159
dt[, binary_logloss_nround2 := 
     label * log(predict_proba_nround2) + 
     (1-label) * log(1-predict_proba_nround2)]
-sum(dt[,binary_logloss_nround2])/dt[,.N]


# PREDICTIONS FROM LEAF_VALUES ###############################

## first tree -----------------------

# this only has 4 splits, easy to gather

dt[,.N, keyby = .(predict_proba_nround1, predict_rawscore_nround1)]
#    predict_proba_nround1 predict_rawscore_nround1   N
# 1:             0.3373846               -0.9642444 679
# 2:             0.3702293               -0.7589048 105
# 3:             0.5472543                0.2708327  37
# 4:             0.5905585                0.5232494 551

tree_chart <- lgb.model.dt.tree(bst)
tree_chart[!is.na(leaf_value) & tree_index == 0, 
           .(predict_proba_nround1=1/(1+ exp(-0.7*leaf_value)),leaf_value, leaf_count)][order(predict_proba_nround1)]

# btw, inverse of the sigmoid function is the logit function
# sigmoid: 
1/(1+exp(-0.5232494 * 0.7)) # 0.5905585
# logit: 
log((0.5905585)/(1-0.5905585))/0.7 # 0.5232495

tree_chart[tree_index == 0 & !is.na(leaf_value), .(tree_index, leaf_index, leaf_value, leaf_count)]


# LEAF_VALUES REPLICATE ###############################

## starting value -------------------------------

# starts training from -0.317839
mean(dt[,label]) # 0.4446064
# put this in the logit func
log((0.4446064)/(1-0.4446064))/0.7 # -0.3178395

## first tree's 4 splits -----------------

# splits: 
# level 1: variance: 0.30942
# level 2: skewness: 7.60565
#          curtosis: -4.48625

dt[variance <= 0.30942 & skewness <= 7.60565, .(leaf_count = .N, num_pos = sum(label), mean_label = mean(label))]
#    leaf_count num_pos mean_label
# 1:        551     512  0.9292196

dt[variance <= 0.30942 & skewness > 7.60565, .(leaf_count = .N, num_pos = sum(label), mean_label = mean(label))]
#    leaf_count num_pos mean_label
# 1:        105      20  0.1904762

# how does it get the 0.5232494 and -0.7589048 for these two groups? 

# first group, with 551 observations, 512 positive
response_pos = -1 * 0.7 / (1 + exp(1 * 0.7 * -0.317839))
response_neg = 1 * 0.7 / (1 + exp(-1 * 0.7 * -0.317839))
gradient = 512 * response_pos + 39 * response_neg
hessian = 512 * (abs(response_pos) * (0.7-abs(response_pos))) + 
  39 * (abs(response_neg) * (0.7-abs(response_neg)))
(-gradient / hessian) * 0.3 - 0.317839 # 0.5232497

response_pos
response_neg

# second group, with 105 observations, 20 positive
response_pos = -1 * 0.7 / (1 + exp(1 * 0.7 * -0.317839))
response_neg = 1 * 0.7 / (1 + exp(-1 * 0.7 * -0.317839))
gradient = 20 * response_pos + 85 * response_neg
hessian = 20 * (abs(response_pos) * (0.7-abs(response_pos))) + 85 * (abs(response_neg) * (0.7-abs(response_neg)))
(-gradient / hessian) * 0.3 - 0.317839 # -0.7589045

## second level ---------------------------

# btw, the second tree is not so symmetrical, one cut by variance, then skewness, then variance again
# the observations are still all assigned to one leaf
586 + 41 + 119 + 626 # 1372

# let's do the simple one that is from the first variance split

dt[variance > 0.77605, .N, keyby = .(label, predict_rawscore_nround1, predict_rawscore_nround2)]

#    label predict_rawscore_nround1 predict_rawscore_nround2   N
# 1:     0               -0.9642444               -1.4947746 575
# 2:     0                0.2708327               -0.2596975   8
# 3:     1               -0.9642444               -1.4947746  23
# 4:     1                0.2708327               -0.2596975  20

dt[variance > 0.77605, .N, keyby = .(label, predict_rawscore_nround1, predict_rawscore_nround2)][
  ,predict_rawscore_nround2-predict_rawscore_nround1] # -0.5305303

response_neg_1 = 1 * 0.7 / (1 + exp(-1 * 0.7 * -0.9642444))
response_neg_2 = 1 * 0.7 / (1 + exp(-1 * 0.7 * 0.2708327))
response_pos_1 = -1 * 0.7 / (1 + exp(1 * 0.7 * -0.9642444))
response_pos_2 = -1 * 0.7 / (1 + exp(1 * 0.7 * 0.2708327))

gradient = 575 * response_neg_1 + 8 * response_neg_2 + 
  23 * response_pos_1 + 20 * response_pos_2

hessian = 575 * (abs(response_neg_1) * (0.7-abs(response_neg_1))) + 
  8 * (abs(response_neg_2) * (0.7-abs(response_neg_2))) + 
  23 * (abs(response_pos_1) * (0.7-abs(response_pos_1))) + 
  20 * (abs(response_pos_2) * (0.7-abs(response_pos_2)))

(-gradient / hessian) * 0.3 # -0.5305302


# NOTES ######################################

# data: 
# https://archive.ics.uci.edu/ml/datasets/banknote+authentication

# loss function details: 
# https://github.com/microsoft/LightGBM/blob/d517ba12f2e7862ac533908304dddbd770655d2b/src/objective/binary_objective.hpp

# good article on binary logloss
# https://towardsdatascience.com/understanding-binary-cross-entropy-log-loss-a-visual-explanation-a3ac6025181a

# sigmoid function: 
# https://en.wikipedia.org/wiki/Sigmoid_function

log(0.4446064 / (1 - 0.4446064)) / 0.7
exp(-0.3178395 * 0.7) / (1 + exp(-0.3178395 * 0.7))


# theory: 
# logloss with positive label: 
# - yi * log(p(yi)) = - log(p(yi))   (because yi is 1)
# the model works with raw_score = log(p / (1-p)) / sigmoid
# so p = exp(raw_score * sigmoid) / (1 + exp(raw_score * sigmoid))
# putting this in loss function and derivative by raw_score: 
# gradient = - sigmoid / (1 + exp(raw_score * sigmoid))
# this is response thingy in the LightGBM calculation

# now, I think that the second order derivative is: 
# sigmoid ^ 2 * exp(raw_score * sigmoid) / ((1 + exp(raw_score * sigmoid)) ^ 2)

# question: is this the same as the response calculation? 

p <- 0.8
raw_score <- log(p / (1 - p)) / 0.7

response <- -1 * 0.7 / (1 + exp(1 * 0.7 * raw_score))

# is this the same as 
- 0.7 / (1 + exp(0.7 * raw_score)) # yeah, sure

# now, with the LightGBM calculations, hessian is: 
(abs(response) * (0.7-abs(response))) # 0.0784

# do I get the same from my own calculations?
0.7 ^ 2 * exp(raw_score * 0.7) / (( 1 + exp(raw_score * 0.7)) ^ 2) # yes, same formula in different format 0.0784

# let's do a chart with the positive labels
plot_tbl <- data.table(p = seq(0.01,0.99, 0.01))
sigma <- 0.7
plot_tbl[,raw_score := log(p / (1-p)) / sigma]
plot_tbl[,hessian := sigma ^ 2 * exp(raw_score * sigma) / (( 1 + exp(raw_score * sigma)) ^ 2) ]

library(ggplot2)

p <- ggplot(data = plot_tbl, aes(x = p, y = hessian)) + geom_line(color = "darkblue") 

ggsave(filename = "/Users/flatiron/Documents/DataScience/quick_projects/charts/binary_logloss_hessian.png", plot = p)

plot_tbl
