
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


split_gain <- function(gradient_l, hessian_l, gradient_r, hessian_r, lambda_l1 = 0, lambda_l2 = 0){
  
  # this is something I basically guessed
  # TODO
  # for now this does not work with lambda_l2!!!
  # lambda_l1 is tested, and I suspect the same logic will need to be applied on lambda_l2
  return(
    (((gradient_l - sign(gradient_l) * lambda_l1)^2 / (hessian_l+lambda_l2)) + 
       ((gradient_r - sign(gradient_r) * lambda_l1)^2 / (hessian_r+lambda_l2)) - 
       ((gradient_l + gradient_r + sign(gradient_l) * sign(gradient_r) * lambda_l1)^2 / (hessian_l+hessian_r+lambda_l2)))) 
}


generate_claim_counts <- function(dt, var_impact){
  # careful, different from other functions of the same name
  # assumes dt only have columns to merge onto var_impact
  dt <- merge.data.table(x = dt, y = var_impact, by = colnames(dt))
  random_pois <- as.numeric(lapply(dt[,lambda], function(x){rpois(n = 1, lambda = x)}))
  dt[, target := random_pois]
  return(dt)
}

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

# in case you want to replicate: 

data_curr[,.N, keyby = .(var1, target)]
#     var1 target   N
# 1:     0      0 349
# 2:     0      1 113
# 3:     0      2  18
# 4:     0      3   3
# 5:     0      4   1
# 6:     1      0 150
# 7:     1      1  93
# 8:     1      2  39
# 9:     1      3  14
# 10:    1      4   1
# 11:    2      0 121
# 12:    2      1  69
# 13:    2      2  22
# 14:    2      3   7

## model fits ---------------------------------

# lambda_l1_to_check <- c(0, 1, 2, 5, 10, 20, 50, 100, 200)
lambda_l1_to_check <- seq(0,200,by=1)
# lambda_l1_to_check <- seq(0,1,by=0.01)

result_dt <- data.table()

for (curr_lambda in lambda_l1_to_check){
  
  cat(paste0('currently checking lambda l1: ', curr_lambda)); cat('\n')
  
  lgb_model <- get_lgbm_model(data_curr, curr_lambda)
  
  dtest_data <- as.matrix(data.table(var1 = c(0,1,2))) # we only need the 3 basic predictions
  curr_predictions <-  predict(lgb_model, dtest_data)
  
  curr_result_dt <- data.table(lambda_l1 = curr_lambda,  
                               group_0_predict = curr_predictions[1],
                               group_1_predict = curr_predictions[2],
                               group_2_predict = curr_predictions[3])
  # kind of a weird solution here
  
  result_dt <- rbind(result_dt, curr_result_dt)
}

## plot --------------------------

plot_line_shift <- 0.005
plot_dt <- melt.data.table(result_dt, id.vars = "lambda_l1")
# need to tweak it a bit so the plot line overlaps won't look weird
plot_dt[variable == "group_0_predict", value := value - plot_line_shift]
plot_dt[variable == "group_1_predict", value := value + plot_line_shift]
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
lgb_model_1 <- get_lgbm_model(data_curr, 1)
lgb_model_15 <- get_lgbm_model(data_curr, 15)
lgb_model_36 <- get_lgbm_model(data_curr, 36)
lgb_model_86 <- get_lgbm_model(data_curr, 86)
lgb_model_87 <- get_lgbm_model(data_curr, 87)
lgb_model_100 <- get_lgbm_model(data_curr, 100)
lgb_model_126 <- get_lgbm_model(data_curr, 126)
lgb_model_127 <- get_lgbm_model(data_curr, 127)

tree_0 <- lgb.model.dt.tree(lgb_model_0)
tree_1 <- lgb.model.dt.tree(lgb_model_1)
tree_15 <- lgb.model.dt.tree(lgb_model_15)
tree_36 <- lgb.model.dt.tree(lgb_model_36)
tree_86 <- lgb.model.dt.tree(lgb_model_86)
tree_87 <- lgb.model.dt.tree(lgb_model_87)
tree_100 <- lgb.model.dt.tree(lgb_model_100)
tree_126 <- lgb.model.dt.tree(lgb_model_126)
tree_127 <- lgb.model.dt.tree(lgb_model_127)

## hypothetical testing -----------------------

# this is going to be a bit weird, but we need to do some tests on whether the regularisation improved anything
# we basically want to say how 'good' approximation a Poisson x is of a real Poisson y...
# there might be some close formulas for this (e.g. which is a bettern prediction for a Poisson 0.5: 0.4 or 0.6?)
# but I think it will be straightforward to just do a giant test

set.seed(100)

data_test <- data.table(
  var1 = sample(c(0,1,2), prob = c(0.5, 0.3, 0.2), size = 1000000, replace = TRUE))

data_test <- generate_claim_counts(data_test, var_impact)

data_test[,mean(target), keyby = var1]
#    var1        V1
# 1:    0 0.3012164
# 2:    1 0.7010834
# 3:    2 0.6980534

# OK, it's much closer to the Poisson lambdas

lambda_l1_to_check <- seq(0,200,by=1)
lambda_l1_to_check <- c(0,20)

likelihood_dt <- data.table()

for (curr_lambda in lambda_l1_to_check){
  
  cat(paste0('currently checking lambda l1: ', curr_lambda)); cat('\n')
  
  data_test[var1==0,pred:=result_dt[lambda_l1==curr_lambda,group_0_predict]]
  data_test[var1==1,pred:=result_dt[lambda_l1==curr_lambda,group_1_predict]]
  data_test[var1==2,pred:=result_dt[lambda_l1==curr_lambda,group_2_predict]]
  data_test[,poisson_likelihood:=pred^target*exp(-pred)/factorial(pred)]
  data_test[,poisson_loglikelihood:=log(poisson_likelihood)]
  
  curr_likelihood_dt <- data.table(
    lambda_l1 = c(curr_lambda), 
    mean_poisson_loglikelihood_group_0 = c(mean(data_test[var1==0,poisson_loglikelihood])),
    mean_poisson_loglikelihood_group_1 = c(mean(data_test[var1==1,poisson_loglikelihood])),
    mean_poisson_loglikelihood_group_2 = c(mean(data_test[var1==2,poisson_loglikelihood])),
    mean_poisson_loglikelihood = c(mean(data_test[,poisson_loglikelihood])))
  
  likelihood_dt <- rbind(likelihood_dt, curr_likelihood_dt)
  
}

likelihood_dt

plot_dt <- melt.data.table(likelihood_dt, id.vars = c("lambda_l1"))
ggplot(plot_dt, aes(x = lambda_l1, y = value, colour = variable)) + geom_line()

likelihood_dt[order(mean_poisson_loglikelihood)]

likelihood_dt[lambda_l1 %in% c(0, 20)]

# this is super interesting, turns out, at lambda_l1 = 20, the loglikelihood is the highest

result_dt[lambda_l1 == 20]
#    lambda_l1 value_0_predict value_1_predict value_2_predict
# 1:        20       0.3760331       0.6632997       0.6118721

# this is apparently bettern than the starting predictions
# ok, great, so it makes some sense to do regularisation 
# based on the chart, increasing over 50 is quite pointless


## lambda l1 0 -----------------------

# let's try to recalculate values for model with 0 reg, the first tree
tree_0[tree_index == 0,]

# the left one, var1 == 0, with leaf_value of -0.7537717
gradient_l <- 484 * 0.513 - 162
hessian_l <- 484 * 0.513 * exp(0.7)
(-gradient_l / hessian_l) * 0.5 +  -0.6674794 # -0.7537716, and it's important to note that the first leaf_values also include the average

# other branch of the first split, interal_value of -0.5865387: 
gradient_r <- 516 * 0.513 - 351
hessian_r <- 516 * 0.513 * exp(0.7)
(-gradient_r / hessian_r) * 0.5 +  -0.6674794  #  -0.5865386

# for completeness' sake, let's to the two additional splits
# the expected value at that point: exp(-0.5865387)

gradient_l <- 297 * exp(-0.6674794  ) - 217
hessian_l <- 297 * exp(-0.6674794  ) * exp(0.7)
(-gradient_l / hessian_l) * 0.5 +  -0.6674794 # -0.5621415
# note: we are not using the internal_value of the split for anything

gradient_r <- 219 * exp(-0.6674794  ) - 134
hessian_r <- 219 * exp(-0.6674794  ) * exp(0.7)
(-gradient_r / hessian_r) * 0.5 +  -0.6674794 # -0.6196252

# can we recalculate the split_gain? 1.721168
split_gain(gradient_l, hessian_l, gradient_r, hessian_r) # 1.721168, yes, great

# it's important to note that mechanically, the leaf values of first tree also include the average predictions
# second tree's values won't be as high: 

tree_0[tree_index == 1,]

## lambda l1 1 ----------------------------

tree_1[tree_index == 0,]

# trying to get var1 = 0 leaf_value, -0.7527717
gradient_l <- 484 * 0.513 - 162
hessian_l <- 484 * 0.513 * exp(0.7)
(-(gradient_l-1) / (hessian_l)) * 0.5 +  -0.6674794  # -0.7527716

# what about other two leaf_values, for var1 = 1 and 2? 
gradient_l <- 297 * 0.513 - 217
hessian_l <- 297 * 0.513 * exp(0.7)
(-(gradient_l+1) / (hessian_l)) * 0.5 +  -0.6674794  # -0.5637711

# well that's great, but why are we suddenly adding the 1...? oh right, because of the sign of the gradient I guess
# so it depends on whether we are going up or down from the current prediction, as in, the sign of current weight
# (even if in practice we are rolling it in the overall average)
# is that the same on the other branch? 
gradient_r <- 219 * 0.513 - 134
hessian_r <- 219 * 0.513 * exp(0.7)
(-(gradient_r+1) / (hessian_r)) * 0.5 +  -0.6674794  # -0.6218352, great

# note that the gradient itself in the first tree is the same as how the 0 started

# now the big question: can we recalculate the split gain? 1.437966 ? we managed to do it for unregularised version
split_gain(gradient_l, hessian_l, gradient_r, hessian_r, lambda_l1 = 1, lambda_l2 = 0)
# 1.437966, wow, i did not expect this!

# other interesting question: why does group 2's prediction not change? the first leaf is different from tree 0 
tree_0[!is.na(leaf_count),.(pred = exp(sum(leaf_value))), keyby = leaf_count]
#   leaf_count      pred
# 1:        219 0.6118722
# 2:        297 0.7306397
# 3:        484 0.3347107
# yes, we knew that, I was just checking 

tree_1[!is.na(leaf_count),.(raw_pred = (sum(leaf_value)), number_of_leaves = .N), keyby = leaf_count]
#   leaf_count    raw_pred number_of_leaves
# 1:        219 -0.52828497               32
# 2:        297 -0.35550676               36
# 3:        484 -1.08833469               62
# 4:        516  0.03705302               30
# 5:        703  0.00000000                4

# interesting, so we have versions where value 1 and 2 are not separated 
# the predictions for value 2 group are: 
exp(-0.52828497 + 0.03705302) # 0.6118721, which I find very weird

# let's check a tree where the two are not separated
tree_1[leaf_count == 516]
tree_1[tree_index == 5,]

# for some reason, the values are not split in the index 5 tree, let's try to find out why

# up until that point, our predictions are going to be: 
tree_1[tree_index < 5 & !is.na(leaf_count), .(raw_pred = sum(leaf_value)), keyby=leaf_count]
#    leaf_count   raw_pred
# 1:        219 -0.5358633
# 2:        297 -0.3882703
# 3:        484 -0.9645786

# the two leaf values, -0.02890242 and 0.01537969, are calculated as: 
gradient_l <- 484 * exp(-0.9645786) - 162
hessian_l <- 484 * exp(-0.9645786) * exp(0.7)
(-(gradient_l-1) / (hessian_l)) * 0.5

gradient_r <- (219 * exp(-0.5358633) + 297 * exp(-0.3882703)) - 351
hessian_r <- (219 * exp(-0.5358633) + 297 * exp(-0.3882703)) * exp(0.7)
(-(gradient_r+1) / hessian_r) * 0.5
  
#just for fun, the split gain: 
split_gain(gradient_l, hessian_l, gradient_r, hessian_r, lambda_l1 = 1, lambda_l2 = 0) # 1.869231


# for now: why don't we split further the values 1 and 2? 

gradient_l <- (219 * exp(-0.5358633)) - 134
hessian_l <- (219 * exp(-0.5358633)) * exp(0.7)
(-(gradient_l+1) / hessian_l) * 0.5
# i think this would have worked

gradient_r <- (297 * exp(-0.3882703)) - 217
hessian_r <- (297 * exp(-0.3882703)) * exp(0.7)
(-(gradient_r+1) / hessian_r) * 0.5


# what about split gain? 
split_gain(gradient_l, hessian_l, gradient_r, hessian_r, lambda_l1 = 1, lambda_l2 = 0)
# -0.01379265
# ah. so the key thing is that the split gain would have been negative
# assuming I calculated it correctly of course

# and now the final question: why aren't we splitting any further? 
# it consists of 66 trees, by the end, the predictions are: 

result_dt[lambda_l1 == 1]

#    lambda_l1 group_0_predict group_1_predict group_2_predict
# 1:         1       0.3367769       0.7272727       0.6118721

# this was the last tree: 
tree_1[tree_index == 65,]

#now let's imagine that we want to split, by group 0 and 1+2

gradient_l <- (484 * 0.3367769) - 162
hessian_l <-  (484 * 0.3367769) * exp(0.7)
(-(gradient_l-1) / hessian_l) * 0.5 # tiny amount

gradient_r <- (297 * 0.7272727 + 219 * 0.6118721) - 351
hessian_r <-  (297 * 0.7272727 + 219 * 0.6118721) * exp(0.7)
(-(gradient_r-1) / hessian_r) * 0.5 # same, tiny amount

split_gain(gradient_l, hessian_l, gradient_r, hessian_r, lambda_l1 = 1, lambda_l2 = 0) # negative, great -0.001008118

# all right, one last question: what does the split gein actually represent? 
# let's try to show why the 66th cut is still made and why the 67th is not







## lambda l1 15 -----------------------

# let's try a higher lambda l1 just to make sure my formulas work 
# (I suspect the third them of the split gain will need to be on the 2nd power)

  

# NOTES ###################

# still the best summary: 
# https://xgboost.readthedocs.io/en/stable/tutorials/model.html

# although it's not really accurate anymore.... lambda_l1 is not documented as it behaves
# to be fair, the two regularisation parameters is the document are different - interestingly, there's one
# that seemingly penalises the number of leaves, which is an interesting concept, but I don't think
# we actually have a hyperparameter for it
