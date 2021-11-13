# 0) INFRASTRUCTURE ############################

library(data.table)
library(ggplot2)
library(stringr)
library(scales)
library(lsr)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

dt <- fread('data/DS_data_test_cleaned.csv')

# SUMMARY CHARTS #########################

dt[,premium:= premium_gross * exposure]

dt[,.(premium_gross = sum(premium_gross), 
      premium = sum(premium),
      claims = sum(claims_incurred),
      diff = (sum(premium) / sum(claims_incurred) - 1) * 100)]

## premium --------------

plot_tbl <- dt[,]

plot_label <- paste0(
  "sum: ", label_comma()(sum(plot_tbl[,premium])),"\n",
  "mean: ", label_comma()(mean(plot_tbl[,premium])),"\n",
  "min: ", label_comma()(min(plot_tbl[,premium])),"\n",
  "max: ", label_comma()(max(plot_tbl[,premium])),"\n")

p <- ggplot(data = plot_tbl, aes(x = premium)) + 
  geom_histogram(fill = "dodgerblue4", color = "grey70") + 
  scale_y_continuous(breaks = pretty_breaks()) + 
  scale_x_continuous(breaks = pretty_breaks()) + 
  labs(title = "Premium histogram", subtitle = "all policies, net") + 
  xlab("net premium (GBP)") + 
  ylab("count") + 
  annotate (geom = "text",x = 700, y = 1000, label = plot_label, hjust = 0, vjust = 1) + 
  theme(plot.margin=unit(c(0.1,0.75,0.1,0.1),"cm"))
p
ggsave(filename = 'charts/premium_hist.PNG', plot = p, width = 5, height = 2, units = "in")

## claims -----------------

plot_tbl <- dt[claims_incurred >0,]

plot_label <- paste0(
  "sum: ", label_comma()(sum(plot_tbl[,claims_incurred])),"\n",
  "mean: ", label_comma()(mean(plot_tbl[,claims_incurred])),"\n",
  "min: ", label_comma()(min(plot_tbl[,claims_incurred])),"\n",
  "max: ", label_comma()(max(plot_tbl[,claims_incurred])),"\n")

p <- ggplot(data = plot_tbl, aes(x = claims_incurred)) + 
  geom_histogram(fill = "red4", color = "grey70") + 
  scale_y_continuous(breaks = pretty_breaks()) + 
  scale_x_continuous(label = comma, breaks = breaks_pretty()) +
  labs(title = "Claims incurred histogram", subtitle = "positive claims only") +
  xlab("claims incurred (GBP)") + 
  ylab("count") + 
  annotate (geom = "text",x = 60000, y = 50, label = plot_label, hjust = 0, vjust = 1) + 
  theme(plot.margin=unit(c(0.1,0.75,0.1,0.1),"cm"))
p
ggsave(filename = 'charts/claims_hist.PNG', plot = p, width = 5, height = 2, units = "in")

# cleanup
rm(plot_tbl, plot_label)

# PREDICTION ANALYSIS ############################

## total error percentage -------------

# we know this, for completeness: 
dt[,.(diff = (sum(premium) / sum(claims_incurred) - 1) * 100)]

## normalised Gini --------------------

# from here: 
# https://www.kaggle.com/c/ClaimPredictionChallenge/discussion/703

normalizedGini <- function(aa, pp) {
  Gini <- function(a, p) {
    if (length(a) !=  length(p)) stop("Actual and Predicted need to be equal lengths!")
    temp.df <- data.frame(actual = a, pred = p, range=c(1:length(a)))
    temp.df <- temp.df[order(-temp.df$pred, temp.df$range),]
    population.delta <- 1 / length(a)
    total.losses <- sum(a)
    null.losses <- rep(population.delta, length(a)) # Hopefully is similar to accumulatedPopulationPercentageSum
    accum.losses <- temp.df$actual / total.losses # Hopefully is similar to accumulatedLossPercentageSum
    gini.sum <- cumsum(accum.losses - null.losses) # Not sure if this is having the same effect or not
    sum(gini.sum) / length(a)
  }
  Gini(aa,pp) / Gini(aa,aa)
}

normalizedGini(dt[,claims_incurred], dt[,premium])

# 0.5544404

## deviance -----------------------

# assume a tweedie 1.5 distribution
# uses formula here: https://scikit-learn.org/stable/modules/model_evaluation.html#mean-tweedie-deviance
tweedie_deviance <- function(actu, pred, tweedie_power){
  # we assume tweedie power is between 1 and 2
  return(
    2 * (
      (pmax(actu, 0) ^ (2-tweedie_power)) / ((1 - tweedie_power) * (2 - tweedie_power)) -
      (actu * pred ^ (1- tweedie_power)) / (1 - tweedie_power) +
      (pred ^ (2- tweedie_power)) / (2 - tweedie_power)
    )
  )
}

tweedie_deviance(dt[,claims_incurred], dt[,premium], 1.5)

residual_deviance <- sum(tweedie_deviance(dt[,claims_incurred], dt[,premium], 1.5))/dt[,.N]
null_deviance <- sum(tweedie_deviance(dt[,claims_incurred], sum(dt[,claims_incurred]) / dt[,.N], 1.5))/dt[,.N]
deviance_explained <- (null_deviance - residual_deviance) / null_deviance

## lift chart ----------------------

# quantile lift chart
plot_tbl <- dt[,.(premium, claims_incurred)]
plot_tbl[,pred_quantile := quantileCut(plot_tbl[,premium], 20, labels = (1:20))]

# check
plot_tbl[,.(min(premium), max(premium),.N), keyby = pred_quantile]
plot_tbl <- plot_tbl[,.(actu = sum(claims_incurred), pred = sum(premium)), keyby = pred_quantile]
plot_tbl <- melt.data.table(plot_tbl, id.vars = "pred_quantile", measure.vars = c("actu", "pred"))


p <- ggplot(data = plot_tbl, aes(x = pred_quantile, y = value, color = variable, group = variable)) + 
  geom_point(size = 3) + geom_line() +
  scale_y_continuous(label = comma, breaks = breaks_pretty()) +
  labs(title = "Lift chart", subtitle = "grouped by premium percentiles") +
  xlab("group") + 
  ylab("total value (GBP)") + 
  scale_color_manual(values = c("red4", "dodgerblue4"))
p

ggsave(filename = 'charts/lift_chart.PNG', plot = p, width = 5, height = 5, units = "in")

# ONE-D ANALYSIS - CATEGORICAL FEATURES ########################

analyse_profitability_1d_cat <- function(
  col_to_analyse, low_exposure_name, exposure_limit, count_limit){
  
  # prepare a table for 1d profitability analysis
  # returns a list of table and plots

  # data prep  
  od_dt <- dt[,.(
    count = .N, 
    exposure = sum(exposure),
    claims_incurred = sum(claims_incurred), 
    premium_gross = sum(premium_gross),
    premium = sum(premium_gross * exposure)
  ), 
  by = eval(col_to_analyse)]
  
  # create LOW_EXPOSURE group under threshold
  
  od_dt[count < count_limit | exposure < exposure_limit, eval(col_to_analyse) := low_exposure_name]
  
  od_dt <- od_dt[,.(
    count = sum(count), 
    exposure = sum(exposure),
    claims_incurred = sum(claims_incurred), 
    premium_gross = sum(premium_gross),
    premium = sum(premium)
  ), 
  by = eval(col_to_analyse)]
  
  od_dt[,profit_ratio := (premium - claims_incurred) / (premium)]
  # re-order analysed column so it's in the correct order in a factor
  od_dt <- rbind(
    od_dt[get(col_to_analyse) != low_exposure_name,][order(profit_ratio)],
    od_dt[get(col_to_analyse) == low_exposure_name,])
  # put low_exposure name at the end
  factor_levels = c(od_dt[get(col_to_analyse) != low_exposure_name,get(col_to_analyse)],low_exposure_name)
  # TODO
  # right now, we assume that we will have a low_exposure name, make it conditional
  od_dt[,eval(col_to_analyse) := factor(get(col_to_analyse), levels = factor_levels)]
  
  # profit ratio plot
  p1 <- ggplot(data = od_dt, aes(x = get(col_to_analyse), y = profit_ratio)) + geom_bar(stat = 'identity') + 
    theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) + 
    labs(x = col_to_analyse, y = "profit ratio")
  
  # total premium and claims incurred plot
  plt_tbl <- od_dt[,.(get(col_to_analyse),premium, claims_incurred)]
  setnames(plt_tbl, old = "V1", new = col_to_analyse)
  plt_tbl <- melt.data.table(plt_tbl, measure_vars = c("claims_incurred", "premium"), id.vars = c(col_to_analyse))
  p2 <- ggplot(data = plt_tbl, aes(x = get(col_to_analyse), y = value, fill  = variable)) + 
    geom_bar(position = 'dodge', stat = 'identity') + 
    theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) + 
    labs(x = col_to_analyse, y = "GBP")
  
  # exposure
  p3 <- ggplot(data = od_dt, aes(x = get(col_to_analyse), y = exposure)) + geom_bar(stat = 'identity') + 
    theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) + 
    labs(x = col_to_analyse, y = "exposure")
  
  # exposure and profit ratio plot
  p4 <- ggplot(data = od_dt, aes(x = profit_ratio, y = exposure, label = get(col_to_analyse))) + 
    geom_point() + geom_text(nudge_y = 0.5)
  
  plots <- list(p1, p2, p3, p4)
  results <- list("table" = od_dt, "plots" = plots)
  
  return(results)
  
}

results <- analyse_profitability_1d_cat(
  col_to_analyse = "make", low_exposure_name = "LOW_EXPOSURE",
  exposure_limit = 10, count_limit = 20)

od_dt <- results$table
od_dt
plots <- results$plots

plots[[1]]
plots[[2]]
plots[[3]]
plots[[4]]

# CHARTS FOR ONE GROUP ###################################

# if we select one group, we want the distribution for premium and claims

value_to_analyse <- "Lancashire South"
col_to_analyse <- "region"

od_dt2 <- dt[get(col_to_analyse) ==  value_to_analyse,]

p5 <- ggplot(data = od_dt2, aes(x = premium)) + geom_histogram()
p5
p6 <- ggplot(data = od_dt2[claims_incurred > 0,], aes(x = claims_incurred)) + geom_histogram() 
p6
