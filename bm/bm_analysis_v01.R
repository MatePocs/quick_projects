# 0) INFRASTRUCTURE ############################

library(data.table)
library(ggplot2)
library(stringr)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

dt <- fread('data/DS_data_test_cleaned.csv')

# SUMMARY CHARTS #########################


dt[,.(premium_gross = sum(premium_gross), 
      premium = sum(premium_gross * exposure),
      claims = sum(claims_incurred))]

# ONE-DIMENSIONAL PROFIT ANALYSIS ########################

col_to_analyse <- "make"


odt <- dt[,.(
  count = .N, 
  exposure = sum(exposure),
  claims_incurred = sum(claims_incurred), 
  premium_gross = sum(premium_gross),
  premium = sum(premium_gross * exposure)
  ), 
  by = (get(col_to_analyse))]

odt[,.(premium_gross = sum(premium_gross), 
      premium = sum(premium),
      claims = sum(claims_incurred))]

odt[,profit_ratio := premium - claims_incurred]

ggplot(data = odt, aes(x = get, y = profit_ratio)) + geom_bar(stat = 'identity')




