# 0) INFRASTRUCTURE ############################

library(data.table)
library(ggplot2)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))


# 1) DATA EXPLORE ##############################


# in this section, not changing anything
# goal is to understand what each 

dt <- fread('data/DS_data_test.csv')
dt
str(dt)
summary(dt)

## policy & policy_version ----------------

dt[,.N, keyby = policy][order(N)]

dt[,.N, keyby = policy][order(N)][,.N, keyby = N]
#    N    N
# 1: 1 3510
# 2: 2  232
# 3: 3   25
# 4: 5    3
# there are policies with 5 instances

dt[,.N, keyby = .(policy, policy_version)][order(N)]

# policy + policy_version is a unique key

# let's have a look at one with 5 lines

dt[policy == 2247,]

# these are different dates, with differennt cars, so probably changed the cover
# I'm going to handle this as separate insurance

## date fields ---------------------------

# convert them to date fields

dt[,inception_date := as.Date(inception_date, format = "%d/%m/%Y")]
dt[,effective_start_date := as.Date(effective_start_date, format = "%d/%m/%Y")]
dt[,effective_end_date := as.Date(effective_end_date, format = "%d/%m/%Y")]

dt[inception_date != effective_start_date][order(policy_version)]

# inception date is same as effective date, with the exception of higher than 1 policy versions

dt[inception_date > effective_start_date,] # 0, good, inception date is always <= start date
dt[effective_start_date > effective_end_date,] # hm, 20 such lines, prob need to drop in the end
dt[effective_start_date == effective_end_date,] # another 20 where it's the same day

dt[,.(diff_days = effective_end_date - effective_start_date)][order(diff_days)] # ranges from -1 to 365
ggplot(data = dt, aes(x = effective_end_date - effective_start_date)) + geom_histogram()
dt[,.(diff_days = effective_end_date - effective_start_date)][diff_days < 365,]

# OK, sensible ranges

## is_monthly ---------------------
dt[,.N, by = is_monthly]

# all lines are FALSE, drop column, won't need it

## unplugged_journeys, num_unplugs, unplugged_miles --------------------
dt[,.N, keyby = unplugged_journeys] # vast majority is no unplugged, max: 13 (lol)
dt[,.N, keyby = num_unplugs] # maybe this is how many thimes they are lost, NAs are probably 0's
dt[unplugged_miles > 0,.N] # only 24 rows 
dt[is.na(unplugged_miles) & num_unplugs >0] # unplugged miles NA's can also be 0's
dt[unplugged_miles > 0 & unplugged_journeys == 0,] # 0, good
dt[unplugged_miles == 0 & unplugged_journeys > 0,] 
# 5, this is a bit weird, but they have lots of journeys, maybe one unplugged journey was not a mile

## num_journeys, total_miles, capped_miles
dt[order(num_journeys)] # I assume this is car start to car end, highest is 2233, technically possible
dt[order(total_miles)] # 24095, possible
dt[num_journeys == 0 & total_miles > 0,] #5, these all have 0 num_journeys, but positive unplugged journeys

ggplot(data = dt, aes(x = num_journeys, y = total_miles)) + 
  geom_point() + 
  scale_y_log10() + 
  scale_x_log10()  + geom_smooth()

dt[total_miles < capped_miles,] # 0, good
dt[total_miles > 0,][order(total_miles / capped_miles)]

ggplot(data = dt[total_miles>0,], aes(x = total_miles, y = capped_miles)) + 
  geom_point() + 
  scale_y_log10() + 
  scale_x_log10()  + geom_smooth()

dt[total_miles < unplugged_miles] # there is one here, which is weird

## exposure --------------

# I assume this is end_date - start_date in years, but let's check
dt[,my_exposure := (effective_end_date - effective_start_date + 1) / 365]
dt[,.(exposure, my_exposure)]
dt[,my_exposure := NULL]

dt[,.(inception_date, effective_start_date, effective_end_date, exposure)][order(exposure)]

## premiums --------------------

ggplot(data = dt, aes(x = premium_gross)) + geom_histogram() # looks reasonable
dt[premium_gross == 0,] # 0, OK

ggplot(data = dt, aes(x = fixed_premium_gross)) + geom_histogram() # looks reasonable
dt[fixed_premium_gross == 0,] # 0, OK
dt[fixed_premium_gross > premium_gross,] # 0, OK

ggplot(data = dt, aes(x = pm_rate_gross)) + geom_histogram() # looks reasonable
dt[pm_rate_gross == 0,] # 0, OK


# can we recalculate total premium from fixed and pm? 

dt[,my_premium_gross := (fixed_premium_gross + pm_rate_gross * total_miles)]
dt[,my_premium_gross := (fixed_premium_gross + (pm_rate_gross * total_miles)/exposure)]
dt[,my_premium_gross := (fixed_premium_gross + pm_rate_gross * total_miles)/exposure]
dt[,my_premium_gross := (fixed_premium_gross + pm_rate_gross * declared_mileage)]
dt[,my_premium_gross := (fixed_premium_gross + pm_rate_gross * declared_mileage / exposure)]

dt[,my_premium_gross := (fixed_premium_gross/exposure + pm_rate_gross * total_miles)]

dt[,.(premium_gross, my_premium_gross, fixed_premium_gross, pm_rate_gross, total_miles)]


dt[,my_premium_gross:=NULL]





# 2) DATA CLEANUP ###########################

# 20 rows where start_date is after end_date

dt <- dt[effective_start_date <= effective_end_date,]

# drop is_monthly column

dt[,is_monthly := NULL]

# num_unplugs and unplugged_miles: change NA to 0

dt[is.na(num_unplugs), num_unplugs:= 0]
dt[is.na(unplugged_miles), unplugged_miles:= 0]


# QUESTIONS ####################
# exposure
