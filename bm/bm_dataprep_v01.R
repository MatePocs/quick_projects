# 0) INFRASTRUCTURE ############################

library(data.table)
library(ggplot2)
library(stringr)

rm(list = ls())

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

dt <- fread('data/DS_data_test.csv')


# 1) DATA EXPLORE ##############################


# in this section, not changing anything
# goal is to understand what each column and row is
dt
str(dt)
summary(dt)

## field categories ---------------------

# id_fields: policy, policy_version

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

dt[inception_date != effective_start_date][order(policy_version)]

# inception date is same as effective date, with the exception of higher than 1 policy versions
# this checks out, inception date is whenever the policy holder first got in contract, start date is the actual policy

dt[inception_date > effective_start_date,] # 0, good, inception date is always <= start date
dt[effective_start_date > effective_end_date,] # hm, 39 such lines, prob need to drop in the end
# won't drop, exposure is 0, so no premium, and claims incurred is 0 too
dt[effective_start_date == effective_end_date,] # another 20 where it's the same day

dt[,.(diff_days = effective_end_date - effective_start_date)][order(diff_days)] # ranges from -1 to 365
ggplot(data = dt, aes(x = effective_end_date - effective_start_date)) + geom_histogram()
dt[,.(diff_days = effective_end_date - effective_start_date)][diff_days < 365,]
# OK, sensible ranges

# min and max dates? 
dt[,.(
  min_inception = min(inception_date),
  max_inception = max(inception_date),
  min_startdate = min(effective_start_date),
  max_startdate = max(effective_start_date),
  min_enddate = min(effective_end_date),
  max_enddate = max(effective_end_date))] 

#    min_inception max_inception min_startdate max_startdate min_enddate max_enddate
# 1:    2018-05-18    2019-04-30    2018-05-18    2019-04-30  2018-06-26  2020-04-29

# roughly two years of data

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
dt[,my_premium_gross := (fixed_premium_gross + pm_rate_gross * capped_miles)]
dt[,my_premium_gross := (fixed_premium_gross + (pm_rate_gross * total_miles)/exposure)]
dt[,my_premium_gross := (fixed_premium_gross + pm_rate_gross * total_miles)/exposure]
dt[,my_premium_gross := (fixed_premium_gross + pm_rate_gross * declared_mileage)]
dt[,my_premium_gross := (fixed_premium_gross + pm_rate_gross * declared_mileage / exposure)]
dt[,my_premium_gross := (fixed_premium_gross/exposure + pm_rate_gross * total_miles)]

dt[,.(premium_gross, my_premium_gross, fixed_premium_gross, pm_rate_gross, total_miles)]


dt[,my_premium_gross:=NULL]

# is there connection between exposure and fixed premium? 
ggplot(data = dt, aes(x = exposure, y = fixed_premium_gross)) + geom_point() + geom_smooth()

# connection between difference in dates and fixed premium? 
ggplot(data = dt, aes(x = effective_end_date - effective_start_date, y = fixed_premium_gross)) + geom_point() + geom_smooth()

# connection between non-fixed portion of insurance vs total_miles?
ggplot(data = dt, aes(x = premium_gross - fixed_premium_gross, y = total_miles)) + geom_point() + geom_smooth()

# something is very weird

dt[,.((premium_gross - fixed_premium_gross) / pm_rate_gross)][order(V1)] # around 7,000 all the time, really weird
dt[,.(total_miles,(premium_gross - fixed_premium_gross) / pm_rate_gross * exposure)] 

## claims -----------------------

dt[total_claims >0,]

# total claims to ws_claims looks like claim numbers
# do they add up? 
dt[total_claims != non_ws_claims + ws_claims,] # 1, looks like fault_claims + non_ws_claims in this case
dt[total_claims != fault_claims + mileage_claims,] # a lot
# I don't think we'll use this level, just the total_claims

dt[,.N, keyby = total_claims]

#    total_claims    N
# 1:            0 3935
# 2:            1  123
# 3:            2    5
# 4:            3    1

# what about claims incurred? 

dt[order(claims_incurred)]
dt[claims_incurred == "-" & total_claims > 0,] # 19, maybe we also have deductibles here....? 
dt[claims_incurred != "-" & total_claims == 0,] # 0, so that's good

# field needs to be converted to float
# once that's done, plot: 

ggplot(data = dt[claims_incurred >0,], aes(x = claims_incurred)) + geom_histogram() + scale_x_log10()

# are the claims and premiums similar? 

dt[,.(claims_incurred = sum(claims_incurred), premium_gross = sum(premium_gross), ratio = sum(claims_incurred) / sum(premium_gross))]
# 21%, that is very low

dt[,.(claims_incurred = sum(claims_incurred), premium = sum(premium_gross * exposure), ratio = sum(claims_incurred) / sum(premium_gross * exposure))]
# 99%, this is probably what we want
# assume that in every line, the actual premium covered

## ncd -----------------------------

# no claim discount

dt[,.N, keyby = ncd]
ggplot(data = dt, aes(x = ncd)) + geom_histogram(bins = 25)

# is there at least connection between ncd and premium gross? 

ggplot(data = dt, aes(x = ncd, y = premium_gross)) + geom_point() + geom_smooth()

dt[,mean(premium_gross), keyby = ncd] # OK, it is going down

## days_to_inception -----------------------------

dt[,.N, keyby = days_to_inception]
dt[days_to_inception == 0,]
dt[days_to_inception == 1,]

dt[inception_date == effective_start_date]

# OK, don't know what this is, won't use

## area and region --------------

dt[,.N, keyby = area]
dt[,.N, keyby = .(area, region)] # these are the same level, area is code for region

## vol_xs ---------------------

dt[,.N, keyby = vol_xs] # no idea what this is

## driving restriction ------------------------

dt[,.N, keyby = driving_restriction]

## car details ------------

# make
dt[,.N, keyby = make] # need to modify this, make all upper-case

# fuel
dt[,.N, keyby = fuel] 
dt[fuel == "4",] # no idea what this is, error probably, recode "4" to "Unknown"

# transmission
dt[,.N, keyby = transmission] # OK, 11 Unknown

# engine size
ggplot(data = dt, aes(x = engine_size)) + geom_histogram() # OK
dt[order(engine_size)] # Mercedes-Benz M156 has 6.2 litre 
dt[engine_size == 0,] # engine_size 0 means Unknown, better replace it to NA actually

# veh age 
dt[,.N, keyby = veh_age] # goes up to 21, assume OK

# years_owned
dt[,.N, keyby = years_owned] # goes up to 16, assume OK... although surprisingly lot of them at 0
ggplot(data = dt, aes(x = years_owned)) + geom_histogram(bins = 17)

# veh_value
ggplot(data = dt, aes(x = veh_value)) + geom_histogram(bins = 30)
dt[veh_value == 0,] # 0, good
dt[order(veh_value)]

# num_veh_seats
dt[,.N, keyby = num_veh_seats] # OK, reasonable

## policyholder details --------------------------------

# ph_age
dt[,.N, keyby = ph_age] # OK

# min_age 
dt[,.N, keyby = min_age] # looks OK but I'm not sure what the field is
dt[,.N, keyby = min_age - ph_age] # wide scope

# ph_license_years
dt[,.N, keyby = ph_licence_years] # weird, lowest is 2 years

# min_license_years
# same as min_age, let's not use this

# preinception_claims
dt[,.N, keyby = preinception_claims] # couple of NA's, make them 0, maximum is 3

# preinception_fault_claims
dt[,.N, keyby = preinception_fault_claims] # same, couple of NA's, make them 0, maximum is 3

# ph conviction_count
dt[,.N, keyby = ph_conviction_count] # mostly 0, max is 2

# ph_employment_status
dt[,.N, keyby = ph_employment_status] 

# ph_business_desc
dt[,.N, keyby = ph_business_desc] # very granular

# ph_occupation_name
dt[,.N, keyby = ph_occupation_name] # very granular

# owner type
dt[,.N, keyby = owner_type] # most is PH, I wonder what the others mean

# policy cancelled
dt[,.N, keyby = policy_cancelled]
# make this a boolean

# 2) DATA CLEANUP ###########################

# convert dates 

dt[,inception_date := as.Date(inception_date, format = "%d/%m/%Y")]
dt[,effective_start_date := as.Date(effective_start_date, format = "%d/%m/%Y")]
dt[,effective_end_date := as.Date(effective_end_date, format = "%d/%m/%Y")]

# drop is_monthly column

dt[,is_monthly := NULL]

# num_unplugs and unplugged_miles: change NA to 0

dt[is.na(num_unplugs), num_unplugs:= 0]
dt[is.na(unplugged_miles), unplugged_miles:= 0]

# claims_incurred converted to numeric
dt[claims_incurred == "-", claims_incurred := 0]
dt[,claims_incurred:= as.numeric(str_replace(claims_incurred,",",""))]

# make: uppercase
dt[,make:=toupper(make)]

# there are "4" in the fuel field, make it Unknown
dt[fuel == "4", fuel:="Unknown"]

# we have 0 engine sizes, should be NA so they are not picked up
dt[engine_size == 0, engine_size := NA]

# preinception_claims NA: change to 0
dt[is.na(preinception_claims), preinception_claims := 0]

# same with preinception_fault_claims
dt[is.na(preinception_fault_claims), preinception_fault_claims := 0]

# policy cancelled should be boolean
dt[,policy_cancelled:=policy_cancelled == 1]

# modify brand names
dt[make == "MERCEDES-BENZ", make:= "MERCEDES"]
dt[make == "LAND ROVER", make:= "LANDROVER"]

dt <- dt[exposure > 0,]

# calculate net premium
dt[,premium := premium_gross * exposure]

# create grp fields

# unplugged_journeys
# dt[,.N, keyby = unplugged_journeys]
dt[unplugged_journeys == 0, grp_unplugged_journeys:= "no"]
dt[unplugged_journeys > 0, grp_unplugged_journeys:= "yes"]

# num_unplugs
dt[,.N, keyby = num_unplugs]
dt[num_unplugs == 0, grp_num_unplugs:= "0"]
dt[num_unplugs == 1, grp_num_unplugs:= "1"]
dt[num_unplugs == 2, grp_num_unplugs:= "2"]
dt[num_unplugs == 3, grp_num_unplugs:= "3"]
dt[num_unplugs >=4 & num_unplugs <= 10, grp_num_unplugs:= "4-10"]
dt[num_unplugs > 10, grp_num_unplugs:= "more than 10"]

label_10 <- as.character(1:10)
label_10 <- paste0("grp ", label_10)
label_6 <- as.character(1:6)
label_6 <- paste0("grp ", label_6)

# num_journeys
dt[,.N, keyby = num_journeys]
dt[,grp_num_journeys := quantileCut(dt[,num_journeys], 10, labels = label_10)]


# total_miles
dt[,grp_total_miles := quantileCut(dt[,total_miles], 10, labels = label_10)]

# ncd
# dt[,.N, keyby = ncd]
dt[,grp_ncd := quantileCut(dt[,ncd], 10, labels = label_10)]

# engine_size
# dt[,.N, keyby = engine_size]
dt[,grp_engine_size := quantileCut(dt[,engine_size], 10, labels = label_10)]

# veh_age
# dt[,.N, keyby = veh_age]
dt[,grp_veh_age := quantileCut(dt[,veh_age], 10, labels = label_10)]

# years_owned
# dt[,.N, keyby = years_owned]
dt[years_owned==0, grp_years_owned:="0"]
dt[years_owned==1, grp_years_owned:="1"]
dt[years_owned==2, grp_years_owned:="2"]
dt[years_owned==3, grp_years_owned:="3"]
dt[years_owned==4, grp_years_owned:="4"]
dt[years_owned==5 | years_owned == 6 , grp_years_owned:="5 or 6"]
dt[years_owned>=7, grp_years_owned:="more than 6"]

# veh_value
dt[,grp_veh_value:=quantileCut(dt[,veh_value],10, labels = label_10)]

# ph_age
# dt[,.N, keyby = ph_age]
dt[,grp_ph_age := quantileCut(dt[,ph_age], 10, labels = label_10)]

# ph_licence_years
# dt[,.N, keyby = ph_licence_years]
dt[,grp_ph_licence_years := quantileCut(dt[,ph_licence_years], 6, labels = label_6)]


# final step: save cleaned data (to be used in visualisation and RShiny)
fwrite(dt, file = 'data/DS_data_test_cleaned.csv')
fwrite(dt, file = '../../R_Work/RShiny_Apps/insurance_profitability/data/data_cleaned.csv')


dt[,.N, keyby = .(grp_ncd, ncd)]


col_to_analyse <- "region"
x <- 1
dt[x,get(col_to_analyse)]

# QUESTIONS ####################
# exposure
# premium_gross



