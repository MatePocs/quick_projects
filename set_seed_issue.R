library(data.table)
library(ggplot2)

# Problem #############

my_random_seed <- 123

set.seed(my_random_seed)
tbl <- data.table(x = runif(10000) * 10)

set.seed(my_random_seed)
tbl[, y := rpois(10000, x)]

tbl

tbl[,.(y_mean = mean(y), count = .N), keyby = .(x_rounded = round(x, 1))]

tbl[,.(sum(x), sum(y))]

p <- ggplot(data = tbl, aes(x = x, y = y)) + geom_line()
p
ggsave("seed_problem.png", p, width = 3, height = 3)

tbl[,.(min(x), max(x)), keyby = y]

# Replicate #####################

tbl[,prob0 := x^0*exp(-x)/factorial(0)]
tbl[,prob1 := prob0 + x^1*exp(-x)/factorial(1)]
tbl[,prob2 := prob1 + x^2*exp(-x)/factorial(2)]
tbl[,prob3 := prob2 + x^3*exp(-x)/factorial(3)]
tbl[,prob4 := prob3 + x^4*exp(-x)/factorial(4)]
tbl[,prob5 := prob4 + x^5*exp(-x)/factorial(5)]

# ... should repeat, will only check till 5

tbl[y2 := NULL]
tbl[x/10 <= prob0, y2:=0]
tbl[x/10 > prob0 & x/10 <= prob1, y2:=1]
tbl[x/10 > prob1 & x/10 <= prob2, y2:=2]
tbl[x/10 > prob2 & x/10 <= prob3, y2:=3]
tbl[x/10 > prob3 & x/10 <= prob4, y2:=4]
tbl[x/10 > prob4 & x/10 <= prob5, y2:=5]

tbl[y <=5, .N, keyby = .(y, y2)]

tbl[x/10 > prob1 & x < prob2]

# Seed testing ##################

my_random_seed_2 <- my_random_seed + 2^32
set.seed(my_random_seed_2)
tbl[, y3 := rpois(10000, x)]

?set.seed

# does not work with such a high number