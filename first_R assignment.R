library(dslabs)
library(tidyverse)
murders %>%
  ggplot(aes(population, total, label=abb, color=region)) +
  geom_label()

a <- 2
b <- -1
c <- -4

(-b + sqrt(b^2 - 4*a*c))/ (2*a)
(-b - sqrt(b^2 - 4*a*c))/ (2*a)

(1+ sqrt(1 - (4*2*4)))/ (2*2)

library(dslabs)
data(movielens)

str(movielens)
levels(movielens$genres)
nlevels(movielens$genres)

#sort, order and rank functions
x <- c(12,19,17,57,3)
sort(x)
order(x)
index <- order(x)
x[index]
rank(x)
min(x)

library(dslabs)
data(murders)

sort(murders$total)
index <- order(murders$total)
murders$state[index]

murders$state[1:10]
rank(murders$total)

#Finding murder rates
murder_rate <- murders$total/murders$population * 100000
murders$state[order(murder_rate, decreasing = TRUE)]

#calculating speed per hour
name <- c("Mandi", "Amy", "Nicole", "Olivia")
distance <- c(0.8, 3.1, 2.8, 4.0)r
time <- c(10, 30, 40, 50)

time_min <- time / 60

speed_hour <- distance / time_min 
speed_hour

my_df <- data.frame(name = name, speed_hour = speed_hour )
my_df 

your_df <- data.frame(name = name, time_min = time_min )
your_df 

#Finding which state has a murder rate less than 0.71
murder_rate <- murders$total/ murders$population *100000
index <- murder_rate <= 0.71
murders$state[index]
sum(index)

#Finding which state is safe and is in the West
safe <- murder_rate <= 1
west <- murders$region == "West"
index <- safe & west
murders$state[index]

#which() returns indexes that are true
x <- c(TRUE, FALSE, FALSE, TRUE, TRUE)
which(x)

#finding the murder rate in Vermont
x <- which(murders$state == "Vermont")
x
murder_rate[x]

m <- which(murder_rate <= 0.45)
m
murder_rate[m]

#match() 
y <- match(c("Vermont", "California", "Massachusetts"), murders$state)
y

z <- match(murder_rate <= 0.45, murders$state)
z

#%in% operator which is used to check whether an element in a first vector is in a second vector
x <- c(0,1,2,3,4)
y <- c(1,3,4,5,6)
x %in% y
y %in% x
!y %in% x

c("Vermont", "California", "Massachusetts") %in% murders$state

#Basic Data Wrangling
#mutate() - used to change a data table by adding a new column or changing an existing function
library(dplyr)
murders <- mutate(murders, rate = total/population * 100000)
head(murders)

#filter() - filter data
filter(murders, rate <= 0.71)

#select() function
new_table <- select(murders, state, region, rate)
new_table

#using the pipe function 
murders %>% mutate(rate = total/population * 100000) %>% filter(rate < 0.5) %>% select(state, population, total, rate)

#Creating data frames
grades <- data.frame(names =c("Kinuthia", "Karanja", "Kimani", "Kindiki"),
                     exam_1 =c(76,81,88,87),
                     exam_2 =c(89, 82, 78, 74))
grades
class(grades$names)
class(grades$exam_1)

# Get content into a data frame
data <- read.csv("credit_card_transaction_flow.csv",
                 header = FALSE, sep = "\t")

X <- subset(data, select = -c(date, bedrooms, bathrooms, sqft_living, sqft_lot,
                              floors, waterfront, view, condition, sqft_above,
                              sqft_basement, yr_built, yr_renovated, street, city,
                              statezip, country))

y <- data$price
