#Catarina Jesus
#June 11st 2023

# Importing all required libraries 
library(rpart)
library(rpart.plot)
library(caret)
library(ggplot2)
library(tidyverse)
library(tseries)
library(rugarch)
library(dplyr)
library(corrplot)

# importing the data source 
spotify <- read.csv("/Users/catarina_jesus1/Library/Mobile Documents/com~apple~CloudDocs/Desktop/Spotify Analysis/tracks.csv")
artists <- read.csv("/Users/catarina_jesus1/Library/Mobile Documents/com~apple~CloudDocs/Desktop/Spotify Analysis/artists.csv")

# Remove characters from the column
spotify$id_artists <- gsub("\\[|'|\\]", "", spotify$id_artists)

#Change name of the columns
colnames(spotify)[7] <- "id"
colnames(spotify)[1] <- "id1"

#merge both tables
finaltable <- merge(spotify,artists, by = "id", all.x = TRUE)

#Just get the date from release_date column
finaltable$year <- substr(finaltable$release_date,1,4)
finaltable$year <- as.numeric(finaltable$year)

#Drop columns that are not important
finaltable1 <- finaltable[, -c(2,3,7,21,22,23,24)]

#See nulls 
colSums(is.na(finaltable1))

#See blanks
colSums(finaltable1 == "")

# Exclude rows with the year 1900
finaltable1 <- subset(finaltable1, year != 1900)

#####################################################################

## Normalize data 

## Removing units using normalization
min_max <- function(x){
  normalize <- (x-min(x))/(max(x)-min(x))
  return(normalize)
}

finaltable1$popularity_norm <- min_max(finaltable1$popularity.x)
finaltable1$duration_ms_norm <- min_max(finaltable1$duration_ms)
finaltable1$explicit_norm <- min_max(finaltable1$explicit)
finaltable1$danceability_norm <- min_max(finaltable1$danceability)
finaltable1$energy_norm <- min_max(finaltable1$energy)
finaltable1$key_norm <- min_max(finaltable1$key)
finaltable1$loudness_norm <- min_max(finaltable1$loudness)
finaltable1$mode_norm <- min_max(finaltable1$mode)
finaltable1$speechiness_norm <- min_max(finaltable1$speechiness)
finaltable1$acousticness_norm <- min_max(finaltable1$acousticness)
finaltable1$instrumentalness_norm <- min_max(finaltable1$instrumentalness)
finaltable1$liveness_norm <- min_max(finaltable1$liveness)
finaltable1$valence_norm <- min_max(finaltable1$valence)
finaltable1$tempo_norm <- min_max(finaltable1$tempo)
finaltable1$time_signature_norm <- min_max(finaltable1$time_signature)

## Correlation between variables

## Strong correlation between loudness and energy - we will eliminate one of them 
## We will eliminate energy because it has higher correlation with acousticness and loudness

cor_matrix <- cor(finaltable1[,c("popularity.x",
                   "duration_ms",
                   "explicit",
                   "danceability",
                   "energy",
                   "key",
                   "loudness",
                   "mode",
                   "speechiness",
                   "acousticness",
                   "instrumentalness",
                   "liveness",
                   "valence",
                   "tempo",
                   "time_signature"
)])

cor_matrix

# Plot the correlation graph
corrplot(cor_matrix, method = "circle", tl.col = "black", tl.srt = 45)

#change popularity for 1 and 0
finaltable1$popularity_norm <- ifelse(finaltable1$popularity_norm > 0.5, 1, 0)

#change popularity for 1 and 0
finaltable1$popularity.x <- ifelse(finaltable1$popularity.x > 50, 1, 0)

## sampling into TRAINIG and TESTING
training_idx <- sample(1:nrow(finaltable1), size=0.8*nrow(finaltable1))
my_df_train <- finaltable1[training_idx,]
my_df_test <- finaltable1[-training_idx,]

#### LOGISTIC REGRESSION MODEL ####

##Normalize variables
## The normalization is made to compare variables between each other
my_logit1 <- glm(popularity_norm ~ duration_ms_norm+explicit_norm+danceability_norm+
                  loudness_norm+speechiness_norm+acousticness_norm+
                  instrumentalness_norm+liveness_norm+valence_norm+tempo_norm+
                  key_norm+mode_norm+time_signature_norm, data = my_df_train, family = "binomial")
summary(my_logit1)

### We excluded key_norm, mode_norm and time_signature_norm because the p-value is < 0.5 which means that they are statically insignificant

my_logit <- glm(popularity_norm ~ duration_ms_norm+explicit_norm+danceability_norm+
                  loudness_norm+speechiness_norm+acousticness_norm+
                  instrumentalness_norm+liveness_norm+valence_norm+tempo_norm
                  ,data = my_df_train, family = "binomial")
summary(my_logit)

#LOUDNESS has the higher impact on the y variable (popularity)
#Followed by explicit and danceability with higher impact 

my_prediction <- predict(my_logit, my_df_test, type="response") 
confusionMatrix(data=as.factor(as.numeric(my_prediction>0.5)), 
                reference=as.factor(as.numeric(my_df_test$popularity_norm)))

###Confusion Matrix and Statistics

##                Reference
##Prediction      0      1
## 0            76038 11630
## 1            831  1117

##Accuracy : 0.861            
##95% CI : (0.8587, 0.8632)
##No Information Rate : 0.8578           
##P-Value [Acc > NIR] : 0.003093       


##Not normalize variables

### We excluded mode, key and time_signature because the p-value is > 0.5 which means that they are statically insignificant

my_logit2 <- glm(popularity.x ~ duration_ms+explicit+danceability+
                  loudness+speechiness+acousticness+
                  instrumentalness+liveness+valence+tempo
                 , data = my_df_train, family = "binomial")
summary(my_logit2)

my_prediction1 <- predict(my_logit2, my_df_test, type="response") 
confusionMatrix(data=as.factor(as.numeric(my_prediction1>0.5)), 
                reference=as.factor(as.numeric(my_df_test$popularity.x)))

###Confusion Matrix and Statistics

##                Reference
##Prediction      0      1
## 0            76038 11630
## 1            831  1117

##Accuracy : 0.861            
##95% CI : (0.8587, 0.8632)
##No Information Rate : 0.8578           
##P-Value [Acc > NIR] : 0.003093   

#################################################################

#We will create a stratified sample and see which one between random sample and stratified samples gives a better accuracy and confusion matrix
#Stratified sample - the population is divided into homogeneous subgroups based on specific characteristics or attributes
#Random sample - observations are selected randomly from the population without any specific consideration for the population's characteristics or subgroups

#Create a stratified sample

set.seed(123)

# Create a stratified sample
training_idx_strat <- createDataPartition(finaltable1$popularity.x, p = 0.8, list = FALSE)

# Subset the data based on the stratified sample indices
training_data_strat <- finaltable1[training_idx_strat, ]
testing_data_strat <- finaltable1[-training_idx_strat, ]

# building a decision tree

my_tree <- rpart(popularity.x ~ duration_ms+explicit+danceability+
                   loudness+speechiness+acousticness+
                   instrumentalness+liveness+valence+tempo, 
                 data=training_data_strat, method="class", cp = 0.0008)

summary(my_tree)

#Plot the decision tree 
rpart.plot(my_tree, type=1, extra=1)

#Accuracy and performance for decision tree

# testing performance of your model
my_df_tree_predict <- predict(my_tree, testing_data_strat, type="prob")

#Accuracy
confusionMatrix(data = as.factor(as.numeric(my_df_tree_predict[,2]>0.5)) ,
                reference= as.factor(as.numeric(testing_data_strat$popularity.x)))

#Confusion Matrix and Statistics

#               Reference
#Prediction      0      1
#0              76030 11437
#1              911  1238

#Accuracy : 0.8622        
#95% CI : (0.8599, 0.8645)
#No Information Rate : 0.8586        
#P-Value [Acc > NIR] : 0.0008446      


# building a decision tree with random sample

my_tree1 <- rpart(popularity.x ~ duration_ms+explicit+danceability+
                   loudness+speechiness+acousticness+
                   instrumentalness+liveness+valence+tempo, 
                 data=my_df_train, method="class", cp = 0.001)

summary(my_tree1)

#Plot the decision tree 
rpart.plot(my_tree1, type=1, extra=1)

## Lets assume that we have a case study:
# if explicit is zero - yes left, no right
## if yes means that popularity is zero which means is not business success

#Accuracy and performance for decision tree

# testing performance of your model
my_df_tree_predict1 <- predict(my_tree1, my_df_test, type="prob")

#Accuracy
confusionMatrix(data = as.factor(as.numeric(my_df_tree_predict1[,2]>0.5)) ,
                reference= as.factor(as.numeric(my_df_test$popularity.x)))

#Confusion Matrix and Statistics

#             Reference
#Prediction     0     1
#0            75840 11443
#1             1029  1304

#Accuracy : 0.8608             
#95% CI : (0.8585, 0.8631)
#No Information Rate : 0.8578         
#P-Value [Acc > NIR] : 0.004245       


########################################################

##Forecasting

## We will forecast 3 variables, which are the variables with higher influence on the y variable of the prediction (popularity)
## The 3 variables are: danceability, loudness and tempo

#Group by variable by year
danceability <- finaltable1 %>%
  group_by(year) %>%
  summarize(avg_danceability = mean(danceability, na.rm = TRUE))

loudness <- finaltable1 %>%
  group_by(year) %>%
  summarize(avg_loudness = mean(loudness, na.rm = TRUE))

tempo <- finaltable1 %>%
  group_by(year) %>%
  summarize(avg_tempo = mean(tempo, na.rm = TRUE))

#plot
ggplot(data=danceability)+
  geom_line(aes(x=year, y=avg_danceability))

ggplot(data=loudness)+
  geom_line(aes(x=year, y=avg_loudness))

ggplot(data=tempo)+
  geom_line(aes(x=year, y=avg_tempo))


#adf test
adf.test(danceability$avg_danceability)
#Since the p-value above .05, we accept the null hypothesis.
#This means the time series is non-stationary. 
#In other words, it has some time-dependent structure and does not have constant variance over time.

adf.test(loudness$avg_loudness)
#Since the p-value above .05, we accept the null hypothesis.
#This means the time series is non-stationary. 
#In other words, it has some time-dependent structure and does not have constant variance over time.

adf.test(tempo$avg_tempo)
#Since the p-value above .05, we accept the null hypothesis.
#This means the time series is non-stationary. 
#In other words, it has some time-dependent structure and does not have constant variance over time.

# Decomposition of the non-stationary data

#On the right side, decompose our non-stationary into semi-stationary or stationary
#We can see that danceability is semi-stationary
ts_dan <- ts(danceability[,c("year", "avg_danceability")], frequency = 5, start=c(1922))
dec_dan <- decompose(ts_dan)
plot(dec_dan)

#We can see that loudness is semi-stationary
ts_lou <- ts(loudness[,c("year", "avg_loudness")], frequency = 5, start=c(1922))
dec_lou <- decompose(ts_lou)
plot(dec_lou)

#We can see that tempo is stationary
ts_tempo <- ts(tempo[,c("year", "avg_tempo")], frequency = 5, start=c(1922))
dec_tempo <- decompose(ts_tempo)
plot(dec_tempo)

#acf and pacf
acf(danceability$avg_danceability)

acf(loudness$avg_loudness)

acf(tempo$avg_tempo)

pacf(danceability$avg_danceability)

pacf(loudness$avg_loudness)

pacf(tempo$avg_tempo)

############################################################

#ARIMA forecasting

##AR - 2
## I - 1 - SEMI-STATIONARY
## MA - 18
danceability_arima <- arima(danceability$avg_danceability, 
                    order=c(2,1,18)) 
predict(danceability_arima, n.ahead =5) #forecast of the next 5 years

#### forecasting for the next 5 years: 0.6689181 0.6761957 0.6721995 0.6941535 0.6851626
#### with a level of uncertainty of (interval of error): 0.02161993 0.02260949 0.02532752 0.02587535 0.03278915

##AR - 2
## I - 1 - SEMI-STATIONARY
## MA - 22
loudness_arima <- arima(loudness$avg_loudness, 
                    order=c(2,1,22)) 
predict(loudness_arima, n.ahead =5) #forecast of the next 5 years

#### forecasting for the next 5 years:-8.574335 -8.155909 -8.814771 -9.463423 -9.092720
#### with a level of uncertainty of (interval of error): 0.9224355 1.0329742 1.1331575 1.1865815 1.4306653

##AR - 2
## I - 0 - STATIONARY
## MA - 22
tempo_arima <- arima(tempo$avg_tempo, 
                                order=c(3,0,22)) 
predict(tempo_arima, n.ahead =5) #forecast of the next 5 years

#### forecasting for the next 5 years: 121.5818 119.0877 121.1205 119.7098 118.4814
#### with a level of uncertainty of (interval of error): 1.951986 2.072900 2.409080 2.826541 2.886078

##############################################################

#GARCH model

model_param <- ugarchspec(mean.model=list(armaOrder=c(0,0)), 
                          variance.model= list(model="sGARCH", garchOrder=c(1,1)),
                          distribution.model="norm")

### danceability
garch_model1 <- ugarchfit(data=danceability$avg_danceability,
                         spec=model_param, out.sample = 20)
print(garch_model1)

## forecasting Bootstrapping GARCH
bootstrap1 <- ugarchboot(garch_model1, method = c("Partial", "Full")[1],
                         n.ahead = 500, n.bootpred = 500)
print(bootstrap1)

### According GARCH model, we expect the forecast to change around 7,8%
## T+1: 0.077818 - 7,8%, variance seems low, which means the future volatility will be low 
## T+2: 0.078014
## T+3: 0.078210
## T+4: 0.078405
## T+5: 0.078600

#GARCH model
### loudness
garch_model2 <- ugarchfit(data=loudness$avg_loudness,
                         spec=model_param, out.sample = 20)
print(garch_model2)

## forecasting Bootstrapping GARCH
bootstrap2 <- ugarchboot(garch_model2, method = c("Partial", "Full")[1],
                         n.ahead = 500, n.bootpred = 500)
print(bootstrap2)

### According GARCH model, we expect the forecast to change around 305%
## T+1: 3.0506 - 305%, variance seems high, which means the future volatility will be high
## T+2: 3.0508
## T+3: 3.0511
## T+4: 3.0514
## T+5: 3.0517

#GARCH model
### tempo
garch_model3 <- ugarchfit(data=tempo$avg_tempo,
                         spec=model_param, out.sample = 20)
print(garch_model3)

## forecasting Bootstrapping GARCH
bootstrap3 <- ugarchboot(garch_model3, method = c("Partial", "Full")[1],
                        n.ahead = 500, n.bootpred = 500)
print(bootstrap3)

### According GARCH model, we expect the forecast to change around 201%
## T+1: 2.0140 - 201%, variance seems high, which means the future volatility will be high
## T+2: 2.0090
## T+3: 2.0042
## T+4: 1.9997
## T+5: 1.9955
