
# Section 3.1.3
HomeIndex <- read.csv("F:/MFE/2.Econometrics/GroupProject/wqu_econometrics_group_project-master/CSUSHPINSA.csv")
HomeIndex$DATE <- as.Date(HomeIndex$DATE, "%Y-%m-%d")
HomeIndex_train <- HomeIndex[1:320,]
HomeIndex_test <- HomeIndex[321:384,]

# Section 3.1.4
# Augmented Dickey-Fuller test
library(tseries)
adf.test(HomeIndex_train$CSUSHPINSA) #adf.test(HomeIndex_train$CSUSHPINSA,k=1)

diff1 = diff(HomeIndex_train$CSUSHPINSA,diff = 1)
adf.test(diff1)

diff2 = diff(HomeIndex_train$CSUSHPINSA,diff = 2)
adf.test(diff2)

acf(diff2)
pacf(diff2)

# Using AIC to determine best parameters for ARIMA model
library(forecast)
p_d_q=auto.arima(diff2)
p_d_q
# fit a simple AR model with 12 lags, no differencing, no moving average terms ¨C i.e. an ARIMA(12,0,0) model:
#AR_model1 <- arima(window(CPI_percent_change,start=1990,end = 2013), order=c(12,0,0), method = "ML")
#summary(AR_model1)

arma_model <- Arima(HomeIndex_train$CSUSHPINSA, order = c(1,2,5))
AR_forecast <- predict (arma_model, n.ahead = 64, se.fit=TRUE)

plot (HomeIndex$CSUSHPINSA)
#lines (HomeIndex$CSUSHPINSA)
#points (y = HomeIndex_test$CSUSHPINSA, x = 361:384, col = "red")
#points (pred$pred, type = "l")
lines(AR_forecast$pred,col="red")
lines(AR_forecast$pred+2*AR_forecast$se,col="red",lty = "dashed")
lines(AR_forecast$pred-2*AR_forecast$se,col="red",lty = "dashed")
