# Section 3.1.1

JP <- read.csv("JPM.csv")

JP$Date <- as.Date(JP$Date, "%Y-%m-%d")

# plot the stock price over time
plot (JP$Adj.Close ~ JP$Date, type = "o", ylab = "Closing Price of JP Morgan stock")

# calculate average stock value
mean(JP$Adj.Close)

# calculate daily stock return
dailyLogReturns <- diff(log(JP$Adj.Close))
print (dailyLogReturns)

# calculate stock volatility
sqrt (sum((dailyLogReturns - mean(dailyLogReturns)) ^ 2)/ (length(dailyLogReturns) -1))

# Section 3.1.2
SP500 <- read.csv("GSPC.csv")
linear_model <- lm (JP$Adj.Close ~ SP500$Adj.Close)
summary (linear_model)

plot (JP$Adj.Close ~ SP500$Adj.Close)
abline (linear_model)

# Section 3.1.3
HomeIndex <- read.csv("CSUSHPINSA.csv")
HomeIndex$DATE <- as.Date(HomeIndex$DATE, "%Y-%m-%d")
HomeIndex_train <- HomeIndex[1:360,]
HomeIndex_test <- HomeIndex[361:384,]

arma_model <- arima(HomeIndex_train$CSUSHPINSA, order = c(5,0,5))
pred <- predict (arma_model, n.ahead = 24)

plot (HomeIndex_train$CSUSHPINSA, xlim = c(0,400), ylim = c(50,250), col = "blue")
points (y = HomeIndex_test$CSUSHPINSA, x = 361:384, col = "red")
points (pred$pred, type = "l")

# Section 3.1.4
# Augmented Dickey-Fuller test
library(aTSA)
aTSA:adf.test(HomeIndex_train$CSUSHPINSA)

# Using AIC to determine best parameters for ARIMA model
library(forecast)
modelAIC <- data.frame()
for(d in 0:1){
  for(p in 0:9){
    for(q in 0:9){
      
      fit=Arima(HomeIndex_train$CSUSHPINSA,order=c(p,d,q))
      modelAIC <- rbind(modelAIC, c(d,p,q,AIC(fit))) #
    }
  }
}
names(modelAIC) <- c("d", "p", "q",  "AIC")
rowNum <- which(modelAIC$AIC==max(modelAIC$AIC))
modelAIC[rowNum,]#Required model parameters
