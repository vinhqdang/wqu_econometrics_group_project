library (quantmod)

JP <- read.csv("JPM.csv")

JP$Date <- as.Date(JP$Date, "%Y-%m-%d")

# plot the stock price over time
plot (JP$Adj.Close ~ JP$Date, type = "o", ylab = "Closing Price of JP Morgan stock")

# calculate average stock value
mean(JP$Adj.Close)

# calculate stock volatility
sd(JP$Adj.Close) / mean(JP$Adj.Close)

# calculate daily stock return
dailyReturn(JP$Adj.Close)

