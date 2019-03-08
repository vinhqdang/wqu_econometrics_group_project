JP <- read.csv("data/JPM.csv")

JP$Date <- as.Date(JP$Date, "%Y-%m-%d")

# plot the stock price over time
plot (JP$Adj.Close ~ JP$Date)

# calculate average stock value