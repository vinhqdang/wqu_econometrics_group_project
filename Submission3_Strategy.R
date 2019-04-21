library("quantmod")
getSymbols("AAPL",src="yahoo")
aapl_close<- AAPL[,"AAPL.Close"]
aapl_close

return <- Delt(aapl_close)
average10<- rollapply(aapl_close,10,mean)
average10
average20<- rollapply(aapl_close,20,mean)
average20
std10<- rollapply(aapl_close,10,sd)
std20<- rollapply(aapl_close,20,sd)
rsi5<- RSI(aapl_close,5,"SMA")
rsi14<- RSI(aapl_close,14,"SMA")
macd12269<- MACD(aapl_close,12,26,9,"SMA")
macd7205<- MACD(aapl_close,7,20,5,"SMA")
bollinger_bands<- BBands(aapl_close,20,"SMA",2)

direction<- data.frame(matrix(NA,dim(aapl_close)[1],1))
lagreturn<- (aapl_close - Lag(aapl_close,20)) / Lag(aapl_close,20)
direction[lagreturn> 0.04] <- "Up"
direction[lagreturn< -0.04] <- "Down"
direction[lagreturn< 0.04 &lagreturn> -0.04] <- "NoWhere"

aapl_close<-cbind(aapl_close,average10,average20,std10,std20,rsi5,rsi14,macd12269,macd7205,bollinger_bands)

train_sdate<- "2013-01-01"
train_edate<- "2016-12-31"
vali_sdate<- "2017-01-01"
vali_edate<- "2017-12-31"
test_sdate<- "2018-01-01"
test_edate<- "2019-04-19"

trainrow<- which(index(aapl_close) >= train_sdate& index(aapl_close) <= train_edate)
train_return<- return[(index(return) >= train_sdate& index(return) <= train_edate),]
train_return

valirow<- which(index(aapl_close) >= vali_sdate& index(aapl_close) <= vali_edate)
vali_return<- return[(index(return) >= vali_sdate& index(return) <= vali_edate),]
vali_return

testrow<- which(index(aapl_close) >= test_sdate& index(aapl_close) <= test_edate)
test_return<- return[(index(return) >= test_sdate& index(return) <= test_edate),]
test_return

traindji<- aapl_close[trainrow,]
validji<- aapl_close[valirow,]
testdji<- aapl_close[testrow,]
trainme<- apply(traindji,2,mean)
trainstd<- apply(traindji,2,sd)

trainidn<- (matrix(1,dim(traindji)[1],dim(traindji)[2]))
valiidn<- (matrix(1,dim(validji)[1],dim(validji)[2]))
testidn<- (matrix(1,dim(testdji)[1],dim(testdji)[2]))
norm_traindji<- (traindji - t(trainme*t(trainidn))) /t(trainstd*t(trainidn))
norm_validji<- (validji - t(trainme*t(valiidn))) / t(trainstd*t(valiidn))
norm_testdji<- (testdji - t(trainme*t(testidn))) / t(trainstd*t(testidn))
traindir<- direction[trainrow,1]
validir<- direction[valirow,1]
testdir<- direction[testrow,1]

#install.packages("nnet")
library(nnet)

set.seed(1)
z = class.ind(traindir)
neural_network<- nnet(norm_traindji,class.ind(traindir),size=4,trace=T)
neural_network
dim(norm_traindji)

vali_pred<- predict(neural_network,norm_validji)
head(vali_pred)
vali_pred_class<- data.frame(matrix(NA,dim(vali_pred)[1],1))
vali_pred_class[vali_pred[,"Down"] > 0.5,1] <- "Down"
vali_pred_class[vali_pred[,"NoWhere"] > 0.5,1] <- "NoWhere"
vali_pred_class[vali_pred[,"Up"] > 0.5,1] <- "Up"
vali_pred_class[,1]
library(caret)
#matrix<- confusionMatrix(vali_pred_class[,1],validir)
l <- union(vali_pred_class[,1], validir) # very important!
table1<-table(factor(vali_pred_class[,1],l),factor(validir,l))
matrix<- confusionMatrix(table1)
matrix
test_pred<- predict(neural_network,norm_testdji)
test_pred
test_pred_class<- data.frame(matrix(NA,dim(test_pred)[1],1))
test_pred_class[test_pred[,"Down"] > 0.5,1] <- "Down"
test_pred_class[test_pred[,"NoWhere"] > 0.5,1] <- "NoWhere"
test_pred_class[test_pred[,"Up"] > 0.5,1] <- "Up"
test_pred_class[,1]
#test_matrix<- confusionMatrix(test_pred_class[,1],testdir)
l <- union(test_pred_class[,1], testdir) # very important!
table2 = table(factor(test_pred_class[,1],l),factor(testdir,l))
test_matrix<- confusionMatrix(table2)
test_matrix

signal<- ifelse(test_pred_class =="Up",1,ifelse(test_pred_class=="Down",-1,0))
signal


#trade_return<- test_return*lag(signal)
rt <- test_return*lag(signal)
trade_return<- xts(rt)
trade_return
library(PerformanceAnalytics)
cumm_return<- Return.cumulative(trade_return)
cumm_return
annual_return<- Return.annualized(trade_return)
annual_return
charts.PerformanceSummary(trade_return)
summary(as.ts(trade_return))
maxDrawdown(trade_return)
StdDev(trade_return)
StdDev.annualized(trade_return)
VaR(trade_return, p = 0.95)
SharpeRatio(as.ts(trade_return), Rf = 0, p = 0.95, FUN = "StdDev")
SharpeRatio.annualized(trade_return, Rf = 0)

