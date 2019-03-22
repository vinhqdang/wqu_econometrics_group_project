import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

from sklearn import linear_model

from statsmodels.tsa.arima_model import ARMA, ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error


class BasicStatistics:

    def __init__(self, file_path):

        self.file_path = file_path
        self.price_col = self.read_csv().loc[:, 'Adj Close'].values
        self.date_col = self.read_csv().loc[:, 'Date']

    def read_csv(self):
        """
            Read csv file
            Return: a data frame
        """

        file = pd.read_csv(self.file_path)

        return file

    def average_stock_value(self):
        """
            Calculate the average stock value for a given period
            Return: average stock value
        """
        average_stock_value = np.mean(self.price_col)

        return average_stock_value

    def daily_stock_return(self):
        """
            Calculate the daily stock return for a given period
            Return: array of daily stock return value
        """
        daily_stock_return = np.diff(np.log(self.price_col))

        return daily_stock_return

    def stock_volatility(self):
        """
            Calculate the stock volatility of a given period
            Return: stock volatility value
        """

        daily_return = self.daily_stock_return()
        mean = np.mean(daily_return)

        volatility = np.sqrt((sum((daily_return - mean) ** 2)) / len(daily_return))

        return volatility

    def plot(self):
        """
            Visualize the stock prices
        """

        plt.plot(self.date_col, self.price_col)
        plt.title('Stock price of JPM in from Feb to Dec 2018')
        plt.show()


class LRegression:

    def __init__(self, x, y):

        self.x = x.T
        self.y = y.T

    def linear_model(self):
        """
            Implement the linear regression model
        """
        reg = linear_model.LinearRegression()
        reg.fit(self.x.reshape(-1, 1), self.y.reshape(-1, 1))


class ARMAForecast:

    def __init__(self, x, y):

        self.x = x
        self.y = y

    def is_stationary_with_adf(self, significance_level=0.05):
        """
            Decide if the given time series is stationary using ADF test.
        """

        test = adfuller(self.x, regression='c', autolag='BIC')
        p_value = test[1]

        print("ADF p-value: {:0.5f}".format(p_value))

        if p_value < significance_level:
            print('Stationary by ADF: Yes')
        else:
            print('Stationary by ADF: No')

    def train_test_split(self, train_test_ratio=0.7):
        """
            Split data set into training data set and testing data set with the given ratio
            Return: x_train, x_test, y_train, y_test
        """

        no_of_train = int(len(self.x) * train_test_ratio)

        x_train = self.x[: no_of_train]
        x_test = self.x[no_of_train:]

        y_train = self.y[:no_of_train]
        y_test = self.y[no_of_train:]

        return x_train, x_test, y_train, y_test

    def fit(self):
        """
            Implement ARMA model to predict future value
        """
        x_train, x_test, y_train, y_test = self.train_test_split()

        model = ARMA(x_train, order=(5, 0, 5))
        model_fit = model.fit()
        fc, se, conf = model_fit.forecast(steps=len(y_test))
        fc_series = pd.Series(fc, index=y_test.index)
        x_test = pd.Series(x_test, index=y_test.index)

        return fc_series, x_test

    def plot(self):
        """
            Visualize the difference between predicted value and real value
        """
        x_train, x_test, y_train, y_test = self.train_test_split()
        fc_series, x_test_series = self.fit()

        plt.figure(figsize=(12, 5), dpi=100)
        plt.plot(x_train, label='training')
        plt.plot(x_test_series, label='actual')
        plt.plot(fc_series, label='forecast')
        plt.title('Forecast the home price')
        plt.show()


# evaluate an ARIMA model for a given order (p,d,q)

def evaluate_arima_model(X, arima_order):
    """
        Evaluate the ARIMA model with a given order

    *** Parameters:
        X:  data set
        arima_order : parameters of ARIMA model in the form of (p,d,p)

    *** Return: Mean square error of the model
    """

    # prepare training dataset
    train_size = int(len(X) * 0.66)
    train, test = X[0:train_size], X[train_size:]
    history = [x for x in train]
    # make predictions
    predictions = list()
    for t in range(len(test)):
        model = ARIMA(history, order=arima_order)
        model_fit = model.fit(disp=0)
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        history.append(test[t])
    # calculate out of sample error
    error = mean_squared_error(test, predictions)
    return error


# evaluate combinations of p, d and q values for an ARIMA model
def evaluate_models(dataset, p_values, d_values, q_values):
    """
        Choose the best parameters for ARIMA model among given (p, d, q) sets

    *** Parameters:
        dataset:  data set
        p_values, d_values, q_values : parameters of ARIMA model

    *** Return: Best set of parameters
    """

    dataset = dataset.astype('float32')
    best_score, best_cfg = float("inf"), None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p, d, q)
                try:
                    mse = evaluate_arima_model(dataset, order)
                    if mse < best_score:
                        best_score, best_cfg = mse, order
                    print('ARIMA%s MSE=%.3f' % (order, mse))

                except:
                    continue

    print('Best ARIMA%s MSE=%.3f' % (best_cfg, best_score))


if __name__ == '__main__':

    # Section 3.1.1:
    print("Section 3.1.1")
    JPM_file = 'JPM.csv'
    JPM_data = BasicStatistics(JPM_file).read_csv()
    JPM_price_col = JPM_data.loc[:, 'Adj Close']
    JPM_average_stock_value = BasicStatistics(JPM_file).average_stock_value()

    JPM_daily_return = BasicStatistics(JPM_file).daily_stock_return()
    print("JPM daily stock return")
    print(JPM_daily_return)

    JPM_stock_volatility = BasicStatistics(JPM_file).stock_volatility()
    print('The average stock value of JPM is:', JPM_average_stock_value)
    print('The stock volatility of JPM is:', JPM_stock_volatility)

    BasicStatistics(JPM_file).plot()

    # Section 3.1.2:

    print("Section 3.1.2 ")

    SP_file = 'GSPC.csv'
    SP_data = BasicStatistics(SP_file).read_csv()
    SP_date_col = SP_data.loc[:, 'Date']
    SP_price_col = SP_data.loc[:, 'Adj Close']


    # Section 3.1.3:

    print("Section 3.1.3 ")

    HP_file = 'CSUSHPINSA.csv'
    HP_data = pd.read_csv(HP_file)
    HP_date_col = HP_data.loc[:, 'DATE']
    HP_price_col = HP_data.loc[:, 'CSUSHPINSA'].values

    print("Using Augmented Dickey-Fuller Test for checking the existence of a unit root in Case-Shiller Index series ")

    print("The Augmented Dickey-Fuller test is used to test for a **unit root** in a time series sample. "
          "The presence of the unit root in time series make it non-stationary (has some time-dependent structure"
          "The null hypothesis of the Augmented Dickey-Fuller is that there is a unit root, with the alternative"
          "that there is no unit root.")

    ARMAForecast(HP_price_col, HP_date_col).is_stationary_with_adf()
    ARMAForecast(HP_price_col, HP_date_col).plot()
