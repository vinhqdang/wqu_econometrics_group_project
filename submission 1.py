import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

# Read data


def read_csv(file_path):

    file = pd.read_csv(file_path)

    return file

# Calculate average stock value:


def average_stock_value(file_path):

    file = read_csv(file_path)
    price_col = file.loc[:, 'Adj Close'].values

    total_stock_value = 0
    no_stock = 0

    for daily_value in price_col:
        total_stock_value += np.float(daily_value)
        no_stock += 1

    average_stock_value = total_stock_value / no_stock

    return average_stock_value


def deviation(value, mean):

    dev = value - mean

    return dev


def stock_volatility(file_path):

    file = read_csv(file_path)
    price_col = file.loc[:, 'Adj Close'].values

    mean = average_stock_value(file_path)

    sum_squared_deviation = 0
    no_of_stock = 0

    for daily_value in price_col:

        dev = deviation(daily_value, mean)
        squared_deviation = dev ** 2
        sum_squared_deviation += squared_deviation
        no_of_stock += 1

    volatility = sum_squared_deviation / no_of_stock

    return volatility


def plot(file_path):

    file = read_csv(file_path)
    price_col = file.loc[:, 'Adj Close'].values
    date_col = file.loc[:, 'Date']


    plt.plot(date_col, price_col)
    plt.show()

if __name__ == "__main__":

    file_path = 'JPM.csv'
    JPM_average_stock_value = average_stock_value(file_path)
    JPM_stock_volatility = stock_volatility(file_path)
    print('The average stock value of JPM is:', JPM_average_stock_value)
    print('The stock volatility of JPM is:', JPM_stock_volatility)
    plot(file_path)