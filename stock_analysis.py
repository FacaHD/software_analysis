import pandas as pd
import yfinance as yf
import ta
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
#import matplotlib.pyplot as plt
import numpy as np
# Define the ticker symbol

def analyze_stock(tickerSymbol):
    #Apple Inc.
    # Get data on this ticker
    tickerData = yf.Ticker(tickerSymbol)
    # Get the historical prices for this ticker
    tickerDf = tickerData.history(period='5d', interval='1m')

    # Calculate the simple moving averages


    # Calculate the RSI
    tickerDf['RSI'] = ta.momentum.rsi(tickerDf['Close'], window=14)



    # Calculate the Bollinger Bands

    tickerDf['BB_high'] = ta.volatility.bollinger_hband(tickerDf['Close'], window=20, window_dev=2)
    tickerDf['BB_mid'] = ta.volatility.bollinger_mavg(tickerDf['Close'], window=20)
    tickerDf['BB_low'] = ta.volatility.bollinger_lband(tickerDf['Close'], window=20, window_dev=2)


    # Calculate MACD
    macd = ta.trend.MACD(tickerDf['Close'])

    # Assign MACD line, signal line, and histogram to new columns 
    tickerDf['MACD_line'] = macd.macd()
    tickerDf['MACD_signal'] = macd.macd_signal()
    tickerDf['MACD_histogram'] = macd.macd_diff()

    # Calculate average volume
    tickerDf['avg_volume'] = tickerDf['Volume'].rolling(window=10).mean()

    # Define the conditions for our signals
    buy_signal_conditions = (
    (tickerDf['Close'] < tickerDf['BB_low']) &
    (tickerDf['RSI'] < 30) &
    (tickerDf['RSI'] > tickerDf['RSI'].shift(1)) &
    (tickerDf['MACD_line'] > tickerDf['MACD_signal']) &
    (tickerDf['Volume'] > tickerDf['avg_volume'])
    )

    sell_signal_conditions = (
    (tickerDf['Close'] > tickerDf['BB_high']) &
    (tickerDf['RSI'] > 70) &
    (tickerDf['RSI'] < tickerDf['RSI'].shift(1)) &
    (tickerDf['MACD_line'] < tickerDf['MACD_signal']) &
    (tickerDf['Volume'] > tickerDf['avg_volume'])
    )





    #fractionated
    # Fractional conditions for buying
    rsi_buy_condition = (30 - tickerDf['RSI']) / 30
    bollinger_buy_condition = (tickerDf['BB_low'] - tickerDf['Close']) / tickerDf['BB_low']
    macd_buy_condition = (tickerDf['MACD_line'] - tickerDf['MACD_signal']) / tickerDf['MACD_line']
    volume_buy_condition = (tickerDf['Volume'] - tickerDf['avg_volume']) / tickerDf['Volume']

    # Fractional conditions for selling
    rsi_sell_condition = (tickerDf['RSI'] - 70) / 30
    bollinger_sell_condition = (tickerDf['Close'] - tickerDf['BB_high']) / tickerDf['BB_high']
    macd_sell_condition = (tickerDf['MACD_signal'] - tickerDf['MACD_line']) / tickerDf['MACD_signal']
    volume_sell_condition = (tickerDf['Volume'] - tickerDf['avg_volume']) / tickerDf['Volume']

    # Clip the conditions between 0 and 1
    buy_conditions = [
        rsi_buy_condition.clip(lower=0, upper=1), 
        bollinger_buy_condition.clip(lower=0, upper=1),
        macd_buy_condition.clip(lower=0, upper=1),
        volume_buy_condition.clip(lower=0, upper=1)
    ]

    sell_conditions = [
        rsi_sell_condition.clip(lower=0, upper=1),
        bollinger_sell_condition.clip(lower=0, upper=1),
        macd_sell_condition.clip(lower=0, upper=1),
        volume_sell_condition.clip(lower=0, upper=1)
    ]

    # Calculate the percentage of conditions met for each day
    tickerDf['Buy_Signal_Score'] = sum(buy_conditions) / len(buy_conditions) * 100
    tickerDf['Sell_Signal_Score'] = sum(sell_conditions) / len(sell_conditions) * 100


    data = tickerDf['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0,1))
    data = scaler.fit_transform(data)
    train_data, _ = train_test_split(data, test_size=0.2, shuffle=False)
    trainX, trainY = create_dataset(train_data)

    # Define the LSTM model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(trainX.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(trainX, trainY, epochs=20, batch_size=64, verbose=0)

    # Predict the close price for the next day
    last_days_data = np.array([data[-60:]])
    next_day_pred = model.predict(last_days_data)
    next_day_pred = scaler.inverse_transform(next_day_pred)

    return {
        'Buy_Signal_Score': tickerDf['Buy_Signal_Score'].iloc[-60:].mean(),
        'Sell_Signal_Score': tickerDf['Sell_Signal_Score'].iloc[-60:].mean()
    }





def create_dataset(data, look_back=60):
    X, Y = [], []
    for i in range(len(data)-look_back-1):
        a = data[i:(i+look_back), :]
        X.append(a)
        Y.append(data[i + look_back, 0]) # Assuming the 'Close' price is at index 0
    return np.array(X), np.array(Y)



# List of stocks to analyze
stocks = ['GOOG', 'AAPL', 'MSFT', 'AMZN', 'TSLA' ,'FTCH', 'RETA', 'NVDA', 'SPGI', 'ROKU', 'TAL', 'SNBR', 'COUR', 'QS','CMRE','TIGR']

# Analyze each stock and store the results in a dictionary
signal_scores = {stock: analyze_stock(stock) for stock in stocks}

# Get the stock with the highest buy signal score
best_buy_stock = max(signal_scores, key=lambda x: signal_scores[x]['Buy_Signal_Score'])

# Get the stock with the highest sell signal score
best_sell_stock = max(signal_scores, key=lambda x: signal_scores[x]['Sell_Signal_Score'])

print('The stock with the highest buy signal score is', best_buy_stock)
print('The stock with the highest sell signal score is', best_sell_stock)

# Analyze each stock and store the results in a dictionary
signal_scores = {stock: analyze_stock(stock) for stock in stocks}

# Prepare data for the graph
stocks = list(signal_scores.keys())
buy_scores = [signal_scores[stock]['Buy_Signal_Score'] for stock in stocks]
sell_scores = [signal_scores[stock]['Sell_Signal_Score'] for stock in stocks]

# Generate graph
x = np.arange(len(stocks))  # the label locations
width = 0.35  # the width of the bars

#fig, ax = plt.subplots()
#rects1 = ax.bar(x - width/2, buy_scores, width, label='Buy Signal Score')
#rects2 = ax.bar(x + width/2, sell_scores, width, label='Sell Signal Score')

# Add some text for labels, title and custom x-axis tick labels, etc.
#ax.set_xlabel('Stocks')
#ax.set_ylabel('Scores')
#ax.set_title('Buy and Sell Signal Scores for Each Stock')
#ax.set_xticks(x)
#ax.set_xticklabels(stocks)
#ax.legend()

#fig.tight_layout()

#plt.show()