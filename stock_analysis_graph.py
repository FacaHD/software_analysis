import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import ta
import time
from IPython.display import clear_output

tickerSymbol = 'FTCH'
tickerData = yf.Ticker(tickerSymbol)

while True: 
    try:
        #Apple Inc.
        # Get data on this ticker
        # Get the historical prices for this ticker
        tickerDf = tickerData.history(period='5d', interval='1m')

        # Calculate the simple moving averages
        tickerDf['SMA_20'] = tickerDf['Close'].rolling(window=20).mean()
        tickerDf['SMA_50'] = tickerDf['Close'].rolling(window=50).mean()

        # Calculate the exponential moving averages
        tickerDf['EMA_20'] = tickerDf['Close'].ewm(span=20, adjust=False).mean()
        tickerDf['EMA_50'] = tickerDf['Close'].ewm(span=50, adjust=False).mean()

        # Calculate the RSI
        tickerDf['RSI'] = ta.momentum.rsi(tickerDf['Close'], window=14)

        # Calculate the MACD
        tickerDf['MACD'] = ta.trend.macd_diff(tickerDf['Close'])

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


        print(tickerDf.head())
        clear_output(wait=True)
        fig, axs = plt.subplots(5, 1, figsize=[10, 40],sharex=True)
        plt.subplots_adjust(hspace=0.1)


        # Plot the close price, SMAs, and EMAs
        tickerDf[['Close', 'SMA_20', 'SMA_50', 'EMA_20', 'EMA_50']].plot(ax=axs[0], title="Close price, SMAs, and EMAs of "+tickerSymbol)

        # Plot the RSI
        tickerDf['RSI'].plot(ax=axs[1], title="RSI of "+tickerSymbol)
        axs[1].hlines(70, xmin=tickerDf.index.min(), xmax=tickerDf.index.max(), colors='r')
        axs[1].hlines(30, xmin=tickerDf.index.min(), xmax=tickerDf.index.max(), colors='r')

        # Plot the MACD
        tickerDf['MACD'].plot(ax=axs[2], title="MACD of "+tickerSymbol)
        axs[2].hlines(0, xmin=tickerDf.index.min(), xmax=tickerDf.index.max(), colors='r')

        # Plot the Bollinger Bands
        tickerDf[['Close', 'BB_high', 'BB_mid', 'BB_low']].plot(ax=axs[3], title="Bollinger Bands of "+tickerSymbol)

        # Plot the Volume
        tickerDf['Volume'].plot(ax=axs[4], title="Volume of "+tickerSymbol)
        axs[4].axhline(y=tickerDf['avg_volume'].mean(), color='r', linestyle='-', label='Average Volume')
        print("YE")
        plt.tight_layout()
        plt.show()
        
        time.sleep(60)
        plt.close()
    except Exception as e:
        print(f"An error occurred: {e}")
        break

