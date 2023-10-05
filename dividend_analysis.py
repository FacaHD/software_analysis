import yfinance as yf

# Get financial data
def get_financial_data(symbol):
    stock = yf.Ticker(symbol)

    # Get stock info
    info = stock.info

    # Get historical market data
    hist = stock.history()

    return info, hist

# Analyze financial data
def analyze_data(info, hist):
    dividend_yield = info.get('dividendYield') * 100 if info.get('dividendYield') else None
    payout_ratio = info.get('payoutRatio') * 100 if info.get('payoutRatio') else None
    trailing_PE = info.get('trailingPE')
    forward_PE = info.get('forwardPE')
    debt_to_equity = info.get('debtToEquity')
    earnings_per_share = info.get('trailingEps')
    return_on_equity = info.get('returnOnEquity') * 100 if info.get('returnOnEquity') else None
    free_cash_flow = info.get('freeCashflow')
    peg_ratio = info.get('pegRatio')

    print(f'Dividend Yield: {dividend_yield}%') # Above 4% is generally considered high, but very high yields (>10%) can be a warning sign
         # Higher is generally better, but very high yields can be a warning sign
    print(f'Payout Ratio: {payout_ratio}%')  
    # Below 80% is generally sustainable, above 100% can be a warning sign
    print(f'Trailing P/E: {trailing_PE}')  
    # Varies by industry, but often 15-25 is considered average
    # # Lower is generally better, but depends on the industry and growth rates
    print(f'Forward P/E: {forward_PE}')  
    # Varies by industry, but often 15-25 is considered average
    # Lower is generally better, but depends on the industry and growth rates
    print(f'Debt to Equity: {debt_to_equity}') 
     # Below 1 is generally considered healthy, but varies by industry
     #  # Lower is generally better, but depends on the industry
    print(f'Earnings Per Share: {earnings_per_share}')  
    #Positive is generally better, and growth over time is a good sign
    # Higher is generally better
    print(f'Return on Equity: {return_on_equity}%')  
    # 15-20% is generally considered good
    # Higher is generally better
    print(f'Free Cash Flow: {free_cash_flow}')  
    #Positive is generally better, and growth over time is a good sign
    # Positive is generally better
    print(f'PEG Ratio: {peg_ratio}')  # Less than 1 can indicate the stock is undervalued

    # You can add more analysis based on your needs

# Main function
def main():
    symbol = input('Enter the stock symbol: ')
    info, hist = get_financial_data(symbol)
    analyze_data(info, hist)

# Run the main function
if __name__ == "__main__":
    main()