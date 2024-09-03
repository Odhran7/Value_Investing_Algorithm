# Stock portfolio class with threading
import yfinance as yf
import pandas as pd
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from stock.stockData import StockData

class StockPortfolio:
    def __init__(self, stocks, max_workers=5):
        self.stocks = stocks
        self.max_workers = max_workers
        self.portfolio = self.build_portfolio()
        self.n = len(stocks)

    def build_portfolio(self, save_to_csv=True):
        failed = []
        portfolio = pd.DataFrame()
        print("Building portfolio...")

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self.build_stock_data, stock): stock for stock in self.stocks}

            for future in as_completed(futures):
                stock = futures[future]
                try:
                    stock_data = future.result()
                    if stock_data is not None:
                        portfolio = pd.concat([portfolio, stock_data], ignore_index=True)
                except Exception as e:
                    print(f"Failed to obtain data for {stock}. Error: {e}")
                    failed.append(stock)

        if save_to_csv:
            # Pay attention to output file -> don't overwrite
            portfolio.to_csv('test_data.csv', index=False)
        return portfolio

    def build_stock_data(self, ticker):
        try:
            stock = StockData(ticker)
            print(stock)
            stock_data = stock.build_data()
            return stock_data
        except Exception as e:
            print(f"Failed to obtain data for {ticker}. Error: {e}")
            return None

    def get_portfolio(self):
        return self.portfolio
    
    def __str__(self):
        return f"StockPortfolio({self.stocks})"
