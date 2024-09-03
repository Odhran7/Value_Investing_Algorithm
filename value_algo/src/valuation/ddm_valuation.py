import numpy as np
import yfinance as yf
import pandas as pd
from valuation import Valuation

# Dividend discount model
class DDMValuation(Valuation):
    def __init__(self, stock):
        super().__init__(stock)
        self.dividend_payout_ratio = self.__calculate_dividend_payout_ratio()

    def __calculate_dividend_payout_ratio(self):
        div_yield = self.stock.df.get('dividendYield', 0) * 100
        pe_ratio = self.stock.df.get('trailingPE', 0)
        dividend_payout_ratio = div_yield / (pe_ratio + 1e-10)
        return 0 if pd.isna(dividend_payout_ratio) else dividend_payout_ratio
    
    def check(self):
        return self.dividend_payout_ratio > 0.15

    def __get_current_stock_price(self):
        return self.stock.df.get('previousClose')
    
    def __calculate_historical_growth(self, periods=0):
        pct_changes = self.stock.dividends.pct_change(periods=periods) if periods else self.stock.dividends.pct_change()
        lower, upper = pct_changes.quantile(0.025), pct_changes.quantile(0.975)
        filtered_pct_changes = pct_changes[(pct_changes >= lower) & (pct_changes <= upper)]
        return filtered_pct_changes.mean()

    def __calculate_rolling_average_growth(self, window=10):
        rolling_growth = self.stock.dividends.pct_change().rolling(window=window).mean()
        return rolling_growth.mean()

    def __calculate_combined_growth(self, short_period=5, long_period=10):
        short_term_growth = self.__calculate_historical_growth(periods=short_period)
        long_term_growth = self.__calculate_historical_growth(periods=long_period)
        combined_growth = (0.4 * short_term_growth) + (0.6 * long_term_growth)
        return combined_growth

    def __get_realistic_growth_rate(self):
        historical_growth = self.__calculate_historical_growth()
        rolling_average_growth = self.__calculate_rolling_average_growth()
        combined_growth = self.__calculate_combined_growth()
        cost_of_equity = self.cost_of_equity()
        realistic_growth_rate = max(min(historical_growth, rolling_average_growth, combined_growth), 0.02)
        optimal_growth_rate = min(max(historical_growth, rolling_average_growth, combined_growth), cost_of_equity * 0.9)
        return {
            'historical_growth': historical_growth,
            'rolling_average_growth': rolling_average_growth,
            'combined_growth': combined_growth,
            'realistic_growth_rate': realistic_growth_rate,
            'optimal_growth_rate': optimal_growth_rate
        }

    def get_value(self):
        if not self.check():
            print("Dividend Payout Ratio is too low to calculate DDM value.")
            return np.nan
        return self.__get_ddm_value()
    
    def cost_of_equity(self):
        beta = self.stock.df.get('beta', 1)
        nominal_risk_free_rate = self.risk_free_rate
        market_rate_of_return = self.last_sp500_return
        cost_of_equity = nominal_risk_free_rate + beta * (market_rate_of_return - nominal_risk_free_rate)
        return cost_of_equity
    
    def __get_current_dividend(self):
        dividends = self.stock.dividends
        current_dividend = dividends.iloc[-1]
        return current_dividend

    def __get_ddm_value(self):
        current_dividend = self.__get_current_dividend()
        if pd.isna(current_dividend):
            return np.nan

        cost_of_equity = self.cost_of_equity()
        growth_data = self.__get_realistic_growth_rate()
        expected_growth_rate = growth_data.get('realistic_growth_rate')
        optimal_growth_rate = growth_data.get('optimal_growth_rate')

        if pd.isna(expected_growth_rate) or cost_of_equity <= expected_growth_rate:
            expected_growth_rate = 0.8 * cost_of_equity

        expected_dividend_next_year = current_dividend * (1 + expected_growth_rate)
        intrinsic_value_min = expected_dividend_next_year / (cost_of_equity - expected_growth_rate)
        intrinsic_value_max = expected_dividend_next_year / (cost_of_equity - optimal_growth_rate)
        return [intrinsic_value_min, intrinsic_value_max]

    def get_valuation_ratios(self):
        ddm_values = self.__get_ddm_value()
        ddm_valuation_ratio_min = ddm_values[0] / self.stock.df.get('marketCap')
        ddm_valuation_ratio_max = ddm_values[1] / self.stock.df.get('marketCap')    
        expected_growth_rate = self.__get_realistic_growth_rate().get('realistic_growth_rate')
        return [ddm_valuation_ratio_min, ddm_valuation_ratio_max, expected_growth_rate]

    def test(self):
        try:
            current_stock_price = self.__get_current_stock_price()
            growth_data = self.__get_realistic_growth_rate()
            expected_growth_rate = growth_data.get('realistic_growth_rate')
            required_rate_of_return = self.cost_of_equity()
            current_dividend = self.__get_current_dividend()
            intrinsic_value_min, intrinsic_value_max = self.__get_ddm_value()
            if self.debug:
                print(f"Ticker: {self.stock.ticker}")
                print(f"Current Dividend: {current_dividend}")
                print(f"Current Stock Price: {current_stock_price}")
                print(f"Expected Growth Rate: {expected_growth_rate:.4f}")
                print(f"Required Rate of Return: {required_rate_of_return:.4f}")
                print(f"Intrinsic Value min: {intrinsic_value_min:.2f}")
                print(f"Intrinsic Value max: {intrinsic_value_max:.2f}")
                print(f"Dividend Payout Ratio: {self.dividend_payout_ratio:.2f}")
        except Exception as e:
            print(f"Failed to obtain data for {self.stock.ticker}.")
            print(e)