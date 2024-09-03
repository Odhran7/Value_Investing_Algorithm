import yfinance as yf
import pandas as pd
from valuation import Valuation
import numpy as np

# Current inflation rate is 0.0160

class DCFValuation(Valuation):
    def __init__(self, stock, inflation_rate, periods=4):
        super().__init__(stock)
        self.inflation_rate = inflation_rate
        self.periods = periods # Periods of FCF to take into account => Yf caps us at four
        self.last_sp500_return = self.__get_market_rate_of_return() # Expected Return
        self.risk_free_rate = self.__get_risk_free_rate() # Risk Free Rate

    def __get_treasury_yields(self):
        treasury_ticker = '^TNX' # We use the ten year treasury yield
        treasury_data = yf.Ticker(treasury_ticker)
        treasury_yield = treasury_data.history(period='5d')['Close'].iloc[-1]
        risk_free_rate = treasury_yield / 100
        return risk_free_rate

    def __get_market_rate_of_return(self):
        sp500 = yf.Ticker('^GSPC')
        sp500_data = sp500.history(period='10y')
        sp500_data['Yearly Return'] = sp500_data['Close'].pct_change(periods=252)
        yearly_returns = sp500_data['Yearly Return'].dropna()
        avg_annual_return = yearly_returns.mean()
        return avg_annual_return

    def __get_risk_free_rate(self):
        raw_risk_free_rate = self.__get_treasury_yields()
        nominal_risk_free_rate = (1 + raw_risk_free_rate) * (1 + self.inflation_rate) - 1
        return nominal_risk_free_rate

    def __get_cash_flows(self):
        try:
            cash_flows = self.stock.cash_flow.loc['Free Cash Flow'].iloc[:self.periods].to_numpy()
        except KeyError:
            cash_flows = self.stock.cash_flow.loc['Operating Cash Flow'].iloc[:self.periods].to_numpy()
        
        # Ensure cash flows are non-negative
        cash_flows = np.where(cash_flows < 0, 0, cash_flows)
        return list(cash_flows)

    def __get_growth_rate(self):
        cash_flows = self.__get_cash_flows()
        growth_rates = []
        for i in range(1, len(cash_flows)):
            yearOne, yearTwo = cash_flows[i - 1], cash_flows[i]
            if yearOne <= 0 or yearTwo <= 0:
                continue
            curr_growth_rate = (yearTwo - yearOne) / yearOne
            growth_rates.append(curr_growth_rate)
        if not growth_rates:
            return 0
        avg_growth_rate = sum(growth_rates) / len(growth_rates)
        # We cap at 20% (to be prudent)
        return min(max(avg_growth_rate, 0), 0.20)

    def cost_of_equity(self):
        beta = self.stock.df.get('beta', 1)
        nominal_risk_free_rate = self.risk_free_rate
        market_rate_of_return = self.last_sp500_return
        cost_of_equity = nominal_risk_free_rate + beta * (market_rate_of_return - nominal_risk_free_rate)
        return cost_of_equity

    def cost_of_debt(self):
        credit_spread = self.__get_credit_spread()
        tax_rate = self.__get_tax_rate()
        nominal_risk_free_rate = self.risk_free_rate    
        current_credit_spread = credit_spread[0]
        cost_of_debt = (nominal_risk_free_rate + current_credit_spread) * (1 - tax_rate)
        return cost_of_debt
    
    def get_wacc(self):
        tax_rate = self.__get_tax_rate()
        cost_of_equity = self.cost_of_equity()
        cost_of_debt = self.cost_of_debt()
        equity_weight, debt_weight = self.get_weights()
        wacc = (equity_weight * cost_of_equity) + (debt_weight * cost_of_debt * (1 - tax_rate))
        return max(wacc, 0)

    def get_weights(self):
        def get_market_value_of_equity():
            market_cap = self.stock.df.get('marketCap')
            return market_cap
        
        def get_market_cap_of_debt():
            total_debt = self.stock.balance_sheet.loc['Total Debt'].iloc[0]
            return total_debt

        equity = get_market_value_of_equity()
        total_debt = get_market_cap_of_debt()
        equity_weight = equity / (equity + total_debt)
        debt_weight = total_debt / (equity + total_debt)
        return equity_weight, debt_weight
    
    def __get_tax_rate(self):
        return self.stock.income_stmt.loc['Tax Rate For Calcs'].iloc[0]
    
    def __get_credit_spread(self):
        real_risk_free_rate = self.risk_free_rate
        interest_expense = self.stock.obtain_interest_expense(periods=self.periods)
        total_debt = self.stock.balance_sheet.loc['Total Debt'].iloc[:self.periods].to_numpy()
        average_interest_expense = [self.risk_free_rate if interest_expense[i] == 0 else interest_expense[i] / total_debt[i] for i in range(len(total_debt))]
        credit_spread = [avg_interest - real_risk_free_rate for avg_interest in average_interest_expense]
        return credit_spread

    def __calculate_terminal_value(self, growth_rate, wacc):
        final_year_fcf = self.stock.cash_flow.loc['Free Cash Flow'].iloc[0]
        perpetuity_growth_rate = 0 if growth_rate < 0 else 0.02
        if wacc <= perpetuity_growth_rate:
            wacc = perpetuity_growth_rate + 0.01
        terminal_value = (final_year_fcf * (1 + perpetuity_growth_rate)) / (wacc - perpetuity_growth_rate)
        return terminal_value

    def get_valuation_ratios(self):
        dcf_valuation_ratio = self.__get_dcf_value() / self.stock.df.get('marketCap')
        free_cash_flow = self.stock.cash_flow.loc['Free Cash Flow'].iloc[:self.periods].to_numpy()
        if len(free_cash_flow) < self.periods:
            free_cash_flow = np.append(free_cash_flow, free_cash_flow[-1] * self.periods - len(free_cash_flow))
        cash_flows = free_cash_flow[:4]
        growth_rate = self.__get_growth_rate()
        wacc = self.get_wacc()
        cost_of_equity = self.cost_of_equity()
        cost_of_debt = self.cost_of_debt()
        return [dcf_valuation_ratio, *cash_flows, growth_rate, wacc, cost_of_equity, cost_of_debt]

    def __get_dcf_value(self):
        growth_rate = self.__get_growth_rate()
        wacc = self.get_wacc()
        cash_flows = self.__get_cash_flows()
        forecasted_fcfs = [cash_flows[-1] * ((1 + growth_rate) ** (i + 1)) for i in range(self.periods)]
        discounted_fcfs = [fcf / ((1 + wacc) ** (i + 1)) for i, fcf in enumerate(forecasted_fcfs)]
        terminal_value = self.__calculate_terminal_value(growth_rate, wacc)
        discounted_terminal_value = terminal_value / ((1 + wacc) ** self.periods)
        dcf_value = sum(discounted_fcfs) + discounted_terminal_value

        # If debug is true
        if self.debug:
            print(f"Growth Rate: {growth_rate}")
            print(f"WACC: {wacc}")
            print(f"Cash Flows: {cash_flows}")
            print(f"Forecasted FCFs: {forecasted_fcfs}")
            print(f"Discounted FCFs: {discounted_fcfs}")
            print(f"Terminal Value: {terminal_value}")
            print(f"Discounted Terminal Value: {discounted_terminal_value}")
            print(f"DCF Value: {dcf_value}")

        return dcf_value

    def get_value(self):
        return self.__get_dcf_value()

    def test(self):
        try:
            market_cap = self.stock.df.get('marketCap')
            wacc = self.get_wacc()
            growth_rate = self.__get_growth_rate()
            terminal_value = self.__calculate_terminal_value(growth_rate, wacc)
            dcf_value = self.__get_dcf_value()
            if self.debug:
                print(f"Ticker: {self.stock.ticker}")
                print(f"Risk Free Rate: {self.risk_free_rate}")
                print(f"Market Cap: {market_cap}")
                print(f"WACC: {wacc}")
                print(f"Growth Rate: {growth_rate}")
                print(f"Terminal Value: {terminal_value}")
                print(f"DCF Value: {dcf_value}")
                print(f"Valuation ratio: {dcf_value / market_cap:.2f}")
        except Exception as e:
            print(f"Failed to obtain data for {self.ticker}.")
            print(e)