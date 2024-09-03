import yfinance as yf
import pandas as pd
import numpy as np

class Stock:
    def __init__(self, ticker, dcf_valuation, ddm_valuation, inflation_rate=0.0160, years=1, isQuarter=False):
        self.ticker = ticker
        self.years = years
        self.isQuarter = isQuarter
        self.inflation_rate = inflation_rate
        self.stock = yf.Ticker(ticker)
        self.df = self.stock.info
        if self.df is None:
            raise ValueError(f"Failed to get data for {self.ticker}.")
        self.income_stmt = self.stock.financials
        self.balance_sheet = self.stock.balance_sheet
        self.cash_flow = self.stock.cashflow
        if self.income_stmt.empty or self.balance_sheet.empty or self.cash_flow.empty:
            raise ValueError(f"Failed to get financial data for {self.ticker}.")
        self.dividends = self.stock.dividends
        self.dcf_valuation = dcf_valuation(self, self.inflation_rate)
        self.ddm_valuation = ddm_valuation(self)
        self.market_cap = self.df.get('marketCap')

        self.validate_data()

    def validate_data(self):
        if not self.ticker or not isinstance(self.ticker, str):
            raise ValueError("Ticker must be a non-empty string.")
        if not isinstance(self.inflation_rate, (int, float)):
            raise ValueError("Inflation rate must be a number.")

    def get_history(self, period="max"):
        try:
            return self.stock.history(period=period)
        except Exception as e:
            print(f"Failed to get history for {self.ticker}: {e}")
            return pd.DataFrame()

    def refresh_data(self):
        try:
            self.stock = yf.Ticker(self.ticker)
            self.df = self.stock.info
            self.income_stmt = self.stock.financials
            self.balance_sheet = self.stock.balance_sheet
            self.cash_flow = self.stock.cashflow
            self.dividends = self.stock.dividends
            print(f"Data refreshed for {self.ticker}.")
        except Exception as e:
            print(f"Failed to refresh data for {self.ticker}: {e}")

    def get_value(self, attribute, default=None, isBig=False):
        return self.df.get(attribute, default) if not isBig else self.df.get(attribute, default) / 100

    def test(self, fn):
        try:
            data = fn()
            print(data)
        except Exception as e:
            print(f"Failed to execute function for {self.ticker}: {e}")

    # Util methods

    def _get_metric_from_df(self, key, default=0):
        return self.df.get(key, default)

    def _get_metric_from_income_stmt(self, key, default=0):
        try:
            return self.income_stmt.loc[key].iloc[0] / 100
        except KeyError:
            return default

    def _get_metric_from_balance_sheet(self, key, default=0):
        try:
            return self.balance_sheet.loc[key].iloc[0] / 100
        except KeyError:
            return default

    def _get_metric_from_cashflow(self, key, default=0):
        try:
            return self.cash_flow.loc[key].iloc[0] / 100
        except KeyError:
            return default

    def _get_sum_of_keys(self, primary_key, keys):
        try:
            return self.balance_sheet.loc[primary_key].iloc[0]
        except KeyError:
            total = 0
            for key in keys:
                if key in self.balance_sheet.index:
                    total += self.balance_sheet.loc[key].sum()
                else:
                    print(f"Key '{key}' not found in balance sheet.")
            return total

    def _get_ratio(self, numerator, denominator, default=0):
        return numerator / denominator if denominator != 0 else default

    def __str__(self):
        return f"Stock({self.ticker})"


class StockData(Stock):
    def __init__(self, ticker, inflation_rate=0.0160):
        super().__init__(ticker, inflation_rate)

    def build_data(self):
        # Basic financial metrics
        financial_metrics = {
            'Ticker': self.ticker,
            'Market Cap': self.market_cap,
            'Revenue': self.df.get('totalRevenue', 0),
            'Sector': self.df.get('sector', 'Unknown'),
            'Free Cash Flow': self.df.get('freeCashflow', 0),
            'Capital Expenditure': abs(self.__get_cap_ex()),
            'Net Income': self.df.get('netIncomeToCommon', 0),
            'Operating Cash Flow': self._get_metric_from_cashflow('Operating Cash Flow', 0),
            'EBITDA': self.df.get('ebitda', 0)
        }

        current_assets, total_assets = self.__get_assets()
        interest_expense = self.obtain_interest_expense()

        valuation_metrics = self.__get_valuation_metrics(financial_metrics, current_assets, total_assets, interest_expense)
        quality_metrics = self.__get_quality_metrics(financial_metrics, current_assets, total_assets, interest_expense)
        economic_moat_metrics = self.__get_economic_moat_metrics(financial_metrics)
        price_data = self.__get_price_data()

        value_data = {**financial_metrics, **valuation_metrics, **quality_metrics, **economic_moat_metrics, **price_data}
        value_data_df = pd.DataFrame([value_data])
        cols_to_divide = ['Market Cap', 'Revenue', 'Free Cash Flow', 'Capital Expenditure', 'Net Income', 'Operating Cash Flow', 'EBITDA', 'DCF Value Billions', 'Operating Cash Flow Billions', 'Retained Earnings Billions', 'Economic Value Added Billions', 'Earnings Power Value']
        for col in cols_to_divide:
            if col in value_data_df.columns:
                value_data_df[col] = value_data_df[col] / 1e9

        return value_data_df

    def __get_total_assets(self):
        return self._get_sum_of_keys('Total Assets', [
            'Cash Cash Equivalents And Short Term Investments',
            'Other Short Term Investments',
            'Cash And Cash Equivalents',
            'Cash Equivalents',
            'Cash Financial',
            'Receivables',
            'Other Receivables',
            'Accounts Receivable',
            'Inventory',
            'Other Current Assets',
            'Net PPE',
            'Investments And Advances',
            'Long Term Equity Investment',
            'Investments In Other Ventures Under Equity Method',
            'Goodwill And Other Intangible Assets',
            'Other Intangible Assets',
            'Goodwill',
            'Properties',
            'Machinery Furniture Equipment',
            'Land And Improvements'
        ])

    def __get_current_assets(self):
        return self._get_sum_of_keys('Current Assets', [
            'Cash Cash Equivalents And Short Term Investments',
            'Other Short Term Investments',
            'Cash And Cash Equivalents',
            'Cash Equivalents',
            'Cash Financial',
            'Receivables',
            'Other Receivables',
            'Accounts Receivable',
            'Inventory',
            'Other Current Assets'
        ])

    def __get_assets(self):
        total_assets = self.__get_total_assets()
        current_assets = self.__get_current_assets()
        return current_assets, total_assets

    def __get_valuation_metrics(self, financial_metrics, current_assets, total_assets, interest_expense):
        dcf_value = self.__get_dcf_value()
        if not self.ddm_valuation.check():
            ddm_valuation_ratio_min, ddm_valuation_ratio_max, dividend_expected_growth_rate = 0, 0, 0
        else:
            ddm_valuation_data = self.__get_ddm_values()
            ddm_valuation_ratio_min, ddm_valuation_ratio_max, dividend_expected_growth_rate = ddm_valuation_data

        dcf_valuation_data = self.__get_dcf_values()
        dcf_valuation_ratio, cash_flow1, cash_flow2, cash_flow3, cash_flow4, growth_rate, wacc, cost_of_equity, cost_of_debt = dcf_valuation_data
        strong_buy, buy, hold, sell, strong_sell = self.__parse_analyst_reccomendations()
        equity_weight, debt_weight = self.dcf_valuation.get_weights()

        pe_ratio = self._get_metric_from_df('trailingPE', 0)
        div_yield = self._get_metric_from_df('dividendYield', 0) * 100
        eps_growth = self.__get_eps_growth()
        return {
            'DCF Value Billions': dcf_value,
            'DCF Valuation Ratio': dcf_valuation_ratio,
            'DDM Valuation Ratio Min': ddm_valuation_ratio_min,
            'DDM Valuation Ratio Max': ddm_valuation_ratio_max,
            'Dividend Expected Growth Rate': dividend_expected_growth_rate,
            'P/E Ratio': pe_ratio,
            'P/B Ratio': self._get_metric_from_df('priceToBook', 0),
            'P/S Ratio': self._get_metric_from_df('priceToSalesTrailing12Months', 0),
            'P/FCF Ratio': self._get_ratio(self.market_cap, financial_metrics['Free Cash Flow']),
            'Dividend Yield': div_yield,
            'EPS': self._get_metric_from_df('trailingEps', 0),
            'EPS Growth': eps_growth,
            'Revenue Growth': self.__get_revenue_growth(),
            'PEG Ratio': self._get_ratio(pe_ratio, eps_growth),
            'Cash Flow 1 Billions': cash_flow1,
            'Cash Flow 2 Billions': cash_flow2,
            'Cash Flow 3 Billions': cash_flow3,
            'Cash Flow 4 Billions': cash_flow4,
            'Growth Rate': growth_rate,
            "NCAVPS": self.__get_net_current_asset_value_per_share(),
            'WACC': wacc,
            'Equity Weight': equity_weight,
            'Debt Weight': debt_weight,
            'Cost of Equity': cost_of_equity,
            'Cost of Debt': cost_of_debt,
            'Enterprise to Revenue': self._get_metric_from_df('enterpriseToRevenue', 0),
            'Enterprise to EBITDA': self._get_metric_from_df('enterpriseToEbitda', 0),
            'Total Cash Per Share': self._get_metric_from_df('totalCashPerShare', 0),
            'Interest Expense Billions': interest_expense / 100,
            'Strong Buy': strong_buy,
            'Buy': buy,
            'Hold': hold,
            'Sell': sell,
            'Strong Sell': strong_sell
        }

    def __get_quality_metrics(self, financial_metrics, current_assets, total_assets, interest_expense):
        current_liabilities = self.__get_current_liabilities()
        net_income = financial_metrics['Net Income']
        roe = self.__get_return_on_equity()
        de_ratio = self._get_metric_from_df('debtToEquity', 0)

        return {
            'Net Margin': self._get_ratio(net_income, financial_metrics['Revenue']),
            'Gross Margin': self.__get_gross_margin(),
            'ROA': self._get_metric_from_df('returnOnAssets', 0),
            'ROE': roe,
            'Current Ratio': self._get_ratio(current_assets, current_liabilities),
            'Quick Ratio': self.__get_quick_ratio(),
            'D/E Ratio': de_ratio,
            'Dividend Payout Ratio': self.__calculate_dividend_payout_ratio(),
            'FCF Yield': self._get_ratio(financial_metrics['Free Cash Flow'], self.market_cap) * 100,
            'Capex_to_sales': self._get_ratio(abs(financial_metrics['Capital Expenditure']), financial_metrics['Revenue']),
            'Operating Margin': self.__get_operating_margin(),
            'Debt to Asset Ratio': self._get_ratio(de_ratio, 1 + de_ratio),
            'Equity to Asset Ratio': self.__get_debt_to_asset(),
            'Interest Coverage Ratio': self._get_ratio(financial_metrics['Revenue'], interest_expense),
            'NCAVPS': self.__get_net_current_asset_value_per_share(),
            'ROIC': self.__get_roic(),
            'Altman Z Score': self.__get_altman_z_score(),
            'Operating Cash Flow Billions': financial_metrics['Operating Cash Flow'],
            'Asset Turnover Ratio': self.__get_turnover_ratios()[0],
            'Inventory Turnover Ratio': self.__get_turnover_ratios()[1],
            'Retained Earnings Billions': self.__get_retained_earnings(),
            'Earnings Power Value': self.__get_earnings_power_value()
        }

    def __get_price_data(self):
        current_price = self.get_history(period="1d")['Close'].iloc[-1]
        historical_data = self.get_history(period="1y")

        moving_average_50 = historical_data['Close'].rolling(window=50).mean().iloc[-1]
        moving_average_200 = historical_data['Close'].rolling(window=200).mean().iloc[-1]

        year_high = historical_data['Close'].max()
        year_low = historical_data['Close'].min()

        average_volume_50 = historical_data['Volume'].rolling(window=50).mean().iloc[-1]

        day_high = self.get_history(period="1d")['High'].iloc[-1]
        day_low = self.get_history(period="1d")['Low'].iloc[-1]

        price_data = {
            'Current Price': current_price,
            '50-Day Moving Average': moving_average_50,
            '200-Day Moving Average': moving_average_200,
            '52-Week High': year_high,
            '52-Week Low': year_low,
            'Average Volume (50 days)': average_volume_50,
            'Day High': day_high,
            'Day Low': day_low,
            'Previous Close': historical_data['Close'].iloc[-2],
            'Open': self.get_history(period='1d')['Open'].iloc[-1]
        }
        
        return price_data

    def __get_economic_moat_metrics(self, financial_metrics):
        free_cash_flow = financial_metrics['Free Cash Flow']
        revenue = financial_metrics['Revenue']
        fcf_ratio = self._get_ratio(free_cash_flow, revenue)
        eva = self.__get_eva()
        sgr = self.__get_sgr()

        return {
            'Economic Value Added Billions': eva,
            'FCF Ratio': fcf_ratio,
            'Sustainable Growth Rate': sgr
        }

    def __get_quick_ratio(self):
        quick_ratio = self._get_metric_from_df('quickRatio', 0)
        if quick_ratio == 0:
            inventory = self._get_metric_from_balance_sheet('Inventory', 0)
            current_assets = self.__get_current_assets()
            current_liabilities = self.__get_current_liabilities()
            quick_ratio = self._get_ratio(current_assets - inventory, current_liabilities)
        return quick_ratio
    
    def __get_debt_to_asset(self):
        debt = self._get_metric_from_balance_sheet('Total Debt', 0)
        assets = self._get_metric_from_balance_sheet('Total Assets', 0)
        return self._get_ratio(debt, assets)

    def __get_return_on_equity(self):
        roe = self._get_metric_from_df('returnOnEquity', 0)
        if roe == 0:
            net_income = self._get_metric_from_df('netIncomeToCommon', 0)
            equity = self._get_metric_from_balance_sheet('Total Equity', 0)
            roe = self._get_ratio(net_income, equity)
        return roe

    def __get_gross_margin(self):
        gross_margin = self._get_metric_from_df('grossMargins', 0)
        if gross_margin == 0:
            gross_profit = self._get_metric_from_income_stmt('Gross Profit', 0)
            revenue = self._get_metric_from_df('totalRevenue', 0)
            gross_margin = self._get_ratio(gross_profit, revenue)
        return gross_margin

    def __get_eps_growth(self):
        eps_growth = self._get_metric_from_df('earningsQuarterlyGrowth', 0)
        if eps_growth == 0:
            try:
                eps_diluted_array = self.income_stmt.loc['Diluted EPS'].to_numpy()[:4]
                eps_growth = self._get_ratio(eps_diluted_array[0] - eps_diluted_array[1], eps_diluted_array[1])
            except KeyError:
                return 0
        return eps_growth

    def __get_revenue_growth(self):
        revenue_growth = self._get_metric_from_df('revenueQuarterlyGrowth', 0)
        if revenue_growth == 0:
            try:
                revenue_array = self.income_stmt.loc['Total Revenue'].to_numpy()[:4]
                revenue_growth = self._get_ratio(revenue_array[0] - revenue_array[1], revenue_array[1])
            except KeyError:
                return 0
        return revenue_growth

    def __get_operating_margin(self):
        operating_margin = self._get_metric_from_df('operatingMargins', 0)
        if operating_margin == 0:
            operating_income = self._get_metric_from_income_stmt('Operating Income', 0)
            revenue = self._get_metric_from_df('totalRevenue', 0)
            operating_margin = self._get_ratio(operating_income, revenue)
        return operating_margin

    def __get_roic(self):
        net_income = self._get_metric_from_df('netIncomeToCommon', 0)
        total_assets = self.__get_total_assets()
        current_liabilities = self.__get_current_liabilities()
        return self._get_ratio(net_income, total_assets - current_liabilities)

    def __get_ebit(self):
        ebit = self._get_metric_from_income_stmt('EBIT', 0)
        if ebit == 0:
            net_income = self._get_metric_from_df('netIncomeToCommon', 0)
            special_income_charges = self._get_metric_from_income_stmt('Special Income Charges', 0)
            tax_provision = self._get_metric_from_income_stmt('Tax Provision', 0)
            interest_expense = self.obtain_interest_expense()
            ebit = net_income + interest_expense + tax_provision + special_income_charges
        return ebit

    def __get_earnings_power_value(self):
        ebit = self.__get_ebit()
        tax_rate = self._get_metric_from_income_stmt('Tax Rate For Calcs', 0)
        wacc = self.dcf_valuation.get_wacc()
        epv = self._get_ratio(ebit * (1 - tax_rate), wacc)
        if np.isnan(epv):
            epv = 0
        return epv

    def __get_altman_z_score(self):
        ebit = self.__get_ebit()
        working_capital = self.__get_current_assets() - self.__get_current_liabilities()
        retained_earnings = self.__get_retained_earnings()
        total_assets = self.__get_total_assets()
        total_liabilities = self.__get_total_liabilities()
        sales = self._get_metric_from_df('totalRevenue', 0)
        z = (1.2 * self._get_ratio(working_capital, total_assets) +
             1.4 * self._get_ratio(retained_earnings, total_assets) +
             3.3 * self._get_ratio(ebit, total_assets) +
             0.6 * self._get_ratio(self.market_cap, total_liabilities) +
             1.0 * self._get_ratio(sales, total_assets))
        return z

    def __get_turnover_ratios(self):
        sales = self._get_metric_from_df('totalRevenue', 0)
        total_assets = self.__get_total_assets()
        inventory = self._get_metric_from_balance_sheet('Inventory', 0)
        return self._get_ratio(sales, total_assets), self._get_ratio(sales, inventory)

    def _get_dividend_coverage(self):
        eps = self._get_metric_from_df('trailingEps', 0)
        dps = self._get_metric_from_df('dividendRate', 0)
        return self._get_ratio(eps, dps) if dps else np.nan

    def obtain_interest_expense(self, periods=0):
        interest_keys = ['Interest Expense Non Operating', 'Interest Expense', 'Net Non Operating Interest Income Expense']
        for key in interest_keys:
            if key in self.income_stmt.index:
                if periods > 0:
                    try:
                        interest_expense = self.income_stmt.loc[key].iloc[:periods].to_numpy()
                    except IndexError:
                        print(f"Not enough data for {periods} periods in key '{key}'. Returning available data.")
                        interest_expense = self.income_stmt.loc[key].to_numpy()
                else:
                    interest_expense = self.income_stmt.loc[key].iloc[0]
                return interest_expense
        if periods > 0:
            return np.zeros(periods)
        return 0

    def __get_eva(self):
        net_income = self._get_metric_from_income_stmt('Net Income', 0)
        tax_rate = self._get_metric_from_income_stmt('Tax Rate For Calcs', 0)
        nopat = net_income * (1 - tax_rate)
        invested_capital = self.__get_total_assets() - self.__get_current_liabilities()
        wacc = self.dcf_valuation.get_wacc()
        eva = nopat - (invested_capital * wacc)
        if np.isnan(eva):
            eva = 0
        return eva

    def __calculate_dividend_payout_ratio(self):
        div_yield = self._get_metric_from_df('dividendYield', 0) * 100
        pe_ratio = self._get_metric_from_df('trailingPE', 0)
        if pe_ratio != 0:
            dividend_payout_ratio = div_yield / pe_ratio
        else:
            dividend_payout_ratio = 0
        return dividend_payout_ratio

    def __get_sgr(self):
        roe = self._get_metric_from_df('returnOnEquity', 0)
        dividend_payout_ratio = self.__calculate_dividend_payout_ratio()
        retention_ratio = 1 - dividend_payout_ratio
        sgr = roe * retention_ratio
        return sgr if not pd.isna(sgr) else 0


    def __get_net_current_asset_value_per_share(self):
        current_assets = self.__get_current_assets()
        total_liabilities = self.__get_total_liabilities()
        ordinary_shares_outstanding = self._get_metric_from_balance_sheet('Ordinary Shares Number', 0)
        preferred_shares = self._get_metric_from_balance_sheet('Preferred Shares Number', 0)
        ncavps = self._get_ratio(current_assets - total_liabilities - preferred_shares, ordinary_shares_outstanding)
        return ncavps

    def __get_cap_ex(self):
        return self._get_metric_from_cashflow('Capital Expenditure', 0)

    def __get_ddm_value(self):
        return self.ddm_valuation.get_value()

    def __get_ddm_values(self):
        return self.ddm_valuation.get_valuation_ratios()

    def __get_dcf_value(self):
        return self.dcf_valuation.get_value()

    def __get_dcf_values(self):
        return self.dcf_valuation.get_valuation_ratios()

    def __get_retained_earnings(self):
        retained_earnings_keys = [
            'Retained Earnings',
            'Accumulated Retained Earnings',
            'Undistributed Profits'
        ]
        return self._get_sum_of_keys('Retained Earnings', retained_earnings_keys)

    def __get_current_liabilities(self):
        current_liabilities_keys = [
            'Accounts Payable',
            'Short Term Debt',
            'Other Current Liabilities',
            'Current Deferred Liabilities',
            'Current Deferred Revenue',
            'Current Debt And Capital Lease Obligation',
            'Current Debt',
            'Other Current Borrowings',
            'Commercial Paper',
            'Payables And Accrued Expenses',
            'Payables'
        ]
        return self._get_sum_of_keys('Current Liabilities', current_liabilities_keys)

    def __get_total_liabilities(self):
        total_liabilities_keys = [
            'Total Liabilities Net Minority Interest',
            'Total Non Current Liabilities Net Minority Interest',
            'Other Non Current Liabilities',
            'Trade and Other Payables Non Current',
            'Long Term Debt And Capital Lease Obligation',
            'Long Term Debt',
            'Current Liabilities',
            'Other Current Liabilities',
            'Current Deferred Liabilities',
            'Current Deferred Revenue',
            'Current Debt And Capital Lease Obligation',
            'Current Debt',
            'Other Current Borrowings',
            'Commercial Paper',
            'Payables And Accrued Expenses',
            'Payables',
            'Accounts Payable'
        ]
        return self._get_sum_of_keys('Total Liabilities Net Minority Interest', total_liabilities_keys)

    def __parse_analyst_reccomendations(self):
        try:
            analyst_recommendations = self.stock.recommendations
            strong_buy = analyst_recommendations['strongBuy'].sum()
            buy = analyst_recommendations['buy'].sum()
            hold = analyst_recommendations['hold'].sum()
            sell = analyst_recommendations['sell'].sum()
            strong_sell = analyst_recommendations['strongSell'].sum()
        except KeyError:
            return 0, 0, 0, 0, 0
        return strong_buy, buy, hold, sell, strong_sell