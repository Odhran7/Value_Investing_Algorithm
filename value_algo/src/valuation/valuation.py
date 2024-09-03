import yfinance as yf
import pandas as pd
import numpy as np

# Valuation abstract class to implement in dcf and ddm
class Valuation:
    def __init__(self, stock, debug = False):
        self.stock = stock
        self.debug = debug

    def get_value(self):
        raise NotImplementedError("Subclasses should implement this method")

    def test(self):
        raise NotImplementedError("Subclasses should implement this method")
    
    def get_valuation_ratios(self):
        raise NotImplementedError("Subclasses should implement this method")