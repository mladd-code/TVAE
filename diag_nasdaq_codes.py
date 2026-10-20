"""
diag_nasdaq_codes.py
--------------------
Quick check: what exchange codes does Refinitiv assign to NASDAQ-listed
stocks? Run this and paste the output.
"""

import refinitiv.data as rd

rd.open_session()

tickers = ["AAPL.O", "MSFT.O", "GOOGL.O", "AMZN.O", "META.O",
           "JPM.N", "XOM.N", "PG.N"]

fields = [
    "TR.ExchangeName",
    "TR.ExchangeMarketIdCode",
    "TR.ExchangeCode",
    "TR.ListingExchange",
]

df = rd.get_data(universe=tickers, fields=fields)
print(df.to_string())

rd.close_session()