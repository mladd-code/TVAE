"""
diag_screener.py
----------------
Diagnose the exchange filter and any result cap.
"""
import refinitiv.data as rd
from refinitiv.data.discovery import Screener

rd.open_session()

# Query A: no exchange filter, just market cap
qA = ('U(IN(Equity(active,public,primary))/*UNV:Public*/), '
      'TR.CompanyMarketCap(Scale=6)>=100, CURN=USD')
dfA = rd.get_data(universe=Screener(qA),
                  fields=["TR.CommonName", "TR.ExchangeCode",
                          "TR.ExchangeName", "TR.CompanyMarketCap"])
print(f"A: no exchange filter, mcap>=100M  ->  {len(dfA)} rows")
print(dfA['Exchange Code'].value_counts().to_dict() if 'Exchange Code' in dfA.columns
      else dfA.columns.tolist())

# Query B: only NASDAQ, no market cap filter
qB = ('U(IN(Equity(active,public,primary))/*UNV:Public*/), '
      'IN(TR.ExchangeCode,"NSQ"), CURN=USD')
dfB = rd.get_data(universe=Screener(qB),
                  fields=["TR.CommonName", "TR.ExchangeCode",
                          "TR.CompanyMarketCap"])
print(f"\nB: NASDAQ only, no mcap filter  ->  {len(dfB)} rows")

# Query C: only NYSE, no market cap filter
qC = ('U(IN(Equity(active,public,primary))/*UNV:Public*/), '
      'IN(TR.ExchangeCode,"NYS"), CURN=USD')
dfC = rd.get_data(universe=Screener(qC),
                  fields=["TR.CommonName", "TR.ExchangeCode",
                          "TR.CompanyMarketCap"])
print(f"C: NYSE only, no mcap filter    ->  {len(dfC)} rows")

# Query D: NASDAQ with a lower mcap threshold
qD = ('U(IN(Equity(active,public,primary))/*UNV:Public*/), '
      'IN(TR.ExchangeCode,"NSQ"), TR.CompanyMarketCap(Scale=6)>=10, CURN=USD')
dfD = rd.get_data(universe=Screener(qD),
                  fields=["TR.CommonName", "TR.ExchangeCode",
                          "TR.CompanyMarketCap"])
print(f"\nD: NASDAQ only, mcap>=10M       ->  {len(dfD)} rows")

rd.close_session()