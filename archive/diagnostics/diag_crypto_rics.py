"""
diag_crypto_rics.py
-------------------
Tests alternative RIC conventions for the missing coins.
"""
import refinitiv.data as rd

rd.open_session()

# Coins to test
coins = ["SOL", "ADA", "DOGE", "LINK", "DOT"]

# Alternative suffixes / patterns to try
patterns = [
    "{c}=",
    "{c}=CCC",
    "{c}=CRY",
    "{c}=BTSP",
    "{c}=GDAX",
    "{c}=KRAKEN",
    "{c}=BITFINEX",
    "{c}=BINANCE",
    "{c}=COINBASE",
    "{c}USD=R",
    "{c}USD=",
    "{c}USDT=",
    "{c}.P",
    "{c}=",
]

for coin in coins:
    print(f"\n=== {coin} ===")
    for pat in patterns:
        ric = pat.format(c=coin)
        try:
            df = rd.get_history(universe=[ric],
                                fields=["TR.MidPrice"],
                                interval="daily",
                                start="2024-01-02", end="2024-01-05")
            if df is not None and not df.empty:
                print(f"  OK    {ric:<18s} -> {len(df)} rows")
            else:
                print(f"  empty {ric:<18s}")
        except Exception as e:
            print(f"  fail  {ric:<18s} -> {str(e)[:60]}")

rd.close_session()