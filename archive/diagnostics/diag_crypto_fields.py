"""
diag_crypto_fields.py
---------------------
Finds the correct field names for FX-style crypto RICs.
"""
import refinitiv.data as rd

rd.open_session()

ric = "BTC="
start, end = "2024-01-02", "2024-01-10"

field_sets = [
    ["TR.ClosePrice"],
    ["TR.MidPrice"],
    ["TR.BidPrice", "TR.AskPrice"],
    ["TR.OpenPrice", "TR.HighPrice", "TR.LowPrice", "TR.ClosePrice"],
    ["TR.PriceClose"],
    ["TR.Volume"],
    ["TR.MidPrice", "TR.Volume"],
    ["BID", "ASK", "MID", "OPEN", "HIGH", "LOW", "CLOSE"],
]

for fs in field_sets:
    try:
        df = rd.get_history(universe=[ric], fields=fs, interval="daily",
                            start=start, end=end)
        if df is not None and not df.empty:
            print(f"OK   fields={fs}")
            print(f"     shape={df.shape}  cols={df.columns.tolist()}")
            print(df.head(3))
            print()
        else:
            print(f"EMPTY fields={fs}")
    except Exception as e:
        print(f"FAIL fields={fs}: {str(e)[:100]}")

rd.close_session()