"""
diag_crypto_rics_full.py
------------------------
For each coin in our target list, tries every known Refinitiv crypto
suffix and records whichever resolves first.

Output: a Python dict {ticker: resolved_ric} printed to stdout and saved
as crypto_ric_map.json. This map is what pull_crypto.py (v4) will use.
"""
import json
import refinitiv.data as rd

rd.open_session()

COINS = [
    "BTC", "ETH", "XRP", "LTC", "BCH", "XLM", "ADA", "SOL", "DOT",
    "LINK", "DOGE", "TRX", "MATIC", "AVAX", "ATOM", "NEO", "EOS",
    "XTZ", "DASH", "XMR", "ZEC", "ETC", "VET", "ALGO", "FIL",
    "THETA", "AAVE", "UNI", "MKR", "COMP", "SNX", "CRV", "SHIB",
    "GRT", "SAND", "MANA", "AXS", "CHZ", "ENJ", "1INCH", "NEAR",
    "FTM", "HBAR", "ICP", "FLOW", "KSM", "BAT", "ZIL", "WAVES",
    "OMG", "LRC", "ANKR", "STORJ", "CELO", "PAXG", "EURT",
    "USDT", "USDC",
]

# Ordered by likelihood based on what we have already observed.
# Bare composite first; then the specific exchange/composite feeds.
SUFFIXES = [
    "",           # bare: {c}=
    "=CCCL",      # CoinDesk D&I composite
    "=BTSP",      # Bitstamp
    "=LUKK",      # Lukka
    "=CCC",       # CoinDesk (older format)
    "=CRY",       # crypto generic
    "=KRAKEN",
    "=BITFINEX",
    "=GDAX",
    "=COINBASE",
    "=GEMINI",
    "=BINANCE",
]

PROBE_START = "2024-01-02"
PROBE_END   = "2024-01-10"

resolved = {}
unresolved = []

for coin in COINS:
    found = False
    for suf in SUFFIXES:
        ric = f"{coin}=" if suf == "" else f"{coin}{suf}"
        try:
            df = rd.get_history(universe=[ric],
                                fields=["TR.MidPrice"],
                                interval="daily",
                                start=PROBE_START, end=PROBE_END)
            if df is not None and not df.empty:
                resolved[coin] = ric
                print(f"  {coin:<8s} -> {ric}")
                found = True
                break
        except Exception:
            pass
    if not found:
        unresolved.append(coin)
        print(f"  {coin:<8s} -> UNRESOLVED")

print(f"\n=== Summary ===")
print(f"Resolved: {len(resolved)} / {len(COINS)}")
if unresolved:
    print(f"Unresolved: {unresolved}")

with open("crypto_ric_map.json", "w") as f:
    json.dump({"resolved": resolved, "unresolved": unresolved}, f, indent=2)
print(f"\nSaved to crypto_ric_map.json")

rd.close_session()