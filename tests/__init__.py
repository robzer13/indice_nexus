import sys

from . import pandas_stub, yfinance_stub

sys.modules.setdefault("pandas", pandas_stub)
sys.modules.setdefault("yfinance", yfinance_stub)
