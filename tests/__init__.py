from pathlib import Path
import sys

from . import fastapi_stub, pandas_stub, streamlit_stub, yfinance_stub

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if SRC.exists():
    sys.path.insert(0, str(SRC))

sys.modules.setdefault("pandas", pandas_stub)
sys.modules.setdefault("yfinance", yfinance_stub)
sys.modules.setdefault("streamlit", streamlit_stub)
sys.modules.setdefault("fastapi", fastapi_stub)
sys.modules.setdefault("fastapi.middleware", fastapi_stub.middleware)
sys.modules.setdefault("fastapi.middleware.cors", fastapi_stub.middleware.cors)
sys.modules.setdefault("fastapi.testclient", fastapi_stub.testclient)
import sys

from . import pandas_stub, yfinance_stub

sys.modules.setdefault("pandas", pandas_stub)
sys.modules.setdefault("yfinance", yfinance_stub)
