# indice_nexus

This repository contains a small utility script (`stock_indicators.py`) that
retrieves historical market data via [yfinance](https://github.com/ranaroussi/yfinance)
and enriches it with both moving averages and fundamental ratios for several
European tickers (`ASML.AS`, `TTE.PA`, `MC.PA`).

## Prerequisites

The script depends on:

- [Python 3.9+](https://www.python.org/downloads/)
- [pandas](https://pandas.pydata.org/)
- [yfinance](https://github.com/ranaroussi/yfinance)

Because these libraries are not bundled with the repository, install them in a
virtual environment before running the script:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use `.venv\\Scripts\\activate`
pip install --upgrade pip
pip install pandas yfinance
```

> **Note:** In some sandboxed environments (including this review workspace),
> outbound network access is restricted, preventing `pip` from downloading the
> dependencies. In such a case, install the packages locally on your machine
> and run the script there.

## Running the script

Once the dependencies are installed, you can execute the script directly to
fetch one year of daily data for each ticker and compute the requested moving
averages:

```bash
python stock_indicators.py
```

The script prints the tail of the enriched DataFrame for every ticker along
with the indicator columns that were computed as well as the fundamental
ratios derived from the most recent financial statements. If you want to
target a different set of tickers or adjust the moving-average windows, edit
the `DEFAULT_TICKERS` or `DEFAULT_CONFIG` constants inside
`stock_indicators.py`.

## Available indicators

### Moving averages

- Exponential moving averages: 7, 9, 20 and 21 sessions.
- Simple moving averages: 20, 50, 100 and 200 sessions.

### Fundamental ratios

All ratios are computed with the latest statements exposed by Yahoo Finance.

- **Bénéfice par action (BPA / EPS)**: \( \frac{\text{Résultat net}}{\text{Nombre d'actions}} \)
- **Price to Earnings (P/E)**: \( \frac{\text{Cours de l'action}}{\text{BPA}} \)
- **Marge nette**: \( \frac{\text{Résultat net}}{\text{Chiffre d'affaires}} \times 100 \)
- **Ratio d'endettement (Debt-to-Equity)**: \( \frac{\text{Dettes totales}}{\text{Capitaux propres}} \)
- **Rendement du dividende**: \( \frac{\text{Dividende par action}}{\text{Cours de l'action}} \times 100 \)
