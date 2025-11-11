# indice_nexus

Suite d'outils modulaires pour analyser des actions européennes à partir de Yahoo Finance : téléchargement multi-threadé avec cache, indicateurs techniques, ratios fondamentaux, scoring pondéré, backtesting EOD, branche machine learning, API FastAPI, dashboard Streamlit et reporting Markdown/HTML avec graphiques.

## Installation

```bash
python -m venv .venv
. .venv/bin/activate  # Sous Windows : .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
```

Les dépendances principales (pandas, yfinance, pyarrow, matplotlib, APScheduler, FastAPI, Streamlit, scikit-learn, xgboost) sont déclarées dans `pyproject.toml` et `requirements.txt`.

## Exécution rapide

```bash
stock-analysis \
  --tickers "ASML.AS,TTE.PA,MC.PA" \
  --period 2y --interval 1d \
  --score --report --save \
  --out-dir out --format parquet \
  --charts-dir charts \
  --bt --strategy sma200_trend \
  --ml-eval
```

Le CLI télécharge les prix (index `Europe/Paris`), calcule SMA/EMA 7/9/20/21/50/100/200, RSI14, MACD 12/26/9, volatilité 20 jours, ratios fondamentaux (EPS, PER, marge nette, dette/capitaux propres, rendement dividende), produit un score pondéré, déclenche un backtest EOD optionnel et génère un rapport Markdown (et HTML si `--html`).

### Flags utiles

- `--cache-ttl` : durée de vie du cache parquet `.cache/` (en secondes). `--cache-ttl 0` désactive le cache.
- `--workers` : nombre de threads de téléchargement (sinon `STOCK_DL_WORKERS`, défaut 8).
- `--ml-eval` / `--ml-model` (`logreg`, `rf`, `xgb`) / `--ml-threshold` (0.55 par défaut) activent la branche machine learning (features ret_1d, vol_20, gap SMA, RSI14, momentum 21j) avec walk-forward AUC, matrice de confusion et Sharpe simulé.
- `--bt` + `--bt-report` : moteur EOD (signaux `sma200_trend`, `rsi_rebound`, `macd_cross`) avec coûts, slippage, stops/take-profit, métriques (CAGR, vol annualisée, Sharpe, Calmar, max drawdown, hit-rate, payoff, exposition) et top/bottom trades.
- `--report` / `--html` / `--charts-dir` : rapports Markdown/HTML, graphiques prix + MACD/RSI (backend matplotlib).
- `--benchmark` : rebasing contre un indice (ex: `^FCHI`) pour le backtest et les KPI.

Les valeurs par défaut sont surchargeables via `stock_analysis.toml` (section `[defaults]`) ou variables d'environnement `STOCK_ANALYSIS_*` (voir `__main__.py`).

## Cache et performances

Le module `stock_analysis.cache` écrit un parquet par combinaison ticker/période/intervalle dans `.cache/`. La TTL est contrôlée par `--cache-ttl` ou `STOCK_CACHE_TTL` (défaut 3600 s). `STOCK_DL_WORKERS` (ou `--workers`) configure le `ThreadPoolExecutor` utilisé par `stock_analysis.data.download_many`.

## Scheduler quotidien

```bash
python -m stock_analysis.scheduler
```

Planifie une exécution à 07:15 CET (`NEXUS_HOUR`, `NEXUS_MINUTE` configurables) en lançant `python -m stock_analysis --tickers ... --save --report ...`. Exemple Windows Task Scheduler :

```powershell
schtasks /Create /SC DAILY /TN "NexusDaily" /TR "powershell -NoLogo -Command \`$env:NEXUS_TICKERS='ASML.AS,TTE.PA,MC.PA'; python -m stock_analysis --score --report --save --out-dir C:\\data\\out" /ST 07:15
```

Les logs tournent via un `RotatingFileHandler` (`nexus.log`, 5 Mo x3).

## API & Dashboard

### Installer les dépendances optionnelles

```powershell
pip install .[api,dashboard]
```

### Lancer l'API FastAPI

```powershell
python -m stock_analysis --serve-api --api-host 0.0.0.0 --api-port 8000
# ou
uvicorn stock_analysis.api:app --reload
```

Endpoints clés :

- `GET /health` → `{ "status": "ok", "version": "x.y.z" }`
- `GET /prices/{ticker}?period=1y&interval=1d` → séries OHLCV normalisées + indicateurs SMA/EMA/RSI/MACD.
- `GET /features/{ticker}` → tableau enrichi (prix, moyennes mobiles, features ML `ret_1d`, `vol_20`, `sma200_gap`, `rsi_14`, `mom_21`).
- `GET /ml/{ticker}` → évaluation ML à la volée (AUC moyen, Sharpe simulé, matrice de confusion, probabilités & signaux).
- `GET /report/{ticker}` → synthèse complète (prix récent, score, qualité, fondamentaux, bloc ML lorsqu’il est disponible).

### Lancer le dashboard Streamlit

```powershell
python -m stock_analysis --dashboard
# ou
streamlit run src/stock_analysis/streamlit_app.py
```

Fonctionnalités : sélection multi-tickers, graphiques interactifs (prix + EMA, RSI/MACD), blocs scores & fondamentaux, évaluation ML (AUC/Sharpe/confusion), tableau des features/labels et export Markdown/HTML du rapport généré par `report.py`. Le bouton *Rafraîchir le cache* purge `.cache/` pour forcer un nouveau téléchargement via yfinance.

## Machine Learning

`stock_analysis.features` expose `add_ta_features` et `make_label_future_ret`. Le module `stock_analysis.ml_pipeline` fournit la liste `FEATURES`, `build_model`, `time_cv`, `walk_forward_signals`, `sharpe_sim` et `confusion` pour enchaîner la cross-validation temporelle, générer des probabilités puis dériver un signal déterministe. L’option CLI `--ml-eval` affiche AUC (moyenne ± écart-type), Sharpe simulé et matrice de confusion, tout en sauvegardant probabilités, signaux et un résumé JSON.

### Évaluation ML (CLI)

```powershell
stock-analysis `
  --tickers "MC.PA" `
  --period 2y --interval 1d `
  --ml-eval --ml-model xgb --ml-horizon 5 --ml-thr 0.0 `
  --ml-retrain 60 --ml-th-proba 0.55 `
  --report --save --out-dir out --format parquet --charts-dir charts
```

## Régimes macro & scoring

`stock_analysis.regimes.infer_regime` applique des seuils heuristiques (VIX ≥25 → `Stress`, CPI YoY ≥4% → `Inflation`, sinon `Normal`). `compute_score_bundle` accepte des pondérations personnalisées (`weights={'trend': 50, ...}`) et renvoie un score 0..100 avec sous-scores trend/momentum/quality/risk et notes de données manquantes.

## Persistance & rapports

`stock_analysis.io.save_analysis` écrit :

- Prix normalisés (`{base}_{ticker}_prices.parquet|csv`).
- Fondamentaux et qualité (`*_fundamentals.json`, `*_quality.json`).
- Tableau des scores (`{base}_scores.parquet|csv`).
- Sorties ML (`{base}_ML_{ticker}_proba.{ext}`, `{base}_ML_{ticker}_signal.{ext}`) et résumé `{base}_ML_SUMMARY.json`.
- Résultats backtest (`{base}_equity|trades|positions.parquet|csv`), KPI (`*_kpis`), drawdown (`*_drawdown`).
- Manifeste `{base}_MANIFEST.json` (provenance, paramètres, versions, graphiques, métriques ML/backtest, timezone Europe/Paris).

Le script `scripts/report_daily.ps1` lance la commande complète (options backtest et ML facultatives). Le template Jinja `templates/report.md.j2` sert de base pour une mise en page personnalisée (Executive summary, score table, régime macro, visuels).

## API Python

```python
from stock_analysis import (
    analyze_tickers, download_many, fetch_price_history, quality_report,
    compute_moving_averages, compute_rsi, compute_macd,
    fetch_fundamentals, add_ta_features, make_label_future_ret,
    compute_score_bundle, compute_volatility, infer_regime,
    run_backtest, summarize_backtest, save_analysis,
    time_cv, walk_forward_signals, sharpe_sim, confusion,
    render_markdown, render_bt_markdown, plot_ticker,
    plot_equity_with_benchmark, plot_drawdown, plot_exposure_heatmap,
    schedule_daily_run
)
```

Toutes les séries retournées sont timezone-aware (`Europe/Paris`) et respectent les colonnes canoniques `Open/High/Low/Close/Adj Close/Volume`.

## Tests & CI

```bash
python -m unittest discover -s tests -t .
python -m compileall .
```

La CI GitHub Actions (`.github/workflows/ci.yml`) exécute `pytest -q --cov` sur Windows (`3.10` et `3.11`).

## Limitations connues

- Yahoo Finance peut renvoyer des données partielles (les champs manquants sont loggés et exposés sous forme de `None`).
- Les graphiques nécessitent `matplotlib` ; l'export HTML requiert `markdown`.
- XGBoost est optionnel : si non installé, choisissez `--ml-model logreg|rf`.
- Le moteur de backtest ignore les journées sans `Open` (warning explicite).
- Vérifiez la fraîcheur des données via `score['as_of']` avant toute décision.
