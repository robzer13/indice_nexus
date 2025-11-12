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
- `--regime` : affiche le régime macro détecté (Stress, Inflation, Recovery, Expansion) et les pondérations Nexus appliquées.
- `--nexus-report` : génère en plus du rapport classique un bloc stratégique Nexus (résumé macro, pondérations adaptatives, top valeurs, recommandation).

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

`stock_analysis.regimes.evaluate_regime` consolide VIX, inflation (CPI YoY), taux 2Y/10Y et spread crédit (cache `.cache/macro`) puis classe chaque date en quatre états : `Stress`, `Inflation`, `Recovery`, `Expansion`. `stock_analysis.weighting.compute_weights` fournit automatiquement les pondérations trend/momentum/fundamental/behavioral adaptées à chaque régime (somme = 1). `compute_score_bundle` renvoie un bundle enrichi (`score`, sous-scores, `weights`, `regime`, notes de données manquantes) utilisé par la CLI, les rapports et la persistance.

## OroTitan / Nexus-grade Intelligence

- `--regime` affiche le régime détecté, les indicateurs macro utilisés et la pondération Nexus appliquée.
- `--nexus-report` ajoute au rapport classique un bloc stratégique (résumé macro, pondérations, Top N, analyse comportementale, recommandation).
- `stock_analysis.report_nexus.generate_nexus_report` produit Markdown/HTML (`Nexus_<date>.md|html`) dans `out/reports/` par défaut.
- `scripts/nexus_daily.py` (et `scripts/nexus_daily.ps1`) exécutent la détection macro, l'analyse pondérée, la sauvegarde et la génération du rapport Nexus à 07:15 CET via le scheduler ou une tâche Windows.
- Le manifeste JSON inclut le régime détecté, ses pondérations et les chemins vers les rapports Nexus.

## V5 OroTitan AI (Cognitive Engine)

La couche "OroTitan AI" fournit un moteur cognitif complet : encodage de signaux multi-régimes, décisions explicables, feedback et reporting.

### CLI

```powershell
python -m stock_analysis `
  --orotitan-ai `
  --ai-task decide `
  --tickers "ASML MC.PA" `
  --risk-budget 0.15 `
  --temperature 0.1 `
  --json
```

- `--ai-task` : `embed`, `decide`, `feedback`, `report` ou `full`.
- `--regime-weights` ou `--regime-file` : pondérations personnalisées (JSON).
- `--feedback-file` : liste d'évènements (hit/miss/neutral) pour ajuster les poids, `--save-state` persiste dans `out/state/orotitan_ai.json`.
- `--ai-report` + `--report-out` génèrent un rapport Markdown/HTML (`render_orotitan_report`).
- `--json` fournit une sortie machine-readable avec les décisions, embeddings, KPI et chemins de rapport.

### API FastAPI

```bash
uvicorn stock_analysis.api.app:app --reload --port 8000
```

Endpoints :

- `POST /ai/embed` → vecteurs OroTitan par ticker.
- `POST /ai/decide` → décisions (BUY/SELL/HOLD), scores, facteurs et confiance.
- `POST /ai/feedback` → applique un feedback bandit-like et renvoie les notes associées.
- `POST /ai/report` → retourne le Markdown du rapport (et le sauvegarde si `SAVE_REPORTS=1`).

### Streamlit

```bash
python -m stock_analysis --dashboard
```

Une nouvelle page "OroTitan AI" (fichier `streamlit_pages/10_OroTitan_AI.py`) permet :

- saisie multi-tickers, régime manuel et pondérations JSON,
- sliders pour `risk_budget` et `temperature`,
- génération des décisions (tableau + export JSON),
- génération d'un rapport Markdown téléchargeable,
- application d'un feedback JSON.

### Automatisation

- `scripts/orotitan_daily.py` et `scripts/orotitan_daily.ps1` orchestrent une exécution quotidienne (`--orotitan-ai --ai-report --save-state`) à 07:20 CET.
- L'extra `orotitan` (`pip install .[orotitan]`) ajoute l'encodage de notes via `sentence-transformers` lorsque disponible.

## V6 OroTitan Behavioral Intelligence & Self-Coaching

La couche comportementale complète OroTitan avec la détection des biais humains/systémiques et des recommandations d'auto-coaching.

- **Score comportemental 0–100** : calculé via `orotitan_ai.behavior.indicators` + `behavior.rules.evaluate_biases`.
- **Ajustement de confiance** : `behavior.scoring.aggregate_scores` applique ±0.25 max sur la confiance finale (`Decision.behavior` attaché à chaque sortie).
- **Coach Corner** : `behavior.coach.build_recommendations` fournit 3–5 actions concrètes (48h) selon les biais dominants.
- **Persistance** : JSONL par défaut (`out/behavior/behavior_records.jsonl`), option `sqlite` (`--behavior-persist sqlite`) ou `none`.
- **Rapports** : sections "Behavioral Insights" + "Self-Coaching Actions" automatiquement ajoutées au Markdown/HTML OroTitan.
- **Interfaces** :
  - CLI → `python -m stock_analysis --orotitan-ai --behavior --behavior-json --tickers "MC.PA TTE.PA" --json`
  - API → `POST /ai/behavior/analyze` (payload `{"tickers":["MC.PA"],"context":{"indicator_overrides":{"loss_hold_bias":0.6}}}`)
  - Streamlit → page `11_OroTitan_Behavior.py` (sliders seuil/influence, table des biais, export Markdown).

Le score et l'ajustement sont visibles dans les sorties JSON (`confidence_adjustment`). Les rapports Markdown incluent désormais la table des biais et les actions d'auto-coaching.

## Persistance & rapports

`stock_analysis.io.save_analysis` écrit :

- Prix normalisés (`{base}_{ticker}_prices.parquet|csv`).
- Fondamentaux et qualité (`*_fundamentals.json`, `*_quality.json`).
- Tableau des scores (`{base}_scores.parquet|csv`).
- Sorties ML (`{base}_ML_{ticker}_proba.{ext}`, `{base}_ML_{ticker}_signal.{ext}`) et résumé `{base}_ML_SUMMARY.json`.
- Résultats backtest (`{base}_equity|trades|positions.parquet|csv`), KPI (`*_kpis`), drawdown (`*_drawdown`).
- Rapport Nexus (`reports/Nexus_<date>.md|html`) lorsque `--nexus-report` ou le script dédié sont utilisés.
- Manifeste `{base}_MANIFEST.json` (provenance, régime détecté + pondérations, paramètres, versions, graphiques, métriques ML/backtest, timezone Europe/Paris).

Les scripts `scripts/report_daily.ps1` et `scripts/nexus_daily.py` automatisent respectivement la commande complète et le rapport Nexus seul (fichier `scripts/nexus_daily.ps1` pour Windows/Task Scheduler). Le template Jinja `templates/report.md.j2` sert de base pour une mise en page personnalisée (Executive summary, score table, régime macro, visuels).

## API Python

```python
from stock_analysis import (
    analyze_tickers, download_many, fetch_price_history, quality_report,
    compute_moving_averages, compute_rsi, compute_macd,
    fetch_fundamentals, add_ta_features, make_label_future_ret,
    compute_score_bundle, compute_volatility,
    evaluate_regime, infer_regime_series, compute_weights,
    generate_nexus_report, run_backtest, summarize_backtest, save_analysis,
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
