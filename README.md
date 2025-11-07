# indice_nexus

Mini-lib modulaire pour analyser des actions européennes à partir de Yahoo Finance.
Elle se compose de quatre modules :

- `stock_analysis.data` : téléchargement et normalisation des prix.
- `stock_analysis.indicators` : calculs SMA/EMA en pur Python.
- `stock_analysis.fundamentals` : ratios fondamentaux robustes avec journalisation.
- `stock_analysis.analyzer` : orchestration multi-tickers avec métadonnées unifiées.
- `stock_analysis.io` : persistance des jeux de données, manifeste, scores et rapports.
- `stock_analysis.scoring` : volatilité, agrégation trend/momentum/quality/risk.
- `stock_analysis.plot` : tracés matplotlib des prix + MACD.
- `stock_analysis.plot_bt` : figures equity/drawdown/heatmap pour les backtests.
- `stock_analysis.report` : synthèses Markdown/HTML et commentaires textuels.
- `stock_analysis.backtest` : génération de signaux et moteur de backtest EOD.
- `stock_analysis.bt_report` : drawdowns, tableaux KPI, commentaires et rapports backtest.

La librairie respecte les colonnes OHLCV canoniques, applique un index timezone
`Europe/Paris`, loggue les problèmes de données et expose des fonctions pures
faciles à tester.

## Installation rapide

```bash
python -m venv .venv
source .venv/bin/activate  # Sous Windows : .venv\Scripts\activate
pip install --upgrade pip
pip install pandas yfinance
```

Si l'installation des dépendances échoue dans cet environnement (internet
restreint), exécutez ces commandes localement puis copiez le dépôt.

## Utilisation

Pour lancer l'analyse complète sur `ASML.AS`, `TTE.PA` et `MC.PA` (période 2 ans,
quotidien) :

```bash
python -m stock_analysis
```

Chaque ticker est téléchargé, enrichi avec les SMA/EMA (7/9/20/21 et 20/50/100/200),
un RSI14, un MACD (12/26/9) et accompagné de cinq ratios fondamentaux :

1. **Bénéfice par action (EPS)** = \(\frac{Résultat\ net}{Nombre\ d'actions}\)
2. **P/E (PER)** = \(\frac{Cours}{BPA}\)
3. **Marge nette** = \(\frac{Résultat\ net}{Chiffre\ d'affaires} \times 100\)
4. **Debt-to-Equity** = \(\frac{Dettes\ totales}{Capitaux\ propres}\)
5. **Rendement du dividende** = \(\frac{Dividende\ par\ action}{Cours} \times 100\)

Les logs `INFO/WARNING` indiquent les champs utilisés ou manquants. Les exceptions
réseau sont relayées avec un message expliquant comment valider hors-ligne. Ajoutez
vos propres tickers ou règles en ligne de commande :

```bash
python -m stock_analysis \
  --tickers "ASML.AS,TTE.PA,MC.PA" \
  --period "2y" \
  --interval "1d" \
  --price-column "Adj Close" \
  --gap-threshold 6.5 \
  --log-level INFO \
  --score \
  --top 5 \
  --save \
  --out-dir ./out \
  --format csv \
  --base-name run \
  --report \
  --report-title "Mon rapport" \
  --html \
  --bt \
  --strategy sma200_trend \
  --capital 15000 \
  --cost-bps 15 \
  --slippage-bps 5 \
  --bt-report \
  --bt-title "Analyse Backtest" \
  --benchmark "^FCHI" \
  --charts-bt-dir "bt_charts"
```

Le flag `--save` active la persistance : un fichier par ticker (prix + indicateurs,
fondamentaux, rapport qualité), un tableau de scores agrégé et un manifeste JSON
global sont écrits dans le répertoire choisi. Les prix sont sérialisés en `parquet`
(pyarrow ou fastparquet) ou `csv` avec un index ISO-8601. Les scores sont enregistrés
dans `{base_name}_scores.{format}` avec les colonnes `Ticker`, `Score`, `Trend`,
`Momentum`, `Quality`, `Risk`, `As Of`, `NotesCount`.

Le flag `--report` produit un rapport Markdown (et optionnellement HTML avec
`--html`) incluant une table récapitulative, un commentaire par valeur et, si les
graphiques ne sont pas désactivés (`--no-charts`), un PNG par ticker généré via
matplotlib. Les images sont sauvegardées dans `{out_dir}/{charts_dir}` et référencées
dans le rapport.

Le flag `--bt` active le backtest journalier. Les options `--strategy`
(`sma200_trend`, `rsi_rebound`, `macd_cross`), `--capital`, `--cost-bps`,
`--slippage-bps`, `--max-positions`, `--stop-pct` et `--tp-pct` contrôlent la taille
des positions, les coûts d'exécution et les sorties anticipées. Un résumé des
métriques (CAGR, volatilité annualisée, Sharpe, drawdown, exposition) et les trois
meilleurs/pires trades sont affichés après l'analyse. Lorsque la persistance est
activée, l'équity curve, le journal des trades, les poids par ticker, ainsi qu'un
tableau consolidé de KPI (`{base_name}_kpis.{format}`) et la table des drawdowns
(`{base_name}_drawdown.{format}`) sont écrits ; le manifeste comporte un bloc
`backtest` détaillant métriques, paramètres et chemins éventuels vers les graphiques
backtest.

Les options `--benchmark`, `--bt-report`, `--charts-bt-dir`, `--no-bt-charts` et
`--bt-title` contrôlent la comparaison à un indice de référence et la génération
d'un bloc backtest dans les rapports Markdown/HTML :

- `--benchmark "^FCHI"` télécharge l'indice choisi et rebasse la courbe d'équité.
- `--bt-report` ajoute, en fin de rapport, un tableau de KPI, un résumé textuel et
  éventuellement des visuels (equity vs benchmark, drawdown, heatmap d'exposition).
- `--charts-bt-dir` précise le sous-répertoire où stocker ces visuels (par défaut
  `bt_charts` sous `out_dir`).
- `--no-bt-charts` désactive les PNG backtest tout en conservant le résumé texte.
- `--bt-title` personnalise l'en-tête du bloc backtest.

### Configuration TOML et variables d'environnement

Les valeurs par défaut peuvent être ajustées via un fichier optionnel
`stock_analysis.toml` à la racine du projet :

```toml
[defaults]
tickers = ["ASML.AS", "TTE.PA", "MC.PA"]
period = "2y"
interval = "1d"
price_column = "Close"
gap_threshold = 5.0
out_dir = "out"
format = "parquet"
base_name = "run"
score = true
top = 10
report = true
html = false
charts_dir = "charts"
include_charts = true
report_title = "Stock Analysis Report"
benchmark = "^FCHI"
bt = false
bt_report = false
charts_bt_dir = "bt_charts"
include_bt_charts = true
bt_title = "Backtest Results"
```

Chaque clé peut aussi être définie par une variable d'environnement (priorité
inférieure aux options CLI) : `STOCK_ANALYSIS_TICKERS`, `STOCK_ANALYSIS_PERIOD`,
`STOCK_ANALYSIS_INTERVAL`, `STOCK_ANALYSIS_PRICE_COLUMN`,
`STOCK_ANALYSIS_GAP_THRESHOLD`, `STOCK_ANALYSIS_OUT_DIR`,
`STOCK_ANALYSIS_FORMAT`, `STOCK_ANALYSIS_BASE_NAME`, `STOCK_ANALYSIS_SAVE`,
`STOCK_ANALYSIS_LOG_LEVEL`, `STOCK_ANALYSIS_SCORE`, `STOCK_ANALYSIS_TOP`,
`STOCK_ANALYSIS_REPORT`, `STOCK_ANALYSIS_HTML`, `STOCK_ANALYSIS_CHARTS_DIR`,
`STOCK_ANALYSIS_INCLUDE_CHARTS`, `STOCK_ANALYSIS_REPORT_TITLE`,
`STOCK_ANALYSIS_BT`, `STOCK_ANALYSIS_STRATEGY`, `STOCK_ANALYSIS_CAPITAL`,
`STOCK_ANALYSIS_COST_BPS`, `STOCK_ANALYSIS_SLIPPAGE_BPS`,
`STOCK_ANALYSIS_MAX_POSITIONS`, `STOCK_ANALYSIS_STOP_PCT`,
`STOCK_ANALYSIS_TP_PCT`, `STOCK_ANALYSIS_BENCHMARK`,
`STOCK_ANALYSIS_BT`, `STOCK_ANALYSIS_BT_REPORT`,
`STOCK_ANALYSIS_BT_CHARTS_DIR`, `STOCK_ANALYSIS_BT_CHARTS`,
`STOCK_ANALYSIS_BT_TITLE`.

## API principale

```python
from stock_analysis import (
    analyze_tickers,
    fetch_price_history,
    quality_report,
    compute_moving_averages,
    compute_rsi,
    compute_macd,
    fetch_fundamentals,
    fetch_benchmark,
    save_analysis,
    compute_volatility,
    compute_score_bundle,
    generate_signals,
    run_backtest,
    compute_drawdown,
    summarize_backtest,
    attach_benchmark,
    build_summary_table,
    format_commentary,
    render_markdown,
    render_html,
    render_bt_markdown,
    plot_ticker,
    plot_equity_with_benchmark,
    plot_drawdown,
    plot_exposure_heatmap,
    save_price_figure,
    save_backtest_figure,
)
```

- `fetch_price_history(ticker, period="1y", interval="1d")` renvoie un `DataFrame`
  trié, timezone `Europe/Paris`, colonnes `Open/High/Low/Close/Adj Close/Volume` et
  métadonnées en `attrs`.
- `quality_report(df, price_column="Close", gap_threshold_pct=5.0)` résume la qualité
  (doublons retirés, anomalies OHLC, gaps %, lignes vides, timezone) sans muter
  l'entrée.
- `compute_moving_averages(df, price_column="Close")` ajoute les colonnes `EMA{n}` et
  `SMA{n}` sur une copie du DataFrame.
- `compute_rsi(df, price_column="Close", period=14)` calcule un RSI (méthode Wilder)
  sur une copie.
- `compute_macd(df, price_column="Close", fast=12, slow=26, signal=9)` ajoute
  `MACD`, `MACD_signal` et `MACD_hist`.
- `fetch_fundamentals(ticker)` renvoie un dictionnaire contenant EPS, P/E, marge
  nette (%), dette/capitaux propres, rendement du dividende (%) + provenance.
- `analyze_tickers([...])` combine les deux précédents et renvoie un dictionnaire par
  ticker avec `prices`, `fundamentals`, `quality`, `score` et `meta`.
- `save_analysis(result, out_dir, base_name="analysis", format="parquet")` écrit
  un fichier prix par ticker (`parquet` ou `csv`), les fondamentaux et rapports
  qualité en JSON, un fichier `{base_name}_scores.{format}`, les fichiers de backtest
  (`{base_name}_equity|trades|positions.{format}`) lorsqu'un backtest est fourni, les
  KPI (`{base_name}_kpis.{format}`) et drawdowns (`{base_name}_drawdown.{format}`),
  ainsi qu'un manifeste (`*_MANIFEST.json`) qui trace période, intervalle, colonne de
  prix, versions (Python/pandas/librairie), paramètres et graphiques éventuels.
- `compute_volatility(df, price_column="Close", window=20)` ajoute `RET` et `VOL20`
  (écart-type des rendements) sur une copie.
- `compute_score_bundle(df, fundamentals)` retourne un dictionnaire avec `trend`,
  `momentum`, `quality`, `risk`, le score global (capé à 100) et les notes
  explicatives.
- `generate_signals(df, strategy)` renvoie une colonne booléenne indiquant les
  régimes longs (`sma200_trend`, `rsi_rebound`, `macd_cross`).
- `run_backtest(results, strategy="sma200_trend", ...)` exécute le moteur EOD
  (signaux calculés sur `t`, exécution à l'open `t+1`, coûts/slippage, stops,
  take-profit) et retourne courbe d'équité, trades, positions, métriques et
  paramètres utilisés.
- `fetch_benchmark(symbol="^FCHI", ...)` télécharge un indice de référence prêt à
  être comparé avec les séries analysées.
- `compute_drawdown(equity)` renvoie un DataFrame equity/peak/drawdown (absolu et %).
- `summarize_backtest(backtest)` convertit la sortie du moteur en dictionnaire plat
  (CAGR, volatilité, Sharpe, drawdown, exposition, nombre de trades, bornes de
  période).
- `attach_benchmark(equity, benchmark_df)` aligne et rebase la stratégie et son
  indice (colonne `Strategy` + label du benchmark).
- `render_bt_markdown(backtest, ...)` compose un bloc Markdown avec KPI, graphiques
  et top/bottom trades.
- `build_summary_table(results)` et `format_commentary(...)` facilitent la génération
  d'un rapport synthétique.
- `render_markdown(results, include_charts=True, charts_dir="...")` compose un
  document Markdown (et `render_html` le convertit si la librairie `markdown` est
  disponible). `render_bt_markdown` complète le rapport avec un bloc backtest.
- `plot_ticker`, `plot_equity_with_benchmark`, `plot_drawdown`,
  `plot_exposure_heatmap` fournissent les figures matplotlib correspondantes
  (`save_price_figure`/`save_backtest_figure` les écrivent sur disque).
- `plot_ticker(df)` trace prix/indicateurs sur un `Figure` matplotlib et
  `save_figure(fig, path)` persiste le graphique.

## Tests hors-ligne

Toutes les fonctions critiques sont testées avec des stubs (`yfinance` est mocké).

```bash
python -m unittest discover -s tests -t .
python -m compileall .
```

## Limites connues

- Yahoo Finance peut renvoyer des données incomplètes : les valeurs manquantes sont
  logguées et renvoyées sous forme de `None` (ou `0.0` pour le dividende).
- Les graphiques et rapports nécessitent `matplotlib` et (optionnellement) la
  librairie `markdown` pour la sortie HTML.
- Les scores reposent sur la dernière ligne disponible : vérifiez la fraîcheur des
  données (`score['as_of']`) avant de prendre des décisions.
- Pour une autre colonne de prix (ex. `Adj Close`), passez `price_column` aux
  fonctions d'indicateurs (`compute_moving_averages`, `compute_rsi`, `compute_macd`)
  ou à `analyze_tickers`/`python -m stock_analysis --price-column`.
