"""End-of-day backtesting utilities for indicator-based strategies."""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Sequence, Tuple

import pandas as pd

LOGGER = logging.getLogger(__name__)

_STRATEGIES = {"sma200_trend", "rsi_rebound", "macd_cross"}


def _coerce_float(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result):
        return None
    return result


def _series_values(series: object, length: int) -> List[object]:
    if series is None:
        return [None] * length
    if hasattr(series, "tolist"):
        try:
            return list(series.tolist())  # type: ignore[no-any-return]
        except TypeError:  # pragma: no cover - defensive
            pass
    data = getattr(series, "_data", None)
    if data is not None:
        return list(data)
    values: List[object] = []
    for position in range(length):
        try:
            accessor = getattr(series, "iloc")
            values.append(accessor[position])
        except Exception:  # pragma: no cover - fallback path
            values.append(None)
    return values


def _column_values(frame: pd.DataFrame, column: str, length: int) -> List[object]:
    columns = getattr(frame, "columns", [])
    if column in columns:
        return _series_values(frame[column], length)
    return [None] * length


def generate_signals(frame: pd.DataFrame, strategy: str) -> pd.DataFrame:
    """Return boolean regime signals for the provided ``strategy``."""

    if strategy not in _STRATEGIES:
        raise ValueError(f"Unsupported strategy {strategy!r}")
    if getattr(frame, "empty", True):
        return pd.DataFrame({"signal": []}, index=getattr(frame, "index", None))

    length = len(frame)
    close_values = _column_values(frame, "Close", length)
    ema21_values = _column_values(frame, "EMA21", length)
    sma200_values = _column_values(frame, "SMA200", length)
    rsi_values = _column_values(frame, "RSI14", length)
    macd_values = _column_values(frame, "MACD", length)
    macd_signal_values = _column_values(frame, "MACD_signal", length)
    macd_hist_values = _column_values(frame, "MACD_hist", length)

    signals: List[bool] = []
    in_position = False

    for idx in range(length):
        close_val = _coerce_float(close_values[idx])
        ema21_val = _coerce_float(ema21_values[idx])
        sma200_val = _coerce_float(sma200_values[idx])
        rsi_val = _coerce_float(rsi_values[idx])
        macd_val = _coerce_float(macd_values[idx])
        macd_signal_val = _coerce_float(macd_signal_values[idx])
        macd_hist_val = _coerce_float(macd_hist_values[idx])

        signal = False
        if strategy == "sma200_trend":
            if (
                close_val is not None
                and sma200_val is not None
                and ema21_val is not None
                and close_val > sma200_val
                and close_val > ema21_val
            ):
                signal = True
        elif strategy == "rsi_rebound":
            if not in_position and rsi_val is not None and rsi_val < 30.0:
                in_position = True
            elif in_position and rsi_val is not None and rsi_val > 50.0:
                in_position = False
            signal = in_position
        elif strategy == "macd_cross":
            if (
                macd_val is not None
                and macd_signal_val is not None
                and macd_hist_val is not None
                and macd_val > macd_signal_val
                and macd_hist_val > 0.0
            ):
                signal = True

        signals.append(bool(signal))

    return pd.DataFrame({"signal": signals}, index=frame.index)


@dataclass
class _Position:
    qty: float
    entry_price: float
    entry_date: datetime
    entry_cost: float
    stop_price: float | None
    take_price: float | None


def _trend_strength(row: Dict[str, object]) -> float:
    close = _coerce_float(row.get("Close"))
    sma200 = _coerce_float(row.get("SMA200"))
    if close is None or sma200 is None or sma200 <= 0:
        return float("-inf")
    return close / sma200


def _prepare_price_lookup(frame: pd.DataFrame, signals: pd.DataFrame) -> Dict[datetime, Dict[str, object]]:
    mapping: Dict[datetime, Dict[str, object]] = {}
    index = list(frame.index)
    columns = list(frame.columns)
    signal_values: List[object] = []
    if len(signals) and "signal" in getattr(signals, "columns", []):
        signal_values = _series_values(signals["signal"], len(signals))

    column_data = {column: _series_values(frame[column], len(frame)) for column in columns}

    for position, timestamp in enumerate(index):
        row: Dict[str, object] = {}
        for column in columns:
            values = column_data[column]
            row[column] = values[position]
        if position < len(signal_values):
            row["signal"] = bool(signal_values[position])
        mapping[timestamp] = row
    return mapping


def _sorted_union(indices: Sequence[Sequence[datetime]]) -> List[datetime]:
    unique: Dict[datetime, None] = {}
    for sequence in indices:
        for value in sequence:
            unique[value] = None
    return sorted(unique.keys())


def _compute_equity_metrics(equity_values: List[float], dates: List[datetime]) -> Dict[str, float]:
    if not equity_values:
        return {
            "CAGR": 0.0,
            "Vol": 0.0,
            "Sharpe": 0.0,
            "MaxDD": 0.0,
            "Calmar": 0.0,
            "ExposurePct": 0.0,
        }

    start = equity_values[0]
    end = equity_values[-1]
    periods = max(1, len(equity_values) - 1)
    years = periods / 252.0
    cagr = (end / start) ** (1.0 / years) - 1.0 if start > 0 and years > 0 else 0.0

    returns: List[float] = []
    for idx in range(1, len(equity_values)):
        previous = equity_values[idx - 1]
        current = equity_values[idx]
        if previous > 0:
            returns.append(current / previous - 1.0)
        else:
            returns.append(0.0)

    if returns:
        mean_ret = sum(returns) / len(returns)
        variance = sum((value - mean_ret) ** 2 for value in returns) / len(returns)
        std_dev = math.sqrt(variance)
    else:
        mean_ret = 0.0
        std_dev = 0.0

    vol = std_dev * math.sqrt(252.0)
    sharpe = (mean_ret / std_dev) * math.sqrt(252.0) if std_dev > 0 else 0.0

    max_drawdown = 0.0
    peak = equity_values[0]
    for value in equity_values:
        if value > peak:
            peak = value
        drawdown = value / peak - 1.0
        if drawdown < max_drawdown:
            max_drawdown = drawdown

    calmar = 0.0
    if max_drawdown < 0:
        calmar = cagr / abs(max_drawdown) if abs(max_drawdown) > 0 else 0.0
        if calmar > 1_000_000:
            calmar = 1_000_000.0

    return {
        "CAGR": cagr,
        "Vol": vol,
        "Sharpe": sharpe,
        "MaxDD": max_drawdown,
        "Calmar": calmar,
    }


def _compute_trade_stats(trades: List[Dict[str, object]]) -> Dict[str, float]:
    if not trades:
        return {
            "HitRate": 0.0,
            "AvgWin": 0.0,
            "AvgLoss": 0.0,
            "Payoff": 0.0,
        }

    wins: List[float] = []
    losses: List[float] = []
    for trade in trades:
        value = _coerce_float(trade.get("ret"))
        if value is None:
            continue
        if value > 0:
            wins.append(value)
        elif value < 0:
            losses.append(value)

    hit_rate = len(wins) / len(trades) if trades else 0.0
    avg_win = sum(wins) / len(wins) if wins else 0.0
    avg_loss = sum(losses) / len(losses) if losses else 0.0
    payoff = abs(avg_win / avg_loss) if avg_win and avg_loss else 0.0

    return {
        "HitRate": hit_rate,
        "AvgWin": avg_win,
        "AvgLoss": avg_loss,
        "Payoff": payoff,
    }


def run_backtest(
    results: Dict[str, Dict[str, object]],
    *,
    strategy: str = "sma200_trend",
    capital: float = 10_000.0,
    weight_scheme: str = "equal",
    max_positions: int | None = None,
    cost_bps: float = 10.0,
    slippage_bps: float = 5.0,
    stop_loss_pct: float | None = None,
    take_profit_pct: float | None = None,
) -> Dict[str, object]:
    """Run an end-of-day backtest on previously analysed ``results``."""

    if strategy not in _STRATEGIES:
        raise ValueError(f"Unsupported strategy {strategy!r}")
    if weight_scheme != "equal":
        raise ValueError("Only the 'equal' weight scheme is implemented")

    tickers = [ticker for ticker in results if isinstance(results.get(ticker), dict)]
    if not tickers:
        raise ValueError("No analysable tickers provided")

    signal_maps: Dict[str, Dict[datetime, Dict[str, object]]] = {}
    indices: List[Sequence[datetime]] = []
    for ticker in tickers:
        payload = results[ticker]
        prices = payload.get("prices") if isinstance(payload, dict) else None
        if prices is None or getattr(prices, "empty", True):
            LOGGER.warning("Ticker %s possède un historique vide, ignoré", ticker)
            continue
        signals = generate_signals(prices, strategy=strategy)
        lookup = _prepare_price_lookup(prices, signals)
        signal_maps[ticker] = lookup
        indices.append(list(lookup.keys()))

    ordered_dates = _sorted_union(indices)
    if not ordered_dates:
        raise ValueError("No price data available for backtest")

    cost_rate = cost_bps / 10_000.0
    slippage_rate = slippage_bps / 10_000.0

    positions: Dict[str, _Position] = {}
    cash = float(capital)
    equity_values: List[float] = []
    equity_dates: List[datetime] = []
    trades: List[Dict[str, object]] = []
    exposure_rows: List[Dict[str, float]] = []
    force_exit: Dict[str, bool] = {}

    for idx, current_date in enumerate(ordered_dates):
        # Execute decisions at the open using previous day's signals
        if idx > 0:
            previous_date = ordered_dates[idx - 1]
            desired: List[Tuple[str, float]] = []
            for ticker, lookup in signal_maps.items():
                previous_row = lookup.get(previous_date)
                if previous_row and previous_row.get("signal"):
                    desired.append((ticker, _trend_strength(previous_row)))

            if max_positions is not None and len(desired) > max_positions:
                desired.sort(key=lambda item: item[1], reverse=True)
                desired = desired[:max_positions]

            desired_set = {ticker for ticker, _ in desired}
            for ticker in list(desired_set):
                if force_exit.get(ticker):
                    desired_set.remove(ticker)
            force_exit.clear()

            todays_rows: Dict[str, Dict[str, object]] = {}
            for ticker in desired_set | set(positions.keys()):
                lookup = signal_maps.get(ticker, {})
                row_today = lookup.get(current_date)
                if row_today is None or _coerce_float(row_today.get("Open")) is None:
                    LOGGER.warning("Donnée d'ouverture manquante pour %s le %s", ticker, current_date)
                    if ticker in desired_set:
                        desired_set.remove(ticker)
                    continue
                todays_rows[ticker] = row_today

            current_positions = {ticker for ticker, pos in positions.items() if pos.qty > 0}
            to_exit = current_positions - desired_set
            to_enter = desired_set - current_positions

            for ticker in list(to_exit):
                row_today = todays_rows.get(ticker)
                if row_today is None:
                    continue
                open_price = _coerce_float(row_today.get("Open"))
                if open_price is None:
                    continue
                position = positions.get(ticker)
                if position is None:
                    continue
                exec_price = open_price * (1.0 - slippage_rate)
                gross_exit = exec_price * position.qty
                exit_cost = gross_exit * cost_rate
                cash += gross_exit - exit_cost
                gross_entry = position.entry_price * position.qty
                pnl = gross_exit - gross_entry - position.entry_cost - exit_cost
                ret = pnl / gross_entry if gross_entry > 0 else 0.0
                trades.append(
                    {
                        "ticker": ticker,
                        "entry_date": position.entry_date,
                        "entry_px": position.entry_price,
                        "exit_date": current_date,
                        "exit_px": exec_price,
                        "pnl": pnl,
                        "ret": ret,
                        "holding_days": (current_date - position.entry_date).days,
                        "costs": position.entry_cost + exit_cost,
                    }
                )
                del positions[ticker]

            entries = list(to_enter)
            for position_index, ticker in enumerate(entries):
                row_today = todays_rows.get(ticker)
                if row_today is None:
                    continue
                open_price = _coerce_float(row_today.get("Open"))
                if open_price is None or open_price <= 0:
                    LOGGER.warning("Impossible d'entrer sur %s faute de prix valide", ticker)
                    continue
                remaining = len(entries) - position_index
                if cash <= 0:
                    break
                allocation = cash / remaining
                exec_price = open_price * (1.0 + slippage_rate)
                if exec_price <= 0:
                    continue
                notional = allocation
                total_cost_factor = 1.0 + cost_rate
                if allocation * total_cost_factor > cash:
                    notional = cash / total_cost_factor
                qty = notional / exec_price if exec_price > 0 else 0.0
                if qty <= 0:
                    continue
                entry_cost = notional * cost_rate
                cash -= notional + entry_cost
                stop_level = exec_price * (1.0 - stop_loss_pct) if stop_loss_pct is not None else None
                take_level = exec_price * (1.0 + take_profit_pct) if take_profit_pct is not None else None
                positions[ticker] = _Position(
                    qty=qty,
                    entry_price=exec_price,
                    entry_date=current_date,
                    entry_cost=entry_cost,
                    stop_price=stop_level,
                    take_price=take_level,
                )

        # Mark-to-market at close
        position_values: Dict[str, float] = {}
        total_position_value = 0.0
        for ticker, position in positions.items():
            row_today = signal_maps.get(ticker, {}).get(current_date)
            if row_today is None:
                continue
            close_price = _coerce_float(row_today.get("Close"))
            open_price = _coerce_float(row_today.get("Open"))
            price = close_price if close_price is not None else open_price
            if price is None:
                LOGGER.warning("Prix de valorisation manquant pour %s le %s", ticker, current_date)
                continue
            value = position.qty * price
            position_values[ticker] = value
            total_position_value += value

        equity = cash + total_position_value
        equity_dates.append(current_date)
        equity_values.append(equity)

        exposure_row: Dict[str, float] = {}
        for ticker in tickers:
            value = position_values.get(ticker, 0.0)
            exposure_row[ticker] = value / equity if equity > 0 else 0.0
        exposure_rows.append(exposure_row)

        if idx < len(ordered_dates) - 1:
            next_forced: Dict[str, bool] = {}
            for ticker, position in positions.items():
                row_today = signal_maps.get(ticker, {}).get(current_date)
                if row_today is None:
                    continue
                low_price = _coerce_float(row_today.get("Low"))
                high_price = _coerce_float(row_today.get("High"))
                if position.stop_price is not None and low_price is not None and low_price <= position.stop_price:
                    next_forced[ticker] = True
                if position.take_price is not None and high_price is not None and high_price >= position.take_price:
                    next_forced[ticker] = True
            force_exit = next_forced

    # Close remaining positions at the last close
    if positions:
        final_date = ordered_dates[-1]
        row_final: Dict[str, Dict[str, object]] = {ticker: signal_maps.get(ticker, {}).get(final_date, {}) for ticker in list(positions.keys())}
        for ticker, position in list(positions.items()):
            row_today = row_final.get(ticker, {})
            price = _coerce_float(row_today.get("Close"))
            if price is None:
                price = _coerce_float(row_today.get("Open"))
            if price is None:
                LOGGER.warning("Impossible de clôturer %s faute de prix final", ticker)
                continue
            exec_price = price * (1.0 - slippage_rate)
            gross_exit = exec_price * position.qty
            exit_cost = gross_exit * cost_rate
            cash += gross_exit - exit_cost
            gross_entry = position.entry_price * position.qty
            pnl = gross_exit - gross_entry - position.entry_cost - exit_cost
            ret = pnl / gross_entry if gross_entry > 0 else 0.0
            trades.append(
                {
                    "ticker": ticker,
                    "entry_date": position.entry_date,
                    "entry_px": position.entry_price,
                    "exit_date": final_date,
                    "exit_px": exec_price,
                    "pnl": pnl,
                    "ret": ret,
                    "holding_days": (final_date - position.entry_date).days,
                    "costs": position.entry_cost + exit_cost,
                }
            )
            del positions[ticker]

    equity_metrics = _compute_equity_metrics(equity_values, equity_dates)
    trade_metrics = _compute_trade_stats(trades)

    exposure_days = sum(1 for row in exposure_rows if sum(abs(value) for value in row.values()) > 1e-9)
    exposure_pct = (exposure_days / len(exposure_rows)) * 100.0 if exposure_rows else 0.0
    equity_metrics["ExposurePct"] = exposure_pct

    equity_series = pd.Series(equity_values, index=equity_dates) if hasattr(pd, "Series") else equity_values
    positions_frame = pd.DataFrame(
        {ticker: [row.get(ticker, 0.0) for row in exposure_rows] for ticker in tickers},
        index=equity_dates,
    )
    trade_columns = [
        "ticker",
        "entry_date",
        "entry_px",
        "exit_date",
        "exit_px",
        "pnl",
        "ret",
        "holding_days",
        "costs",
    ]
    trades_frame = pd.DataFrame({column: [trade.get(column) for trade in trades] for column in trade_columns})

    metrics = {**equity_metrics, **trade_metrics}

    return {
        "equity": equity_series,
        "trades": trades_frame,
        "positions": positions_frame,
        "metrics": metrics,
        "params": {
            "strategy": strategy,
            "capital": capital,
            "weight_scheme": weight_scheme,
            "max_positions": max_positions,
            "cost_bps": cost_bps,
            "slippage_bps": slippage_bps,
            "stop_loss_pct": stop_loss_pct,
            "take_profit_pct": take_profit_pct,
        },
    }


__all__ = ["generate_signals", "run_backtest"]
