"""Lightweight stub of pandas used for offline unit tests."""
from __future__ import annotations

import csv
import math
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Sequence
from zoneinfo import ZoneInfo


def date_range(start: datetime, periods: int, freq: str = "D", tz: str | None = None):
    if not isinstance(start, datetime):
        raise TypeError("start must be a datetime instance in this stub")
    if freq != "D":
        raise ValueError("Only daily frequency is supported in the stub")
    tzinfo = ZoneInfo(tz) if isinstance(tz, str) else tz
    if tzinfo is not None:
        start = start.replace(tzinfo=tzinfo)
    values = [start + timedelta(days=offset) for offset in range(periods)]
    return DatetimeIndex(values)


def to_datetime(values: Iterable[Any]) -> "DatetimeIndex":
    parsed = []
    all_datetimes = True
    for value in values:
        if isinstance(value, datetime):
            parsed.append(value)
        elif isinstance(value, str):
            try:
                parsed.append(datetime.fromisoformat(value))
            except ValueError:
                parsed.append(value)
                all_datetimes = False
        else:
            parsed.append(value)
            if not isinstance(value, datetime):
                all_datetimes = False
    if all_datetimes:
        return DatetimeIndex(parsed)
    return parsed  # type: ignore[return-value]


def Timestamp(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            raise TypeError(f"Unsupported Timestamp value: {value!r}") from None
    raise TypeError(f"Unsupported Timestamp value: {value!r}")


class DatetimeIndex(List[datetime]):
    @property
    def tz(self):
        for value in self:
            if value is not None:
                return value.tzinfo
        return None

    def notna(self) -> List[bool]:
        return [value is not None for value in self]

    def tz_localize(self, tz: ZoneInfo | str) -> "DatetimeIndex":
        if isinstance(tz, str):
            tz = ZoneInfo(tz)
        localized = []
        for value in self:
            if value is None:
                localized.append(None)
            elif value.tzinfo is not None:
                localized.append(value)
            else:
                localized.append(value.replace(tzinfo=tz))
        return DatetimeIndex(localized)

    def tz_convert(self, tz: ZoneInfo | str) -> "DatetimeIndex":
        if isinstance(tz, str):
            tz = ZoneInfo(tz)
        converted = []
        for value in self:
            if value is None:
                converted.append(None)
            elif value.tzinfo is None:
                raise ValueError("Cannot convert timezone on naive datetime")
            else:
                converted.append(value.astimezone(tz))
        return DatetimeIndex(converted)

    def sort_values(self) -> "DatetimeIndex":
        return DatetimeIndex(sorted([value for value in self if value is not None]))

    @property
    def is_monotonic_increasing(self) -> bool:
        values = [value for value in self if value is not None]
        return all(values[i] <= values[i + 1] for i in range(len(values) - 1))

    def __ge__(self, other: datetime) -> List[bool]:
        return [value is not None and value >= other for value in self]

    def __getitem__(self, key):  # type: ignore[override]
        result = super().__getitem__(key)
        if isinstance(key, slice):
            return DatetimeIndex(result)
        return result


class Series:
    def __init__(self, data: Sequence[Any] | None = None, index: Sequence[Any] | None = None, dtype: str | None = None):
        self._data = list(data or [])
        if isinstance(index, DatetimeIndex):
            self.index: Sequence[Any] = index
        elif index is None:
            self.index = []
        else:
            self.index = list(index)
        self.dtype = dtype
        self.iloc = _SeriesILoc(self)

    def dropna(self) -> "Series":
        filtered_data = []
        filtered_index = []
        for value, label in zip(self._data, self.index):
            if value is None:
                continue
            if isinstance(value, float) and math.isnan(value):
                continue
            filtered_data.append(value)
            filtered_index.append(label)
        return Series(filtered_data, DatetimeIndex(filtered_index), self.dtype)

    def tolist(self) -> List[Any]:
        return list(self._data)

    def values(self) -> List[Any]:
        return list(self._data)

    def sum(self) -> float:
        return float(sum(value for value in self._data if value is not None))

    @property
    def empty(self) -> bool:
        return len(self._data) == 0

    def pct_change(self) -> "Series":
        changes: List[float] = []
        previous = None
        for value in self._data:
            if value is None or (isinstance(value, float) and math.isnan(value)) or previous is None:
                changes.append(math.nan)
            else:
                try:
                    changes.append((value - previous) / previous)
                except ZeroDivisionError:
                    changes.append(math.nan)
            if value is not None and not (isinstance(value, float) and math.isnan(value)):
                previous = value
        return Series(changes, self.index, self.dtype)

    def rolling(self, window: int) -> "_RollingSeries":
        return _RollingSeries(self, window)

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self):
        return iter(self._data)

    def __getitem__(self, key):
        if isinstance(key, slice):
            sliced_index = self.index[key] if isinstance(self.index, DatetimeIndex) else list(self.index)[key]
            return Series(self._data[key], sliced_index, self.dtype)
        if isinstance(key, list):
            filtered_data = [value for value, flag in zip(self._data, key) if flag]
            if isinstance(self.index, DatetimeIndex):
                filtered_index = DatetimeIndex([label for label, flag in zip(self.index, key) if flag])
            else:
                filtered_index = [label for label, flag in zip(self.index, key) if flag]
            return Series(filtered_data, filtered_index, self.dtype)
        raise KeyError("Series only supports slicing or boolean masks in this stub")

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"Series({self._data})"


class _ILocAccessor:
    def __init__(self, frame: "DataFrame") -> None:
        self._frame = frame

    def __getitem__(self, key):
        if isinstance(key, slice):
            new_data = {col: values[key] for col, values in self._frame._data.items()}
            new_index = self._frame.index[key]
            return DataFrame(new_data, index=new_index)
        position = key if key >= 0 else len(self._frame) + key
        row = {col: values[position] for col, values in self._frame._data.items()}
        return Series(list(row.values()), index=list(row.keys()))


class _SeriesILoc:
    def __init__(self, series: Series) -> None:
        self._series = series

    def __getitem__(self, key: int) -> Any:
        position = key if key >= 0 else len(self._series._data) + key
        return self._series._data[position]


class _LocAccessor:
    def __init__(self, frame: DataFrame) -> None:
        self._frame = frame

    def __getitem__(self, label: Any) -> Series:
        try:
            position = self._frame._index.index(label)
        except ValueError as exc:
            raise KeyError(label) from exc
        row_values = [self._frame._data[col][position] for col in self._frame._columns]
        return Series(row_values, index=list(self._frame._columns))


class DataFrame:
    def __init__(self, data: Dict[str, Sequence[Any]] | None = None, index: Sequence[Any] | None = None, columns: Sequence[str] | None = None):
        data = data or {}
        if columns is not None and not data:
            data = {col: [] for col in columns}
        self._columns = list(data.keys())
        length = len(next(iter(data.values()))) if data else (len(index) if index is not None else 0)
        if index is None:
            index_values = list(range(length))
        else:
            index_values = list(index)
        converted_index = index_values if isinstance(index_values, DatetimeIndex) else to_datetime(index_values)
        if isinstance(converted_index, DatetimeIndex):
            self._index = converted_index
        else:
            self._index = converted_index
        self._data: Dict[str, List[Any]] = {col: list(values) for col, values in data.items()}
        for col in self._columns:
            if len(self._data[col]) < length:
                self._data[col].extend([None] * (length - len(self._data[col])))
        self.attrs: Dict[str, Any] = {}
        self.iloc = _ILocAccessor(self)
        self.loc = _LocAccessor(self)

    def copy(self, deep: bool = True) -> "DataFrame":
        data_copy = {col: list(values) for col, values in self._data.items()}
        index_copy = DatetimeIndex(list(self._index)) if isinstance(self._index, DatetimeIndex) else list(self._index)
        return DataFrame(data_copy, index=index_copy, columns=self._columns)

    @property
    def index(self) -> DatetimeIndex:
        return self._index

    @index.setter
    def index(self, value: DatetimeIndex) -> None:
        self._index = value

    @property
    def columns(self) -> List[str]:
        return list(self._columns)

    @property
    def empty(self) -> bool:
        return len(self) == 0

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, key):
        if isinstance(key, str):
            return Series(self._data.get(key, []), self._index)
        if isinstance(key, list):
            filtered_data = {col: [value for value, flag in zip(values, key) if flag] for col, values in self._data.items()}
            filtered_index = [label for label, flag in zip(self._index, key) if flag]
            return DataFrame(filtered_data, index=filtered_index, columns=self._columns)
        raise KeyError("Unsupported key type for DataFrame in stub")

    def __setitem__(self, key: str, values: Sequence[Any]) -> None:
        self._data[key] = list(values)
        if key not in self._columns:
            self._columns.append(key)

    def sort_index(self, inplace: bool = True) -> None:
        paired = sorted(zip(self._index, range(len(self._index))), key=lambda item: (item[0], item[1]))
        sorted_index = [value for value, _ in paired]
        sorted_positions = [position for _, position in paired]
        sorted_data = {col: [self._data[col][pos] for pos in sorted_positions] for col in self._columns}
        self._index = DatetimeIndex(sorted_index)
        self._data = sorted_data

    def dropna(self, axis: int = 0, how: str = "any", subset: Sequence[str] | None = None, inplace: bool = False):
        subset = list(subset or self._columns)
        keep_mask = []
        for row_values in zip(*(self._data[col] for col in subset)):
            if how == "all":
                keep = not all(value is None or (isinstance(value, float) and math.isnan(value)) for value in row_values)
            else:
                keep = any(value is not None and not (isinstance(value, float) and math.isnan(value)) for value in row_values)
            keep_mask.append(keep)
        filtered = self[keep_mask]
        if inplace:
            self._data = filtered._data
            self._index = filtered._index
            self._columns = filtered._columns
            return None
        return filtered

    def drop(self, columns: Sequence[str] | None = None, errors: str = "raise") -> "DataFrame":
        columns = list(columns or [])
        missing = [column for column in columns if column not in self._columns]
        if missing and errors != "ignore":
            raise KeyError(f"Columns not found: {missing}")
        retained_columns = [column for column in self._columns if column not in columns]
        new_data = {column: list(self._data[column]) for column in retained_columns}
        return DataFrame(new_data, index=self._index, columns=retained_columns)

    def to_csv(self, path, index: bool = True, date_format: str | None = None) -> None:
        close_handle = False
        if hasattr(path, "write"):
            handle = path
        else:
            handle = open(path, "w", newline="", encoding="utf-8")
            close_handle = True

        writer = csv.writer(handle)
        header = []
        if index:
            header.append("index")
        header.extend(self._columns)
        writer.writerow(header)

        for position in range(len(self)):
            row = []
            if index:
                label = self._index[position]
                if isinstance(label, datetime) and date_format:
                    row.append(label.strftime(date_format))
                elif label is None:
                    row.append("")
                else:
                    row.append(str(label))
            for column in self._columns:
                value = self._data[column][position]
                if value is None:
                    row.append("")
                else:
                    row.append(str(value))
            writer.writerow(row)

        if close_handle:
            handle.close()

    def tail(self, n: int = 5) -> "DataFrame":  # pragma: no cover - debug helper
        start = max(0, len(self) - n)
        return self.iloc[start:]


NA = None


class _RollingSeries:
    def __init__(self, series: Series, window: int) -> None:
        if window <= 0:
            raise ValueError("window must be positive")
        self._series = series
        self._window = window

    def std(self) -> Series:
        results: List[float] = []
        data = self._series._data
        for index in range(len(data)):
            start = max(0, index - self._window + 1)
            window_values = [
                value
                for value in data[start : index + 1]
                if value is not None and not (isinstance(value, float) and math.isnan(value))
            ]
            if len(window_values) < self._window or not window_values:
                results.append(math.nan)
                continue
            mean = sum(window_values) / len(window_values)
            variance = sum((value - mean) ** 2 for value in window_values) / len(window_values)
            results.append(math.sqrt(variance))
        return Series(results, self._series.index, self._series.dtype)
