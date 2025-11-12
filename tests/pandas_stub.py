"""Lightweight stub of pandas used for offline unit tests."""
from __future__ import annotations

import csv
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence
from zoneinfo import ZoneInfo


def _is_nan(value: Any) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


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
        if data is None:
            values: List[Any] = []
        elif isinstance(data, Sequence) and not isinstance(data, (str, bytes)):
            values = list(data)
        else:
            length = len(index) if index is not None else 1
            values = [data] * length

        if index is None:
            computed_index: Sequence[Any] = list(range(len(values)))
        elif isinstance(index, DatetimeIndex):
            computed_index = index
        else:
            computed_index = list(index)

        self._data = values
        self.index: Sequence[Any] = computed_index
        self.dtype = dtype
        self.iloc = _SeriesILoc(self)
        self.loc = _SeriesLoc(self)
        self.name: str | None = None

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

    def mean(self) -> float:
        values = [float(value) for value in self._data if not _is_nan(value)]
        if not values:
            return math.nan
        return sum(values) / len(values)

    @property
    def empty(self) -> bool:
        return len(self._data) == 0

    def pct_change(self, periods: int = 1) -> "Series":
        if periods == 0:
            return Series([math.nan] * len(self._data), self.index, self.dtype)
        changes: List[float] = []
        for position, value in enumerate(self._data):
            reference_position = position - periods
            if reference_position < 0:
                changes.append(math.nan)
                continue
            previous = self._data[reference_position]
            if _is_nan(value) or _is_nan(previous):
                changes.append(math.nan)
                continue
            try:
                changes.append((value - previous) / previous)
            except ZeroDivisionError:
                changes.append(math.nan)
        if periods > 0 and changes:
            tail_start = max(0, len(changes) - periods)
            for idx in range(tail_start, len(changes)):
                changes[idx] = math.nan
        return Series(changes, self.index, self.dtype)

    def rolling(self, window: int, min_periods: int | None = None) -> "_RollingSeries":
        return _RollingSeries(self, window, min_periods=min_periods)

    def shift(self, periods: int) -> "Series":
        if periods == 0:
            return Series(list(self._data), self.index, self.dtype)
        filler = [math.nan] * abs(periods)
        if periods > 0:
            shifted_data = filler + list(self._data[:-periods]) if periods < len(self._data) else filler
        else:
            shifted_data = list(self._data[-periods:]) + filler if -periods < len(self._data) else filler
        return Series(shifted_data[: len(self._data)], self.index, self.dtype)

    def isna(self) -> "Series":
        flags = [_is_nan(value) for value in self._data]
        return Series(flags, self.index, "bool")

    def notna(self) -> "Series":
        flags = [not _is_nan(value) for value in self._data]
        return Series(flags, self.index, "bool")

    def isin(self, values: Iterable[Any]) -> "Series":
        lookup = set(values)
        flags = [value in lookup for value in self._data]
        return Series(flags, self.index, "bool")

    def all(self) -> bool:
        result = True
        for value in self._data:
            if _is_nan(value):
                continue
            result = result and bool(value)
            if not result:
                break
        return result

    def astype(self, dtype: Any) -> "Series":
        dtype_str = str(dtype)
        if dtype_str in {"float", "float64", "float32"} or dtype is float:
            converted = [float(value) if not _is_nan(value) else math.nan for value in self._data]
            result = Series(converted, self.index, "float64")
            result.name = self.name
            return result
        raise ValueError(f"Unsupported dtype conversion in stub: {dtype}")

    def tail(self, n: int = 5) -> "Series":
        start = max(0, len(self._data) - n)
        tail_data = self._data[start:]
        if isinstance(self.index, DatetimeIndex):
            tail_index = DatetimeIndex(list(self.index)[start:])
        else:
            tail_index = list(self.index)[start:]
        result = Series(tail_data, tail_index, self.dtype)
        result.name = self.name
        return result

    def multiply(self, other: Any) -> "Series":
        return self._binary_op(other, lambda a, b: a * b)

    def __mul__(self, other: Any) -> "Series":
        return self.multiply(other)

    def __rmul__(self, other: Any) -> "Series":
        return self.multiply(other)

    def __truediv__(self, other: Any) -> "Series":
        return self._binary_op(other, lambda a, b: a / b)

    def __rtruediv__(self, other: Any) -> "Series":
        result_values = []
        for value in self._data:
            if _is_nan(value) or _is_nan(other):
                result_values.append(math.nan)
                continue
            try:
                result_values.append(other / value)
            except ZeroDivisionError:
                result_values.append(math.nan)
        result = Series(result_values, self.index, self.dtype)
        result.name = self.name
        return result

    def __sub__(self, other: Any) -> "Series":
        return self._binary_op(other, lambda a, b: a - b)

    def __rsub__(self, other: Any) -> "Series":
        result_values = []
        for value in self._data:
            if _is_nan(value) or _is_nan(other):
                result_values.append(math.nan)
                continue
            result_values.append(other - value)
        result = Series(result_values, self.index, self.dtype)
        result.name = self.name
        return result

    def __add__(self, other: Any) -> "Series":
        return self._binary_op(other, lambda a, b: a + b)

    def __radd__(self, other: Any) -> "Series":
        return self.__add__(other)

    def _comparison_op(self, other: Any, comparator) -> "Series":
        if isinstance(other, Series):
            left_index = list(self.index)
            right_index = list(other.index)
            if left_index != right_index:
                lookup = {label: value for label, value in zip(right_index, other._data)}
                right_values = [lookup.get(label, math.nan) for label in left_index]
            else:
                right_values = list(other._data)
            result_values = []
            for left_value, right_value in zip(self._data, right_values):
                if _is_nan(left_value) or _is_nan(right_value):
                    result_values.append(False)
                else:
                    result_values.append(bool(comparator(left_value, right_value)))
            return Series(result_values, self.index, "bool")
        result_values = []
        for value in self._data:
            if _is_nan(value) or _is_nan(other):
                result_values.append(False)
            else:
                result_values.append(bool(comparator(value, other)))
        return Series(result_values, self.index, "bool")

    def __gt__(self, other: Any) -> "Series":
        return self._comparison_op(other, lambda a, b: a > b)

    def __ge__(self, other: Any) -> "Series":
        return self._comparison_op(other, lambda a, b: a >= b)

    def __lt__(self, other: Any) -> "Series":
        return self._comparison_op(other, lambda a, b: a < b)

    def __le__(self, other: Any) -> "Series":
        return self._comparison_op(other, lambda a, b: a <= b)

    def __eq__(self, other: Any) -> "Series":
        return self._comparison_op(other, lambda a, b: a == b)

    def __ne__(self, other: Any) -> "Series":
        comparison = self.__eq__(other)
        inverted = [not flag for flag in comparison._data]
        return Series(inverted, comparison.index, comparison.dtype)

    def _binary_op(self, other: Any, operator) -> "Series":
        if isinstance(other, Series):
            left_index = list(self.index)
            right_index = list(other.index)
            if left_index != right_index:
                lookup = {label: value for label, value in zip(right_index, other._data)}
                right_values = [lookup.get(label, math.nan) for label in left_index]
            else:
                right_values = list(other._data)
            result_values: List[float] = []
            for left_value, right_value in zip(self._data, right_values):
                if _is_nan(left_value) or _is_nan(right_value):
                    result_values.append(math.nan)
                    continue
                try:
                    result_values.append(operator(left_value, right_value))
                except ZeroDivisionError:
                    result_values.append(math.nan)
            result = Series(result_values, self.index, self.dtype)
            result.name = self.name
            return result
        else:
            result_values = []
            for value in self._data:
                if _is_nan(value) or _is_nan(other):
                    result_values.append(math.nan)
                    continue
                try:
                    result_values.append(operator(value, other))
                except ZeroDivisionError:
                    result_values.append(math.nan)
            result = Series(result_values, self.index, self.dtype)
            result.name = self.name
            return result

    def divide(self, other: Any) -> "Series":
        return self._binary_op(other, lambda a, b: a / b)

    def subtract(self, other: Any) -> "Series":
        return self._binary_op(other, lambda a, b: a - b)

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


class _SeriesLoc:
    def __init__(self, series: Series) -> None:
        self._series = series

    def _labels(self) -> List[Any]:
        if isinstance(self._series.index, DatetimeIndex):
            return list(self._series.index)
        return list(self._series.index)

    def _mask_from_key(self, key: Any) -> List[bool]:
        labels = self._labels()
        if isinstance(key, Series):
            lookup = {label: bool(value) for label, value in zip(list(key.index), key._data)}
            return [bool(lookup.get(label, False)) for label in labels]
        if isinstance(key, list):
            if len(key) != len(labels):
                raise ValueError("Boolean index length mismatch in stub")
            return [bool(value) for value in key]
        raise TypeError("Unsupported key type for Series.loc in stub")

    def __getitem__(self, key: Any):
        labels = self._labels()
        if isinstance(key, slice):
            selected_labels = labels[key]
            selected_data = self._series._data[key]
            if isinstance(self._series.index, DatetimeIndex):
                selected_index = DatetimeIndex(list(selected_labels))
            else:
                selected_index = list(selected_labels)
            return Series(selected_data, selected_index, self._series.dtype)
        if isinstance(key, (Series, list)):
            mask = self._mask_from_key(key)
            data = [value for value, flag in zip(self._series._data, mask) if flag]
            selected_labels = [label for label, flag in zip(labels, mask) if flag]
            if isinstance(self._series.index, DatetimeIndex):
                selected_index = DatetimeIndex(selected_labels)
            else:
                selected_index = selected_labels
            return Series(data, selected_index, self._series.dtype)
        try:
            position = labels.index(key)
        except ValueError as exc:
            raise KeyError(key) from exc
        return self._series._data[position]

    def __setitem__(self, key: Any, value: Any) -> None:
        labels = self._labels()
        if isinstance(key, (Series, list)):
            mask = self._mask_from_key(key)
            if isinstance(value, Series):
                value_map = {label: val for label, val in zip(list(value.index), value._data)}
                for position, (flag, label) in enumerate(zip(mask, labels)):
                    if flag:
                        self._series._data[position] = value_map.get(label, math.nan)
                return
            if isinstance(value, (list, tuple)):
                expected = sum(1 for flag in mask if flag)
                values = list(value)
                if len(values) != expected:
                    raise ValueError("Value length mismatch for assignment in stub")
                iterator = iter(values)
                for position, flag in enumerate(mask):
                    if flag:
                        self._series._data[position] = next(iterator)
                return
            for position, flag in enumerate(mask):
                if flag:
                    self._series._data[position] = value
            return
        try:
            position = labels.index(key)
        except ValueError as exc:
            raise KeyError(key) from exc
        if isinstance(value, Series):
            if len(value._data) != 1:
                raise ValueError("Single value expected for Series assignment in stub")
            self._series._data[position] = value._data[0]
        else:
            self._series._data[position] = value


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

    def get(self, key: str, default: Any = None):
        if key in self._columns:
            return self.__getitem__(key)
        return default

    def __setitem__(self, key: str, values: Sequence[Any]) -> None:
        self._data[key] = list(values)
        if key not in self._columns:
            self._columns.append(key)

    def iterrows(self):
        for position in range(len(self)):
            row_values = {column: self._data[column][position] for column in self._columns}
            yield self._index[position], Series(list(row_values.values()), index=list(row_values.keys()))

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

    def reindex(self, index: Sequence[Any]) -> "DataFrame":
        target_index = list(index)
        lookup: Dict[Any, int] = {label: pos for pos, label in enumerate(self._index)}
        data: Dict[str, List[Any]] = {}
        for column in self._columns:
            column_values: List[Any] = []
            for label in target_index:
                position = lookup.get(label)
                if position is None:
                    column_values.append(None)
                else:
                    column_values.append(self._data[column][position])
            data[column] = column_values
        converted_index = target_index if isinstance(target_index, DatetimeIndex) else to_datetime(target_index)
        return DataFrame(data, index=converted_index, columns=self._columns)

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

    def to_parquet(self, path, engine: str = "pyarrow") -> None:  # type: ignore[override]
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        self.to_csv(target, index=True, date_format="%Y-%m-%dT%H:%M:%S%z")

    def tail(self, n: int = 5) -> "DataFrame":  # pragma: no cover - debug helper
        start = max(0, len(self) - n)
        return self.iloc[start:]


NA = None


class _RollingSeries:
    def __init__(self, series: Series, window: int, min_periods: int | None = None) -> None:
        if window <= 0:
            raise ValueError("window must be positive")
        self._series = series
        self._window = window
        self._min_periods = min_periods if min_periods is not None else window

    def mean(self) -> Series:
        results: List[float] = []
        data = self._series._data
        for index in range(len(data)):
            start = max(0, index - self._window + 1)
            window_values = [
                value
                for value in data[start : index + 1]
                if not _is_nan(value)
            ]
            if len(window_values) < self._min_periods or not window_values:
                results.append(math.nan)
                continue
            results.append(sum(window_values) / len(window_values))
        return Series(results, self._series.index, self._series.dtype)

    def std(self, ddof: int = 0) -> Series:
        results: List[float] = []
        data = self._series._data
        for index in range(len(data)):
            start = max(0, index - self._window + 1)
            window_values = [
                value
                for value in data[start : index + 1]
                if not _is_nan(value)
            ]
            if len(window_values) < self._min_periods or not window_values:
                results.append(math.nan)
                continue
            mean = sum(window_values) / len(window_values)
            divisor = max(len(window_values) - ddof, 1)
            variance = sum((value - mean) ** 2 for value in window_values) / divisor
            results.append(math.sqrt(variance))
        return Series(results, self._series.index, self._series.dtype)


def read_parquet(path, engine: str = "pyarrow") -> DataFrame:
    target = Path(path)
    with target.open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        rows = list(reader)
    if not rows:
        return DataFrame({})
    header = rows[0]
    has_index = header and header[0] == "index"
    columns = header[1:] if has_index else header
    data: Dict[str, List[Any]] = {column: [] for column in columns}
    index_values: List[Any] = []
    for raw_row in rows[1:]:
        if not raw_row:
            continue
        row = list(raw_row)
        if has_index:
            index_raw = row.pop(0)
            if index_raw:
                try:
                    index_values.append(datetime.fromisoformat(index_raw))
                except ValueError:
                    index_values.append(index_raw)
            else:
                index_values.append(None)
        for column, value in zip(columns, row):
            if value == "":
                data[column].append(None)
            else:
                try:
                    data[column].append(float(value))
                except ValueError:
                    data[column].append(value)
        for column in columns[len(row) :]:
            data[column].append(None)
    index = DatetimeIndex(index_values) if index_values else []
    return DataFrame(data, index=index, columns=columns)
