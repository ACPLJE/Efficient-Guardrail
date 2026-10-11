"""JSON (de)serialisation for frozen dataclasses (SPEC 2.0).

Rules:
  * IntEnum / StrEnum are written as their member name.
  * bytes are written as base64.
  * Mapping int keys are written as strings and restored from the type hints.
  * numpy arrays are written as lists (large arrays belong in Record npz files, not JSON).
  * +-inf / nan are written as the strings "inf", "-inf", "nan".
"""
from __future__ import annotations

import base64
import dataclasses
import enum
import math
import types
import typing
from collections.abc import Mapping, Sequence
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints

import numpy as np


def _float_out(x: float) -> float | str:
    if math.isnan(x):
        return "nan"
    if math.isinf(x):
        return "inf" if x > 0 else "-inf"
    return x


def to_jsonable(x: Any) -> Any:
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        return {f.name: to_jsonable(getattr(x, f.name)) for f in dataclasses.fields(x)}
    if isinstance(x, enum.Enum):
        return x.name
    if isinstance(x, bool) or x is None or isinstance(x, (str, int)) and not isinstance(x, enum.Enum):
        return x
    if isinstance(x, float):
        return _float_out(x)
    if isinstance(x, (bytes, bytearray)):
        return base64.b64encode(bytes(x)).decode("ascii")
    if isinstance(x, np.ndarray):
        return [to_jsonable(v) for v in x.tolist()]
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return _float_out(float(x))
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, Mapping):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set, frozenset)):
        return [to_jsonable(v) for v in x]
    raise TypeError(f"cannot serialise {type(x).__name__}")


_HINTS: dict[type, dict[str, Any]] = {}


def _hints(cls: type) -> dict[str, Any]:
    h = _HINTS.get(cls)
    if h is None:
        h = get_type_hints(cls)
        _HINTS[cls] = h
    return h


def _float_in(v: Any) -> float:
    if isinstance(v, str):
        if v in ("inf", "-inf", "nan"):
            return float(v)
        raise TypeError(f"not a float: {v!r}")
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise TypeError(f"not a float: {v!r}")
    return float(v)


def from_jsonable(tp: Any, v: Any) -> Any:  # noqa: C901 - one dispatcher on purpose
    if tp is Any or tp is object:
        return v
    if tp is type(None):
        if v is not None:
            raise TypeError("expected null")
        return None
    origin = get_origin(tp)
    if origin is Union or (types.UnionType is not None and isinstance(tp, types.UnionType)):
        args = get_args(tp)
        if v is None and type(None) in args:
            return None
        errors = []
        for a in args:
            if a is type(None):
                continue
            try:
                return from_jsonable(a, v)
            except (TypeError, ValueError, KeyError) as e:
                errors.append(str(e))
        raise TypeError(f"no union member matched {v!r}: {errors}")
    if origin is Literal:
        if v not in get_args(tp):
            raise ValueError(f"{v!r} not in {get_args(tp)}")
        return v
    if isinstance(tp, type) and dataclasses.is_dataclass(tp):
        if not isinstance(v, Mapping):
            raise TypeError(f"expected object for {tp.__name__}")
        hints = _hints(tp)
        kwargs = {}
        for f in dataclasses.fields(tp):
            if f.name in v:
                kwargs[f.name] = from_jsonable(hints[f.name], v[f.name])
        return tp(**kwargs)
    if isinstance(tp, type) and issubclass(tp, enum.Enum):
        if isinstance(v, str) and v in tp.__members__:
            return tp[v]
        return tp(v)
    if tp is bool:
        if not isinstance(v, bool):
            raise TypeError(f"not a bool: {v!r}")
        return v
    if tp is int:
        if isinstance(v, bool) or not isinstance(v, int):
            raise TypeError(f"not an int: {v!r}")
        return v
    if tp is float:
        return _float_in(v)
    if tp is str:
        if not isinstance(v, str):
            raise TypeError(f"not a str: {v!r}")
        return v
    if tp is bytes:
        return base64.b64decode(v)
    if tp is np.ndarray:
        return np.asarray(v)
    if origin in (tuple,):
        args = get_args(tp)
        if not isinstance(v, (list, tuple)):
            raise TypeError(f"expected list for {tp}")
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(from_jsonable(args[0], x) for x in v)
        if not args:
            return tuple(v)
        if len(args) != len(v):
            raise TypeError(f"tuple length {len(v)} != {len(args)}")
        return tuple(from_jsonable(a, x) for a, x in zip(args, v))
    if origin in (list, Sequence, typing.Sequence):
        (a,) = get_args(tp) or (Any,)
        if not isinstance(v, (list, tuple)):
            raise TypeError(f"expected list for {tp}")
        return [from_jsonable(a, x) for x in v]
    if origin in (dict, Mapping, typing.Mapping):
        ka, va = get_args(tp) or (Any, Any)
        if not isinstance(v, Mapping):
            raise TypeError(f"expected object for {tp}")
        out = {}
        for k, x in v.items():
            key = int(k) if ka is int else k
            out[key] = from_jsonable(va, x)
        return out
    raise TypeError(f"unsupported type {tp!r}")


class JsonMixin:
    """Adds to_json / from_json to a dataclass."""

    def to_json(self) -> dict:
        return to_jsonable(self)

    @classmethod
    def from_json(cls, d: Mapping[str, Any]):
        return from_jsonable(cls, d)
