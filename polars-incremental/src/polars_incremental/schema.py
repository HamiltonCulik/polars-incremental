from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from typing import Any

import polars as pl

from .errors import SchemaEvolutionError


@dataclass(frozen=True)
class SchemaPolicy:
    mode: str = "strict"  # strict | add_new_columns | coerce | type_widen
    rescue_mode: str = "none"  # none | column
    rescue_column: str = "_rescued"


def _dtype_to_str(dtype: pl.DataType) -> str:
    return str(dtype)


def _dtype_from_str(value: str) -> pl.DataType:
    simple_map = {
        "Int8": pl.Int8,
        "Int16": pl.Int16,
        "Int32": pl.Int32,
        "Int64": pl.Int64,
        "UInt8": pl.UInt8,
        "UInt16": pl.UInt16,
        "UInt32": pl.UInt32,
        "UInt64": pl.UInt64,
        "Float32": pl.Float32,
        "Float64": pl.Float64,
        "Utf8": pl.Utf8,
        "String": pl.Utf8,
        "Boolean": pl.Boolean,
        "Date": pl.Date,
        "Time": pl.Time,
        "Binary": pl.Binary,
        "Categorical": pl.Categorical,
    }
    if value in simple_map:
        return simple_map[value]

    if value == "Utf8View":
        return getattr(pl, "Utf8View", pl.Utf8)

    if value == "BinaryView":
        return getattr(pl, "BinaryView", pl.Binary)

    if value.startswith("Enum("):
        match = re.search(r"Enum\(categories=(\[.*\])\)", value)
        if match:
            try:
                categories = ast.literal_eval(match.group(1))
            except (SyntaxError, ValueError) as exc:
                raise SchemaEvolutionError(f"Unsupported Enum categories: {value}") from exc
            if not hasattr(pl, "Enum"):
                raise SchemaEvolutionError("Enum dtype not supported in this Polars version")
            return pl.Enum(categories)  # type: ignore[arg-type]

    if value.startswith("Datetime"):
        match = re.search(r"time_unit='(\w+)'", value)
        unit = match.group(1) if match else "ms"
        tz_match = re.search(r"time_zone='([^']*)'", value)
        tz = tz_match.group(1) if tz_match else None
        if tz == "None":
            tz = None
        return pl.Datetime(unit, tz)

    if value.startswith("Duration"):
        match = re.search(r"time_unit='(\w+)'", value)
        unit = match.group(1) if match else "ms"
        return pl.Duration(unit)

    if value.startswith("Decimal"):
        match = re.search(r"Decimal\((\d+),\s*(\d+)\)", value)
        if match:
            return pl.Decimal(int(match.group(1)), int(match.group(2)))

    if value.startswith("List(") and value.endswith(")"):
        inner = value[len("List(") : -1].strip()
        return pl.List(_dtype_from_str(inner))

    if value.startswith("Struct(") and value.endswith(")"):
        inner = value[len("Struct(") : -1].strip()
        fields = _parse_struct_fields(inner)
        return pl.Struct(fields)

    raise SchemaEvolutionError(f"Unsupported dtype for schema: {value}")


def _split_top_level(text: str) -> list[str]:
    parts: list[str] = []
    buf: list[str] = []
    depth_paren = 0
    depth_brace = 0
    in_quote: str | None = None

    for ch in text:
        if in_quote is not None:
            buf.append(ch)
            if ch == in_quote:
                in_quote = None
            continue

        if ch in ("'", '"'):
            in_quote = ch
            buf.append(ch)
            continue
        if ch == "(":
            depth_paren += 1
        elif ch == ")":
            depth_paren -= 1
        elif ch == "{":
            depth_brace += 1
        elif ch == "}":
            depth_brace -= 1

        if ch == "," and depth_paren == 0 and depth_brace == 0:
            part = "".join(buf).strip()
            if part:
                parts.append(part)
            buf = []
            continue
        buf.append(ch)

    tail = "".join(buf).strip()
    if tail:
        parts.append(tail)
    return parts


def _parse_struct_fields(text: str) -> list[pl.Field]:
    inner = text.strip()
    if inner.startswith("{") and inner.endswith("}"):
        inner = inner[1:-1].strip()
    if not inner:
        return []
    parts = _split_top_level(inner)
    fields: list[pl.Field] = []
    for part in parts:
        if ":" not in part:
            continue
        name_part, dtype_part = part.split(":", 1)
        name = name_part.strip().strip("'\"")
        dtype = _dtype_from_str(dtype_part.strip())
        fields.append(pl.Field(name, dtype))
    return fields


_INT_WIDTH = {
    pl.Int8: 8,
    pl.Int16: 16,
    pl.Int32: 32,
    pl.Int64: 64,
}
_UINT_WIDTH = {
    pl.UInt8: 8,
    pl.UInt16: 16,
    pl.UInt32: 32,
    pl.UInt64: 64,
}
_FLOAT_WIDTH = {
    pl.Float32: 32,
    pl.Float64: 64,
}

_INT_FROM_WIDTH = {v: k for k, v in _INT_WIDTH.items()}
_UINT_FROM_WIDTH = {v: k for k, v in _UINT_WIDTH.items()}
_FLOAT_FROM_WIDTH = {v: k for k, v in _FLOAT_WIDTH.items()}


def _is_int(dtype: pl.DataType) -> bool:
    return dtype in _INT_WIDTH


def _is_uint(dtype: pl.DataType) -> bool:
    return dtype in _UINT_WIDTH


def _is_float(dtype: pl.DataType) -> bool:
    return dtype in _FLOAT_WIDTH


def _is_bool(dtype: pl.DataType) -> bool:
    return dtype == pl.Boolean


def _is_utf8(dtype: pl.DataType) -> bool:
    return dtype == pl.Utf8


def _is_numeric(dtype: pl.DataType) -> bool:
    return _is_int(dtype) or _is_uint(dtype) or _is_float(dtype)


def _is_string_widenable(dtype: pl.DataType) -> bool:
    return _is_numeric(dtype) or _is_bool(dtype) or _is_utf8(dtype)


def _widen_numeric(expected: pl.DataType, actual: pl.DataType) -> pl.DataType | None:
    if _is_int(expected) and _is_int(actual):
        width = max(_INT_WIDTH[expected], _INT_WIDTH[actual])
        return _INT_FROM_WIDTH[width]
    if _is_uint(expected) and _is_uint(actual):
        width = max(_UINT_WIDTH[expected], _UINT_WIDTH[actual])
        return _UINT_FROM_WIDTH[width]
    if (_is_int(expected) and _is_uint(actual)) or (_is_uint(expected) and _is_int(actual)):
        return pl.Float64
    if _is_float(expected) and _is_float(actual):
        width = max(_FLOAT_WIDTH[expected], _FLOAT_WIDTH[actual])
        return _FLOAT_FROM_WIDTH[width]
    if (_is_float(expected) and _is_numeric(actual)) or (_is_float(actual) and _is_numeric(expected)):
        return pl.Float64
    return None


def _widen_target(expected: pl.DataType, actual: pl.DataType) -> pl.DataType | None:
    if expected == actual:
        return expected

    if _is_utf8(expected) or _is_utf8(actual):
        if _is_string_widenable(expected) and _is_string_widenable(actual):
            return pl.Utf8
        return None

    if _is_bool(expected) or _is_bool(actual):
        other = actual if _is_bool(expected) else expected
        if _is_int(other):
            return other
        if _is_uint(other):
            return other
        if _is_float(other):
            return pl.Float64
        return None

    if _is_numeric(expected) or _is_numeric(actual):
        return _widen_numeric(expected, actual)

    return None


def serialize_schema(schema: list[tuple[str, pl.DataType]]) -> list[dict[str, Any]]:
    return [{"name": name, "dtype": _dtype_to_str(dtype)} for name, dtype in schema]


def deserialize_schema(payload: list[dict[str, Any]]) -> list[tuple[str, pl.DataType]]:
    schema: list[tuple[str, pl.DataType]] = []
    for entry in payload:
        schema.append((str(entry["name"]), _dtype_from_str(str(entry["dtype"]))))
    return schema


def normalize_schema_input(schema: Any) -> list[dict[str, Any]] | None:
    if schema is None:
        return None
    if isinstance(schema, dict):
        items = [(name, dtype) for name, dtype in schema.items()]
    elif isinstance(schema, (list, tuple)):
        items = list(schema)
    else:
        raise SchemaEvolutionError("Schema must be dict or list of (name, dtype)")

    parsed: list[tuple[str, pl.DataType]] = []
    for name, dtype in items:
        if isinstance(dtype, pl.DataType):
            parsed.append((str(name), dtype))
        elif isinstance(dtype, str):
            parsed.append((str(name), _dtype_from_str(dtype)))
        else:
            raise SchemaEvolutionError(f"Unsupported dtype for schema: {dtype}")
    return serialize_schema(parsed)


def _schema_items_from_lazy(lf: pl.LazyFrame) -> list[tuple[str, pl.DataType]]:
    if hasattr(lf, "collect_schema"):
        schema = lf.collect_schema()
    else:
        schema = lf.schema
    return list(schema.items())


def apply_schema(
    df: pl.DataFrame,
    stored_schema: list[dict[str, Any]] | None,
    explicit_schema: list[dict[str, Any]] | None,
    policy: SchemaPolicy,
) -> tuple[pl.DataFrame, list[dict[str, Any]], bool]:
    if explicit_schema is not None:
        schema_payload = explicit_schema
    elif stored_schema is not None:
        schema_payload = stored_schema
    else:
        inferred = serialize_schema(list(df.schema.items()))
        return df, inferred, True

    schema = deserialize_schema(schema_payload)
    schema_names = [name for name, _ in schema]
    df_schema = df.schema

    new_columns = [name for name in df.columns if name not in schema_names]
    missing_columns = [name for name in schema_names if name not in df.columns]

    if policy.mode == "strict":
        if new_columns:
            raise SchemaEvolutionError(f"Unexpected columns: {new_columns}")
        if missing_columns:
            raise SchemaEvolutionError(f"Missing columns: {missing_columns}")
    elif policy.mode not in ("add_new_columns", "coerce", "type_widen"):
        raise SchemaEvolutionError(f"Unsupported schema mode: {policy.mode}")

    updated_schema = list(schema)
    changed = False
    if new_columns and policy.mode in ("add_new_columns", "coerce", "type_widen"):
        for name in new_columns:
            updated_schema.append((name, df_schema[name]))
        changed = True
        schema_names = [name for name, _ in updated_schema]

    if missing_columns and policy.mode != "strict":
        for name in missing_columns:
            dtype = dict(updated_schema)[name]
            df = df.with_columns(pl.lit(None).cast(dtype).alias(name))

    mismatch_columns: list[tuple[str, pl.DataType]] = []
    if policy.mode == "type_widen":
        updated = list(updated_schema)
        for idx, (name, dtype) in enumerate(updated):
            if name not in df.schema:
                continue
            actual = df.schema[name]
            if actual == dtype:
                continue
            target = _widen_target(dtype, actual)
            if target is None:
                raise SchemaEvolutionError(
                    f"Type mismatch for columns: {[name]} (no safe widening rule)"
                )
            mismatch_columns.append((name, target))
            if target != dtype:
                updated[idx] = (name, target)
                changed = True
        updated_schema = updated
    else:
        for name, dtype in updated_schema:
            if name not in df.schema:
                continue
            if df.schema[name] != dtype:
                mismatch_columns.append((name, dtype))

        if mismatch_columns and policy.mode != "coerce":
            raise SchemaEvolutionError(
                f"Type mismatch for columns: {[name for name, _ in mismatch_columns]}"
            )

    rescue_fields = []
    cast_expressions = []
    for name, dtype in mismatch_columns:
        casted = pl.col(name).cast(dtype, strict=False)
        cast_expressions.append(casted.alias(name))
        if policy.rescue_mode == "column":
            failure = pl.col(name).is_not_null() & casted.is_null()
            rescue_fields.append(
                pl.when(failure).then(pl.col(name).cast(pl.Utf8)).otherwise(None).alias(name)
            )

    if cast_expressions or (rescue_fields and policy.rescue_mode == "column"):
        expressions = list(cast_expressions)
        if rescue_fields and policy.rescue_mode == "column":
            rescue_struct = pl.struct(rescue_fields).alias(policy.rescue_column)
            expressions.append(rescue_struct)
        df = df.with_columns(expressions)

    ordered_cols = [name for name, _ in updated_schema]
    if policy.rescue_mode == "column" and policy.rescue_column in df.columns:
        ordered_cols.append(policy.rescue_column)
    df = df.select(ordered_cols)

    payload = serialize_schema(updated_schema)
    if payload != schema_payload:
        changed = True

    return df, payload, changed


def apply_schema_lazy(
    lf: pl.LazyFrame,
    stored_schema: list[dict[str, Any]] | None,
    explicit_schema: list[dict[str, Any]] | None,
    policy: SchemaPolicy,
) -> tuple[pl.LazyFrame, list[dict[str, Any]], bool]:
    if explicit_schema is not None:
        schema_payload = explicit_schema
    elif stored_schema is not None:
        schema_payload = stored_schema
    else:
        inferred = serialize_schema(_schema_items_from_lazy(lf))
        return lf, inferred, True

    schema = deserialize_schema(schema_payload)
    schema_names = [name for name, _ in schema]
    lf_schema = dict(_schema_items_from_lazy(lf))

    new_columns = [name for name in lf_schema if name not in schema_names]
    missing_columns = [name for name in schema_names if name not in lf_schema]

    if policy.mode == "strict":
        if new_columns:
            raise SchemaEvolutionError(f"Unexpected columns: {new_columns}")
        if missing_columns:
            raise SchemaEvolutionError(f"Missing columns: {missing_columns}")
    elif policy.mode not in ("add_new_columns", "coerce", "type_widen"):
        raise SchemaEvolutionError(f"Unsupported schema mode: {policy.mode}")

    updated_schema = list(schema)
    changed = False
    if new_columns and policy.mode in ("add_new_columns", "coerce", "type_widen"):
        for name in new_columns:
            updated_schema.append((name, lf_schema[name]))
        changed = True
        schema_names = [name for name, _ in updated_schema]

    lf_out = lf
    if missing_columns and policy.mode != "strict":
        missing_exprs = []
        for name in missing_columns:
            dtype = dict(updated_schema)[name]
            missing_exprs.append(pl.lit(None).cast(dtype).alias(name))
        lf_out = lf_out.with_columns(missing_exprs)

    mismatch_columns: list[tuple[str, pl.DataType]] = []
    if policy.mode == "type_widen":
        updated = list(updated_schema)
        for idx, (name, dtype) in enumerate(updated):
            if name not in lf_schema:
                continue
            actual = lf_schema[name]
            if actual == dtype:
                continue
            target = _widen_target(dtype, actual)
            if target is None:
                raise SchemaEvolutionError(
                    f"Type mismatch for columns: {[name]} (no safe widening rule)"
                )
            mismatch_columns.append((name, target))
            if target != dtype:
                updated[idx] = (name, target)
                changed = True
        updated_schema = updated
    else:
        for name, dtype in updated_schema:
            if name not in lf_schema:
                continue
            if lf_schema[name] != dtype:
                mismatch_columns.append((name, dtype))

        if mismatch_columns and policy.mode != "coerce":
            raise SchemaEvolutionError(
                f"Type mismatch for columns: {[name for name, _ in mismatch_columns]}"
            )

    rescue_fields = []
    cast_expressions = []
    for name, dtype in mismatch_columns:
        casted = pl.col(name).cast(dtype, strict=False)
        cast_expressions.append(casted.alias(name))
        if policy.rescue_mode == "column":
            failure = pl.col(name).is_not_null() & casted.is_null()
            rescue_fields.append(
                pl.when(failure).then(pl.col(name).cast(pl.Utf8)).otherwise(None).alias(name)
            )

    if cast_expressions or (rescue_fields and policy.rescue_mode == "column"):
        expressions = list(cast_expressions)
        if rescue_fields and policy.rescue_mode == "column":
            rescue_struct = pl.struct(rescue_fields).alias(policy.rescue_column)
            expressions.append(rescue_struct)
        lf_out = lf_out.with_columns(expressions)

    ordered_cols = [name for name, _ in updated_schema]
    include_rescue = policy.rescue_mode == "column" and (
        policy.rescue_column in lf_schema or bool(rescue_fields)
    )
    if include_rescue:
        ordered_cols.append(policy.rescue_column)
    lf_out = lf_out.select(ordered_cols)

    payload = serialize_schema(updated_schema)
    if payload != schema_payload:
        changed = True

    return lf_out, payload, changed


@dataclass(frozen=True)
class SchemaEvolution:
    mode: str = "strict"
    rescue_mode: str = "none"
    rescue_column: str = "_rescued"
    schema: Any | None = None

    @classmethod
    def from_options(
        cls, options: dict[str, Any]
    ) -> tuple["SchemaEvolution" | None, dict[str, Any]]:
        schema_keys = {"schema_mode", "rescue_mode", "rescue_column", "schema"}
        found = {key: options.pop(key) for key in list(options.keys()) if key in schema_keys}
        if not found:
            return None, options
        schema_mode = found.get("schema_mode") or "strict"
        rescue_mode = found.get("rescue_mode") or "none"
        rescue_column = found.get("rescue_column") or "_rescued"
        schema = found.get("schema")
        return (
            cls(
                mode=schema_mode,
                rescue_mode=rescue_mode,
                rescue_column=rescue_column,
                schema=schema,
            ),
            options,
        )

    def apply(self, data: Any, checkpoint: Any) -> Any:
        if not isinstance(data, (pl.DataFrame, pl.LazyFrame)):
            return data
        policy = SchemaPolicy(
            mode=str(self.mode),
            rescue_mode=str(self.rescue_mode),
            rescue_column=str(self.rescue_column),
        )
        explicit_schema = normalize_schema_input(self.schema)
        stored_schema = None
        if checkpoint is not None and hasattr(checkpoint, "get_schema"):
            stored_schema = checkpoint.get_schema()

        if isinstance(data, pl.LazyFrame):
            lf, schema_payload, changed = apply_schema_lazy(
                data, stored_schema, explicit_schema, policy
            )
            if changed and checkpoint is not None and hasattr(checkpoint, "set_schema"):
                checkpoint.set_schema(schema_payload)
            return lf

        df, schema_payload, changed = apply_schema(data, stored_schema, explicit_schema, policy)
        if changed and checkpoint is not None and hasattr(checkpoint, "set_schema"):
            checkpoint.set_schema(schema_payload)
        return df
