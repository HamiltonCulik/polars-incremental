import sys
import unittest
from unittest import mock
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_ROOT))

from polars_incremental.errors import SchemaEvolutionError
from polars_incremental.schema import (
    SchemaEvolution,
    SchemaPolicy,
    _dtype_from_str,
    _widen_target,
    apply_schema,
    apply_schema_lazy,
    deserialize_schema,
    normalize_schema_input,
)


class TestSchemaUtils(unittest.TestCase):
    def test_dtype_from_str_parses_complex_types(self) -> None:
        dt = _dtype_from_str("Datetime(time_unit='us', time_zone='UTC')")
        self.assertEqual(dt, pl.Datetime("us", "UTC"))
        dt_none = _dtype_from_str("Datetime(time_unit='ms', time_zone='None')")
        self.assertEqual(dt_none, pl.Datetime("ms", None))
        self.assertEqual(_dtype_from_str("Duration(time_unit='ns')"), pl.Duration("ns"))
        self.assertEqual(_dtype_from_str("Decimal(10, 2)"), pl.Decimal(10, 2))
        with self.assertRaises(SchemaEvolutionError):
            _dtype_from_str("NotAType")

    def test_dtype_from_str_parses_list_and_struct(self) -> None:
        self.assertEqual(_dtype_from_str("List(Int64)"), pl.List(pl.Int64))
        self.assertEqual(
            _dtype_from_str("Struct({'a': Int64, 'b': String})"),
            pl.Struct([pl.Field("a", pl.Int64), pl.Field("b", pl.Utf8)]),
        )
        self.assertEqual(
            _dtype_from_str("List(Struct({'a': Int64}))"),
            pl.List(pl.Struct([pl.Field("a", pl.Int64)])),
        )
        self.assertEqual(_dtype_from_str("List(List(Int64))"), pl.List(pl.List(pl.Int64)))
        self.assertEqual(
            _dtype_from_str("Struct({'a': List(Int64), 'b': String})"),
            pl.Struct([pl.Field("a", pl.List(pl.Int64)), pl.Field("b", pl.Utf8)]),
        )

    def test_dtype_from_str_parses_view_types(self) -> None:
        utf8_view = getattr(pl, "Utf8View", None)
        binary_view = getattr(pl, "BinaryView", None)
        expected_utf8 = utf8_view if utf8_view is not None else pl.Utf8
        expected_binary = binary_view if binary_view is not None else pl.Binary
        self.assertEqual(_dtype_from_str("Utf8View"), expected_utf8)
        self.assertEqual(_dtype_from_str("BinaryView"), expected_binary)
        self.assertEqual(
            _dtype_from_str("List(Utf8View)"),
            pl.List(expected_utf8),
        )

    def test_dtype_from_str_parses_enum(self) -> None:
        if not hasattr(pl, "Enum"):
            self.skipTest("Enum dtype not available in this Polars version")
        enum_dtype = pl.Enum(["a", "b"])  # type: ignore[arg-type]
        parsed = _dtype_from_str("Enum(categories=['a', 'b'])")
        self.assertEqual(parsed, enum_dtype)
        parsed_no_spaces = _dtype_from_str("Enum(categories=['a','b'])")
        self.assertEqual(parsed_no_spaces, enum_dtype)

    def test_normalize_schema_input_accepts_dict_and_list(self) -> None:
        payload = normalize_schema_input({"a": "Int64", "b": "Utf8"})
        schema = deserialize_schema(payload)
        self.assertEqual(schema, [("a", pl.Int64), ("b", pl.Utf8)])

        payload = normalize_schema_input([("x", "Boolean"), ("y", "Float32")])
        schema = deserialize_schema(payload)
        self.assertEqual(schema, [("x", pl.Boolean), ("y", pl.Float32)])

        with self.assertRaises(SchemaEvolutionError):
            normalize_schema_input(123)
        with self.assertRaises(SchemaEvolutionError):
            normalize_schema_input([("bad", 123)])

    def test_apply_schema_adds_missing_columns(self) -> None:
        df = pl.DataFrame({"a": [1, 2]})
        schema = normalize_schema_input({"a": "Int64", "b": "Utf8"})
        policy = SchemaPolicy(mode="add_new_columns")
        out, payload, changed = apply_schema(df, stored_schema=schema, explicit_schema=None, policy=policy)

        self.assertEqual(out.columns, ["a", "b"])
        self.assertEqual(out["b"].to_list(), [None, None])
        self.assertFalse(changed)
        self.assertEqual(deserialize_schema(payload), [("a", pl.Int64), ("b", pl.Utf8)])

    def test_apply_schema_lazy_with_rescue(self) -> None:
        df = pl.DataFrame({"a": ["1", "x"]})
        lf = df.lazy()
        schema = normalize_schema_input({"a": "Int64"})
        policy = SchemaPolicy(mode="coerce", rescue_mode="column", rescue_column="_rescued")
        lf_out, payload, changed = apply_schema_lazy(
            lf, stored_schema=schema, explicit_schema=None, policy=policy
        )
        out = lf_out.collect()
        self.assertEqual(out["a"].to_list(), [1, None])
        rescued = out.select(pl.col("_rescued").struct.field("a")).to_series().to_list()
        self.assertEqual(rescued, [None, "x"])
        self.assertFalse(changed)
        self.assertEqual(deserialize_schema(payload), [("a", pl.Int64)])

    def test_schema_evolution_from_options_and_apply(self) -> None:
        options = {"schema_mode": "coerce", "rescue_mode": "column", "other": 123}
        schema_evolution, remaining = SchemaEvolution.from_options(dict(options))
        self.assertIsNotNone(schema_evolution)
        self.assertEqual(remaining, {"other": 123})

        df = pl.DataFrame({"a": ["1", "x"]})
        checkpoint = mock.Mock()
        checkpoint.get_schema.return_value = normalize_schema_input({"a": "Int64"})

        out = schema_evolution.apply(df, checkpoint)
        self.assertIsInstance(out, pl.DataFrame)
        checkpoint.set_schema.assert_not_called()

        marker = {"a": 1}
        self.assertIs(schema_evolution.apply(marker, checkpoint), marker)

    def test_parse_timestamp_round_trip(self) -> None:
        schema_evolution = SchemaEvolution(mode="strict")
        ts = datetime(2024, 1, 1, tzinfo=timezone.utc).isoformat()
        payload = {"a": "Int64", "b": "Datetime(time_unit='ms', time_zone='UTC')"}
        normalized = normalize_schema_input(payload)
        schema = deserialize_schema(normalized)
        self.assertEqual(schema[1][1], pl.Datetime("ms", "UTC"))
        self.assertIsInstance(ts, str)

    def test_widen_target_variants(self) -> None:
        self.assertEqual(_widen_target(pl.Int32, pl.Int64), pl.Int64)
        self.assertEqual(_widen_target(pl.UInt8, pl.UInt16), pl.UInt16)
        self.assertEqual(_widen_target(pl.Int32, pl.UInt32), pl.Float64)
        self.assertEqual(_widen_target(pl.Float32, pl.Int64), pl.Float64)
        self.assertEqual(_widen_target(pl.Utf8, pl.Int32), pl.Utf8)
        self.assertEqual(_widen_target(pl.Boolean, pl.Int8), pl.Int8)

    def test_apply_schema_lazy_strict_rejects_missing(self) -> None:
        lf = pl.DataFrame({"a": [1]}).lazy()
        schema = normalize_schema_input({"a": "Int64", "b": "Utf8"})
        policy = SchemaPolicy(mode="strict")
        with self.assertRaises(SchemaEvolutionError):
            apply_schema_lazy(lf, stored_schema=schema, explicit_schema=None, policy=policy)

    def test_schema_evolution_apply_lazy_updates_schema(self) -> None:
        schema_evolution = SchemaEvolution(mode="add_new_columns")
        checkpoint = mock.Mock()
        checkpoint.get_schema.return_value = None
        lf = pl.DataFrame({"a": [1]}).lazy()
        out = schema_evolution.apply(lf, checkpoint)
        self.assertIsInstance(out, pl.LazyFrame)
        checkpoint.set_schema.assert_called_once()
