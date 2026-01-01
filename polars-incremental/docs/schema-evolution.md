# Schema evolution

Schema evolution lets you control how new columns and type changes are handled across batches. The effective schema is stored in the checkpoint metadata.

## Configuration

Use `SchemaEvolution` and attach it to a `Pipeline`:

```python
schema_evolution = pli.SchemaEvolution(
    mode="add_new_columns",
    rescue_mode="column",
    rescue_column="_rescued",
)

pipeline = pli.Pipeline(
    source=source,
    checkpoint_dir="data/checkpoints/raw",
    reader=reader,
    writer=writer,
    schema_evolution=schema_evolution,
)
```

Fields:
- `mode`: `strict`, `add_new_columns`, `coerce`, `type_widen`.
- `schema`: explicit schema (dict or list of `(name, dtype)` pairs).
- `rescue_mode`: `none` or `column`.
- `rescue_column`: rescue column name (default `_rescued`).

## Modes

- `strict`: new/missing columns or type changes are errors.
- `add_new_columns`: allow new columns; missing columns are added as nulls.
- `coerce`: cast mismatched types with `strict=False`.
- `type_widen`: widen numeric/string types when safe (otherwise error).

## Rescue mode

If `rescue_mode="column"`, casting failures are stored in a structured rescue column (default `_rescued`).

## Example

```python
pli.Pipeline(
    source=pli.FilesSource(path="data/raw", file_format="parquet"),
    checkpoint_dir="data/checkpoints/raw",
    reader=reader,
    writer=writer,
    schema_evolution=pli.SchemaEvolution(
        mode="add_new_columns",
        rescue_mode="column",
        rescue_column="_rescued",
    ),
).run(once=True)
```

Explicit schema example:

```python
import polars as pl

pli.Pipeline(
    source=pli.FilesSource(path="data/raw", file_format="parquet"),
    checkpoint_dir="data/checkpoints/raw",
    reader=reader,
    writer=writer,
    schema_evolution=pli.SchemaEvolution(
        mode="strict",
        schema={"id": pl.Int64, "value": pl.Float64},
    ),
).run(once=True)
```

## DataFrame and LazyFrame

Schema evolution applies to both DataFrame and LazyFrame inputs. For LazyFrames, schema comparisons use lazy schema inference, and casts are applied as lazy expressions (evaluated when you collect or sink the frame).

## Precedence

If you pass an explicit `schema`, it takes precedence over any stored checkpoint schema. If no explicit schema is provided, the stored checkpoint schema is used. If neither exists, the schema is inferred from the current batch and stored.
