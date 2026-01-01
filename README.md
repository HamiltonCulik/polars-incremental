# polars-incremental (experimental)

This repo is just a small experiment to see how feasible it would be to
generalize incremental utilities for Polars. It is not actively maintained.
And I don't really intend to maintain a compatibility matrix for polars + deltalake. 

The installable package and all source files live in `polars-incremental/`.
For local development, install it in editable mode from the repo root:

```bash
pip install -e polars-incremental
```

To build and install a wheel:

```bash
python -m pip install --upgrade build
python -m build polars-incremental
python -m pip install polars-incremental/dist/*.whl
```

`scripts/` simply contains some showcases / ad-hoc testing that was done
to double check all features were working properly with "semi real" examples.

License: Don't really care. You are welcome to expand on this work,
and if you do, I would be honored to be credited. That's it.
