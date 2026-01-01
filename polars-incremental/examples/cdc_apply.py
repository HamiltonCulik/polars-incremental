import polars as pl
import polars_incremental as pli


def main() -> None:
    existing = pl.DataFrame({"id": [1, 2], "value": [10, 20]})
    changes = pl.DataFrame(
        {
            "id": [2, 1, 3],
            "value": [25, None, 30],
            "_change_type": ["update_postimage", "delete", "insert"],
            "_commit_version": [1, 2, 3],
        }
    )

    updated = pli.apply_cdc(changes, existing, keys=["id"])

    print("existing:")
    print(existing.sort("id"))
    print("changes:")
    print(changes.sort("id"))
    print("updated:")
    print(updated.sort("id"))


if __name__ == "__main__":
    main()
