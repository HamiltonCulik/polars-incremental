from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

import polars_incremental as pli


def main() -> None:
    parser = argparse.ArgumentParser(description="Consume a Delta table with checkpoints.")
    parser.add_argument("--source", default="data/delta/source_events", help="Source Delta table path")
    parser.add_argument("--sink", default="data/delta/sink_events", help="Sink Delta table path")
    parser.add_argument(
        "--checkpoint",
        default="data/checkpoints/delta_source",
        help="Checkpoint directory",
    )
    parser.add_argument("--max-files", type=int, default=1000, help="Max files per batch")
    parser.add_argument("--max-bytes", type=int, default=None, help="Max bytes per batch")
    parser.add_argument("--ignore-deletes", action="store_true", help="Ignore delete-only commits")
    parser.add_argument("--ignore-changes", action="store_true", help="Ignore updates/deletes")
    parser.add_argument(
        "--strict-changes",
        action="store_true",
        help="Fail on updates/deletes (default is to ignore changes)",
    )
    parser.add_argument(
        "--read-change-feed",
        action="store_true",
        help="Read Delta Change Data Feed (CDF) if available",
    )
    parser.add_argument("--starting-version", type=int, default=None, help="Start version")
    parser.add_argument("--starting-timestamp", type=str, default=None, help="Start timestamp")
    parser.add_argument("--loop", action="store_true", help="Poll in a loop")
    parser.add_argument("--sleep", type=float, default=1.0, help="Sleep between polls")
    args = parser.parse_args()

    def reader(files, batch=None):
        if args.read_change_feed and batch is not None:
            return pli.read_cdf_batch(batch)
        return pl.read_parquet(files)

    def writer(df, _batch=None):
        df.write_delta(Path(args.sink), mode="append")

    def run_once() -> int:
        ignore_changes = args.ignore_changes or not args.strict_changes
        pipeline = pli.Pipeline(
            source=pli.DeltaSource(
                path=args.source,
                starting_version=args.starting_version,
                starting_timestamp=args.starting_timestamp,
                max_files_per_trigger=args.max_files,
                max_bytes_per_trigger=args.max_bytes,
                ignore_deletes=args.ignore_deletes,
                ignore_changes=ignore_changes,
                read_change_feed=args.read_change_feed,
            ),
            checkpoint_dir=args.checkpoint,
            reader=reader,
            writer=writer,
        )
        result = pipeline.run(once=True)
        return result.batches

    if args.loop:
        while True:
            batches = run_once()
            if batches == 0:
                print("no new delta files found")
            else:
                preview = pl.read_delta(str(Path(args.sink)))
                print("sink preview:")
                print(preview.head(5))
            time.sleep(args.sleep)
    else:
        batches = run_once()
        if batches == 0:
            print("no new delta files found")
            return
        preview = pl.read_delta(str(Path(args.sink)))
        print("sink preview:")
        print(preview.head(5))


if __name__ == "__main__":
    main()
