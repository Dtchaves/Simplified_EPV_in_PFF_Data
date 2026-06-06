from __future__ import annotations

import argparse

from PFFDataset_src.dataset import PFFDataset


def _parse_match_ids(raw_value: str | None) -> list[str] | None:
    if raw_value is None:
        return None
    match_ids = [token.strip() for token in raw_value.split(",") if token.strip()]
    return match_ids or None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate reusable PFF parquet triplets under data/processed/pff_match_triplets.",
    )
    parser.add_argument("--competition", default="PL", help="Competition code, for example PL or UCL")
    parser.add_argument("--season", default="24-25", help="Season folder, for example 24-25")
    parser.add_argument("--n-matches", type=int, default=None, help="Optional limit on matches to process")
    parser.add_argument(
        "--match-ids",
        default=None,
        help="Optional comma-separated match ids. Overrides --n-matches when provided.",
    )
    parser.add_argument(
        "--event-type",
        default="all",
        choices=["all", "shot", "pass", "carry"],
        help="Optional event filter applied before saving the triplets.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Rebuild triplets even when cached files exist")
    parser.add_argument(
        "--filter-tracking-to-event-frames",
        action="store_true",
        help="Keep only tracking rows aligned to the filtered events.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help="Worker processes for cache generation. Defaults to an automatic CPU-based value.",
    )
    args = parser.parse_args()

    dataset = PFFDataset(args.competition, args.season)
    dataset.load_data(
        n_matches=args.n_matches,
        match_ids=_parse_match_ids(args.match_ids),
        add_velocity=True,
        event_type=args.event_type,
        save=True,
        overwrite=args.overwrite,
        filter_tracking_to_event_frames=args.filter_tracking_to_event_frames,
        n_jobs=args.n_jobs,
        store_in_memory=False,
    )

    total_matches = len(dataset.last_load_summary)
    cache_hits = sum(1 for item in dataset.last_load_summary if item.get("cache_hit"))
    rebuilt_matches = total_matches - cache_hits

    print(f"Triplet root: {dataset.save_path}")
    print(f"Matches scanned: {total_matches}")
    print(f"Triplets reused: {cache_hits}")
    print(f"Triplets built: {rebuilt_matches}")

    for item in dataset.last_load_summary[:10]:
        print(item)


if __name__ == "__main__":
    main()