from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.experiment_baselines import (
    LLM_WEATHER_OUTPUT_COLUMNS,
    aggregate_nws_llm_weather_features,
    load_nws_llm_weather_features,
    resolve_llm_api_config,
)
from scripts.rebuild_nws_llm_cache import rebuild_cache


DEFAULT_PILS = ("AFDLOX", "AFDSGX", "AFDHNX", "AFDSTO", "AFDMTR")


def _parse_pils(value: str) -> tuple[str, ...]:
    return tuple(part.strip().upper() for part in value.split(",") if part.strip())


def _backup_existing_aggregate(path: Path, source: str) -> None:
    if not path.exists():
        return
    try:
        existing = pd.read_csv(path, usecols=["llm_feature_source"])
    except Exception:
        existing = pd.DataFrame()
    if not existing.empty and existing["llm_feature_source"].astype(str).eq(source).all():
        return
    backup = path.with_name(f"{path.stem}.pre_{source}_backup{path.suffix}")
    if not backup.exists():
        shutil.copy2(path, backup)


def rebuild_multi_cache(
    *,
    raw_nws_dir: Path,
    processed_dir: Path,
    suffix: str,
    pils: tuple[str, ...],
    provider: str,
    model: str,
    max_workers: int,
    flush_every: int,
    limit: int | None,
    pending_limit: int | None,
) -> Path:
    config = resolve_llm_api_config(mode=provider, provider=provider, model=model)
    source = config["source"]
    if limit == 0:
        for pil in pils:
            nws_path = raw_nws_dir / f"{pil}_{suffix}.txt"
            if not nws_path.exists():
                raise FileNotFoundError(f"Missing NWS text file: {nws_path}")
        aggregate_path = processed_dir / f"nws_llm_weather_features_multi_nws_{suffix}.csv"
        print(
            f"dry_run paths_ok={len(pils)} provider={config['provider']} model={config['model']} "
            f"source={source} aggregate_path={aggregate_path}",
            flush=True,
        )
        return aggregate_path

    per_office_paths: list[Path] = []

    for pil in pils:
        nws_path = raw_nws_dir / f"{pil}_{suffix}.txt"
        if not nws_path.exists():
            raise FileNotFoundError(f"Missing NWS text file: {nws_path}")
        output_path = processed_dir / f"nws_llm_weather_features_{pil}_{suffix}.csv"
        print(f"rebuild {pil}: {nws_path.name} -> {output_path.name}", flush=True)
        rebuild_cache(
            nws_path=nws_path,
            output_path=output_path,
            provider=provider,
            model=model,
            max_workers=max_workers,
            flush_every=flush_every,
            limit=limit,
            pending_limit=pending_limit,
        )
        per_office_paths.append(output_path)

    frames = [load_nws_llm_weather_features(path) for path in per_office_paths]
    aggregate = aggregate_nws_llm_weather_features(frames)
    aggregate_path = processed_dir / f"nws_llm_weather_features_multi_nws_{suffix}.csv"
    _backup_existing_aggregate(aggregate_path, source)
    aggregate.to_csv(aggregate_path, index=False, columns=LLM_WEATHER_OUTPUT_COLUMNS)
    print(
        f"aggregate_written={aggregate_path} rows={len(aggregate)} source={source}",
        flush=True,
    )
    return aggregate_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild and aggregate multi-office NWS LLM weather caches.")
    parser.add_argument("--raw-nws-dir", type=Path, default=Path("data_multi_weather_2022_2025/raw/nws_text"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data_multi_weather_2022_2025/processed"))
    parser.add_argument("--suffix", default="2022-01-01_2026-01-01")
    parser.add_argument("--pils", default=",".join(DEFAULT_PILS))
    parser.add_argument("--provider", default="deepseek", choices=["deepseek", "openai"])
    parser.add_argument("--model", default="")
    parser.add_argument("--max-workers", type=int, default=3)
    parser.add_argument("--flush-every", type=int, default=25)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--pending-limit", type=int, default=None)
    args = parser.parse_args()

    rebuild_multi_cache(
        raw_nws_dir=args.raw_nws_dir,
        processed_dir=args.processed_dir,
        suffix=args.suffix,
        pils=_parse_pils(args.pils),
        provider=args.provider,
        model=args.model,
        max_workers=args.max_workers,
        flush_every=args.flush_every,
        limit=args.limit,
        pending_limit=args.pending_limit,
    )


if __name__ == "__main__":
    main()
