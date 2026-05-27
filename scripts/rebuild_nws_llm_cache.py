from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import sys
import time

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.experiment_baselines import (
    LLM_WEATHER_OUTPUT_COLUMNS,
    _parse_nws_issue_local_date,
    _split_nws_products,
    extract_api_llm_weather_features,
    load_nws_llm_weather_features_from_frame,
    normalize_llm_weather_payload,
    resolve_llm_api_config,
)


def _daily_products(nws_path: Path) -> list[tuple[str, str, int]]:
    text = nws_path.read_text(encoding="utf-8", errors="ignore")
    rows = []
    for product in _split_nws_products(text):
        issue_date = _parse_nws_issue_local_date(product)
        if issue_date is not None:
            rows.append({"issue_local_date": issue_date, "product_text": product})
    if not rows:
        return []

    products = pd.DataFrame(rows)
    daily = []
    for issue_date, group in products.groupby("issue_local_date", sort=True):
        combined = "\n\n".join(group["product_text"].astype(str).tail(3))
        daily.append((str(issue_date), combined, int(len(group))))
    return daily


def _extract_one(task: tuple[str, str, int], config: dict[str, str]) -> dict[str, object]:
    issue_date, text, product_count = task
    features = extract_api_llm_weather_features(text, config=config)
    normalized = normalize_llm_weather_payload(
        features,
        text_product_count=product_count,
        source=str(features.get("llm_feature_source", config["source"])),
    )
    return {"issue_local_date": issue_date, **normalized}


def _write_cache(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = load_nws_llm_weather_features_from_frame(pd.DataFrame(rows))
    out = out.sort_values("issue_local_date").drop_duplicates("issue_local_date", keep="last")
    out.to_csv(path, index=False, columns=LLM_WEATHER_OUTPUT_COLUMNS)


def rebuild_cache(
    nws_path: Path,
    output_path: Path,
    provider: str,
    model: str,
    max_workers: int,
    flush_every: int,
    limit: int | None,
    pending_limit: int | None,
) -> None:
    config = resolve_llm_api_config(mode=provider, provider=provider, model=model)
    tasks = _daily_products(nws_path)
    if limit is not None:
        tasks = tasks[:limit]
    expected_dates = {issue_date for issue_date, _, _ in tasks}

    existing_rows: list[dict[str, object]] = []
    completed_dates: set[str] = set()
    if output_path.exists():
        existing = pd.read_csv(output_path)
        source_mask = existing.get("llm_feature_source", pd.Series(dtype=str)).astype(str).eq(config["source"])
        existing = existing[source_mask & existing["issue_local_date"].astype(str).isin(expected_dates)].copy()
        if not existing.empty:
            existing_rows = existing.to_dict("records")
            completed_dates = set(existing["issue_local_date"].astype(str))

    pending = [task for task in tasks if task[0] not in completed_dates]
    if pending_limit is not None:
        pending = pending[:pending_limit]
    rows = existing_rows.copy()
    print(
        f"provider={config['provider']} model={config['model']} source={config['source']} "
        f"total={len(tasks)} completed={len(completed_dates)} pending={len(pending)}",
        flush=True,
    )
    if not pending:
        _write_cache(output_path, rows)
        return

    started = time.perf_counter()
    done = 0
    errors = []
    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as executor:
        futures = {executor.submit(_extract_one, task, config): task[0] for task in pending}
        for future in as_completed(futures):
            issue_date = futures[future]
            try:
                rows.append(future.result())
            except Exception as exc:
                print(f"error issue_local_date={issue_date}: {exc}", flush=True)
                errors.append(issue_date)
            done += 1
            if done % flush_every == 0 or done == len(pending):
                _write_cache(output_path, rows)
                elapsed = time.perf_counter() - started
                print(
                    f"completed_new={done}/{len(pending)} total_written={len(rows)} "
                    f"elapsed_sec={elapsed:.1f}",
                    flush=True,
                )
    if errors:
        print(f"failed_dates={len(errors)} first_failed={','.join(errors[:10])}", flush=True)
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild NWS LLM weather feature cache with an API provider.")
    parser.add_argument("--nws-path", type=Path, default=Path("data/raw/nws_text/AFDLOX_2022-01-01_2025-01-01.txt"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/nws_llm_weather_features_2022-01-01_2025-01-01.csv"),
    )
    parser.add_argument("--provider", default="deepseek", choices=["deepseek", "openai"])
    parser.add_argument("--model", default="")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--flush-every", type=int, default=25)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--pending-limit", type=int, default=None)
    args = parser.parse_args()
    rebuild_cache(
        nws_path=args.nws_path,
        output_path=args.output,
        provider=args.provider,
        model=args.model,
        max_workers=args.max_workers,
        flush_every=args.flush_every,
        limit=args.limit,
        pending_limit=args.pending_limit,
    )


if __name__ == "__main__":
    main()
