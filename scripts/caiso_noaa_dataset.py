from __future__ import annotations

import argparse
import csv
import datetime as dt
import gzip
import json
import re
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, TextIO
from zoneinfo import ZoneInfo


PACIFIC_TZ = ZoneInfo("America/Los_Angeles")
UTC = dt.timezone.utc

OASIS_BASE_URL = "https://oasis.caiso.com/oasisapi/SingleZip"
IEM_AFOS_URL = "https://mesonet.agron.iastate.edu/cgi-bin/afos/retrieve.py"
STORM_EVENTS_BASE_URL = "https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/"
TODAYS_OUTLOOK_HISTORY_BASE_URL = "https://www.caiso.com/outlook/history"

DEFAULT_CONFIG: dict[str, Any] = {
    "time_range": {
        "start_date": "2024-07-01",
        "end_date": "2024-07-08",
        "date_semantics": "start inclusive, end exclusive, California local dates",
    },
    "region": {
        "name": "CAISO SP15 / Southern California",
        "caiso_node": "TH_SP15_GEN-APND",
        "solar_trading_hub": "SP15",
        "weather_point": {
            "name": "Los Angeles downtown proxy",
            "latitude": 34.0522,
            "longitude": -118.2437,
        },
        "nws_afos_pils": ["AFDLOX"],
        "storm_event_state": "CALIFORNIA",
        "storm_event_counties": [
            "LOS ANGELES",
            "ORANGE",
            "RIVERSIDE",
            "SAN BERNARDINO",
            "SAN DIEGO",
            "IMPERIAL",
            "VENTURA",
            "SANTA BARBARA",
            "SAN LUIS OBISPO",
            "KERN",
        ],
    },
    "datasets": {
        "caiso_oasis": {
            "enabled": True,
            "items": ["lmp_da", "lmp_rt_5min", "solar_dam", "solar_rtd", "solar_rtpd"],
        },
        "caiso_todays_outlook": {
            "enabled": True,
            "items": ["fuelsource", "demand", "netdemand", "wind_solar_wo_hybrids"],
        },
        "nws_text": {"enabled": True},
        "storm_events": {"enabled": True},
        "hrrr_zarr_point": {
            "enabled": True,
            "cycle_hours_utc": [12],
            "forecast_hours": list(range(1, 19)),
            "variables": [
                {"name": "tmp_2m", "level": "2m_above_ground", "var": "TMP"},
                {"name": "dpt_2m", "level": "2m_above_ground", "var": "DPT"},
                {"name": "ugrd_10m", "level": "10m_above_ground", "var": "UGRD"},
                {"name": "vgrd_10m", "level": "10m_above_ground", "var": "VGRD"},
                {"name": "gust_surface", "level": "surface", "var": "GUST"},
                {"name": "tcdc_entire_atmosphere", "level": "entire_atmosphere", "var": "TCDC"},
                {"name": "prate_surface", "level": "surface", "var": "PRATE"},
            ],
        },
    },
    "request": {
        "timeout_seconds": 120,
        "sleep_seconds": 0.2,
        "retries": 3,
        "user_agent": "scireport-caiso-noaa-downloader/0.1 (research data collection)",
    },
}

OASIS_DATASETS: dict[str, dict[str, Any]] = {
    "lmp_da": {
        "queryname": "PRC_LMP",
        "market_run_id": "DAM",
        "version": 12,
        "node_param": True,
        "description": "CAISO day-ahead hourly LMP at configured node",
    },
    "lmp_rt_5min": {
        "queryname": "PRC_INTVL_LMP",
        "market_run_id": "RTM",
        "version": 3,
        "node_param": True,
        "description": "CAISO real-time 5-minute LMP at configured node",
    },
    "solar_dam": {
        "queryname": "SLD_REN_FCST",
        "market_run_id": "DAM",
        "version": 1,
        "node_param": False,
        "description": "CAISO wind and solar day-ahead forecast",
    },
    "solar_rtd": {
        "queryname": "SLD_REN_FCST",
        "market_run_id": "RTD",
        "version": 1,
        "node_param": False,
        "description": "CAISO wind and solar real-time dispatch / actual stream",
    },
    "solar_rtpd": {
        "queryname": "SLD_REN_FCST",
        "market_run_id": "RTPD",
        "version": 1,
        "node_param": False,
        "description": "CAISO wind and solar real-time pre-dispatch forecast",
    },
}

TODAYS_OUTLOOK_DATASETS: dict[str, dict[str, str]] = {
    "fuelsource": {
        "file_name": "fuelsource.csv",
        "description": "CAISO Today's Outlook actual supply by fuel source, including solar generation",
    },
    "demand": {
        "file_name": "demand.csv",
        "description": "CAISO Today's Outlook actual demand with day-ahead and hour-ahead forecasts",
    },
    "netdemand": {
        "file_name": "netdemand.csv",
        "description": "CAISO Today's Outlook actual net demand with forecasts",
    },
    "wind_solar_wo_hybrids": {
        "file_name": "wind_solar_wo_hybrids.csv",
        "description": "CAISO Today's Outlook wind and solar without hybrids",
    },
}

HRRR_FIELDNAMES = [
    "run_time_utc",
    "forecast_hour",
    "valid_time_utc",
    "valid_time_local",
    "point_name",
    "latitude",
    "longitude",
    "hrrr_y_index",
    "hrrr_x_index",
    "variable_name",
    "hrrr_level",
    "hrrr_var",
    "value",
    "units",
    "source_s3",
    "status",
    "note",
]


@dataclass(frozen=True)
class DownloadResult:
    dataset: str
    local_date: str
    start_utc: str
    end_utc: str
    region: str
    path: str
    url: str
    status: str
    bytes: int
    note: str = ""


class Manifest:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(
            self._file,
            fieldnames=[
                "dataset",
                "local_date",
                "start_utc",
                "end_utc",
                "region",
                "path",
                "url",
                "status",
                "bytes",
                "note",
            ],
        )
        self._writer.writeheader()

    def add(self, result: DownloadResult) -> None:
        self._writer.writerow(result.__dict__)
        self._file.flush()

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> "Manifest":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()


class ProgressReporter:
    def __init__(self, total: int, stream: TextIO | None = None, enabled: bool = True) -> None:
        self.total = max(total, 0)
        self.current = 0
        self.stream = stream if stream is not None else sys.stdout
        self.enabled = enabled

    def advance(self, category: str, detail: str, status: str) -> None:
        if not self.enabled:
            return
        self.current += 1
        percent = 100.0 if self.total == 0 else min(100.0, self.current * 100.0 / self.total)
        timestamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(
            f"{timestamp} [{percent:5.1f}%] {self.current}/{self.total} {category}: {detail} -> {status}",
            file=self.stream,
            flush=True,
        )


def parse_date(value: str) -> dt.date:
    return dt.date.fromisoformat(value)


def daily_windows(start_date: dt.date, end_date: dt.date) -> Iterable[tuple[dt.date, dt.date]]:
    if end_date <= start_date:
        raise ValueError("end_date must be after start_date")

    cursor = start_date
    while cursor < end_date:
        next_day = cursor + dt.timedelta(days=1)
        yield cursor, next_day
        cursor = next_day


def count_days(start_date: dt.date, end_date: dt.date) -> int:
    return (end_date - start_date).days


def dated_csv_files(root: Path, start_date: dt.date, end_date: dt.date) -> list[Path]:
    files: list[Path] = []
    if not root.exists():
        return files
    for csv_file in sorted(root.glob("*.csv")):
        try:
            day = dt.date.fromisoformat(csv_file.stem)
        except ValueError:
            continue
        if start_date <= day < end_date:
            files.append(csv_file)
    return files


def hrrr_part_path(
    output_dir: Path,
    start_date: dt.date,
    end_date: dt.date,
    run_date: dt.date,
    cycle_hour: int,
) -> Path:
    return (
        output_dir
        / "processed"
        / f"hrrr_zarr_point_parts_{start_date}_{end_date}"
        / f"{run_date:%Y%m%d}_{cycle_hour:02d}z.csv"
    )


def configured_weather_points(config: dict[str, Any]) -> list[dict[str, Any]]:
    region = config.get("region", {})
    points = region.get("weather_points")
    if points:
        return [normalize_weather_point(point) for point in points]
    point = region.get("weather_point")
    if not point:
        raise ValueError("region.weather_point or region.weather_points is required for HRRR extraction")
    return [normalize_weather_point(point)]


def normalize_weather_point(point: dict[str, Any]) -> dict[str, Any]:
    latitude = float(point["latitude"])
    longitude = float(point["longitude"])
    name = str(point.get("name") or f"{latitude:.4f}_{longitude:.4f}")
    return {"name": name, "latitude": latitude, "longitude": longitude}


def combine_csv_parts(
    part_paths: Iterable[Path],
    out_path: Path,
    fieldnames: list[str],
    *,
    status_field: str | None = None,
) -> tuple[int, int]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    errors = 0
    with out_path.open("w", newline="", encoding="utf-8") as out_handle:
        writer = csv.DictWriter(out_handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for part_path in sorted(part_paths):
            if not part_path.exists() or part_path.stat().st_size == 0:
                continue
            with part_path.open("r", newline="", encoding="utf-8-sig", errors="replace") as in_handle:
                reader = csv.DictReader(in_handle)
                for row in reader:
                    writer.writerow(row)
                    rows_written += 1
                    if status_field and row.get(status_field) == "error":
                        errors += 1
    return rows_written, errors


def count_progress_steps(config: dict[str, Any], *, skip_hrrr: bool) -> int:
    start_date = parse_date(config["time_range"]["start_date"])
    end_date = parse_date(config["time_range"]["end_date"])
    days = count_days(start_date, end_date)
    steps = 0

    caiso_oasis = config["datasets"].get("caiso_oasis", {})
    if caiso_oasis.get("enabled"):
        items = caiso_oasis.get("items", [])
        steps += days * len(items)
        steps += len(items)

    todays_outlook = config["datasets"].get("caiso_todays_outlook", {})
    if todays_outlook.get("enabled"):
        items = todays_outlook.get("items", [])
        steps += days * len(items)
        steps += len(items)

    if config["datasets"].get("nws_text", {}).get("enabled"):
        steps += len(config["region"].get("nws_afos_pils", []))

    if config["datasets"].get("storm_events", {}).get("enabled"):
        years = range(start_date.year, (end_date - dt.timedelta(days=1)).year + 1)
        steps += len(list(years))
        steps += 1

    if config["datasets"].get("hrrr_zarr_point", {}).get("enabled") and not skip_hrrr:
        steps += len(list(iter_hrrr_run_specs(config)))
        steps += 1

    return steps


def california_local_day_bounds_utc(day: dt.date) -> tuple[dt.datetime, dt.datetime]:
    start_local = dt.datetime.combine(day, dt.time.min, tzinfo=PACIFIC_TZ)
    end_local = start_local + dt.timedelta(days=1)
    return start_local.astimezone(UTC), end_local.astimezone(UTC)


def format_oasis_datetime(value: dt.datetime) -> str:
    value_utc = value.astimezone(UTC)
    return value_utc.strftime("%Y%m%dT%H:%M-0000")


def build_oasis_url(
    *,
    queryname: str,
    market_run_id: str,
    version: int,
    start_utc: dt.datetime,
    end_utc: dt.datetime,
    node: str | None = None,
    extra_params: dict[str, str] | None = None,
) -> str:
    params: dict[str, str] = {
        "resultformat": "6",
        "queryname": queryname,
        "version": str(version),
        "market_run_id": market_run_id,
        "startdatetime": format_oasis_datetime(start_utc),
        "enddatetime": format_oasis_datetime(end_utc),
    }
    if node:
        params["node"] = node
    if extra_params:
        params.update(extra_params)
    return f"{OASIS_BASE_URL}?{urllib.parse.urlencode(params)}"


def build_iem_afos_url(pil: str, start_date: dt.date, end_date: dt.date) -> str:
    params = {
        "limit": "9999",
        "pil": pil,
        "fmt": "text",
        "order": "asc",
        "sdate": start_date.isoformat(),
        "edate": end_date.isoformat(),
    }
    return f"{IEM_AFOS_URL}?{urllib.parse.urlencode(params)}"


def build_todays_outlook_history_url(item: str, day: dt.date) -> str:
    spec = TODAYS_OUTLOOK_DATASETS[item]
    return f"{TODAYS_OUTLOOK_HISTORY_BASE_URL}/{day:%Y%m%d}/{spec['file_name']}"


def pick_latest_storm_events_file(directory_html: str, year: int) -> str:
    pattern = re.compile(rf"StormEvents_details-ftp_v1\.0_d{year}_c(\d+)\.csv\.gz")
    matches = [(m.group(1), m.group(0)) for m in pattern.finditer(directory_html)]
    if not matches:
        raise ValueError(f"No StormEvents details file found for {year}")
    return sorted(matches)[-1][1]


def request_url(
    url: str,
    *,
    timeout: int,
    user_agent: str,
    retries: int,
    sleep_seconds: float,
) -> bytes:
    headers = {"User-Agent": user_agent}
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as response:
                return response.read()
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(sleep_seconds * attempt)
    assert last_error is not None
    raise last_error


def download_url_to_file(
    url: str,
    path: Path,
    *,
    request_config: dict[str, Any],
    force: bool,
) -> tuple[str, int]:
    if path.exists() and path.stat().st_size > 0 and not force:
        return "exists", path.stat().st_size

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = request_url(
        url,
        timeout=int(request_config["timeout_seconds"]),
        user_agent=str(request_config["user_agent"]),
        retries=int(request_config["retries"]),
        sleep_seconds=float(request_config["sleep_seconds"]),
    )
    path.write_bytes(payload)
    return "downloaded", len(payload)


def extract_zip_if_valid(zip_path: Path, out_dir: Path) -> str:
    if not zipfile.is_zipfile(zip_path):
        return "not_zip"
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(out_dir)
    return "extracted"


def load_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return json.loads(json.dumps(DEFAULT_CONFIG))
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    return deep_merge(json.loads(json.dumps(DEFAULT_CONFIG)), config)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def write_effective_config(config: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "effective_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, ensure_ascii=True)


def apply_runtime_overrides(config: dict[str, Any], args: argparse.Namespace) -> None:
    if getattr(args, "start_date", None):
        config["time_range"]["start_date"] = args.start_date
    if getattr(args, "end_date", None):
        config["time_range"]["end_date"] = args.end_date
    only = getattr(args, "only", None)
    if only:
        enabled = set(only)
        for dataset_name, dataset_config in config["datasets"].items():
            if isinstance(dataset_config, dict):
                dataset_config["enabled"] = dataset_name in enabled


def download_caiso_oasis(
    config: dict[str, Any],
    output_dir: Path,
    manifest: Manifest,
    force: bool,
    progress: ProgressReporter | None = None,
) -> None:
    start_date = parse_date(config["time_range"]["start_date"])
    end_date = parse_date(config["time_range"]["end_date"])
    request_config = config["request"]
    region = config["region"]["name"]
    node = config["region"]["caiso_node"]
    enabled_items = config["datasets"]["caiso_oasis"]["items"]

    for day, _ in daily_windows(start_date, end_date):
        start_utc, end_utc = california_local_day_bounds_utc(day)
        for item in enabled_items:
            spec = OASIS_DATASETS[item]
            url = build_oasis_url(
                queryname=spec["queryname"],
                market_run_id=spec["market_run_id"],
                version=spec["version"],
                start_utc=start_utc,
                end_utc=end_utc,
                node=node if spec["node_param"] else None,
            )
            raw_path = output_dir / "raw" / "caiso_oasis" / item / f"{day.isoformat()}.zip"
            try:
                status, size = download_url_to_file(
                    url,
                    raw_path,
                    request_config=request_config,
                    force=force,
                )
                extract_status = extract_zip_if_valid(
                    raw_path,
                    output_dir / "raw" / "caiso_oasis" / item / day.isoformat(),
                )
                note = spec["description"] if extract_status == "extracted" else f"{spec['description']}; {extract_status}"
                manifest.add(
                    DownloadResult(
                        dataset=f"caiso_oasis/{item}",
                        local_date=day.isoformat(),
                        start_utc=start_utc.isoformat(),
                        end_utc=end_utc.isoformat(),
                        region=region,
                        path=str(raw_path),
                        url=url,
                        status=status,
                        bytes=size,
                        note=note,
                    )
                )
                if progress:
                    progress.advance("CAISO OASIS", f"{day.isoformat()} {item}", status)
            except Exception as exc:
                manifest.add(
                    DownloadResult(
                        dataset=f"caiso_oasis/{item}",
                        local_date=day.isoformat(),
                        start_utc=start_utc.isoformat(),
                        end_utc=end_utc.isoformat(),
                        region=region,
                        path=str(raw_path),
                        url=url,
                        status="error",
                        bytes=0,
                        note=repr(exc),
                    )
                )
                if progress:
                    progress.advance("CAISO OASIS", f"{day.isoformat()} {item}", "error")


def caiso_row_matches_region(item: str, row: dict[str, str], solar_trading_hub: str) -> bool:
    if item.startswith("solar_") and solar_trading_hub.upper() != "ALL":
        return row.get("TRADING_HUB", "").upper() == solar_trading_hub.upper()
    return True


def safe_token(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def write_processed_caiso_files(
    config: dict[str, Any],
    output_dir: Path,
    manifest: Manifest,
    progress: ProgressReporter | None = None,
) -> None:
    start_date = parse_date(config["time_range"]["start_date"])
    end_date = parse_date(config["time_range"]["end_date"])
    region = config["region"]["name"]
    node = config["region"]["caiso_node"]
    solar_trading_hub = config["region"].get("solar_trading_hub", "ALL")
    enabled_items = config["datasets"]["caiso_oasis"]["items"]

    for item in enabled_items:
        csv_files: list[Path] = []
        for csv_file in sorted((output_dir / "raw" / "caiso_oasis" / item).rglob("*.csv")):
            try:
                day = dt.date.fromisoformat(csv_file.parent.name)
            except ValueError:
                continue
            if start_date <= day < end_date:
                csv_files.append(csv_file)
        fieldnames: list[str] | None = None
        for csv_file in csv_files:
            with csv_file.open("r", newline="", encoding="utf-8-sig", errors="replace") as handle:
                reader = csv.DictReader(handle)
                if reader.fieldnames:
                    fieldnames = list(reader.fieldnames) + ["source_file"]
                    break

        location_token = solar_trading_hub if item.startswith("solar_") else node
        out_path = (
            output_dir
            / "processed"
            / f"caiso_{item}_{safe_token(location_token)}_{start_date}_{end_date}.csv"
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        rows_written = 0

        if fieldnames is None:
            out_path.write_text("", encoding="utf-8")
        else:
            with out_path.open("w", newline="", encoding="utf-8") as out_handle:
                writer = csv.DictWriter(out_handle, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                for csv_file in csv_files:
                    with csv_file.open("r", newline="", encoding="utf-8-sig", errors="replace") as in_handle:
                        reader = csv.DictReader(in_handle)
                        for row in reader:
                            if not caiso_row_matches_region(item, row, solar_trading_hub):
                                continue
                            row["source_file"] = str(csv_file)
                            writer.writerow(row)
                            rows_written += 1

        manifest.add(
            DownloadResult(
                dataset=f"processed/caiso_oasis/{item}",
                local_date=f"{start_date}_{end_date}",
                start_utc="",
                end_utc="",
                region=region,
                path=str(out_path),
                url="",
                status="ok",
                bytes=out_path.stat().st_size if out_path.exists() else 0,
                note=f"rows={rows_written}; location={location_token}",
            )
        )
        if progress:
            progress.advance("Process CAISO OASIS", item, "ok")


def download_caiso_todays_outlook(
    config: dict[str, Any],
    output_dir: Path,
    manifest: Manifest,
    force: bool,
    progress: ProgressReporter | None = None,
) -> None:
    start_date = parse_date(config["time_range"]["start_date"])
    end_date = parse_date(config["time_range"]["end_date"])
    request_config = config["request"]
    region = config["region"]["name"]
    enabled_items = config["datasets"]["caiso_todays_outlook"]["items"]

    for day, _ in daily_windows(start_date, end_date):
        start_utc, end_utc = california_local_day_bounds_utc(day)
        for item in enabled_items:
            spec = TODAYS_OUTLOOK_DATASETS[item]
            url = build_todays_outlook_history_url(item, day)
            raw_path = output_dir / "raw" / "caiso_todays_outlook" / item / f"{day.isoformat()}.csv"
            try:
                status, size = download_url_to_file(
                    url,
                    raw_path,
                    request_config=request_config,
                    force=force,
                )
                manifest.add(
                    DownloadResult(
                        dataset=f"caiso_todays_outlook/{item}",
                        local_date=day.isoformat(),
                        start_utc=start_utc.isoformat(),
                        end_utc=end_utc.isoformat(),
                        region=region,
                        path=str(raw_path),
                        url=url,
                        status=status,
                        bytes=size,
                        note=spec["description"],
                    )
                )
                if progress:
                    progress.advance("CAISO Today's Outlook", f"{day.isoformat()} {item}", status)
            except Exception as exc:
                manifest.add(
                    DownloadResult(
                        dataset=f"caiso_todays_outlook/{item}",
                        local_date=day.isoformat(),
                        start_utc=start_utc.isoformat(),
                        end_utc=end_utc.isoformat(),
                        region=region,
                        path=str(raw_path),
                        url=url,
                        status="error",
                        bytes=0,
                        note=repr(exc),
                    )
                )
                if progress:
                    progress.advance("CAISO Today's Outlook", f"{day.isoformat()} {item}", "error")


def todays_outlook_interval_start(day: dt.date, interval_index: int) -> dt.datetime:
    start_local = dt.datetime.combine(day, dt.time.min, tzinfo=PACIFIC_TZ)
    return start_local + dt.timedelta(minutes=5 * interval_index)


def write_processed_todays_outlook_files(
    config: dict[str, Any],
    output_dir: Path,
    manifest: Manifest,
    progress: ProgressReporter | None = None,
) -> None:
    start_date = parse_date(config["time_range"]["start_date"])
    end_date = parse_date(config["time_range"]["end_date"])
    region = config["region"]["name"]
    enabled_items = config["datasets"]["caiso_todays_outlook"]["items"]

    for item in enabled_items:
        csv_files = dated_csv_files(output_dir / "raw" / "caiso_todays_outlook" / item, start_date, end_date)
        fieldnames: list[str] | None = None
        for csv_file in csv_files:
            if csv_file.stat().st_size == 0:
                continue
            with csv_file.open("r", newline="", encoding="utf-8-sig", errors="replace") as handle:
                reader = csv.DictReader(handle)
                if reader.fieldnames:
                    fieldnames = [
                        "local_date",
                        "interval_index",
                        "interval_start_local",
                        "interval_start_utc",
                        *reader.fieldnames,
                        "source_file",
                    ]
                    break

        out_path = output_dir / "processed" / f"caiso_todays_outlook_{item}_{start_date}_{end_date}.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        rows_written = 0

        if fieldnames is None:
            out_path.write_text("", encoding="utf-8")
        else:
            with out_path.open("w", newline="", encoding="utf-8") as out_handle:
                writer = csv.DictWriter(out_handle, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                for csv_file in csv_files:
                    try:
                        day = dt.date.fromisoformat(csv_file.stem)
                    except ValueError:
                        continue
                    with csv_file.open("r", newline="", encoding="utf-8-sig", errors="replace") as in_handle:
                        reader = csv.DictReader(in_handle)
                        for interval_index, row in enumerate(reader):
                            interval_local = todays_outlook_interval_start(day, interval_index)
                            row["local_date"] = day.isoformat()
                            row["interval_index"] = str(interval_index)
                            row["interval_start_local"] = interval_local.isoformat()
                            row["interval_start_utc"] = interval_local.astimezone(UTC).isoformat()
                            row["source_file"] = str(csv_file)
                            writer.writerow(row)
                            rows_written += 1

        manifest.add(
            DownloadResult(
                dataset=f"processed/caiso_todays_outlook/{item}",
                local_date=f"{start_date}_{end_date}",
                start_utc="",
                end_utc="",
                region=region,
                path=str(out_path),
                url="",
                status="ok",
                bytes=out_path.stat().st_size if out_path.exists() else 0,
                note=f"rows={rows_written}",
            )
        )
        if progress:
            progress.advance("Process Today's Outlook", item, "ok")


def download_nws_text(
    config: dict[str, Any],
    output_dir: Path,
    manifest: Manifest,
    force: bool,
    progress: ProgressReporter | None = None,
) -> None:
    start_date = parse_date(config["time_range"]["start_date"])
    end_date = parse_date(config["time_range"]["end_date"])
    request_config = config["request"]
    region = config["region"]["name"]

    for pil in config["region"]["nws_afos_pils"]:
        url = build_iem_afos_url(pil, start_date, end_date)
        raw_path = output_dir / "raw" / "nws_text" / f"{pil}_{start_date}_{end_date}.txt"
        try:
            status, size = download_url_to_file(
                url,
                raw_path,
                request_config=request_config,
                force=force,
            )
            manifest.add(
                DownloadResult(
                    dataset=f"nws_text/{pil}",
                    local_date=f"{start_date}_{end_date}",
                    start_utc="",
                    end_utc="",
                    region=region,
                    path=str(raw_path),
                    url=url,
                    status=status,
                    bytes=size,
                    note="IEM AFOS NWS text archive",
                )
            )
            if progress:
                progress.advance("NWS text", pil, status)
        except Exception as exc:
            manifest.add(
                DownloadResult(
                    dataset=f"nws_text/{pil}",
                    local_date=f"{start_date}_{end_date}",
                    start_utc="",
                    end_utc="",
                    region=region,
                    path=str(raw_path),
                    url=url,
                    status="error",
                    bytes=0,
                    note=repr(exc),
                )
            )
            if progress:
                progress.advance("NWS text", pil, "error")


def fetch_storm_directory_html(config: dict[str, Any], output_dir: Path, force: bool) -> str:
    listing_path = output_dir / "raw" / "noaa_storm_events" / "csvfiles_listing.html"
    status, _ = download_url_to_file(
        STORM_EVENTS_BASE_URL,
        listing_path,
        request_config=config["request"],
        force=force,
    )
    if status in {"downloaded", "exists"}:
        return listing_path.read_text(encoding="utf-8", errors="replace")
    raise RuntimeError("Could not download Storm Events directory listing")


def storm_event_begin_date(row: dict[str, str]) -> dt.date | None:
    try:
        year_month = int(row.get("BEGIN_YEARMONTH", ""))
        day = int(row.get("BEGIN_DAY", ""))
        return dt.date(year_month // 100, year_month % 100, day)
    except (TypeError, ValueError):
        return None


def row_matches_storm_region(row: dict[str, str], state: str, counties: set[str]) -> bool:
    if row.get("STATE", "").upper() != state.upper():
        return False
    if not counties:
        return True
    cz_name = row.get("CZ_NAME", "").upper().strip()
    return cz_name in counties


def filter_storm_events(
    gz_path: Path,
    *,
    start_date: dt.date,
    end_date: dt.date,
    state: str,
    counties: set[str],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with gzip.open(gz_path, "rt", newline="", encoding="utf-8", errors="replace") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            begin_date = storm_event_begin_date(row)
            if begin_date is None or not (start_date <= begin_date < end_date):
                continue
            if row_matches_storm_region(row, state, counties):
                rows.append(row)
    return rows


def write_storm_labels(rows: list[dict[str, str]], start_date: dt.date, end_date: dt.date, path: Path) -> None:
    events_by_day: dict[dt.date, set[str]] = {}
    for row in rows:
        begin_date = storm_event_begin_date(row)
        if begin_date is None:
            continue
        events_by_day.setdefault(begin_date, set()).add(row.get("EVENT_TYPE", "").strip())

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["date", "has_extreme_event", "event_types"])
        writer.writeheader()
        for day, _ in daily_windows(start_date, end_date):
            event_types = sorted(t for t in events_by_day.get(day, set()) if t)
            writer.writerow(
                {
                    "date": day.isoformat(),
                    "has_extreme_event": "1" if event_types else "0",
                    "event_types": "|".join(event_types),
                }
            )


def download_storm_events(
    config: dict[str, Any],
    output_dir: Path,
    manifest: Manifest,
    force: bool,
    progress: ProgressReporter | None = None,
) -> None:
    start_date = parse_date(config["time_range"]["start_date"])
    end_date = parse_date(config["time_range"]["end_date"])
    region = config["region"]["name"]
    state = config["region"]["storm_event_state"]
    counties = {county.upper() for county in config["region"].get("storm_event_counties", [])}
    directory_html = fetch_storm_directory_html(config, output_dir, force)
    all_rows: list[dict[str, str]] = []

    for year in range(start_date.year, (end_date - dt.timedelta(days=1)).year + 1):
        try:
            file_name = pick_latest_storm_events_file(directory_html, year)
            url = urllib.parse.urljoin(STORM_EVENTS_BASE_URL, file_name)
            gz_path = output_dir / "raw" / "noaa_storm_events" / file_name
            status, size = download_url_to_file(
                url,
                gz_path,
                request_config=config["request"],
                force=force,
            )
            rows = filter_storm_events(
                gz_path,
                start_date=start_date,
                end_date=end_date,
                state=state,
                counties=counties,
            )
            all_rows.extend(rows)
            manifest.add(
                DownloadResult(
                    dataset="noaa_storm_events/details",
                    local_date=str(year),
                    start_utc="",
                    end_utc="",
                    region=region,
                    path=str(gz_path),
                    url=url,
                    status=status,
                    bytes=size,
                    note=f"filtered_rows={len(rows)}; state={state}; counties={len(counties)}",
                )
            )
            if progress:
                progress.advance("NOAA Storm Events", str(year), status)
        except Exception as exc:
            manifest.add(
                DownloadResult(
                    dataset="noaa_storm_events/details",
                    local_date=str(year),
                    start_utc="",
                    end_utc="",
                    region=region,
                    path="",
                    url=STORM_EVENTS_BASE_URL,
                    status="error",
                    bytes=0,
                    note=repr(exc),
                )
            )
            if progress:
                progress.advance("NOAA Storm Events", str(year), "error")

    filtered_path = output_dir / "processed" / f"storm_events_{start_date}_{end_date}.csv"
    filtered_path.parent.mkdir(parents=True, exist_ok=True)
    if all_rows:
        with filtered_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)
    else:
        filtered_path.write_text("", encoding="utf-8")

    labels_path = output_dir / "processed" / f"extreme_weather_labels_{start_date}_{end_date}.csv"
    write_storm_labels(all_rows, start_date, end_date, labels_path)
    if progress:
        progress.advance("Process Storm Events", f"{start_date}_{end_date}", "ok")


def hrrr_requires() -> None:
    missing: list[str] = []
    for module in ["numpy", "xarray", "s3fs", "zarr"]:
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    if missing:
        packages = " ".join("zarr" if m == "zarr" else m for m in missing)
        raise RuntimeError(f"Missing HRRR Zarr dependencies: {packages}. Install with: python -m pip install {packages}")


def close_s3_filesystem(fs: object) -> None:
    try:
        from fsspec.asyn import sync

        s3_client = getattr(fs, "s3", None)
        s3_exit = getattr(s3_client, "__aexit__", None)
        if callable(s3_exit):
            sync(getattr(fs, "loop"), s3_exit, None, None, None)
            return
    except Exception:
        pass

    close_session = getattr(fs, "close_session", None)
    if callable(close_session):
        try:
            close_session(getattr(fs, "loop"), getattr(fs, "s3"))
            return
        except Exception:
            pass

    close = getattr(fs, "close", None)
    if callable(close):
        close()


def nearest_hrrr_grid_point(latitude: float, longitude: float) -> tuple[int, int]:
    hrrr_requires()
    import numpy as np
    import s3fs
    import xarray as xr

    fs = s3fs.S3FileSystem(anon=True)
    chunk_index = None
    try:
        chunk_index = xr.open_zarr(s3fs.S3Map("hrrrzarr/grid/HRRR_chunk_index.zarr", s3=fs), consolidated=False)
        latitudes = chunk_index["latitude"].load().values
        longitudes = chunk_index["longitude"].load().values
        lon_scale = np.cos(np.deg2rad(latitude))
        distance = (latitudes - latitude) ** 2 + ((longitudes - longitude) * lon_scale) ** 2
        y_index, x_index = np.unravel_index(np.nanargmin(distance), distance.shape)
        return int(y_index), int(x_index)
    finally:
        if chunk_index is not None:
            chunk_index.close()
        close_s3_filesystem(fs)


def hrrr_valid_time(run_date: dt.date, cycle_hour: int, forecast_hour: int) -> dt.datetime:
    return dt.datetime(run_date.year, run_date.month, run_date.day, cycle_hour, tzinfo=UTC) + dt.timedelta(
        hours=forecast_hour
    )


def iter_hrrr_run_dates(start_date: dt.date, end_date: dt.date) -> Iterable[dt.date]:
    # Include the previous UTC date so lead times that verify early on the first local day are available.
    start_utc_date = (dt.datetime.combine(start_date, dt.time.min, PACIFIC_TZ).astimezone(UTC).date()) - dt.timedelta(
        days=1
    )
    end_utc_date = dt.datetime.combine(end_date, dt.time.min, PACIFIC_TZ).astimezone(UTC).date()
    cursor = start_utc_date
    while cursor <= end_utc_date:
        yield cursor
        cursor += dt.timedelta(days=1)


def iter_hrrr_run_specs(config: dict[str, Any]) -> Iterable[tuple[dt.date, int]]:
    start_date = parse_date(config["time_range"]["start_date"])
    end_date = parse_date(config["time_range"]["end_date"])
    cycle_hours = [int(hour) for hour in config["datasets"]["hrrr_zarr_point"]["cycle_hours_utc"]]
    for run_date in iter_hrrr_run_dates(start_date, end_date):
        for cycle_hour in cycle_hours:
            yield run_date, cycle_hour


def download_hrrr_zarr_point(
    config: dict[str, Any],
    output_dir: Path,
    manifest: Manifest,
    force: bool = False,
    progress: ProgressReporter | None = None,
) -> None:
    hrrr_requires()
    import s3fs
    import xarray as xr

    start_date = parse_date(config["time_range"]["start_date"])
    end_date = parse_date(config["time_range"]["end_date"])
    start_local = dt.datetime.combine(start_date, dt.time.min, PACIFIC_TZ)
    end_local = dt.datetime.combine(end_date, dt.time.min, PACIFIC_TZ)
    region = config["region"]["name"]
    point_specs = configured_weather_points(config)
    indexed_points = []
    for point in point_specs:
        y_index, x_index = nearest_hrrr_grid_point(float(point["latitude"]), float(point["longitude"]))
        indexed_points.append({**point, "hrrr_y_index": y_index, "hrrr_x_index": x_index})
    hrrr_config = config["datasets"]["hrrr_zarr_point"]
    forecast_hours = [int(hour) for hour in hrrr_config["forecast_hours"]]
    variables = hrrr_config["variables"]
    fs = s3fs.S3FileSystem(anon=True)
    out_path = output_dir / "processed" / f"hrrr_zarr_point_{start_date}_{end_date}.csv"
    parts_dir = output_dir / "processed" / f"hrrr_zarr_point_parts_{start_date}_{end_date}"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    parts_dir.mkdir(parents=True, exist_ok=True)

    try:
        for run_date, cycle_hour in iter_hrrr_run_specs(config):
            date_token = run_date.strftime("%Y%m%d")
            run_time = dt.datetime(run_date.year, run_date.month, run_date.day, cycle_hour, tzinfo=UTC)
            part_path = hrrr_part_path(output_dir, start_date, end_date, run_date, cycle_hour)
            if part_path.exists() and part_path.stat().st_size > 0 and not force:
                if progress:
                    progress.advance("HRRR Zarr point", f"{date_token} {cycle_hour:02d}z", "exists")
                continue

            tmp_path = part_path.with_suffix(".tmp")
            chunk_errors = 0
            chunk_rows = 0
            with tmp_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=HRRR_FIELDNAMES)
                writer.writeheader()
                for variable in variables:
                    level = variable["level"]
                    var = variable["var"]
                    name = variable["name"]
                    store_key = f"hrrrzarr/sfc/{date_token}/{date_token}_{cycle_hour:02d}z_fcst.zarr/{level}/{var}/{level}"
                    source_s3_url = f"s3://{store_key}"
                    ds = None
                    try:
                        ds = xr.open_zarr(s3fs.S3Map(store_key, s3=fs), consolidated=False)
                        data = ds[var]
                        dims = list(data.dims)
                        y_dim = next(dim for dim in dims if dim in {"y", "projection_y_coordinate"})
                        x_dim = next(dim for dim in dims if dim in {"x", "projection_x_coordinate"})
                        time_dims = [dim for dim in dims if dim not in {y_dim, x_dim}]
                        time_dim = time_dims[0] if time_dims else None
                        units = str(data.attrs.get("units", ""))

                        for point in indexed_points:
                            point_data = data.isel(
                                {
                                    y_dim: int(point["hrrr_y_index"]),
                                    x_dim: int(point["hrrr_x_index"]),
                                }
                            )
                            loaded_point_data = point_data.load()
                            for forecast_hour in forecast_hours:
                                valid_utc = hrrr_valid_time(run_date, cycle_hour, forecast_hour)
                                valid_local = valid_utc.astimezone(PACIFIC_TZ)
                                if not (start_local <= valid_local < end_local):
                                    continue
                                try:
                                    if time_dim is None:
                                        value_obj = loaded_point_data.item()
                                    else:
                                        value_obj = loaded_point_data.isel({time_dim: forecast_hour - 1}).item()
                                    value = "" if value_obj is None else str(value_obj)
                                    status = "ok"
                                    note = ""
                                except Exception as exc:
                                    value = ""
                                    status = "error"
                                    note = repr(exc)
                                    chunk_errors += 1

                                writer.writerow(
                                    {
                                        "run_time_utc": run_time.isoformat(),
                                        "forecast_hour": forecast_hour,
                                        "valid_time_utc": valid_utc.isoformat(),
                                        "valid_time_local": valid_local.isoformat(),
                                        "point_name": point["name"],
                                        "latitude": point["latitude"],
                                        "longitude": point["longitude"],
                                        "hrrr_y_index": point["hrrr_y_index"],
                                        "hrrr_x_index": point["hrrr_x_index"],
                                        "variable_name": name,
                                        "hrrr_level": level,
                                        "hrrr_var": var,
                                        "value": value,
                                        "units": units,
                                        "source_s3": source_s3_url,
                                        "status": status,
                                        "note": note,
                                    }
                                )
                                chunk_rows += 1
                    except Exception as exc:
                        chunk_errors += 1
                        for point in indexed_points:
                            writer.writerow(
                                {
                                    "run_time_utc": run_time.isoformat(),
                                    "forecast_hour": "",
                                    "valid_time_utc": "",
                                    "valid_time_local": "",
                                    "point_name": point["name"],
                                    "latitude": point["latitude"],
                                    "longitude": point["longitude"],
                                    "hrrr_y_index": point["hrrr_y_index"],
                                    "hrrr_x_index": point["hrrr_x_index"],
                                    "variable_name": name,
                                    "hrrr_level": level,
                                    "hrrr_var": var,
                                    "value": "",
                                    "units": "",
                                    "source_s3": source_s3_url,
                                    "status": "error",
                                    "note": repr(exc),
                                }
                            )
                            chunk_rows += 1
                    finally:
                        if ds is not None:
                            ds.close()

            tmp_path.replace(part_path)
            if progress:
                status = "ok" if chunk_errors == 0 else "partial"
                progress.advance("HRRR Zarr point", f"{date_token} {cycle_hour:02d}z rows={chunk_rows}", status)

        part_paths = sorted(parts_dir.glob("*.csv"))
        rows_written, errors = combine_csv_parts(part_paths, out_path, HRRR_FIELDNAMES, status_field="status")
        if progress:
            progress.advance("Process HRRR parts", f"{start_date}_{end_date}", "ok" if errors == 0 else "partial")
        manifest.add(
            DownloadResult(
                dataset="noaa_hrrr_zarr_point",
                local_date=f"{start_date}_{end_date}",
                start_utc="",
                end_utc="",
                region=region,
                path=str(out_path),
                url="s3://hrrrzarr/",
                status="ok" if errors == 0 else "partial",
                bytes=out_path.stat().st_size if out_path.exists() else 0,
                note=(
                    f"rows={rows_written}; errors={errors}; points={len(indexed_points)}; "
                    f"point_names={','.join(str(point['name']) for point in indexed_points)}; "
                    f"parts={len(part_paths)}"
                ),
            )
        )
    finally:
        close_s3_filesystem(fs)


def copy_default_config(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(DEFAULT_CONFIG, handle, indent=2, ensure_ascii=True)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download unified CAISO + NOAA/NWS research data.")
    parser.add_argument("--config", type=Path, default=Path("config/dataset_config.json"))
    parser.add_argument("--output", type=Path, default=Path("data"))
    parser.add_argument("--start-date", help="Override config start_date, inclusive, as YYYY-MM-DD.")
    parser.add_argument("--end-date", help="Override config end_date, exclusive, as YYYY-MM-DD.")
    parser.add_argument(
        "--only",
        action="append",
        choices=["caiso_oasis", "caiso_todays_outlook", "nws_text", "storm_events", "hrrr_zarr_point"],
        help="Run only the named dataset group. May be passed multiple times.",
    )
    parser.add_argument("--force", action="store_true", help="Re-download existing files.")
    parser.add_argument("--no-progress", action="store_true", help="Disable progress percentage logging.")
    parser.add_argument("--write-default-config", action="store_true", help="Write the default JSON config and exit.")
    parser.add_argument(
        "--skip-hrrr",
        action="store_true",
        help="Skip HRRR Zarr extraction even if enabled in config.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    if args.write_default_config:
        copy_default_config(args.config)
        print(f"Wrote {args.config}")
        return 0

    config_path = args.config if args.config.exists() else None
    config = load_config(config_path)
    apply_runtime_overrides(config, args)
    args.output.mkdir(parents=True, exist_ok=True)
    write_effective_config(config, args.output)
    progress = ProgressReporter(
        count_progress_steps(config, skip_hrrr=args.skip_hrrr),
        enabled=not args.no_progress,
    )

    with Manifest(args.output / "manifest.csv") as manifest:
        if config["datasets"]["caiso_oasis"]["enabled"]:
            download_caiso_oasis(config, args.output, manifest, args.force, progress)
            write_processed_caiso_files(config, args.output, manifest, progress)
        if config["datasets"]["caiso_todays_outlook"]["enabled"]:
            download_caiso_todays_outlook(config, args.output, manifest, args.force, progress)
            write_processed_todays_outlook_files(config, args.output, manifest, progress)
        if config["datasets"]["nws_text"]["enabled"]:
            download_nws_text(config, args.output, manifest, args.force, progress)
        if config["datasets"]["storm_events"]["enabled"]:
            download_storm_events(config, args.output, manifest, args.force, progress)
        if config["datasets"]["hrrr_zarr_point"]["enabled"] and not args.skip_hrrr:
            download_hrrr_zarr_point(config, args.output, manifest, args.force, progress)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
