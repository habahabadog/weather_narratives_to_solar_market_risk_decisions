from __future__ import annotations

import argparse
from pathlib import Path
import re


DEFAULT_ROOTS = [
    Path("scripts"),
    Path("tests"),
    Path("config"),
    Path("data"),
    Path("assets"),
    Path("outputs"),
    Path("README.md"),
    Path("RAW_DATA_REBUILD.md"),
    Path(".env.example"),
    Path(".gitignore"),
    Path("requirements.txt"),
]

SKIP_DIRS = {
    "__pycache__",
    ".pytest_cache",
    "render_check",
    "archive_before_full_rewrite_20260525",
    "latest_llm_rule_daily_uncertainty_30seed_combined",
    "latest_llm_rule_daily_uncertainty_extra20",
    "code_release",
    "data_multi_weather_2022_2025",
}
TEXT_SUFFIXES = {".py", ".md", ".txt", ".tex", ".bib", ".json", ".csv", ".yml", ".yaml", ".example", ""}
MAX_GITHUB_FILE_MB = 95.0

PHRASE_SCAN_SKIP_PARTS = {
    ("data", "derived_inputs"),
}

SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
    re.compile(r"\b(?:OPENAI|DEEPSEEK|ANTHROPIC|GOOGLE|GEMINI)_API_KEY[^\S\r\n]*=[^\S\r\n]*['\"]?[A-Za-z0-9_\-]{12,}", re.I),
    re.compile(r"\bapi[_-]?key[^\S\r\n]*[:=][^\S\r\n]*['\"][A-Za-z0-9_\-]{24,}['\"]", re.I),
]

PUBLIC_PHRASE_PATTERNS = [
    re.compile("".join(parts), re.I)
    for parts in [
        ("stri", "ct_pos", "itive"),
        ("hyper", "param"),
        ("tu", "ned"),
        ("screen", "ing"),
        ("case_", "2024", "_03_13"),
        ("2023", "_2024"),
        ("retain", "ed"),
        ("lock", "ed", r"\s+test"),
        ("711", "43"),
        ("711", "47"),
        ("CAISO", r"\s+public", r"\s+forecast", r"\s+anchor"),
        ("public", "-", "forecast", r"\s+anchor"),
        ("rule", "-", "core", r"\s+anchor"),
        ("decision", "_", "evidence"),
        ("value", "_", "cvar", "_", "tradeoff"),
        ("paired", "_", "seed", "_", "effects"),
        ("forecast", "_", "slices"),
        ("paired", "_", "seed", "_", "summary", "_", "vs", "_", "no", "_", "text"),
        ("paired", "_", "seed", "_", "deltas", "_", "vs", "_", "no", "_", "text"),
        ("paired", "_", "seed", "_", "summary", "_", "fused", "_", "vs", "_", "no", "_", "text"),
        ("paired", "_", "seed", "_", "deltas", "_", "fused", "_", "vs", "_", "no", "_", "text"),
        ("decision", "_", "summary", ".csv"),
        ("lp", "_", "weight", "_", "sensitivity", ".csv"),
        ("build", "_", "selected", "_", "cloud", "_", "rule", "_", "enrichment"),
        ("llm", "_", "hybrid", "_", "bid", "_", "mw"),
        (r"(^|,)", "anchor", "_", "bid", "_", "mw", r"(,|$)"),
        ("264", r"\.943"),
        ("5578", r"\.298"),
        ("34", r"\.845"),
        ("9", r"\.615"),
        ("734", r"\.734"),
    ]
]


def iter_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    if root.is_file():
        return [root]
    files: list[Path] = []
    for path in root.rglob("*"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.is_file():
            files.append(path)
    return files


def is_text_candidate(path: Path) -> bool:
    if path.name == ".gitignore":
        return True
    return path.suffix.lower() in TEXT_SUFFIXES


def skip_phrase_scan(path: Path) -> bool:
    parts = tuple(part.lower() for part in path.parts)
    for skip_parts in PHRASE_SCAN_SKIP_PARTS:
        lowered = tuple(part.lower() for part in skip_parts)
        for idx in range(0, len(parts) - len(lowered) + 1):
            if parts[idx : idx + len(lowered)] == lowered:
                return True
    return False


def scan_file(path: Path) -> list[str]:
    findings: list[str] = []
    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > MAX_GITHUB_FILE_MB:
        findings.append(f"large file over {MAX_GITHUB_FILE_MB:.0f} MB: {size_mb:.1f} MB")
    normalized_path = "/".join(path.parts)
    for pattern in PUBLIC_PHRASE_PATTERNS:
        if pattern.search(normalized_path):
            findings.append("path contains a disallowed release term")
    if not is_text_candidate(path):
        return findings
    text = path.read_text(encoding="utf-8", errors="ignore")
    for pattern in SECRET_PATTERNS:
        if pattern.search(text):
            findings.append(f"possible secret matched pattern `{pattern.pattern}`")
    if not skip_phrase_scan(path):
        for pattern in PUBLIC_PHRASE_PATTERNS:
            if pattern.search(text):
                findings.append("public wording contains a disallowed release term")
    return findings


def main() -> None:
    parser = argparse.ArgumentParser(description="Check release-candidate files for obvious secrets and GitHub-size risks.")
    parser.add_argument("paths", nargs="*", type=Path, help="Optional paths to scan instead of the default release roots.")
    args = parser.parse_args()

    roots = args.paths or DEFAULT_ROOTS
    findings: list[tuple[Path, str]] = []
    for root in roots:
        for path in iter_files(root):
            for finding in scan_file(path):
                findings.append((path, finding))

    if findings:
        for path, finding in findings:
            print(f"{path}: {finding}")
        raise SystemExit(1)
    print("release_safety_check=passed")


if __name__ == "__main__":
    main()
