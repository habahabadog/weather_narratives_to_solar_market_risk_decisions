from __future__ import annotations

import argparse
from pathlib import Path
import re


DEFAULT_ROOTS = [
    Path("scripts"),
    Path("tests"),
    Path("config"),
    Path("data"),
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
    "outputs",
    "data_multi_weather_2023_2024",
}
TEXT_SUFFIXES = {".py", ".md", ".txt", ".tex", ".bib", ".json", ".csv", ".yml", ".yaml", ".example", ""}
MAX_GITHUB_FILE_MB = 95.0

SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
    re.compile(r"\b(?:OPENAI|DEEPSEEK|ANTHROPIC|GOOGLE|GEMINI)_API_KEY[^\S\r\n]*=[^\S\r\n]*['\"]?[A-Za-z0-9_\-]{12,}", re.I),
    re.compile(r"\bapi[_-]?key[^\S\r\n]*[:=][^\S\r\n]*['\"][A-Za-z0-9_\-]{24,}['\"]", re.I),
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


def scan_file(path: Path) -> list[str]:
    findings: list[str] = []
    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > MAX_GITHUB_FILE_MB:
        findings.append(f"large file over {MAX_GITHUB_FILE_MB:.0f} MB: {size_mb:.1f} MB")
    if not is_text_candidate(path):
        return findings
    text = path.read_text(encoding="utf-8", errors="ignore")
    for pattern in SECRET_PATTERNS:
        if pattern.search(text):
            findings.append(f"possible secret matched pattern `{pattern.pattern}`")
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
