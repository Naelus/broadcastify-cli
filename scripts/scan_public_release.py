from __future__ import annotations

import argparse
import os
import re
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


FORBIDDEN_EXACT_NAMES = {
    ".env",
    "archive-quota.sqlite3",
    "broadcastify-analysis.sqlite3",
    "broadcastify-desktop.env",
    "broadcastify-desktop.log",
    "cookies.json",
    "credentials.json",
    "direct_url.json",
    "session.json",
    "settings.json",
    "support_handoff.md",
}
FORBIDDEN_SUFFIXES = (
    ".gguf",
    ".log",
    ".pdb",
    ".pt",
    ".safetensors",
    ".sqlite",
    ".sqlite3",
)
SECRET_NAME_PATTERN = re.compile(
    r"(?:API[_-]?KEY|COOKIE|PASSWORD|SECRET|SESSION|TOKEN)$",
    re.IGNORECASE,
)
CHUNK_BYTES = 2 * 1024 * 1024


@dataclass(frozen=True)
class ByteNeedle:
    label: str
    values: tuple[bytes, ...]


def _encoded_values(value: str) -> tuple[bytes, ...]:
    normalized = value.casefold()
    variants = {normalized, normalized.replace("\\", "/")}
    values: set[bytes] = set()
    for variant in variants:
        if not variant:
            continue
        values.add(variant.encode("utf-8"))
        values.add(variant.encode("utf-16-le"))
        values.add(variant.encode("utf-16-be"))
    return tuple(sorted(values))


def _is_generic_github_hosted_profile(
    value: str,
    environment: Mapping[str, str],
) -> bool:
    normalized = value.strip().replace("/", "\\").rstrip("\\").casefold()
    return (
        environment.get("GITHUB_ACTIONS", "").casefold() == "true"
        and environment.get("RUNNER_ENVIRONMENT", "").casefold()
        == "github-hosted"
        and re.fullmatch(r"[a-z]:\\users\\runneradmin", normalized) is not None
    )


def _needles(forbidden_paths: list[str]) -> list[ByteNeedle]:
    results = [
        ByteNeedle(f"build path {index}", _encoded_values(value))
        for index, value in enumerate(forbidden_paths, start=1)
        if value.strip()
    ]
    for name, value in sorted(os.environ.items()):
        if (
            SECRET_NAME_PATTERN.search(name)
            and len(value) >= 8
            and not value.isspace()
        ):
            results.append(
                ByteNeedle(f"environment secret {name}", _encoded_values(value))
            )
    return [value for value in results if value.values]


def _file_matches(path: Path, needles: list[ByteNeedle]) -> set[str]:
    if not needles:
        return set()
    byte_values = [
        (needle.label, value)
        for needle in needles
        for value in needle.values
    ]
    max_length = max(len(value) for _, value in byte_values)
    overlap = b""
    matches: set[str] = set()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_BYTES):
            data = (overlap + chunk).lower()
            for label, value in byte_values:
                if label not in matches and value in data:
                    matches.add(label)
            if len(matches) == len(needles):
                break
            overlap = data[-(max_length - 1) :] if max_length > 1 else b""
    return matches


def scan_public_release(
    root: Path,
    forbidden_paths: list[str],
) -> tuple[int, list[str]]:
    root = root.resolve()
    needles = _needles(forbidden_paths)
    violations: list[str] = []
    files_scanned = 0
    for path in sorted(value for value in root.rglob("*") if value.is_file()):
        files_scanned += 1
        relative = path.relative_to(root).as_posix()
        lowered_name = path.name.casefold()
        if (
            lowered_name in FORBIDDEN_EXACT_NAMES
            or lowered_name.endswith(FORBIDDEN_SUFFIXES)
        ):
            violations.append(f"{relative}: forbidden release file")
            continue
        for label in sorted(_file_matches(path, needles)):
            violations.append(f"{relative}: contains {label}")
    return files_scanned, violations


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Reject secrets, machine paths, and private files in a public stage."
    )
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--forbid-path", action="append", default=[])
    parser.add_argument("--forbid-user-profile")
    arguments = parser.parse_args(argv)
    if not arguments.root.is_dir():
        parser.error(f"release root is not a directory: {arguments.root}")
    forbidden_paths = [str(value) for value in arguments.forbid_path]
    if arguments.forbid_user_profile and not _is_generic_github_hosted_profile(
        arguments.forbid_user_profile,
        os.environ,
    ):
        forbidden_paths.append(arguments.forbid_user_profile)
    files_scanned, violations = scan_public_release(
        arguments.root,
        forbidden_paths,
    )
    if violations:
        print("Public release scan failed:", file=sys.stderr)
        for violation in violations:
            print(f"- {violation}", file=sys.stderr)
        return 1
    print(f"Public release scan passed: {files_scanned} files checked.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
