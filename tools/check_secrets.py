#!/usr/bin/env python3
"""Refuse secrets in the repository (CLAUDE.md 3.2): Hugging Face and GitHub tokens in git-tracked text
files, and tracked env/*.env files (only env/*.env.example may be tracked; .gitignore has env/*.env).

Prints `path:line: kind` per finding and never the matched text. Binary files (NUL byte in the first
8 KiB) and symlinks are skipped. Placeholders such as `hf_...` or `hf_xxx` are shorter than the patterns
and do not match; there is no escape-hatch comment.

python tools/check_secrets.py               # files in the git index (what CI sees)
python tools/check_secrets.py --untracked   # + untracked, not-ignored files (tools/pr_check.sh: before `git add`);
                                            #   an env/*.env among them means .gitignore lost its rule
python tools/check_secrets.py PATH ...      # + these files / directories (outside git is fine)
exit 0 clean, 1 on any finding, 2 when there is nothing to scan (not a git checkout and no PATH) or a PATH is missing
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path

PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("hf_token", re.compile(r"hf_[A-Za-z0-9]{30,}")),
    # ghp_ classic PAT; gho_/ghu_/ghs_/ghr_ are the other GitHub token prefixes with the same 36-char body
    ("github_token", re.compile(r"gh[pousr]_[A-Za-z0-9]{36}")),
    ("github_pat", re.compile(r"github_pat_[A-Za-z0-9_]{50,}")),
)
SNIFF_BYTES = 8192
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", ".venv", "node_modules"}


def is_env_file(rel: str) -> bool:
    """True for <repo>/env/<name>.env (the anchored .gitignore rule env/*.env); env/*.env.example is allowed."""
    parts = rel.replace(os.sep, "/").split("/")
    return len(parts) == 2 and parts[0] == "env" and parts[1].endswith(".env")


def scan_text(text: str) -> Iterator[tuple[int, str]]:
    """(1-based line, kind) for every pattern hit; one entry per kind per line."""
    for i, line in enumerate(text.split("\n"), 1):
        for kind, rx in PATTERNS:
            if rx.search(line):
                yield i, kind


def scan_file(path: Path) -> list[tuple[int, str]] | None:
    """Findings in one file, or None when it is skipped (binary, symlink, missing, unreadable)."""
    try:
        if path.is_symlink() or not path.is_file():
            return None
        data = path.read_bytes()
    except OSError:
        return None
    if b"\0" in data[:SNIFF_BYTES]:
        return None
    return list(scan_text(data.decode("utf-8", errors="replace")))


def git_toplevel(cwd: Path) -> tuple[Path | None, str]:
    """(work-tree root or None, git's complaint when there is none)."""
    try:
        r = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=cwd, capture_output=True, text=True)
    except OSError as e:                                             # git not installed
        return None, str(e)
    if r.returncode == 0 and r.stdout.strip():
        return Path(r.stdout.strip()), ""
    return None, (r.stderr.strip().splitlines() or [""])[0]


def git_files(top: Path, *, untracked: bool = False) -> tuple[list[str], list[str]]:
    """(paths in the index (`git ls-files -z`), untracked not-ignored paths when asked), relative to `top`."""
    def ls(*opts: str) -> list[str]:
        out = subprocess.run(["git", "ls-files", "-z", *opts], cwd=top, capture_output=True, check=True).stdout
        return sorted({n.decode("utf-8", "surrogateescape") for n in out.split(b"\0") if n})
    return ls("--cached"), (ls("--others", "--exclude-standard") if untracked else [])


def walk(path: Path) -> Iterator[Path]:
    if path.is_dir() and not path.is_symlink():
        for dirpath, dirnames, filenames in os.walk(path):
            dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
            for f in sorted(filenames):
                yield Path(dirpath) / f
    else:
        yield path


def check(top: Path | None, tracked: Iterable[str], untracked: Iterable[str] = (),
          extra: Sequence[Path] = ()) -> tuple[list[str], int, int]:
    """(finding lines, files scanned, files skipped). `tracked` / `untracked` are relative to `top`."""
    base = top or Path.cwd()
    findings: list[str] = []
    scanned = skipped = 0
    targets = [(r, base / r, "tracked") for r in tracked] + [(r, base / r, "untracked") for r in untracked]
    targets += [(str(f), f, "arg") for p in extra for f in walk(p)]           # shown as given on the command line
    seen: set[str] = set()
    for shown, path, origin in targets:
        key = os.path.realpath(path)
        if key in seen:
            continue
        seen.add(key)
        if origin == "tracked" and is_env_file(shown):
            findings.append(f"{shown}:0: tracked_env_file (only env/*.env.example may be tracked)")
        elif origin == "untracked" and is_env_file(shown):
            findings.append(f"{shown}:0: env_file_not_ignored (.gitignore must keep env/*.env out of git)")
        hits = scan_file(path)
        if hits is None:
            skipped += 1
            continue
        scanned += 1
        findings += [f"{shown}:{line}: {kind}" for line, kind in hits]
    return findings, scanned, skipped


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("paths", nargs="*", type=Path, help="extra files or directories to scan")
    ap.add_argument("--untracked", action="store_true", help="also scan untracked files that .gitignore does not hide")
    args = ap.parse_args(argv)
    missing = [str(p) for p in args.paths if not p.exists()]
    if missing:
        print(f"check_secrets: no such path: {', '.join(missing)}", file=sys.stderr)
        return 2
    top, why = git_toplevel(Path.cwd())
    if top is None and not args.paths:
        print(f"check_secrets: not inside a git checkout ({why}); pass paths to scan", file=sys.stderr)
        return 2
    tracked, untracked = git_files(top, untracked=args.untracked) if top is not None else ([], [])
    findings, scanned, skipped = check(top, tracked, untracked, args.paths)
    for f in findings:
        print(f)
    print(f"check_secrets: {scanned} files scanned, {skipped} skipped (binary, symlink or missing), "
          f"{len(findings)} findings")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
