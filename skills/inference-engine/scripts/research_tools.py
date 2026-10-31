#!/usr/bin/env python3
"""Search bundled inference references and run pinned CPU knowledge tools."""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
TOOLS = {
    "wiki-query": "kernelwiki/scripts/query.py",
    "wiki-page": "kernelwiki/scripts/get_page.py",
    "wiki-grep": "kernelwiki/scripts/grep_wiki.py",
    "kda-export": "kda/scripts/export_huggingface.py",
}

def corpus_status(library: Path) -> tuple[bool, str]:
    wiki = library / "kernelwiki"
    marker = wiki / "INSTALLATION.json"
    try:
        info = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False, "missing installation manifest"
    if info.get("edition") != "complete":
        return False, "technical pages only; full PR/source/code corpus is not installed"
    files = info.get("required_files")
    if not isinstance(files, list) or not files:
        return False, "missing corpus file inventory"
    missing = [rel for rel in files if not (wiki / rel).is_file()]
    if missing:
        return False, f"incomplete corpus: {len(missing)} files missing; first: {missing[0]}"
    return True, f"complete pinned corpus ({len(files)} files)"

def search(root: Path, arguments: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="research_tools.py find")
    parser.add_argument("query", nargs="*")
    parser.add_argument("--repo", default="")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--include-excluded", action="store_true")
    args = parser.parse_args(arguments)
    if not 1 <= args.limit <= 100:
        parser.error("--limit must be between 1 and 100")
    try:
        rows = json.loads((root / "reference-index.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        parser.error(f"Reference index unavailable: {error}")
    terms = " ".join(args.query).lower().split()
    found = []
    for row in rows:
        if not row["selected"] and not args.include_excluded:
            continue
        if args.repo.lower() not in row["repo"].lower():
            continue
        name = row["path"].rsplit("/SKILL.md", 1)[0].lower()
        haystack = " ".join([row["repo"], name, row.get("description", ""), " ".join(row.get("topics", []))]).lower()
        if not all(term in haystack for term in terms):
            continue
        score = sum(5 if term in name else 1 for term in terms)
        found.append((score, row))
    found.sort(key=lambda item: (-item[0], item[1]["repo"], item[1]["path"]))
    print(f"{len(found)} matching entries; showing {min(args.limit, len(found))}")
    for _, row in found[:args.limit]:
        print(f"\n{row['repo']} @ {row['sha'][:12]}: {row['path']}")
        print(row.get("description", "").strip())
        if row.get("guide"):
            guide = root / row["guide"]
            status = "installed" if guide.is_file() else "not installed in this copy"
            print(f"Guide ({status}): {guide}")
        else:
            print(f"Disposition: {row['reason']}")
        print(f"Source: https://github.com/{row['repo']}/blob/{row['sha']}/{row['path']}")
    return 0

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library-root", type=Path, default=Path(os.environ.get("INFERENCE_SKILL_LIBRARY") or ROOT / "library"))
    parser.add_argument("--list", action="store_true")
    parser.add_argument("tool", choices=["find", *TOOLS], nargs="?")
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    library = args.library_root.expanduser().resolve()
    forwarded = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
    if args.list:
        print("find\tbuilt-in reference index")
        complete, status = corpus_status(library)
        for name, rel in TOOLS.items():
            exists = (library / rel).is_file()
            extra = status if name.startswith("wiki-") else "Python >=3.11; kernel-builder required for export"
            print(f"{name}\t{'script installed' if exists else 'script missing'}\t{extra}")
        return 0
    if args.tool == "find":
        index_root = library.parent if (library.parent / "reference-index.json").is_file() else ROOT
        return search(index_root, forwarded)
    if not args.tool:
        parser.print_help()
        return 0
    script = library / TOOLS[args.tool]
    if not script.is_file():
        parser.error(f"Tool script is not installed: {script}")
    help_only = any(arg in ("--help", "-h") for arg in forwarded)
    if args.tool.startswith("wiki-"):
        if importlib.util.find_spec("yaml") is None:
            parser.error("PyYAML is required in this Python environment; no automatic installation was attempted.")
        if not help_only:
            complete, status = corpus_status(library)
            if not complete:
                parser.error(status + ". Select the complete server library with --library-root or INFERENCE_SKILL_LIBRARY.")
    if args.tool == "kda-export" and sys.version_info < (3, 11):
        parser.error("KDA export requires Python >=3.11 (tomllib).")
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES="", NVIDIA_VISIBLE_DEVICES="void", PYTHONDONTWRITEBYTECODE="1")
    # Bind the chosen library; do not let an unrelated upstream override redirect it.
    if args.tool.startswith("wiki-"):
        env["BLACKWELL_WIKI_ROOT"] = str(library / "kernelwiki")
    return subprocess.run([sys.executable, str(script), *forwarded], env=env, check=False).returncode

if __name__ == "__main__":
    raise SystemExit(main())
