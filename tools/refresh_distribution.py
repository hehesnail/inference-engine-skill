#!/usr/bin/env python3
"""Refresh/check source hashes and install plans after maintaining skill files."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import stat
import sys
sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[1]
FULL = ROOT / "skills/inference-engine"
OVERLAYS = ROOT / "profiles/desktop"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def encoded(value, compact=False):
    return json.dumps(value, ensure_ascii=False, indent=None if compact else 2,
                      separators=(",", ":") if compact else None) + "\n"


def refresh(check=False, accept_overrides=False):
    original = json.loads((FULL / "source-manifest.json").read_text())
    base_file = ROOT / "profiles/desktop-base.json"
    base_hashes = json.loads(base_file.read_text()) if base_file.exists() else {}
    overrides = {
        p.relative_to(OVERLAYS).as_posix(): p for p in OVERLAYS.rglob("*")
        if p.is_file() and p.name != "source-manifest.json"}
    changed = [p for p in overrides if base_hashes.get(p) not in (None, sha(FULL / p))]
    if changed and not accept_overrides:
        raise ValueError("Desktop overrides need review after base edits: "
                         + ", ".join(changed) + ". Review them, then use --accept-overrides.")
    full = copy.deepcopy(original)
    old_rows = {x["path"]: x for x in original["files"]}
    rows = []
    for p in sorted(FULL.rglob("*")):
        if not p.is_file() or p == FULL / "source-manifest.json":
            continue
        rel = p.relative_to(FULL).as_posix()
        if p.is_symlink() or "__pycache__" in p.parts or ".git" in p.parts:
            raise ValueError("Unexpected generated file or symlink: " + rel)
        row = copy.deepcopy(old_rows.get(rel, {
            "path": rel, "placement": "desktop-and-server", "repository": "local",
            "source_path": None, "adapted": False}))
        current_sha = sha(p)
        if row.get("packaged_sha256", current_sha) != current_sha and row["repository"] != "local":
            row["adapted"] = True
        row.update(bytes=p.stat().st_size, packaged_sha256=current_sha, installed=True)
        rows.append(row)
    full["files"] = rows
    full["edition"] = "complete"
    desktop = copy.deepcopy(full)
    desktop["edition"] = "desktop"
    selected = {x["path"] for x in rows if x["placement"] == "desktop-and-server"}
    if set(overrides) - selected:
        raise ValueError("Orphan desktop overrides: " + str(sorted(set(overrides) - selected)))
    for row in desktop["files"]:
        row["installed"] = row["path"] in selected
        if row["path"] in overrides:
            p = overrides[row["path"]]
            row.update(bytes=p.stat().st_size, packaged_sha256=sha(p), adapted=True)
    products = {
        FULL / "source-manifest.json": encoded(full),
        OVERLAYS / "source-manifest.json": encoded(desktop),
        base_file: encoded({p: sha(FULL / p) for p in sorted(overrides)}),
    }
    distribution = {"schema": 1, "skill_name": "inference-engine",
                    "repository": "hehesnail/inference-engine-skill", "editions": {}}
    for edition, manifest in (("complete", full), ("desktop", desktop)):
        plan = []
        entries = [x for x in manifest["files"] if x["installed"]]
        for item in entries:
            rel = item["path"]
            p = overrides[rel] if edition == "desktop" and rel in overrides else FULL / rel
            plan.append({"path": rel, "source": p.relative_to(ROOT).as_posix(),
                         "bytes": item["bytes"], "sha256": item["packaged_sha256"],
                         "mode": 0o755 if p.stat().st_mode & stat.S_IXUSR else 0o644})
        mp = (FULL if edition == "complete" else OVERLAYS) / "source-manifest.json"
        raw = products[mp].encode("utf-8")
        plan.append({"path": "source-manifest.json", "source": mp.relative_to(ROOT).as_posix(),
                     "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(), "mode": 0o644})
        plan.sort(key=lambda x: x["path"])
        distribution["editions"][edition] = {"files": plan}
    products[ROOT / "distribution.json"] = encoded(distribution, compact=True)
    differences = []
    for path, text in products.items():
        if not path.exists() or path.read_bytes() != text.encode("utf-8"):
            differences.append(path.relative_to(ROOT).as_posix())
            if not check:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(text.encode("utf-8"))
    if check and differences:
        raise ValueError("Stale distribution metadata: " + ", ".join(differences))
    for edition, item in distribution["editions"].items():
        print("%s: %d files, %.1f MiB" % (edition, len(item["files"]),
              sum(x["bytes"] for x in item["files"]) / 1024**2))
    print("Metadata verified." if check else "Metadata refreshed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--accept-overrides", action="store_true")
    args = parser.parse_args()
    try:
        refresh(args.check, args.accept_overrides)
    except (OSError, ValueError) as error:
        raise SystemExit(str(error))
