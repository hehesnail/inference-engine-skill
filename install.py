#!/usr/bin/env python3
"""Install/update the inference-engine Codex skill. Python 3.9+, stdlib only."""
import argparse
import contextlib
import datetime
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tempfile
import uuid

sys.dont_write_bytecode = True
REPOSITORY = "hehesnail/inference-engine-skill"
NAME = "inference-engine"


def native_path(value):
    path = os.path.abspath(os.path.expanduser(str(value)))
    if os.name == "nt" and not path.startswith("\\\\?\\"):
        path = ("\\\\?\\UNC\\" + path[2:]) if path.startswith("\\\\") else "\\\\?\\" + path
    return Path(path)


def safe_relative(value):
    p = PurePosixPath(value)
    if (not value or p.is_absolute() or "\\" in value or ":" in value
            or any(x in ("", ".", "..") for x in value.split("/"))):
        raise ValueError("Unsafe package path: " + repr(value))
    return p


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_plan(root, edition):
    data = json.loads((root / "distribution.json").read_text(encoding="utf-8"))
    if data["schema"] != 1 or data["skill_name"] != NAME:
        raise ValueError("Unsupported distribution format")
    rows = data["editions"][edition]["files"]
    seen = set()
    for row in rows:
        safe_relative(row["path"])
        safe_relative(row["source"])
        key = row["path"].casefold()
        if key in seen:
            raise ValueError("Duplicate/case-colliding package path: " + row["path"])
        seen.add(key)
        if not re.fullmatch(r"[0-9a-f]{64}", row["sha256"]):
            raise ValueError("Invalid SHA256: " + row["path"])
    if "skill.md" not in seen:
        raise ValueError("Package is missing SKILL.md")
    return data, rows


def checked_source(root, row):
    source = root.joinpath(*safe_relative(row["source"]).parts)
    if not source.resolve().is_relative_to(root.resolve()):
        raise ValueError("Source escapes repository: " + row["source"])
    if source.is_symlink() or not source.is_file():
        raise ValueError("Missing or non-regular source: " + row["source"])
    if source.stat().st_size != row["bytes"] or digest(source) != row["sha256"]:
        raise ValueError("Integrity check failed: " + row["source"])
    return source


def matches(destination, rows):
    if not destination.is_dir() or destination.is_symlink():
        return False
    for row in rows:
        p = destination.joinpath(*safe_relative(row["path"]).parts)
        if (p.is_symlink() or not p.is_file()
                or not p.resolve().is_relative_to(destination.resolve())
                or p.stat().st_size != row["bytes"] or digest(p) != row["sha256"]):
            return False
    # A complete installation is not silently downgraded to desktop.
    receipt = destination / ".installation.json"
    if receipt.exists():
        data = json.loads(receipt.read_text(encoding="utf-8"))
        if data.get("file_count") != len(rows):
            return False
    return True


def install(root, edition, skills_dir, update=False, check=False):
    data, rows = read_plan(root, edition)
    skills_dir = native_path(skills_dir)
    dest = skills_dir / NAME
    if dest.is_symlink():
        raise ValueError("Destination is a symlink; choose its intended real parent with --dest")
    for row in rows:
        checked_source(root, row)
    if matches(dest, rows):
        print("Already current: " + str(dest))
        return dest
    if check:
        raise ValueError("Installed skill differs from " + edition + " package, or is absent")
    if dest.exists() and not update:
        raise ValueError("Skill exists. Re-run with --update to back it up and replace it.")
    skills_dir.mkdir(parents=True, exist_ok=True)
    backup = None
    # Stage outside skills/ so Codex never discovers a partial or duplicate skill.
    with tempfile.TemporaryDirectory(prefix=".inference-engine-stage-", dir=str(skills_dir.parent)) as temp:
        staged = native_path(temp) / NAME
        staged.mkdir()
        for row in rows:
            source = checked_source(root, row)
            target = staged.joinpath(*safe_relative(row["path"]).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            target.chmod(row.get("mode", 0o644))
            if digest(target) != row["sha256"]:
                raise ValueError("Copy verification failed: " + row["path"])
        receipt = {
            "repository": REPOSITORY, "edition": edition, "file_count": len(rows),
            "distribution_sha256": digest(root / "distribution.json"),
            "installed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        (staged / ".installation.json").write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if dest.exists():
            backup_root = skills_dir.parent / "skill-backups"
            backup_root.mkdir(parents=True, exist_ok=True)
            stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            backup = backup_root / (NAME + "-" + stamp + "-" + uuid.uuid4().hex[:8])
            dest.rename(backup)
        try:
            staged.rename(dest)
        except BaseException:
            if backup is not None and not dest.exists():
                backup.rename(dest)
            raise
    print("Installed %s: %d files, %.1f MiB at %s" % (
        edition, len(rows), sum(x["bytes"] for x in rows) / 1024**2, dest))
    if backup is not None:
        print("Previous version preserved at: " + str(backup))
    print("The skill is available to Codex on the next turn.")
    return dest


def git(args, cwd=None, input_text=None):
    prefix = ["git", "-c", "core.autocrlf=false", "-c", "core.longpaths=true"]
    if shutil.which("gh"):
        prefix += ["-c", "credential.helper=", "-c", "credential.helper=!gh auth git-credential"]
    env = dict(os.environ, GIT_LFS_SKIP_SMUDGE="1", GIT_TERMINAL_PROMPT="0")
    return subprocess.run(prefix + args, cwd=cwd, input=input_text, text=True,
                          check=True, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def sparse_pattern(path):
    safe_relative(path)
    # gitignore syntax, rooted exact file paths, not directory cones.
    return "/" + re.sub(r"([\\*?\[\]!# ])", r"\\\1", path)


@contextlib.contextmanager
def checkout(repository, ref, edition):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]*", ref) or ".." in ref or "//" in ref:
        raise ValueError("Invalid --ref; use a branch or tag name")
    with tempfile.TemporaryDirectory(prefix="inference-engine-download-") as temp:
        root = native_path(temp) / "repo"
        git(["clone", "--filter=blob:none", "--no-checkout", "--depth=1",
             "--single-branch", "--branch", ref, "--no-tags", repository, str(root)])
        git(["sparse-checkout", "init", "--no-cone"], cwd=root)
        git(["sparse-checkout", "set", "--no-cone", "--stdin"], cwd=root,
            input_text="/distribution.json\n")
        git(["checkout", "--force"], cwd=root)
        _, rows = read_plan(root, edition)
        patterns = ["/distribution.json"] + [sparse_pattern(x["source"]) for x in rows]
        git(["sparse-checkout", "set", "--no-cone", "--stdin"], cwd=root,
            input_text="\n".join(sorted(set(patterns))) + "\n")
        yield root


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edition", choices=("desktop", "complete"), default="desktop")
    parser.add_argument("--dest", type=Path, help="Parent skills directory (default CODEX_HOME/skills)")
    parser.add_argument("--source", type=Path, help="Install from an existing repository checkout")
    parser.add_argument("--repo", default="https://github.com/" + REPOSITORY + ".git")
    parser.add_argument("--ref", default="main", help="Branch or release tag for remote installation")
    parser.add_argument("--update", action="store_true", help="Back up and replace an existing installation")
    parser.add_argument("--check", action="store_true", help="Only verify an existing installation")
    args = parser.parse_args(argv)
    home = native_path(os.environ.get("CODEX_HOME", str(Path.home() / ".codex")))
    destination = args.dest if args.dest is not None else home / "skills"
    source = args.source
    if source is None:
        own_file = globals().get("__file__", "")
        if own_file and not own_file.startswith("<"):
            candidate = native_path(own_file).parent
            if (candidate / "distribution.json").is_file():
                source = candidate
    try:
        if source is not None:
            install(native_path(source), args.edition, destination, args.update, args.check)
        else:
            with checkout(args.repo, args.ref, args.edition) as root:
                install(root, args.edition, destination, args.update, args.check)
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        detail = error.stderr if isinstance(error, subprocess.CalledProcessError) else str(error)
        print("Installation failed: " + str(detail).strip(), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
