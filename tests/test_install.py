import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("installer", ROOT / "install.py")
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)


class InstallTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = installer.native_path(self.temp.name)
        self.repo = self.base / "source repo"
        self.repo.mkdir()
        self.dest = self.base / "home with spaces" / "skills"
        self.make_plan()

    def make_plan(self, version="one"):
        entries = []
        for path, data in {
            "SKILL.md": ("---\nname: inference-engine\n---\n" + version).encode(),
            "references/notes [1].md": "中文说明\n".encode(),
            "corpus/large.txt": b"server-only" * 200000,
        }.items():
            p = self.repo / "skills/inference-engine" / path
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)
            entries.append({"path": path, "source": p.relative_to(self.repo).as_posix(),
                            "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                            "mode": 0o644})
        plan = {"schema": 1, "skill_name": "inference-engine", "editions": {
            "complete": {"files": entries},
            "desktop": {"files": [x for x in entries if not x["path"].startswith("corpus/")]}}}
        plan["schema"] = 2
        plan["companions"] = {}
        for name in installer.COMPANIONS:
            source = self.repo / "skills" / name / "SKILL.md"
            source.parent.mkdir(parents=True, exist_ok=True)
            data = ("---\\nname: " + name + "\\n---\\n").encode()
            source.write_bytes(data)
            plan["companions"][name] = [{"path": "SKILL.md", "source": source.relative_to(self.repo).as_posix(),
                                         "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}]
        (self.repo / "distribution.json").write_text(json.dumps(plan), encoding="utf-8")
        return plan

    def test_desktop_and_idempotent_install(self):
        dest = installer.install(self.repo, "desktop", self.dest)
        self.assertTrue((dest / "references/notes [1].md").is_file())
        self.assertFalse((dest / "corpus").exists())
        for name in installer.COMPANIONS:
            self.assertEqual((self.dest / name / "SKILL.md").read_bytes(),
                             (self.repo / "skills" / name / "SKILL.md").read_bytes())
            self.assertTrue((self.dest / name / ".installation.json").is_file())
        installer.install(self.repo, "desktop", self.dest, update=True)
        self.assertFalse((self.dest.parent / "skill-backups").exists())

    def test_update_preserves_local_edits_and_old_version(self):
        dest = installer.install(self.repo, "desktop", self.dest)
        (dest / "SKILL.md").write_text("local changes", encoding="utf-8")
        (dest / "my-note.md").write_text("keep me", encoding="utf-8")
        self.make_plan("two")
        with self.assertRaisesRegex(ValueError, "exists"):
            installer.install(self.repo, "desktop", self.dest)
        installer.install(self.repo, "desktop", self.dest, update=True)
        backups = list((self.dest.parent / "skill-backups").iterdir())
        self.assertEqual(len(backups), 1)
        self.assertEqual((backups[0] / "SKILL.md").read_text(), "local changes")
        self.assertEqual((backups[0] / "my-note.md").read_text(), "keep me")
        self.assertTrue((dest / "SKILL.md").read_text().endswith("two"))

    def test_corrupt_source_does_not_change_installation(self):
        dest = installer.install(self.repo, "desktop", self.dest)
        before = (dest / "SKILL.md").read_bytes()
        (self.repo / "skills/inference-engine/SKILL.md").write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "Integrity"):
            installer.install(self.repo, "desktop", self.dest, update=True)
        self.assertEqual((dest / "SKILL.md").read_bytes(), before)
        self.assertFalse((self.dest.parent / "skill-backups").exists())

    def test_failed_final_move_restores_previous_version(self):
        dest = installer.install(self.repo, "desktop", self.dest)
        before = (dest / "SKILL.md").read_bytes()
        self.make_plan("two")
        original = Path.rename
        def fail_once(path, target):
            if ".inference-engine-stage-" in str(path):
                raise OSError("simulated rename failure")
            return original(path, target)
        with patch.object(Path, "rename", fail_once):
            with self.assertRaisesRegex(OSError, "simulated"):
                installer.install(self.repo, "desktop", self.dest, update=True)
        self.assertEqual((dest / "SKILL.md").read_bytes(), before)

    def test_update_from_single_skill_and_manual_companion(self):
        path = self.repo / "distribution.json"
        plan = json.loads(path.read_bytes())
        plan["schema"] = 1
        del plan["companions"]
        path.write_text(json.dumps(plan))
        original = installer.install(self.repo, "desktop", self.dest)
        self.make_plan()
        name, missing = installer.COMPANIONS
        shutil.copytree(self.repo / "skills" / name, self.dest / name)
        with self.assertRaisesRegex(ValueError, missing):
            installer.install(self.repo, "desktop", self.dest, check=True)
        self.assertFalse((self.dest / missing).exists())
        installer.install(self.repo, "desktop", self.dest, update=True)
        self.assertFalse((self.dest.parent / "skill-backups").exists())
        for skill in [original, self.dest / name, self.dest / missing]:
            receipt = json.loads((skill / ".installation.json").read_bytes())
            self.assertEqual(receipt["distribution_sha256"], installer.digest(path))

    def test_corrupt_companion_stops_the_whole_bundle(self):
        (self.repo / "skills/pytorch-refactor/SKILL.md").write_bytes(b"truncated")
        with self.assertRaisesRegex(ValueError, "Integrity"):
            installer.install(self.repo, "desktop", self.dest, update=True)
        self.assertFalse(self.dest.exists())

    def test_late_failure_rolls_back_replacement_and_receipt_adoption(self):
        first = installer.install(self.repo, "desktop", self.dest)
        second, third = (self.dest / name for name in installer.COMPANIONS)
        (first / "SKILL.md").write_bytes(b"my edit")
        (second / ".installation.json").unlink()
        shutil.rmtree(third)
        original = Path.rename

        def fail_last(path, target):
            if target == third and ".inference-engine-stage-" in str(path):
                raise OSError("last skill failed")
            return original(path, target)

        with patch.object(Path, "rename", fail_last), self.assertRaisesRegex(OSError, "last skill"):
            installer.install(self.repo, "desktop", self.dest, update=True)
        self.assertEqual((first / "SKILL.md").read_bytes(), b"my edit")
        self.assertFalse((second / ".installation.json").exists())
        self.assertFalse(third.exists())
        self.assertFalse(list((self.dest.parent / "skill-backups").iterdir()))

    def test_traversal_rejected(self):
        plan = json.loads((self.repo / "distribution.json").read_text())
        plan["editions"]["desktop"]["files"][0]["path"] = "../outside"
        (self.repo / "distribution.json").write_text(json.dumps(plan))
        with self.assertRaisesRegex(ValueError, "Unsafe"):
            installer.install(self.repo, "desktop", self.dest)
        self.assertFalse(self.dest.exists())

    @unittest.skipIf(os.name == "nt", "Windows symlink creation needs separate privilege")
    def test_source_symlink_escape_rejected(self):
        source = self.repo / "skills/inference-engine/SKILL.md"
        outside = self.base / "outside.md"
        shutil.copyfile(source, outside)
        source.unlink()
        source.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "escapes"):
            installer.install(self.repo, "desktop", self.dest)

    def test_complete_and_downgrade_are_explicit(self):
        dest = installer.install(self.repo, "complete", self.dest)
        self.assertTrue((dest / "corpus/large.txt").exists())
        with self.assertRaisesRegex(ValueError, "exists"):
            installer.install(self.repo, "desktop", self.dest)
        installer.install(self.repo, "desktop", self.dest, update=True)
        self.assertFalse((dest / "corpus").exists())

    def test_check_only_and_custom_codex_home(self):
        with patch.dict(os.environ, {"CODEX_HOME": str(self.base / "custom codex")}):
            self.assertEqual(installer.main(["--source", str(self.repo)]), 0)
            self.assertEqual(installer.main(["--source", str(self.repo), "--check"]), 0)
        self.assertTrue((self.base / "custom codex/skills/inference-engine/SKILL.md").exists())

    def test_git_sparse_install_does_not_download_complete_corpus(self):
        def run(*args):
            return subprocess.run(["git", "-c", "core.autocrlf=false", *args], cwd=self.repo,
                                  check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True).stdout
        run("init", "-b", "main")
        run("config", "user.name", "Installer test")
        run("config", "user.email", "test@example.invalid")
        run("config", "uploadpack.allowFilter", "true")
        run("add", ".")
        run("commit", "-m", "fixture")
        large_oid = run("hash-object", "skills/inference-engine/corpus/large.txt").strip()
        with installer.checkout((Path(self.temp.name) / "source repo").as_uri(), "main", "desktop") as remote:
            missing = installer.git(["rev-list", "--objects", "--missing=print", "HEAD"], cwd=remote).stdout
            self.assertIn("?" + large_oid, missing)
            self.assertFalse((remote / "skills/inference-engine/corpus/large.txt").exists())
            installer.install(remote, "desktop", self.dest)
        with installer.checkout((Path(self.temp.name) / "source repo").as_uri(), "main", "complete") as remote:
            installer.install(remote, "complete", self.dest, update=True)
            self.assertTrue((self.dest / "inference-engine/corpus/large.txt").exists())


if __name__ == "__main__":
    unittest.main()
