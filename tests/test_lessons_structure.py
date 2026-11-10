from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LESSONS = ROOT / "lessons"


def test_every_lesson_has_a_readme() -> None:
    lesson_dirs = sorted(path for path in LESSONS.iterdir() if path.is_dir())
    assert lesson_dirs, "no lesson directories found"
    missing = [path.name for path in lesson_dirs if not (path / "README.md").is_file()]
    assert not missing, missing


def test_foundations_lab_is_valid_python() -> None:
    lab = LESSONS / "00_foundations" / "lab.py"
    source = lab.read_text(encoding="utf-8")
    compile(source, str(lab), "exec")
