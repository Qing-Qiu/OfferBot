import os
import sys
import tempfile
from pathlib import Path

from mkdocs.__main__ import cli


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TMP_ROOT = ROOT / ".mkdocs_tmp"
TMP_ROOT = Path(os.environ.get("OFFERBOT_MKDOCS_TMP", str(DEFAULT_TMP_ROOT)))
TMP_DIR = TMP_ROOT.resolve()
TMP_DIR.mkdir(parents=True, exist_ok=True)

for name in ("TMPDIR", "TEMP", "TMP"):
    os.environ[name] = str(TMP_DIR)

tempfile.tempdir = str(TMP_DIR)

if len(sys.argv) == 1:
    watch_paths = [
        "README.md",
        "KNOWLEDGE_TREE_INDEX.md",
        "LEARNING_MAP.md",
        "MKDOCS_GUIDE.md",
        "00_Python_Syntax",
        "01_Math_Foundations",
        "02_ML_Foundations",
        "03_PyTorch",
        "04_RecSys",
        "05_LLM",
        "06_AI_Engineering",
        "07_Algorithm",
    ]
    sys.argv.extend(["serve", "--dev-addr", "127.0.0.1:8000"])
    for path in watch_paths:
        sys.argv.extend(["--watch", str(ROOT / path)])

cli()
