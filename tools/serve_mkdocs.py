import os
import sys
import tempfile
from pathlib import Path

from mkdocs.__main__ import cli


ROOT = Path(__file__).resolve().parents[1]
TMP_ROOT = Path(os.environ.get("OFFERBOT_MKDOCS_TMP", str(ROOT / "site" / ".mkdocs_tmp")))
TMP_DIR = TMP_ROOT.resolve()
TMP_DIR.mkdir(parents=True, exist_ok=True)

for name in ("TMPDIR", "TEMP", "TMP"):
    os.environ[name] = str(TMP_DIR)

tempfile.tempdir = str(TMP_DIR)

if len(sys.argv) == 1:
    sys.argv.extend(["serve", "--dev-addr", "127.0.0.1:8000"])

cli()
