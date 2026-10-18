import os
import tempfile
from pathlib import Path

# Isolate state *before* the app modules are imported.
_tmp = tempfile.mkdtemp()
os.environ["INDEX_DIR"] = str(Path(_tmp) / "index")
os.environ["AUDIT_LOG"] = str(Path(_tmp) / "audit.jsonl")
os.environ["LLM_MODE"] = "offline"
os.environ.pop("API_KEYS", None)
