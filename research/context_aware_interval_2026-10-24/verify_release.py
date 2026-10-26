"""Verify the delivered file payload before reproduction (Python standard library)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "verification" / "Content_Manifest.json").read_text(encoding="utf-8"))
    seen = set()
    for entry in manifest["files"]:
        name = entry["path"]
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or name in seen:
            raise SystemExit(f"Unsafe or duplicate manifest path: {name}")
        seen.add(name)
        path = root / relative
        if not path.is_file():
            raise SystemExit(f"Missing payload: {name}")
        data = path.read_bytes()
        if len(data) != entry["bytes"] or hashlib.sha256(data).hexdigest() != entry["sha256"]:
            raise SystemExit(f"Payload checksum mismatch: {name}")
    print(json.dumps({"status": "passed", "payload_files": len(seen),
                      "scope": "Integrity of released payload; not a fresh neural-inference result"}))


if __name__ == "__main__":
    main()

