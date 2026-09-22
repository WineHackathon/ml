#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

from ocr_rerank import OCR_CONFIG, load_engine, predict_image


def main() -> None:
    with redirect_stdout(sys.stderr):
        engine = load_engine(4)
    print(json.dumps({"ready": True, "config": OCR_CONFIG}), flush=True)
    for line in sys.stdin:
        try:
            request = json.loads(line)
            response = {"result": predict_image(engine, Path(request["path"]))}
        except Exception as exc:
            response = {"error": f"{type(exc).__name__}: {exc}"}
        print(json.dumps(response, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
