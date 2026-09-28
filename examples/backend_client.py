#!/usr/bin/env python3
import json
import mimetypes
import sys
import urllib.request
import uuid
from pathlib import Path


def recognize(base_url: str, image: Path, token: str | None = None, timeout: float = 12) -> dict:
    boundary = uuid.uuid4().hex
    content_type = mimetypes.guess_type(image.name)[0] or "application/octet-stream"
    body = (
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{image.name}\"\r\n"
        f"Content-Type: {content_type}\r\n\r\n"
    ).encode() + image.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
    headers = {"Content-Type": f"multipart/form-data; boundary={boundary}"}
    if token:
        headers["X-Token"] = token
    request = urllib.request.Request(f"{base_url.rstrip('/')}/v1/recognize", body, headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


if __name__ == "__main__":
    print(json.dumps(recognize(sys.argv[1], Path(sys.argv[2])), ensure_ascii=False, indent=2))
