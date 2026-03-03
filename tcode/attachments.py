from __future__ import annotations
from typing import Dict, Any
import os
import uuid

class AttachmentStore:
    def __init__(self, base_dir: str = None):
        self.base_dir = base_dir or os.path.join(os.getcwd(), ".tcode_attachments")
        os.makedirs(self.base_dir, exist_ok=True)

    def store(self, content: bytes, filename: str, mime: str) -> str:
        id = str(uuid.uuid4())
        path = os.path.join(self.base_dir, id + "_" + filename)
        with open(path, "wb") as f:
            f.write(content)
        return f"attachment://{id}"

    def get_path(self, url: str) -> str:
        # url is attachment://id
        if not url.startswith("attachment://"):
            raise ValueError("invalid url")
        id = url.split("attachment://", 1)[1]
        for fn in os.listdir(self.base_dir):
            if fn.startswith(id + "_"):
                return os.path.join(self.base_dir, fn)
        raise FileNotFoundError()

    def get(self, url: str) -> bytes:
        path = self.get_path(url)
        with open(path, "rb") as f:
            return f.read()
