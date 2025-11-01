from pathlib import Path
from werkzeug.utils import secure_filename
import uuid

UPLOAD_DIR = Path.cwd() / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

def save_upload_file(contents: bytes, filename: str) -> str:
    fname = secure_filename(filename)
    # 一意化
    fname = f"{uuid.uuid4().hex}_{fname}"
    p = UPLOAD_DIR / fname
    p.write_bytes(contents)
    return str(p)