"""Read-only tools reused from Milestone 1's fs_tools.py.

Keep this standalone copy so the submission does not depend on a sibling repo.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zipfile import BadZipFile

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".docx"}


def _metadata(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "name": path.name,
        "size_bytes": stat.st_size,
        "modified_at": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
        "extension": path.suffix.lower(),
    }


def _extract_text(path: Path) -> str:
    if path.suffix.lower() == ".txt":
        return path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader
        from pypdf.errors import PdfReadError

        try:
            reader = PdfReader(str(path))
        except PdfReadError as exc:
            raise ValueError(f"Invalid PDF document: {exc}") from exc
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    if path.suffix.lower() == ".docx":
        from docx import Document
        from docx.opc.exceptions import PackageNotFoundError

        try:
            document = Document(str(path))
        except PackageNotFoundError as exc:
            raise ValueError(f"Invalid DOCX document: {exc}") from exc
        paragraphs = [paragraph.text for paragraph in document.paragraphs]
        tables = [
            "\n".join("\t".join(cell.text for cell in row.cells) for row in table.rows)
            for table in document.tables
        ]
        return "\n".join(part for part in [*paragraphs, *tables] if part)
    raise ValueError(f"Unsupported file type: {path.suffix}")


def read_file(filepath: str) -> dict[str, Any]:
    path = Path(filepath)
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        return {"success": False, "filepath": filepath, "error": "Unsupported file type."}
    if not path.is_file():
        return {"success": False, "filepath": filepath, "error": "File does not exist."}
    try:
        return {
            "success": True,
            "filepath": str(path),
            "content": _extract_text(path),
            "metadata": _metadata(path),
        }
    except (OSError, UnicodeError, BadZipFile, ValueError, ImportError) as exc:
        return {"success": False, "filepath": filepath, "error": str(exc)}


def list_files(directory: str, extension: str | None = None) -> list[dict[str, Any]]:
    path = Path(directory)
    normalized = f".{extension.lstrip('.').lower()}" if extension else None
    try:
        if not path.is_dir():
            return [{"success": False, "directory": directory, "error": "Directory does not exist."}]
        return [
            _metadata(item)
            for item in sorted(path.iterdir(), key=lambda entry: entry.name.lower())
            if item.is_file() and (normalized is None or item.suffix.lower() == normalized)
        ]
    except OSError as exc:
        return [{"success": False, "directory": directory, "error": str(exc)}]
