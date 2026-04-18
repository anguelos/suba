"""
suba.io.resumable_download
~~~~~~~~~~~~~~~~~~~~~~~~~~
HTTP download with resume support via Range requests.
"""
from __future__ import annotations
from pathlib import Path


def download_resumable(url: str, dest: Path, chunk_size: int = 1 << 20) -> None:
    """Download *url* to *dest*, resuming a partial file if present.

    Uses HTTP ``Range`` requests.  Returns immediately if the server responds
    with 416 (file already complete).

    Parameters
    ----------
    url:
        HTTP/HTTPS URL.
    dest:
        Destination path (parent must exist).
    chunk_size:
        Streaming chunk size in bytes (default 1 MiB).
    """
    import requests
    from tqdm import tqdm

    existing = dest.stat().st_size if dest.exists() else 0
    headers = {"Range": f"bytes={existing}-"} if existing else {}

    with requests.get(url, headers=headers, stream=True, timeout=60) as r:
        if r.status_code == 416:
            return
        r.raise_for_status()

        total = int(r.headers.get("Content-Length", 0)) + existing
        mode = "ab" if existing else "wb"

        with open(dest, mode) as fh:
            with tqdm(
                total=total if total > 0 else None,
                initial=existing,
                unit="B",
                unit_scale=True,
                desc=dest.name,
            ) as pbar:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    fh.write(chunk)
                    pbar.update(len(chunk))
