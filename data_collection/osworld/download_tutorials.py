#!/usr/bin/env python3
"""Download tutorial materials for OSWorld tasks.

Reads URL mapping JSON file(s), fetches each tutorial page,
saves raw HTML and downloads images to {output_dir}/{task_id}/tutorial/.
Skips tasks that already have a completed download (metadata.json exists).

Usage:
    # Download all from osworld_active_urls.json (skip existing)
    python download_tutorials.py

    # Download specific file(s)
    python download_tutorials.py --urls-file osworld_active_urls.json

    # Force re-download with 8 threads
    python download_tutorials.py --force --workers 8

    # Custom output directory
    python download_tutorials.py --output-dir ./my_tutorials
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import shutil
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup


SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data_tutorial" / "html" / "osworld"
DEFAULT_URL_FILES = [
    SCRIPT_DIR / "osworld_urls.json",
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp"}

log = logging.getLogger(__name__)

# Thread-local storage for per-thread requests.Session
_thread_local = threading.local()


def _get_session() -> requests.Session:
    """Get a thread-local requests.Session (thread-safe)."""
    if not hasattr(_thread_local, "session"):
        session = requests.Session()
        session.headers.update(HEADERS)
        _thread_local.session = session
    return _thread_local.session


def fetch_page(url: str, timeout: int = 30) -> requests.Response | None:
    """Fetch a URL with retries."""
    for attempt in range(3):
        try:
            resp = _get_session().get(url, timeout=timeout, allow_redirects=True)
            resp.raise_for_status()
            return resp
        except requests.RequestException as e:
            log.warning("  [attempt %d/3] Error fetching %s: %s", attempt + 1, url, e)
            if attempt < 2:
                import time
                time.sleep(2 ** attempt)
    return None


def download_image(
    img_url: str,
    save_dir: Path,
    base_url: str,
    _seen: set[str] | None = None,
    _seen_hashes: dict[str, str] | None = None,
) -> str | None:
    """Download an image and return the local filename."""
    if _seen is None:
        _seen = set()
    if _seen_hashes is None:
        _seen_hashes = {}

    full_url = urljoin(base_url, img_url)

    parsed = urlparse(full_url)
    if not parsed.scheme.startswith("http"):
        return None

    url_path = Path(parsed.path)
    stem = url_path.stem or "image"
    ext = url_path.suffix.lower()
    if ext not in IMAGE_EXTENSIONS:
        ext = ".png"
    stem = re.sub(r"[^\w\-]", "_", stem)

    filename = f"{stem}{ext}"
    counter = 2
    while filename in _seen or (save_dir / filename).exists():
        filename = f"{stem}_{counter}{ext}"
        counter += 1
    _seen.add(filename)

    filepath = save_dir / filename

    try:
        resp = _get_session().get(full_url, timeout=15)
        resp.raise_for_status()

        ct = resp.headers.get("content-type", "")
        if "image" not in ct and "svg" not in ct:
            return None

        content_length = resp.headers.get("content-length")
        if content_length:
            size = int(content_length)
            if size < 200 or size > 10 * 1024 * 1024:
                return None

        content = resp.content
        content_hash = hashlib.md5(content).hexdigest()

        if content_hash in _seen_hashes:
            return _seen_hashes[content_hash]

        with open(filepath, "wb") as f:
            f.write(content)

        _seen_hashes[content_hash] = filename
        return filename
    except requests.RequestException:
        return None


def download_single_url(
    url: str,
    task_dir: Path,
    html_filename: str = "page.html",
) -> tuple[bool, int]:
    """Download a single tutorial URL, save HTML + images.

    Returns (success, image_count).
    """
    images_dir = task_dir / "images"
    images_dir.mkdir(exist_ok=True)

    resp = fetch_page(url)
    if resp is None:
        log.error("  FAILED: Could not fetch %s", url)
        return False, 0

    soup = BeautifulSoup(resp.text, "html.parser")

    img_count = 0
    seen_names: set[str] = set()
    seen_hashes: dict[str, str] = {}
    for img_tag in soup.find_all("img"):
        src = img_tag.get("src") or img_tag.get("data-src") or ""
        if not src:
            continue
        local_name = download_image(
            src, images_dir, url, _seen=seen_names, _seen_hashes=seen_hashes,
        )
        if local_name:
            img_tag["src"] = f"images/{local_name}"
            img_count += 1

    html_path = task_dir / html_filename
    html_path.write_text(str(soup), encoding="utf-8")
    log.info("  Saved %s (%d bytes, %d images)", html_filename, len(str(soup)), img_count)

    if img_count == 0:
        try:
            images_dir.rmdir()
        except OSError:
            pass

    return True, img_count


def process_task(task: dict, output_dir: Path, force: bool = False) -> str:
    """Process a single task. Returns status: 'success', 'failed', 'skipped', 'no_url'."""
    task_id = task["task_id"]
    instruction = task["instruction"]
    single_url = task.get("tutorial_url")
    multi_urls = task.get("tutorial_urls", [])

    # Collect all URLs to download
    urls: list[str] = []
    if single_url:
        urls.append(single_url)
    if multi_urls:
        for u in multi_urls:
            if u and u not in urls:
                urls.append(u)

    if not urls:
        log.info("[%s] SKIPPED: No tutorial URL", task_id)
        return "no_url"

    task_dir = output_dir / task_id / "tutorial"
    meta_path = task_dir / "metadata.json"

    # Skip if already downloaded (metadata.json = completion marker)
    if not force and meta_path.exists():
        log.info("[%s] SKIPPED: Already downloaded", task_id)
        return "skipped"

    log.info("[%s] Downloading %d URL(s)...", task_id, len(urls))

    # Clean up partial state from previous interrupted download
    if task_dir.exists():
        shutil.rmtree(task_dir)
    task_dir.mkdir(parents=True, exist_ok=True)

    total_images = 0
    all_source_urls = []

    if len(urls) == 1:
        ok, img_count = download_single_url(urls[0], task_dir, "page.html")
        if not ok:
            return "failed"
        total_images += img_count
        all_source_urls.append(urls[0])
    else:
        any_ok = False
        for i, url in enumerate(urls):
            log.info("  [%s] Sub-skill %d: %s", task_id, i, url)
            ok, img_count = download_single_url(url, task_dir, f"page_{i}.html")
            if ok:
                any_ok = True
                total_images += img_count
                all_source_urls.append(url)
            else:
                all_source_urls.append(None)
        if not any_ok:
            return "failed"

    # Write metadata.json LAST as completion marker
    metadata: dict = {
        "task_id": task_id,
        "instruction": instruction,
    }
    if len(all_source_urls) == 1:
        metadata["source_url"] = all_source_urls[0]
    else:
        metadata["source_urls"] = all_source_urls

    meta_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8",
    )

    log.info("[%s] Done: %d images", task_id, total_images)
    return "success"


def resolve_urls_file(path_str: str) -> Path:
    """Resolve a URL file path, checking script directory as fallback."""
    p = Path(path_str)
    if p.exists():
        return p
    if not p.is_absolute():
        fallback = SCRIPT_DIR / p
        if fallback.exists():
            return fallback
    return p


def main():
    parser = argparse.ArgumentParser(description="Download OSWorld tutorial materials")
    parser.add_argument(
        "--urls-file",
        nargs="*",
        default=None,
        help="URL mapping JSON file(s). Default: osworld_active_urls.json",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory (default: data_tutorial/html/osworld)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of download threads (default: 4)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if already exists",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(threadName)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # Resolve URL files
    if args.urls_file is None:
        urls_files = [f for f in DEFAULT_URL_FILES if f.exists()]
    else:
        urls_files = [resolve_urls_file(f) for f in args.urls_file]

    output_dir = Path(args.output_dir)

    # Load and merge all tasks
    all_tasks: list[dict] = []
    for urls_file in urls_files:
        if not urls_file.exists():
            log.error("File not found: %s", urls_file)
            sys.exit(1)
        with open(urls_file) as f:
            tasks = json.load(f)
        log.info("Loaded %d tasks from %s", len(tasks), urls_file.name)
        all_tasks.extend(tasks)

    log.info("Total: %d tasks, output: %s, workers: %d, force: %s",
             len(all_tasks), output_dir, args.workers, args.force)

    # Run downloads with thread pool
    results: dict[str, int] = {"success": 0, "failed": 0, "skipped": 0, "no_url": 0}
    failed_tasks: list[str] = []

    with ThreadPoolExecutor(max_workers=args.workers, thread_name_prefix="dl") as executor:
        futures = {
            executor.submit(process_task, task, output_dir, args.force): task
            for task in all_tasks
        }
        for future in as_completed(futures):
            task = futures[future]
            try:
                status = future.result()
            except Exception as e:
                log.error("[%s] ERROR: %s", task["task_id"], e)
                status = "failed"
            results[status] += 1
            if status == "failed":
                failed_tasks.append(task["task_id"])

    log.info("=" * 60)
    log.info("DONE: %d success, %d failed, %d skipped, %d no_url (total %d)",
             results["success"], results["failed"], results["skipped"],
             results["no_url"], len(all_tasks))
    if failed_tasks:
        log.info("Failed tasks: %s", failed_tasks)


if __name__ == "__main__":
    main()
