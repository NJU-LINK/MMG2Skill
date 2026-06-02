#!/usr/bin/env python3
"""Download tutorial materials for RLCard games.

Reads rlcard_urls.json, fetches tutorial pages, saves as multi-page HTML
(page_0.html, page_1.html, ...) following the OSWorld convention.

Output: data_tutorial/html/{game}/{task_id}/tutorial/
  page.html  (single URL) or page_0.html, page_1.html, ... (multiple URLs)
  metadata.json
  images/

Usage:
    python data_collection/rlcard/download_tutorials.py
    python data_collection/rlcard/download_tutorials.py --game doudizhu
    python data_collection/rlcard/download_tutorials.py --force
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup, Comment


SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
URLS_FILE = SCRIPT_DIR / "rlcard_urls.json"
OUTPUT_DIR = PROJECT_ROOT / "data_tutorial" / "html"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

SESSION = requests.Session()
SESSION.headers.update(HEADERS)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp"}


def fetch_page(url: str, timeout: int = 30) -> requests.Response | None:
    """Fetch a URL with retries."""
    for attempt in range(3):
        try:
            resp = SESSION.get(url, timeout=timeout, allow_redirects=True)
            resp.raise_for_status()
            return resp
        except requests.RequestException as e:
            print(f"  [attempt {attempt + 1}/3] Error fetching {url}: {e}")
            if attempt < 2:
                time.sleep(2 ** attempt)
    return None


def extract_main_content(soup: BeautifulSoup) -> BeautifulSoup:
    """Extract the main content area and clean noise."""
    selectors = [
        ".entry-content",
        ".post-content",
        ".article-content",
        ".article-body",
        ".post-body",
        ".td-post-content",
        "article",
        "main",
        "[role='main']",
        "#content",
        ".content",
    ]
    content = None
    for selector in selectors:
        el = soup.select_one(selector)
        if el and len(el.get_text(strip=True)) > 100:
            content = el
            break

    if content is None:
        content = soup.find("body") or soup

    # Clean noise within the extracted content
    for tag_name in ["script", "style", "nav", "aside", "noscript", "footer", "header"]:
        for tag in content.find_all(tag_name):
            tag.decompose()

    for comment in content.find_all(string=lambda text: isinstance(text, Comment)):
        comment.extract()

    noise_patterns = [
        "sidebar", "menu", "cookie", "banner", "advert",
        "social", "share", "signup", "newsletter", "popup",
        "related", "recommend",
    ]
    for pattern in noise_patterns:
        for el in content.find_all(class_=re.compile(pattern, re.I)):
            el.decompose()
        for el in content.find_all(id=re.compile(pattern, re.I)):
            el.decompose()

    return content


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
        resp = SESSION.get(full_url, timeout=15, stream=True)
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
    html_filename: str,
    images_dir: Path,
    seen_names: set[str],
    seen_hashes: dict[str, str],
) -> tuple[bool, int]:
    """Fetch one URL, save HTML with rewritten image paths.

    Returns (success, image_count).
    """
    resp = fetch_page(url)
    if resp is None:
        print(f"  FAILED: Could not fetch {url}")
        return False, 0

    soup = BeautifulSoup(resp.text, "html.parser")
    content = extract_main_content(soup)

    img_count = 0
    for img_tag in content.find_all("img"):
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
    html_path.write_text(str(content), encoding="utf-8")
    print(f"  Saved {html_filename} ({len(str(content))} bytes, {img_count} images)")
    return True, img_count


def process_task(task: dict, force: bool = False) -> bool:
    """Process a single task: fetch all pages, save HTML + images + metadata."""
    task_id = task["task_id"]
    instruction = task["instruction"]
    game = task.get("game", "unknown")

    # Support both old (tutorial_url) and new (tutorial_urls) format
    urls = task.get("tutorial_urls", [])
    if not urls:
        single = task.get("tutorial_url")
        if single:
            urls = [single]
    if not urls:
        print(f"  SKIPPED: No URLs for {task_id}")
        return False

    print(f"\n{'='*60}")
    print(f"Task:        {task_id}")
    print(f"Game:        {game}")
    print(f"URLs:        {len(urls)}")
    print(f"Instruction: {instruction[:80]}...")

    # Create output directory
    task_dir = OUTPUT_DIR / game / task_id / "tutorial"
    images_dir = task_dir / "images"
    task_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(exist_ok=True)

    # Skip if already downloaded (check for metadata.json as completion marker)
    meta_path = task_dir / "metadata.json"
    if meta_path.exists() and not force:
        print(f"  SKIPPED: Already downloaded at {task_dir}")
        return True

    # Clean existing pages if forcing re-download
    if force:
        for old_page in task_dir.glob("page*.html"):
            old_page.unlink()

    # Shared image dedup state across all pages
    seen_names: set[str] = set()
    seen_hashes: dict[str, str] = {}
    total_images = 0
    source_urls: list[str | None] = []

    if len(urls) == 1:
        ok, img_count = download_single_url(
            urls[0], task_dir, "page.html", images_dir, seen_names, seen_hashes,
        )
        if not ok:
            return False
        total_images += img_count
        source_urls.append(urls[0])
    else:
        any_ok = False
        for i, url in enumerate(urls):
            print(f"  [{i+1}/{len(urls)}] {url}")
            ok, img_count = download_single_url(
                url, task_dir, f"page_{i}.html", images_dir, seen_names, seen_hashes,
            )
            if ok:
                any_ok = True
                total_images += img_count
                source_urls.append(url)
            else:
                source_urls.append(None)
            time.sleep(1)  # Be polite between pages

        if not any_ok:
            return False

    # Save metadata (completion marker)
    metadata: dict = {
        "task_id": task_id,
        "instruction": instruction,
        "game": game,
    }
    if len(source_urls) == 1:
        metadata["source_url"] = source_urls[0]
    else:
        metadata["source_urls"] = source_urls

    meta_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(f"  Done: {total_images} images, {len([u for u in source_urls if u])} pages")

    # Clean up empty images dir
    if total_images == 0:
        try:
            images_dir.rmdir()
        except OSError:
            pass

    return True


def main():
    parser = argparse.ArgumentParser(description="Download RLCard game tutorials")
    parser.add_argument(
        "--game",
        type=str,
        default=None,
        help="Filter by game name (e.g. doudizhu, mahjong). Default: download all.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if already exists.",
    )
    args = parser.parse_args()

    if not URLS_FILE.exists():
        print(f"Error: {URLS_FILE} not found")
        sys.exit(1)

    with open(URLS_FILE) as f:
        tasks = json.load(f)

    # Filter by game if specified
    if args.game:
        tasks = [t for t in tasks if t.get("game") == args.game]
        if not tasks:
            print(f"No tasks found for game '{args.game}'")
            sys.exit(1)

    print(f"Loaded {len(tasks)} tasks from {URLS_FILE}")
    print(f"Output directory: {OUTPUT_DIR}")

    success = 0
    failed = 0
    failed_tasks = []

    for task in tasks:
        try:
            if process_task(task, force=args.force):
                success += 1
            else:
                failed += 1
                failed_tasks.append(task["task_id"])
        except Exception as e:
            print(f"  ERROR: {e}")
            failed += 1
            failed_tasks.append(task["task_id"])

    print(f"\n{'='*60}")
    print(f"DONE: {success} succeeded, {failed} failed out of {len(tasks)} tasks")
    if failed_tasks:
        print(f"Failed tasks: {failed_tasks}")


if __name__ == "__main__":
    main()
