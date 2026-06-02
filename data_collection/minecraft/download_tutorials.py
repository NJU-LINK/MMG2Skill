#!/usr/bin/env python3
"""Download tutorial materials for Minecraft (OpenHA) tasks.

Reads URL mapping JSON file(s), fetches each tutorial page,
saves raw HTML and downloads images to {output_dir}/{task_id}/tutorial/.
Skips tasks that already have a completed download (metadata.json exists).

Default manifests:
  1. `beginners_guide.json` — shared Beginner's Guide page (general
     mechanics / GUI / controls), loaded for every Minecraft task.
  2. `minecraft_urls.json` — per-task wiki pages (recipe / block specifics).

Both are fetched together; the runner's `MinecraftKit.tutorial_ids_for(task)`
layers the per-task page on top of the shared guide before extraction. The
legacy `shared_urls.json` is retained on disk for archival and can be
invoked explicitly via `--urls-file`.

Usage:
    # Download default (shared Beginner's Guide), skip if already cached
    python download_tutorials.py

    # Force re-download with 8 threads
    python download_tutorials.py --force --workers 8

    # Use the legacy per-task manifest instead
    python download_tutorials.py --urls-file minecraft_urls.json \\
        --output-dir ./data_tutorial/html/minecraft
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# curl_cffi impersonates a real browser's TLS fingerprint; plain `requests` gets
# 403-blocked by Cloudflare on sites like minecraft.wiki.
from curl_cffi import requests
from bs4 import BeautifulSoup


SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data_tutorial" / "html" / "minecraft"
DEFAULT_URL_FILES = [
    SCRIPT_DIR / "beginners_guide.json",
    SCRIPT_DIR / "minecraft_urls.json",
]

IMPERSONATE = "chrome"

# Selectors whose matched elements are stripped before saving. Targets MediaWiki
# (minecraft.wiki) chrome + media but works on other sites too.
#
# `link` / `meta` are deliberately omitted: although both are head-only void
# elements, MediaWiki emits `<link rel="mw-deduplicated-inline-style">` inside
# body, and bs4's `html.parser` mis-treats it as an opened container so
# decomposing one swallows the entire article (observed on zh.minecraft.wiki).
# Their stray presence in body is harmless and not worth that footgun.
NOISE_SELECTORS = [
    "script", "style", "noscript",
    "nav", "aside", "header", "footer",
    "img", "picture", "source", "video", "audio", "figure", ".thumb",
    ".navbox", ".hatnote", ".ambox", ".mw-editsection",
    ".toc", "#toc", ".infobox", ".gallery",
    ".references", ".mw-references-wrap", ".reflist",
    ".catlinks", ".printfooter", ".mw-jump-link",
    ".noprint",
]

# H2/H3 headings at which we trim everything after (inclusive).
TRIM_HEADINGS = {
    "see also", "references", "external links", "gallery",
    "issues", "history", "video", "videos", "trivia",
    "notes", "navigation", "data values", "achievements", "advancements",
}

log = logging.getLogger(__name__)

_thread_local = threading.local()


def _get_session():
    """Get a thread-local curl_cffi Session (thread-safe)."""
    if not hasattr(_thread_local, "session"):
        _thread_local.session = requests.Session(impersonate=IMPERSONATE)
    return _thread_local.session


def fetch_page(url: str, timeout: int = 30):
    """Fetch a URL with retries."""
    for attempt in range(3):
        try:
            resp = _get_session().get(url, timeout=timeout, allow_redirects=True)
            resp.raise_for_status()
            return resp
        except Exception as e:
            log.warning("  [attempt %d/3] Error fetching %s: %s", attempt + 1, url, e)
            if attempt < 2:
                import time
                time.sleep(2 ** attempt)
    return None


def _extract_content(html: str) -> str:
    """Extract article body, strip navigation/infoboxes/images/etc.

    Returns a minimal HTML document with only the cleaned main content.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Prefer MediaWiki/article containers; fall back to <body>.
    content = (
        soup.select_one(".mw-parser-output")
        or soup.select_one("article")
        or soup.select_one("main")
        or soup.select_one(".entry-content")
        or soup.select_one(".post-content")
        or soup.body
        or soup
    )

    for sel in NOISE_SELECTORS:
        for el in content.select(sel):
            el.decompose()

    # Trim everything from the first matching heading onwards.
    for h in content.find_all(["h2", "h3"]):
        text = h.get_text(" ", strip=True).lower()
        # MediaWiki wraps heading text in a span.mw-headline
        if not text:
            hl = h.find(class_="mw-headline")
            if hl:
                text = hl.get_text(" ", strip=True).lower()
        if text in TRIM_HEADINGS:
            cursor = h
            while cursor is not None:
                nxt = cursor.find_next_sibling()
                cursor.decompose()
                cursor = nxt
            break

    title = soup.title.string.strip() if soup.title and soup.title.string else ""
    return (
        "<!DOCTYPE html><html><head>"
        f'<meta charset="utf-8"><title>{title}</title>'
        f'</head><body>{content}</body></html>'
    )


def download_single_url(
    url: str,
    task_dir: Path,
    html_filename: str = "page.html",
) -> bool:
    """Fetch a URL, strip noise + media, save cleaned HTML to task_dir."""
    resp = fetch_page(url)
    if resp is None:
        log.error("  FAILED: Could not fetch %s", url)
        return False

    cleaned = _extract_content(resp.text)
    html_path = task_dir / html_filename
    html_path.write_text(cleaned, encoding="utf-8")
    log.info("  Saved %s (%d bytes)", html_filename, len(cleaned))
    return True


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

    all_source_urls: list[str | None] = []

    if len(urls) == 1:
        if not download_single_url(urls[0], task_dir, "page.html"):
            return "failed"
        all_source_urls.append(urls[0])
    else:
        any_ok = False
        for i, url in enumerate(urls):
            log.info("  [%s] Sub-skill %d: %s", task_id, i, url)
            if download_single_url(url, task_dir, f"page_{i}.html"):
                any_ok = True
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

    log.info("[%s] Done", task_id)
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
        help="URL mapping JSON file(s). Default: os + multi_apps",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory (default: data_tutorial/html/minecraft)",
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
