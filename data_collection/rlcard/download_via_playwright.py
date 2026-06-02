#!/usr/bin/env python3
"""Capture RLCard tutorials as viewport-sized scrolling screenshots.

Reads ``rlcard_urls.json`` (same schema as the html downloader), opens each
tutorial URL in a headless Chromium via Playwright, and scrolls the page
in viewport-sized steps, capturing one PNG frame per scroll position.

Output layout (mirrors the html flow but with type=screenshot, plus the
per-game subdir that RLCard uses — osworld is benchmark-scoped and has
no game layer):

    data_tutorial/screenshot/{game}/{task_id}/tutorial/
      metadata.json   # {"task_id","instruction","game","source_url","content_type":"screenshot"}
      images/
        frame_001.png  frame_002.png  ...

Usage:
    python data_collection/rlcard/download_via_playwright.py
    python data_collection/rlcard/download_via_playwright.py --game doudizhu
    python data_collection/rlcard/download_via_playwright.py --task <TASK_ID>
    python data_collection/rlcard/download_via_playwright.py --force

Dependencies (not in main requirements.txt — see requirements-playwright.txt):
    pip install playwright>=1.42
    playwright install chromium
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import shutil
import sys
from pathlib import Path

from playwright.async_api import Browser, Page, async_playwright


VIEWPORT_WIDTH = 1280
VIEWPORT_HEIGHT = 2000
MAX_FRAMES = 30
SCROLL_RATIO = 0.9        # 10% overlap so key elements aren't sliced

NAVIGATION_TIMEOUT_MS = 30_000
SETTLE_AFTER_LOAD_MS = 1500
SETTLE_AFTER_SCROLL_MS = 400


SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_URLS_FILE = SCRIPT_DIR / "rlcard_urls.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data_tutorial" / "screenshot"


log = logging.getLogger(__name__)


async def capture_url(
    page: Page, url: str, images_dir: Path, frame_seq_start: int,
) -> int:
    """Scroll-capture one URL into ``frame_{N:03d}.png`` files.

    Returns the next available frame index so multi-url tasks continue
    numbering without collisions.
    """
    seq = frame_seq_start
    try:
        # domcontentloaded + static settle is robust against pages with
        # persistent connections where networkidle never fires.
        await page.goto(
            url, wait_until="domcontentloaded", timeout=NAVIGATION_TIMEOUT_MS,
        )
        await page.wait_for_timeout(SETTLE_AFTER_LOAD_MS)
    except Exception as e:
        log.warning("  goto failed for %s: %s", url, e)
        return seq

    step = max(1, int(VIEWPORT_HEIGHT * SCROLL_RATIO))
    scroll_y = 0

    while seq <= MAX_FRAMES:
        await page.evaluate(f"window.scrollTo(0, {scroll_y})")
        await page.wait_for_timeout(SETTLE_AFTER_SCROLL_MS)

        frame_path = images_dir / f"frame_{seq:03d}.png"
        await page.screenshot(path=str(frame_path), full_page=False)
        seq += 1

        at_bottom = await page.evaluate(
            "window.innerHeight + window.scrollY >= document.body.scrollHeight - 1"
        )
        if at_bottom:
            break

        scroll_y += step

    return seq


async def process_task(
    task: dict, output_dir: Path, browser: Browser, force: bool,
) -> str:
    """Process a single task. Returns 'success' / 'failed' / 'skipped' / 'no_url'."""
    task_id = task["task_id"]
    instruction = task.get("instruction", "")
    game = task.get("game", "unknown")

    single_url = task.get("tutorial_url")
    multi_urls = task.get("tutorial_urls", []) or []

    urls: list[str] = []
    if single_url:
        urls.append(single_url)
    for u in multi_urls:
        if u and u not in urls:
            urls.append(u)

    if not urls:
        log.info("[%s] SKIPPED: No tutorial URL", task_id)
        return "no_url"

    task_dir = output_dir / game / task_id / "tutorial"
    meta_path = task_dir / "metadata.json"
    images_dir = task_dir / "images"

    if not force and meta_path.exists():
        log.info("[%s] SKIPPED: Already captured", task_id)
        return "skipped"

    if task_dir.exists():
        shutil.rmtree(task_dir)
    images_dir.mkdir(parents=True, exist_ok=True)

    log.info("[%s] (%s) Capturing %d URL(s)...", task_id, game, len(urls))

    context = await browser.new_context(
        viewport={"width": VIEWPORT_WIDTH, "height": VIEWPORT_HEIGHT},
    )
    page = await context.new_page()

    next_seq = 1
    captured_urls: list[str] = []
    try:
        for url in urls:
            before = next_seq
            next_seq = await capture_url(page, url, images_dir, next_seq)
            if next_seq > before:
                captured_urls.append(url)
            if next_seq > MAX_FRAMES:
                log.info(
                    "[%s] Hit MAX_FRAMES=%d, stopping after this URL",
                    task_id, MAX_FRAMES,
                )
                break
    finally:
        await context.close()

    frame_count = next_seq - 1
    if frame_count == 0:
        log.error("[%s] FAILED: No frames captured", task_id)
        return "failed"

    metadata: dict = {
        "task_id": task_id,
        "instruction": instruction,
        "game": game,
        "content_type": "screenshot",
    }
    if len(captured_urls) == 1:
        metadata["source_url"] = captured_urls[0]
    else:
        metadata["source_urls"] = captured_urls

    meta_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    log.info("[%s] Done: %d frames", task_id, frame_count)
    return "success"


async def run(args) -> int:
    urls_file = Path(args.urls_file) if args.urls_file else DEFAULT_URLS_FILE
    if not urls_file.exists():
        log.error("File not found: %s", urls_file)
        return 1

    with open(urls_file) as f:
        tasks = json.load(f)
    log.info("Loaded %d tasks from %s", len(tasks), urls_file.name)

    if args.game:
        tasks = [t for t in tasks if t.get("game") == args.game]
        if not tasks:
            log.error("No tasks for game '%s'", args.game)
            return 1

    if args.task:
        tasks = [t for t in tasks if t["task_id"] == args.task]
        if not tasks:
            log.error("Task %s not found in %s", args.task, urls_file.name)
            return 1

    output_dir = Path(args.output_dir)
    log.info(
        "Total: %d tasks, output: %s, force: %s",
        len(tasks), output_dir, args.force,
    )

    results = {"success": 0, "failed": 0, "skipped": 0, "no_url": 0}

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        try:
            for task in tasks:
                try:
                    status = await process_task(
                        task, output_dir, browser, args.force,
                    )
                except Exception as e:
                    log.error("[%s] ERROR: %s", task.get("task_id", "?"), e)
                    status = "failed"
                results[status] = results.get(status, 0) + 1
        finally:
            await browser.close()

    log.info(
        "Summary: success=%d failed=%d skipped=%d no_url=%d",
        results["success"], results["failed"],
        results["skipped"], results["no_url"],
    )
    return 0 if results["failed"] == 0 else 2


def main():
    parser = argparse.ArgumentParser(
        description="Capture RLCard tutorials as viewport-sized scrolling screenshots",
    )
    parser.add_argument(
        "--urls-file", type=str, default=None,
        help=f"URL mapping JSON (default: {DEFAULT_URLS_FILE.name})",
    )
    parser.add_argument(
        "--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory (default: data_tutorial/screenshot)",
    )
    parser.add_argument(
        "--game", type=str, default=None,
        help="Filter by game (doudizhu|mahjong|nolimit_holdem)",
    )
    parser.add_argument(
        "--task", type=str, default=None,
        help="Process only the task with this id",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%H:%M:%S",
    )
    sys.exit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
