#!/usr/bin/env python3
"""Capture OSWorld tutorials as deterministic viewport-tall segments.

Sibling to ``download_via_playwright.py``. Replaces the original's scroll-
loop-per-frame strategy (prone to popup/scroll-reset races producing duplicate
frames, and to a window/inner-element fallback that scrolls invisible widgets)
with a fixed ``ceil(page_height / VIEWPORT_HEIGHT)`` scroll-then-clip pass.
Each segment scrolls to its target Y and waits ``SETTLE_AFTER_SCROLL_MS`` so
lazy-loaded images and async iframe ad slots have time to render before the
clip snapshot — something a single ``full_page=True`` shot cannot guarantee.

Output layout (identical to the scrolling script):

    data_tutorial/screenshot/osworld/{task_id}/tutorial/
      metadata.json
      images/frame_001.png frame_002.png ...

Usage:
    python download_via_playwright_fullpage.py
    python download_via_playwright_fullpage.py --task <UUID>
    python download_via_playwright_fullpage.py --force
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import re
import shutil
import sys
import urllib.request
from pathlib import Path

from playwright.async_api import Browser, Page, async_playwright


# ---------------------------------------------------------------------------
# Tunable constants
# ---------------------------------------------------------------------------

VIEWPORT_WIDTH = 1280
VIEWPORT_HEIGHT = 2000
MAX_FRAMES = 30                       # cap on segments emitted per task

NAVIGATION_TIMEOUT_MS = 30_000
SETTLE_AFTER_LOAD_MS = 1500
SETTLE_AFTER_SCROLL_MS = 400
POST_CHALLENGE_TIMEOUT_MS = 8_000

# Lazy-load priming: walk the page top-to-bottom in steps so intersection-
# observer-driven images/components render before the clip snapshots.
LAZY_LOAD_STEP_RATIO = 0.9
LAZY_LOAD_STEP_PAUSE_MS = 150

# Cap on how long to wait for <img> resources to finish loading after the
# lazy-load priming pass (per page).
IMAGE_LOAD_TIMEOUT_MS = 10_000


# ---------------------------------------------------------------------------
# Anti-bot tweaks (same set as the scrolling script)
# ---------------------------------------------------------------------------

STEALTH_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)

STEALTH_LAUNCH_ARGS = [
    "--disable-blink-features=AutomationControlled",
    "--disable-features=IsolateOrigins,site-per-process",
]

STEALTH_INIT_SCRIPT = """
Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'] });
Object.defineProperty(navigator, 'plugins', {
    get: () => [1, 2, 3, 4, 5],
});
"""

POPUP_CLOSE_TEXTS = [
    "accept", "accept all", "agree", "allow all", "got it", "ok",
    "close", "no thanks", "not now", "continue", "dismiss", "skip",
    "maybe later", "×",
]

POPUP_OVERLAY_SELECTORS = [
    "dialog",
    "[role='dialog']",
    "[aria-modal='true']",
    "[aria-label*='popup' i]",
    "[class*='modal' i]",
    "[class*='popup' i]",
    "[class*='newsletter' i]",
    "[class*='cookie' i]",
    "[class*='consent' i]",
    "[class*='adblock' i]",
    ".fc-consent-root",
    ".qc-cmp2-container",
    ".onetrust-pc-dark-filter",
    "#onetrust-banner-sdk",
    ".adsbygoogle",
    ".login-modal-div",
]


SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data_tutorial" / "screenshot" / "osworld"
DEFAULT_URL_FILES = [
    SCRIPT_DIR / "osworld_urls.json",
]


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Page helpers
# ---------------------------------------------------------------------------


async def goto_with_html_fallback(page: Page, url: str) -> None:
    try:
        await page.goto(
            url, wait_until="commit", timeout=NAVIGATION_TIMEOUT_MS,
        )
        return
    except Exception as e:
        log.warning("  goto failed for %s, falling back to static HTML: %s", url, e)

    request = urllib.request.Request(url, headers={"User-Agent": STEALTH_USER_AGENT})
    html = await asyncio.to_thread(
        lambda: urllib.request.urlopen(request, timeout=30).read().decode(
            "utf-8", errors="replace",
        )
    )
    await page.set_content(html, wait_until="domcontentloaded")


async def dismiss_popups(page: Page) -> None:
    for text in POPUP_CLOSE_TEXTS:
        try:
            button = page.get_by_role("button", name=re.compile(text, re.I)).first
            if await button.count():
                await button.click(timeout=800)
        except Exception:
            pass

    try:
        await page.evaluate(
            """(selectors) => {
                for (const selector of selectors) {
                    for (const el of document.querySelectorAll(selector)) {
                        const style = window.getComputedStyle(el);
                        const rect = el.getBoundingClientRect();
                        const isOverlay =
                            style.position === 'fixed' ||
                            style.position === 'sticky' ||
                            rect.width > window.innerWidth * 0.5;
                        if (isOverlay) el.remove();
                    }
                }
                for (const el of document.querySelectorAll('body *')) {
                    const style = window.getComputedStyle(el);
                    const rect = el.getBoundingClientRect();
                    const zIndex = Number.parseInt(style.zIndex, 10) || 0;
                    const coversViewport =
                        rect.width > window.innerWidth * 0.6 &&
                        rect.height > window.innerHeight * 0.25;
                    const blocksView =
                        style.position === 'fixed' &&
                        zIndex >= 1000 &&
                        coversViewport;
                    if (blocksView) el.remove();
                }
                // Only undo inline overflow:hidden (set by popup libs to lock
                // background scroll). Forcing 'auto' onto pages whose CSS
                // expects 'visible' shifts document.scrollingElement and
                // breaks window.scrollTo on some sites (e.g. baeldung).
                if (document.documentElement.style.overflow === 'hidden') {
                    document.documentElement.style.overflow = '';
                }
                if (document.body && document.body.style.overflow === 'hidden') {
                    document.body.style.overflow = '';
                }
            }""",
            POPUP_OVERLAY_SELECTORS,
        )
    except Exception as e:
        log.debug("  popup dismissal failed: %s", e)


async def wait_for_images(page: Page) -> None:
    """Force-load every <img> on the page, then wait for completion.

    First rewrites `data-src`/`data-srcset` (used by lazysizes, EIO loader,
    lozad, etc.) onto `src`/`srcset` so custom-JS lazy loaders that ignore
    programmatic scrollTo() still resolve to real URLs. Then resolves on
    `load`/`error` events per <img>, capped by IMAGE_LOAD_TIMEOUT_MS so a
    single broken/slow asset can't block the entire capture.
    """
    try:
        await page.evaluate(
            """async (timeoutMs) => {
                // Common lazy-load attribute names across libs:
                //   data-src / data-srcset       — lazysizes, EIO, lozad
                //   data-lazy-src / -srcset      — Rocket LazyLoad / WP Rocket
                //   data-original / -srcset      — jQuery Lazy (legacy)
                const SRC_ATTRS = ['data-src', 'data-lazy-src', 'data-original'];
                const SRCSET_ATTRS = [
                    'data-srcset', 'data-lazy-srcset', 'data-original-srcset',
                ];
                const pickAttr = (el, names) => {
                    for (const n of names) {
                        const v = el.getAttribute(n);
                        if (v) return v;
                    }
                    return null;
                };

                const imgs = Array.from(document.querySelectorAll('img'));
                // Force lazyload libs to commit their real URL. Many libs
                // keep the real URL in data-* and only swap on intersection
                // — programmatic scrollTo() doesn't always trigger them.
                for (const img of imgs) {
                    const realSrc = pickAttr(img, SRC_ATTRS);
                    if (realSrc && img.src !== realSrc) img.src = realSrc;
                    const realSrcset = pickAttr(img, SRCSET_ATTRS);
                    if (realSrcset && img.srcset !== realSrcset) img.srcset = realSrcset;
                    // Strip native lazy attribute so decode kicks in now.
                    if (img.loading === 'lazy') img.loading = 'eager';
                    img.classList.remove('lazyload', 'lazyloading', 'lazyloaded');
                }
                // Some libs additionally put data-* on <source> inside
                // <picture>; rewrite those too.
                for (const src of document.querySelectorAll('picture source')) {
                    const realSrcset = pickAttr(src, SRCSET_ATTRS);
                    if (realSrcset) src.srcset = realSrcset;
                }

                const pending = imgs.filter(img => !img.complete);
                if (pending.length === 0) return;
                await Promise.race([
                    Promise.all(pending.map(img => new Promise(resolve => {
                        img.addEventListener('load', resolve, { once: true });
                        img.addEventListener('error', resolve, { once: true });
                    }))),
                    new Promise(r => setTimeout(r, timeoutMs)),
                ]);
            }""",
            IMAGE_LOAD_TIMEOUT_MS,
        )
    except Exception as e:
        log.debug("  wait_for_images failed: %s", e)


async def prime_lazy_load(page: Page) -> int:
    """Walk the page top→bottom in viewport steps, then return to top.

    Returns the final measured ``document.documentElement.scrollHeight`` so
    the caller knows how many segments to capture.
    """
    return await page.evaluate(
        """async ({ stepRatio, pauseMs }) => {
            const sleep = ms => new Promise(r => setTimeout(r, ms));
            const step = Math.max(200, window.innerHeight * stepRatio);
            const get_max = () => Math.max(
                document.documentElement.scrollHeight,
                document.body ? document.body.scrollHeight : 0,
            );
            let y = 0;
            let last_max = 0;
            // Loop with a guard: lazy loading can grow scrollHeight; stop
            // once scrollHeight stabilises across two passes.
            for (let pass = 0; pass < 3; pass++) {
                let max = get_max();
                while (y < max) {
                    window.scrollTo(0, y);
                    await sleep(pauseMs);
                    y += step;
                    max = get_max();
                }
                if (max === last_max) break;
                last_max = max;
                y = 0;
            }
            window.scrollTo(0, 0);
            await sleep(pauseMs * 2);
            return get_max();
        }""",
        {"stepRatio": LAZY_LOAD_STEP_RATIO, "pauseMs": LAZY_LOAD_STEP_PAUSE_MS},
    )


# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------


async def capture_segments(
    page: Page, page_height: int, images_dir: Path, seq_start: int,
) -> int:
    """Scroll-then-clip per segment, emitting frames numbered from ``seq_start``.

    Each iteration scrolls the window to the segment's top, waits
    ``SETTLE_AFTER_SCROLL_MS`` so lazy-loaded images / iframe ads can render,
    then takes a viewport-clip screenshot.
    """
    max_segments = math.ceil(page_height / VIEWPORT_HEIGHT)
    budget = max(0, MAX_FRAMES - (seq_start - 1))
    n = min(max_segments, budget)
    for i in range(n):
        y_target = i * VIEWPORT_HEIGHT
        h = min(VIEWPORT_HEIGHT, page_height - y_target)
        try:
            await page.evaluate("(y) => window.scrollTo(0, y)", y_target)
        except Exception as e:
            log.warning("  scroll evaluate raced a navigation: %s", e)
            return i
        await page.wait_for_timeout(SETTLE_AFTER_SCROLL_MS)
        try:
            await page.screenshot(
                path=str(images_dir / f"frame_{seq_start + i:03d}.png"),
                clip={"x": 0, "y": 0, "width": VIEWPORT_WIDTH, "height": h},
            )
        except Exception as e:
            log.warning("  screenshot raced a navigation: %s", e)
            return i
    return n


# ---------------------------------------------------------------------------
# Per-task driver
# ---------------------------------------------------------------------------


async def process_task(
    task: dict, output_dir: Path, browser: Browser, force: bool,
) -> str:
    task_id = task["task_id"]
    instruction = task["instruction"]
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

    task_dir = output_dir / task_id / "tutorial"
    meta_path = task_dir / "metadata.json"
    images_dir = task_dir / "images"

    if not force and meta_path.exists():
        log.info("[%s] SKIPPED: Already captured", task_id)
        return "skipped"

    if task_dir.exists():
        shutil.rmtree(task_dir)
    images_dir.mkdir(parents=True, exist_ok=True)

    log.info("[%s] Capturing %d URL(s)...", task_id, len(urls))

    context = await browser.new_context(
        viewport={"width": VIEWPORT_WIDTH, "height": VIEWPORT_HEIGHT},
        user_agent=STEALTH_USER_AGENT,
        locale="en-US",
        timezone_id="America/Los_Angeles",
    )
    await context.add_init_script(STEALTH_INIT_SCRIPT)
    page = await context.new_page()

    captured_urls: list[str] = []
    next_seq = 1
    try:
        for url in urls:
            log.info("[%s]   → %s", task_id, url)
            try:
                await goto_with_html_fallback(page, url)
                try:
                    await page.wait_for_load_state(
                        "domcontentloaded",
                        timeout=POST_CHALLENGE_TIMEOUT_MS,
                    )
                except Exception:
                    pass
                await page.wait_for_timeout(SETTLE_AFTER_LOAD_MS)
                await dismiss_popups(page)

                page_height = await prime_lazy_load(page)
                # popups can re-appear after lazy-load scroll; sweep again
                await dismiss_popups(page)
                # block until <img> tags finish decoding so clipped frames
                # don't capture half-rendered or placeholder graphics
                await wait_for_images(page)

                written = await capture_segments(
                    page, page_height, images_dir, next_seq,
                )
                log.info(
                    "[%s]     %dpx → %d frames",
                    task_id, page_height, written,
                )

                if written > 0:
                    captured_urls.append(url)
                    next_seq += written

                if next_seq > MAX_FRAMES:
                    log.info(
                        "[%s] Hit MAX_FRAMES=%d, stopping after this URL",
                        task_id, MAX_FRAMES,
                    )
                    break
            except Exception as e:
                log.error("[%s] URL failed (%s): %s", task_id, url, e)
                continue

        if not captured_urls:
            log.error("[%s] No frames captured for any URL", task_id)
            return "failed"

        metadata = {
            "task_id": task_id,
            "instruction": instruction,
            "content_type": "screenshot",
            "source_url": (
                captured_urls[0] if len(captured_urls) == 1 else captured_urls
            ),
        }
        meta_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        log.info(
            "[%s] DONE: %d frame(s) from %d URL(s)",
            task_id, next_seq - 1, len(captured_urls),
        )
        return "success"
    finally:
        await context.close()


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def resolve_urls_file(path_str: str) -> Path:
    p = Path(path_str)
    if p.exists():
        return p
    if not p.is_absolute():
        fallback = SCRIPT_DIR / p
        if fallback.exists():
            return fallback
    return p


async def run(args) -> int:
    if args.urls_file is None:
        urls_files = [f for f in DEFAULT_URL_FILES if f.exists()]
    else:
        urls_files = [resolve_urls_file(f) for f in args.urls_file]

    output_dir = Path(args.output_dir)

    all_tasks: list[dict] = []
    for urls_file in urls_files:
        if not urls_file.exists():
            log.error("File not found: %s", urls_file)
            return 1
        with open(urls_file) as f:
            tasks = json.load(f)
        log.info("Loaded %d tasks from %s", len(tasks), urls_file.name)
        all_tasks.extend(tasks)

    if args.task:
        all_tasks = [t for t in all_tasks if t["task_id"] == args.task]
        if not all_tasks:
            log.error("Task %s not found in URL files", args.task)
            return 1

    log.info(
        "Total: %d tasks, output: %s, force: %s",
        len(all_tasks), output_dir, args.force,
    )

    results = {"success": 0, "failed": 0, "skipped": 0, "no_url": 0}

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True, args=STEALTH_LAUNCH_ARGS,
        )
        try:
            for task in all_tasks:
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
        description=(
            "Capture OSWorld tutorials as deterministic viewport-tall "
            "scroll-then-clip segments"
        ),
    )
    parser.add_argument("--urls-file", nargs="*", default=None)
    parser.add_argument(
        "--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory (default: data_tutorial/screenshot/osworld)",
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
