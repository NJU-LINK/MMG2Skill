"""Load tutorial materials from disk.

Directory layout: data_tutorial/{tutorial_type}/{bench}/{task_id}/tutorial/

  - html type:        page.html (or page_0.html, page_1.html, ...) + images/ + metadata.json
  - screenshot type:  images/ (non-empty) + metadata.json
  - video type:       reserved (NotImplementedError)

The ``tutorial_type`` is passed in by the runner from ``data.tutorial_type``
config and used both to validate the on-disk artifacts and to decide how
``TutorialMaterial.body`` is populated. It is the source of truth — if
``metadata.json`` carries a conflicting ``content_type``, a warning is
logged and the caller-supplied value wins.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path

from anything2skill.parser.data_types import TutorialMaterial

logger = logging.getLogger("anything2skill.parser")

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp"}

_SUPPORTED_TYPES = {"html", "screenshot"}


def _collect_local_images(images_dir: Path) -> list[str]:
    """Collect all image file paths from the images/ directory."""
    if not images_dir.is_dir():
        return []
    return [
        str(img.resolve())
        for img in sorted(images_dir.iterdir())
        if img.suffix.lower() in _IMAGE_EXTENSIONS
    ]


def _load_metadata(tutorial_dir: Path, task_id: str) -> dict:
    """Load metadata from metadata.json."""
    import json

    meta_path = tutorial_dir / "metadata.json"
    if not meta_path.exists():
        return {"task_id": task_id}

    return json.loads(meta_path.read_text(encoding="utf-8"))


def _resolve_tutorial_dir(
    task_id: str, data_dir: str, tutorial_type: str,
) -> Path:
    """Find the tutorial directory for a task and validate per type.

    Layout: ``{data_dir}/{task_id}/tutorial/`` — note ``data_dir`` already
    includes the ``{tutorial_type}/{benchmark}`` segments resolved upstream.

    Raises:
        FileNotFoundError: when the type-specific required artifact is
            missing (page*.html for html; non-empty images/ for screenshot).
        NotImplementedError: when ``tutorial_type == "video"``.
        ValueError: when ``tutorial_type`` is otherwise unsupported.
    """
    if tutorial_type == "video":
        raise NotImplementedError(
            "tutorial_type='video' is reserved and not implemented",
        )
    if tutorial_type not in _SUPPORTED_TYPES:
        raise ValueError(
            f"Unsupported tutorial_type {tutorial_type!r}; "
            f"expected one of {sorted(_SUPPORTED_TYPES)} or 'video' (reserved)",
        )

    root = Path(data_dir)
    tutorial_dir = root / task_id / "tutorial"

    if not tutorial_dir.is_dir():
        raise FileNotFoundError(
            f"Tutorial directory not found for {task_id} at {tutorial_dir}",
        )

    if tutorial_type == "html":
        if not any(tutorial_dir.glob("page*.html")):
            raise FileNotFoundError(
                f"tutorial_type='html' requires page*.html under {tutorial_dir}",
            )
    elif tutorial_type == "screenshot":
        images_dir = tutorial_dir / "images"
        if not _collect_local_images(images_dir):
            raise FileNotFoundError(
                f"tutorial_type='screenshot' requires non-empty images/ "
                f"under {tutorial_dir}",
            )

    return tutorial_dir


def load_tutorial(
    task_id: str, data_dir: str, tutorial_type: str,
) -> TutorialMaterial:
    """Load a single tutorial from disk.

    Args:
        task_id: UUID of the task.
        data_dir: Root data directory already qualified with
            ``{tutorial_type}/{benchmark}`` (e.g.
            ``data_tutorial/html/osworld``).
        tutorial_type: The declared modality. Must match the on-disk
            artifacts; mismatched ``metadata.json:content_type`` only
            triggers a warning, the caller-supplied value wins.

    Returns:
        TutorialMaterial. ``body`` is the concatenated HTML for html
        type, and ``""`` for screenshot type (images carry the content).

    Raises:
        FileNotFoundError: If the required artifact is missing.
    """
    tutorial_dir = _resolve_tutorial_dir(task_id, data_dir, tutorial_type)
    metadata = _load_metadata(tutorial_dir, task_id)
    images_dir = tutorial_dir / "images"

    declared = metadata.get("content_type")
    if declared and declared != tutorial_type:
        logger.warning(
            "metadata.json content_type=%r conflicts with caller "
            "tutorial_type=%r for task %s; using caller value",
            declared, tutorial_type, task_id,
        )

    if tutorial_type == "html":
        html_files = sorted(tutorial_dir.glob("page*.html"))
        body = "\n".join(f.read_text(encoding="utf-8") for f in html_files)
    else:  # screenshot
        body = ""

    image_paths = _collect_local_images(images_dir)

    return TutorialMaterial(
        task_id=metadata.get("task_id", task_id),
        instruction=metadata.get("instruction", ""),
        content_type=tutorial_type,
        body=body,
        image_paths=image_paths,
        source_url=metadata.get("source_url", ""),
    )


def load_tutorials(
    tutorial_ids: list[str],
    data_dir: str,
    tutorial_type: str,
    task_id: str | None = None,
) -> TutorialMaterial:
    """Load a tutorial stack and merge it into a single ``TutorialMaterial``.

    Despite the plural name, this returns **one** ``TutorialMaterial`` — the
    concatenation of the requested tutorials' bodies (separated by a heading
    marker so the VLM can tell them apart) with deduplicated image paths.
    The single-entry case short-circuits and returns the only loaded tutorial
    directly (no wrapping). Missing entries are logged and skipped so a
    missing supplementary guide does not block the primary per-task tutorial.

    All requested tutorials must share the same ``tutorial_type``; mixing is
    not supported.

    Args:
        tutorial_ids: Ordered list of tutorial directory names to load.
            The first one is treated as primary (its instruction / source_url
            populate the merged material).
        data_dir: Root data directory already qualified with
            ``{tutorial_type}/{benchmark}``.
        tutorial_type: The declared modality, propagated to every
            ``load_tutorial`` call.
        task_id: Optional override for the returned material's ``task_id``.
            When ``None`` (default), the first loaded tutorial's id is used
            unchanged. When provided, the returned material's ``task_id`` is
            set to this value.

    Returns:
        Single merged ``TutorialMaterial``.

    Raises:
        FileNotFoundError: If none of ``tutorial_ids`` resolved to disk.
    """
    loaded: list[TutorialMaterial] = []
    for tid in tutorial_ids:
        try:
            loaded.append(load_tutorial(tid, data_dir, tutorial_type))
        except FileNotFoundError as e:
            logger.warning("Tutorial %s missing, skipping: %s", tid, e)

    if not loaded:
        raise FileNotFoundError(
            f"None of the requested tutorials exist under {data_dir}: {tutorial_ids}"
        )

    if len(loaded) == 1:
        only = loaded[0]
        if task_id is not None and task_id != only.task_id:
            return replace(only, task_id=task_id)
        return only

    primary = loaded[0]
    merged_image_paths: list[str] = []
    seen_images: set[str] = set()
    for tut in loaded:
        for img in tut.image_paths:
            if img not in seen_images:
                merged_image_paths.append(img)
                seen_images.add(img)

    if primary.content_type == "html":
        merged_body = "\n\n".join(
            f"<!-- tutorial: {tut.task_id} -->\n{tut.body}" for tut in loaded
        )
    else:
        # screenshot (and future video) carry content via images only;
        # keep body empty per the loader contract.
        merged_body = ""

    return TutorialMaterial(
        task_id=task_id if task_id is not None else primary.task_id,
        instruction=primary.instruction,
        content_type=primary.content_type,
        body=merged_body,
        image_paths=merged_image_paths,
        source_url=primary.source_url,
    )
