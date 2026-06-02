"""Unified VLM client using OpenAI-compatible API."""

import base64
import json
import logging
import os
from pathlib import Path

import backoff
import openai

logger = logging.getLogger("anything2skill.vlm")


_MODEL_OVERRIDES_PATH = Path(__file__).resolve().parents[2] / "configs" / "api" / "model_overrides.json"
_MODEL_OVERRIDES_CACHE: dict | None = None


def _load_model_overrides() -> dict:
    """Load (and cache) ``configs/api/model_overrides.json``.

    Schema: ``{"<model_id_or_prefix>": {"use_max_completion_tokens": bool, ...}}``.
    Missing file → empty dict. Parse failure is logged and returns {}.
    """
    global _MODEL_OVERRIDES_CACHE
    if _MODEL_OVERRIDES_CACHE is not None:
        return _MODEL_OVERRIDES_CACHE
    if not _MODEL_OVERRIDES_PATH.is_file():
        _MODEL_OVERRIDES_CACHE = {}
        return _MODEL_OVERRIDES_CACHE
    try:
        _MODEL_OVERRIDES_CACHE = json.loads(
            _MODEL_OVERRIDES_PATH.read_text(encoding="utf-8"),
        )
    except Exception as e:
        logger.warning("Failed to parse %s: %s", _MODEL_OVERRIDES_PATH, e)
        _MODEL_OVERRIDES_CACHE = {}
    return _MODEL_OVERRIDES_CACHE


def _lookup_model_override(model: str) -> dict:
    """Return override dict for ``model``: exact match first, then longest prefix."""
    table = _load_model_overrides()
    if model in table:
        return table[model]
    best_key = ""
    for key in table:
        if model.startswith(key) and len(key) > len(best_key):
            best_key = key
    return table[best_key] if best_key else {}


class RetryableVLMResponseError(RuntimeError):
    """Raised when an API call returns an unusable chat completion payload."""


def encode_image(image_content: bytes) -> str:
    """Encode raw image bytes to base64 string."""
    return base64.b64encode(image_content).decode("utf-8")


def _svg_to_jpeg_bytes(svg_path: Path) -> bytes:
    """Convert SVG file to JPEG bytes via cairosvg + Pillow."""
    import io

    import cairosvg
    from PIL import Image

    png_data = cairosvg.svg2png(url=str(svg_path))
    img = Image.open(io.BytesIO(png_data)).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


_SUPPORTED_FORMATS = {
    "PNG": "image/png",
    "JPEG": "image/jpeg",
    "GIF": "image/gif",
    "WEBP": "image/webp",
}

# Caps to stay under typical VLM API limits (Anthropic / OpenAI both reject
# very large images). Enforced by _shrink_to_caps().
_MAX_SIDE_PX = 2000
_MAX_BYTES = 5 * 1024 * 1024


def _shrink_to_caps(img, src_bytes_len: int):
    """Return (img, scaled) with max side <= _MAX_SIDE_PX.

    For the byte cap, pixel count scales roughly linearly with encoded
    size, so we derive a scale from ``sqrt(_MAX_BYTES / src_bytes_len)``
    and take the more aggressive of the two ratios. A single-pass resize
    is approximate for the byte cap; a post-encode fallback in the caller
    drops JPEG quality if the output still overruns.
    """
    from PIL import Image

    w, h = img.size
    scales = [1.0]
    if max(w, h) > _MAX_SIDE_PX:
        scales.append(_MAX_SIDE_PX / max(w, h))
    if src_bytes_len > _MAX_BYTES:
        scales.append((_MAX_BYTES / src_bytes_len) ** 0.5)
    scale = min(scales)
    if scale >= 1.0:
        return img, False
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return img.resize((new_w, new_h), Image.LANCZOS), True


def encode_image_file(path: str) -> str:
    """Read a local image file and return a base64 data URL string.

    SVG files are converted via cairosvg. For other formats, the actual
    image format is detected with PIL — unsupported formats (AVIF, HEIF,
    BMP, TIFF, etc.) are automatically converted to JPEG. Corrupted
    images raise an exception so the caller can skip them.

    Images with max side > 2000 px or file size > 5 MB are shrunk once
    to the more restrictive of the two ratios, re-encoded as JPEG. An
    animated GIF is reduced to its first frame.
    """
    import io

    from PIL import Image

    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".svg":
        png_data = _svg_to_jpeg_bytes(path)
        img = Image.open(io.BytesIO(png_data))
        img, _ = _shrink_to_caps(img, len(png_data))
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=90)
        data = buf.getvalue()
        if len(data) > _MAX_BYTES:
            buf = io.BytesIO()
            img.convert("RGB").save(buf, format="JPEG", quality=70)
            data = buf.getvalue()
        b64 = base64.b64encode(data).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"

    img = Image.open(path)
    w, h = img.size
    if w < 10 or h < 10:
        raise ValueError(f"Image too small ({w}x{h}), min 10x10")
    fmt = img.format
    is_animated_gif = fmt == "GIF" and getattr(img, "is_animated", False)
    if is_animated_gif:
        img.seek(0)

    src_bytes_len = path.stat().st_size
    must_reencode = (
        is_animated_gif
        or fmt not in _SUPPORTED_FORMATS
        or max(w, h) > _MAX_SIDE_PX
        or src_bytes_len > _MAX_BYTES
    )

    if not must_reencode:
        mime = _SUPPORTED_FORMATS[fmt]
        data = path.read_bytes()
    else:
        reason = (
            "animated GIF" if is_animated_gif
            else fmt if fmt not in _SUPPORTED_FORMATS
            else f"{w}x{h}/{src_bytes_len}B over caps"
        )
        logger.info("Re-encoding %s (%s) to JPEG: %s", path.name, fmt, reason)
        img, _ = _shrink_to_caps(img, src_bytes_len)
        img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        data = buf.getvalue()
        if len(data) > _MAX_BYTES:
            logger.warning(
                "Image %s still %d bytes after resize; dropping JPEG quality",
                path.name, len(data),
            )
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=70)
            data = buf.getvalue()
        mime = "image/jpeg"

    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def encode_image_files(
    paths: list[str], max_images: int | None = None,
) -> list[tuple[str, str]]:
    """Encode multiple image files to ``(filename, data_url)`` entries.

    Failed encodings are skipped with a ``logger.warning``. ``max_images=None``
    means no cap. The ordering of entries matches the (capped) input order.
    """
    if max_images is not None and len(paths) > max_images:
        paths = paths[:max_images]
    entries: list[tuple[str, str]] = []
    for p in paths:
        try:
            entries.append((Path(p).name, encode_image_file(p)))
        except Exception as e:
            logger.warning("Failed to encode image %s: %s", p, e)
    return entries


class VLMClient:
    """Unified VLM client wrapping openai.OpenAI for all LLM requests.

    Supports any OpenAI-compatible endpoint (OpenAI, vLLM, LiteLLM, etc.)
    by setting base_url.
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        base_url: str | None = None,
        api_key: str | None = None,
        use_max_completion_tokens: bool | None = None,
    ):
        self.model = model
        override = _lookup_model_override(model)
        if use_max_completion_tokens is None:
            use_max_completion_tokens = override.get(
                "use_max_completion_tokens", False,
            )
        self.use_max_completion_tokens = bool(use_max_completion_tokens)
        self.client = openai.OpenAI(
            base_url=base_url or os.environ.get("OPENAI_BASE_URL"),
            api_key=api_key or os.environ.get("OPENAI_API_KEY", ""),
            timeout=600.0,
        )

    @backoff.on_exception(
        backoff.constant,
        (
            openai.RateLimitError,
            openai.InternalServerError,
            openai.APIConnectionError,
            openai.APIStatusError,
            RetryableVLMResponseError,
        ),
        giveup=lambda e: isinstance(e, openai.APIStatusError)
        and e.status_code not in (429, 499, 500, 502, 503, 504),
        interval=30,
        max_tries=30,
    )
    def _create(self, **kwargs):
        """Create a completion and return validated text for backoff reuse."""
        response = self.client.chat.completions.create(**kwargs)
        return self._extract_text(response)

    def _extract_text(self, response) -> str:
        """Return assistant text or raise a retryable error for bad payloads."""
        try:
            content = response.choices[0].message.content
        except RetryableVLMResponseError:
            raise
        except (AttributeError, IndexError, TypeError) as e:
            raise RetryableVLMResponseError(
                f"Malformed chat completion response: {e}",
            ) from e

        if not content:
            raise RetryableVLMResponseError(
                "Chat completion returned empty content",
            )

        return content

    def chat(
        self,
        messages: list[dict],
        max_tokens: int = 1500,
        temperature: float | None = None,
    ) -> str:
        """Send a chat completion request and return the response text.

        Args:
            messages: OpenAI-format messages (supports text + image_url content).
            max_tokens: Max tokens in response.
            temperature: Sampling temperature. ``None`` (default) omits the
                parameter entirely — required for reasoning endpoints
                (o1/o3, Qwen/Kimi thinking mode) that reject it. Pass a
                float to enable classic sampling.
        """
        logger.debug("VLM request: model=%s, messages=%d", self.model, len(messages))
        token_key = "max_completion_tokens" if self.use_max_completion_tokens else "max_tokens"
        kwargs: dict = {
            "model": self.model,
            "messages": messages,
            token_key: max_tokens,
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        content = self._create(**kwargs)
        logger.debug("VLM response: %s chars", len(content))
        return content
