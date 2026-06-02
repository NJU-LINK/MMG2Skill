"""Claude-specific screenshot/coordinate adapters for OSWorld.

Mirrors OSWorld official mm_agents/anthropic/main.py:
- send screenshot resized to 1280x720
- rescale model-returned (x, y) from 1280x720 back to real screen size
"""

from __future__ import annotations

import ast
import io
import logging

from PIL import Image

from anything2skill.benchmarks.osworld.pyautogui_sanitizer import _numeric_constant

logger = logging.getLogger("anything2skill.benchmarks.osworld.claude_resize")


CLAUDE_BASE_SIZE: tuple[int, int] = (1280, 720)

# Encrypted/aliased gateway IDs that don't contain the "claude" substring.
# Extend here when onboarding a new Claude relay/alias.
CLAUDE_MODEL_IDS: frozenset[str] = frozenset(
    {
        # e.g. "anthropic-prod-xxx", "x-claude-relay-3"
    }
)

_COORDINATE_FUNCTION_PARAMETERS = {
    "click": ["x", "y", "clicks", "interval", "button", "duration", "pause"],
    "rightClick": ["x", "y", "duration", "tween", "pause"],
    "middleClick": ["x", "y", "duration", "tween", "pause"],
    "doubleClick": ["x", "y", "interval", "button", "duration", "pause"],
    "tripleClick": ["x", "y", "interval", "button", "duration", "pause"],
    "moveTo": ["x", "y", "duration", "tween", "pause"],
    "dragTo": ["x", "y", "duration", "button", "mouseDownUp", "pause"],
    # OSWorld official emits pyautogui.scroll(clicks, x, y) and hscroll(...).
    # The first positional is the wheel amount, NOT a coord, so x/y sit at indices 1/2.
    "scroll": ["clicks", "x", "y"],
    "hscroll": ["clicks", "x", "y"],
}


def is_claude_model(model_name: str) -> bool:
    name = (model_name or "").strip().lower()
    if not name:
        return False
    if "claude" in name:
        return True
    return name in CLAUDE_MODEL_IDS


def resize_screenshot_for_claude(png_bytes: bytes) -> bytes:
    """LANCZOS-resize a PNG screenshot to CLAUDE_BASE_SIZE."""
    img = Image.open(io.BytesIO(png_bytes))
    resized = img.resize(CLAUDE_BASE_SIZE, Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    resized.save(buf, format="PNG")
    return buf.getvalue()


def rewrite_claude_pixel_coordinates(
    code: str,
    screen_size: tuple[int, int] | list[int],
) -> str:
    """Rescale (x, y) from CLAUDE_BASE_SIZE space to real screen pixels.

    Rewrites int and float literals on pyautogui.{click,moveTo,dragTo,...} calls.
    Variables / expressions are left alone.
    """
    try:
        screen_width = int(screen_size[0])
        screen_height = int(screen_size[1])
    except (IndexError, TypeError, ValueError):
        logger.warning(
            "Invalid screen_size for Claude coordinate rewrite: %r",
            screen_size,
        )
        return code

    sx = screen_width / CLAUDE_BASE_SIZE[0]
    sy = screen_height / CLAUDE_BASE_SIZE[1]
    if sx == 1.0 and sy == 1.0:
        return code

    try:
        tree = ast.parse(code)

        class _CoordinateRewriter(ast.NodeTransformer):
            def __init__(self):
                self.rewrote_any = False

            def visit_Call(self, node: ast.Call):  # type: ignore[override]
                self.generic_visit(node)
                if not (
                    isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "pyautogui"
                ):
                    return node

                param_names = _COORDINATE_FUNCTION_PARAMETERS.get(node.func.attr)
                if not param_names:
                    return node

                coord_nodes: dict[str, ast.AST] = {}
                coord_locations: dict[str, tuple[str, int]] = {}
                for coord_name in ("x", "y"):
                    param_index = param_names.index(coord_name)
                    if len(node.args) > param_index:
                        coord_nodes[coord_name] = node.args[param_index]
                        coord_locations[coord_name] = ("arg", param_index)

                for keyword_index, keyword in enumerate(node.keywords):
                    if keyword.arg in ("x", "y"):
                        coord_nodes[keyword.arg] = keyword.value
                        coord_locations[keyword.arg] = ("keyword", keyword_index)

                if "x" not in coord_nodes or "y" not in coord_nodes:
                    return node

                x_value = _numeric_constant(coord_nodes["x"])
                y_value = _numeric_constant(coord_nodes["y"])
                if x_value is None or y_value is None:
                    return node

                # Match OSWorld official mm_agents/anthropic/main.py:145-146
                # which truncates via int(...) rather than rounding.
                projected = {
                    "x": int(x_value * sx),
                    "y": int(y_value * sy),
                }
                changed = False
                for coord_name, projected_value in projected.items():
                    old_node = coord_nodes[coord_name]
                    if isinstance(old_node, ast.Constant) and old_node.value == projected_value:
                        continue
                    new_node = ast.copy_location(
                        ast.Constant(value=projected_value),
                        old_node,
                    )
                    location_type, location_index = coord_locations[coord_name]
                    if location_type == "arg":
                        node.args[location_index] = new_node
                    else:
                        node.keywords[location_index].value = new_node
                    changed = True

                if changed:
                    self.rewrote_any = True
                return node

        rewriter = _CoordinateRewriter()
        tree = rewriter.visit(tree)
        if not rewriter.rewrote_any:
            return code

        tree = ast.fix_missing_locations(tree)
        return ast.unparse(tree)
    except (SyntaxError, ValueError):
        return code
    except Exception:
        logger.exception("Unexpected Claude coordinate rewrite failure.")
        return code
