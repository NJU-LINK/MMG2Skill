"""Sanitize fragile pyautogui text input calls before OSWorld execution."""

from __future__ import annotations

import ast
import logging
import re

logger = logging.getLogger("anything2skill.benchmarks.osworld.sanitizer")


_TEXT_CALL_PATTERN = re.compile(r"pyautogui\.(?:write|typewrite)\s*\(")

_COORDINATE_FUNCTION_PARAMETERS = {
    "click": ["x", "y", "clicks", "interval", "button", "duration", "pause"],
    "rightClick": ["x", "y", "duration", "tween", "pause"],
    "middleClick": ["x", "y", "duration", "tween", "pause"],
    "doubleClick": ["x", "y", "interval", "button", "duration", "pause"],
    "tripleClick": ["x", "y", "interval", "button", "duration", "pause"],
    "moveTo": ["x", "y", "duration", "tween", "pause"],
    "dragTo": ["x", "y", "duration", "button", "mouseDownUp", "pause"],
}


def _build_press_sequence(text: str) -> str:
    commands: list[str] = []
    for char in text:
        press_value = "enter" if char == "\n" else char
        escaped_value = press_value.replace("\\", "\\\\").replace("'", "\\'")
        commands.append(f"pyautogui.press('{escaped_value}')")
    return "; ".join(commands)


def _consume_string_literal(code: str, start: int) -> tuple[str, int] | None:
    if code.startswith(("'''", '"""'), start):
        quote = code[start : start + 3]
        index = start + 3
        end = code.find(quote, index)
        if end == -1:
            return code[index:], len(code)
        literal = code[start : end + 3]
        return ast.literal_eval(literal), end + 3

    if start >= len(code) or code[start] not in ("'", '"'):
        return None

    quote = code[start]
    index = start + 1
    content_chars: list[str] = []
    while index < len(code):
        char = code[index]
        if char == "\\":
            if index + 1 >= len(code):
                content_chars.append("\\")
                index += 1
                continue
            next_char = code[index + 1]
            if next_char == "n":
                content_chars.append("\n")
            elif next_char == "r":
                content_chars.append("\r")
            elif next_char == "t":
                content_chars.append("\t")
            else:
                content_chars.append(next_char)
            index += 2
            continue
        if char == quote:
            return "".join(content_chars), index + 1
        content_chars.append(char)
        index += 1
    return "".join(content_chars), len(code)


def _find_call_end(code: str, start: int, literal_end: int) -> int:
    index = literal_end
    while index < len(code) and code[index].isspace():
        index += 1

    closing_paren = code.find(")", index)
    if closing_paren == -1:
        return len(code)
    return closing_paren + 1


def _rewrite_text_call(code: str, start: int) -> tuple[str, int] | None:
    match = _TEXT_CALL_PATTERN.match(code, start)
    if not match:
        return None

    index = match.end()
    while index < len(code) and code[index].isspace():
        index += 1

    keyword_match = re.match(r"(?:message|text)\s*=\s*", code[index:])
    if keyword_match:
        index += keyword_match.end()

    literal = _consume_string_literal(code, index)
    if literal is None:
        return None

    text_content, literal_end = literal
    call_end = _find_call_end(code, index, literal_end)
    return _build_press_sequence(text_content), call_end


def _fallback_rewrite_pyautogui_text_inputs(code: str) -> str:
    """Regex-based fallback for malformed pyautogui.write/typewrite calls."""
    logger.info(
        "SyntaxError detected in pyautogui action, using fallback rewrite."
    )

    result_parts: list[str] = []
    last_index = 0
    rewrote_any = False

    for match in _TEXT_CALL_PATTERN.finditer(code):
        rewritten = _rewrite_text_call(code, match.start())
        if rewritten is None:
            continue

        replacement, end_index = rewritten
        result_parts.append(code[last_index : match.start()])
        result_parts.append(replacement)
        last_index = end_index
        rewrote_any = True

    if not rewrote_any:
        return code

    result_parts.append(code[last_index:])
    return "".join(result_parts)


def rewrite_pyautogui_text_inputs(code: str) -> str:
    """Expand pyautogui.write/typewrite string literals into presses."""
    try:
        tree = ast.parse(code)

        class _TextCallRewriter(ast.NodeTransformer):
            def __init__(self):
                self.rewrote_any = False

            def _extract_text(self, call: ast.Call) -> str | None:
                if not (
                    isinstance(call.func, ast.Attribute)
                    and isinstance(call.func.value, ast.Name)
                    and call.func.value.id == "pyautogui"
                    and call.func.attr in ("write", "typewrite")
                ):
                    return None

                message_node = call.args[0] if call.args else None
                if message_node is None:
                    for keyword in call.keywords:
                        if keyword.arg in ("message", "text"):
                            message_node = keyword.value
                            break

                if (
                    isinstance(message_node, ast.Constant)
                    and isinstance(message_node.value, str)
                ):
                    return message_node.value
                return None

            def visit_Expr(self, node: ast.Expr):  # type: ignore[override]
                self.generic_visit(node)
                if isinstance(node.value, ast.Call):
                    text = self._extract_text(node.value)
                    if text is not None:
                        new_nodes: list[ast.Expr] = []
                        for char in text:
                            press_value = "enter" if char == "\n" else char
                            press_call = ast.Expr(
                                value=ast.Call(
                                    func=ast.Attribute(
                                        value=ast.Name(id="pyautogui", ctx=ast.Load()),
                                        attr="press",
                                        ctx=ast.Load(),
                                    ),
                                    args=[ast.Constant(value=press_value)],
                                    keywords=[],
                                )
                            )
                            new_nodes.append(press_call)
                        if new_nodes:
                            self.rewrote_any = True
                            return new_nodes
                return node

        rewriter = _TextCallRewriter()
        tree = rewriter.visit(tree)
        if not rewriter.rewrote_any:
            return code

        tree = ast.fix_missing_locations(tree)
        return ast.unparse(tree)
    except (SyntaxError, ValueError):
        return _fallback_rewrite_pyautogui_text_inputs(code)
    except Exception:
        logger.exception("Unexpected pyautogui rewrite failure, using fallback.")
        return _fallback_rewrite_pyautogui_text_inputs(code)


def _numeric_constant(node: ast.AST) -> float | None:
    if not isinstance(node, ast.Constant):
        return None
    if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
        return None
    return float(node.value)


def rewrite_kimi_normalized_coordinates(
    code: str,
    screen_size: tuple[int, int] | list[int],
) -> str:
    """Project Kimi-style normalized pyautogui coordinates to screen pixels."""
    try:
        screen_width = int(screen_size[0])
        screen_height = int(screen_size[1])
    except (IndexError, TypeError, ValueError):
        logger.warning(
            "Invalid screen_size for Kimi coordinate rewrite: %r",
            screen_size,
        )
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
                if not (0 <= x_value <= 1 and 0 <= y_value <= 1):
                    return node

                projected = {
                    "x": int(round(x_value * screen_width)),
                    "y": int(round(y_value * screen_height)),
                }
                for coord_name, projected_value in projected.items():
                    old_node = coord_nodes[coord_name]
                    new_node = ast.copy_location(
                        ast.Constant(value=projected_value),
                        old_node,
                    )
                    location_type, location_index = coord_locations[coord_name]
                    if location_type == "arg":
                        node.args[location_index] = new_node
                    else:
                        node.keywords[location_index].value = new_node

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
        logger.exception("Unexpected Kimi coordinate rewrite failure.")
        return code
