"""Incremental speech text with code-point offsets into the Markdown source."""
from __future__ import annotations

import re
import random
import string
from dataclasses import dataclass


_PREFIX = re.compile(r"^ {0,3}(?:#{1,6}[ \t]+|>[ \t]?|[-+*][ \t]+|\d{1,9}[.)][ \t]+)")
_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})")
_RULE = re.compile(r"^ {0,3}([-*_])(?:[ \t]*\1){2,}[ \t]*(?:\n|$)")
_LINK = re.compile(r"^!?\[([^\]\n]{0,256})\]\(")
_TABLE_RULE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$")
_NUMBER = r"-?\d+(?:\.\d+)?"
_SIMPLE_MATH = re.compile(rf"^({_NUMBER})(?:\s*([+\-*/×÷=])\s*({_NUMBER})){{1,3}}$")
_MATH_SYNTAX = re.compile(r"\\[A-Za-z]|[=+*/^_{}<>∑∫√×÷]|[A-Za-z]\s*-")
_COMPLEX_MATH = re.compile(r"\\[A-Za-z]|[{}^∑∫]|[A-Za-z]\w*\(")
_CUES = {
    "zh": {
        "code": ("代码放在对话里了。", "这段实现可以在对话里查看。", "具体代码我写在对话里。", "代码部分可以看对话。", "这段代码留在界面里供你查看。"),
        "math": ("公式写在对话里了。", "具体算式可以在对话里查看。", "这一步的表达式放在对话里。", "公式部分可以看对话。", "完整表达式留在界面里供你查看。"),
        "inline_code": ("对话里的代码", "这段代码", "界面中的代码"),
        "inline_math": ("对话里的公式", "这个表达式", "界面中的公式"),
    },
    "en": {
        "code": ("The code is in the conversation.", "You can view this implementation in the conversation.", "I've put the code in the conversation.", "The code is available on screen.", "You can check the code in the chat."),
        "math": ("The formula is in the conversation.", "You can view the expression in the conversation.", "I've put this expression in the chat.", "The formula is available on screen.", "You can check the full expression in the chat."),
        "inline_code": ("the code in the chat", "this code", "the code on screen"),
        "inline_math": ("the formula in the chat", "this expression", "the formula on screen"),
    },
}


def _simple_math(text: str, language: str) -> str | None:
    """Speak only a bounded numeric expression, never evaluate model output."""
    text = text.strip()
    if re.fullmatch(_NUMBER, text) or re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,15}", text):
        return text
    if not _SIMPLE_MATH.fullmatch(text):
        return None
    words = ({"+": "加", "-": "减", "*": "乘以", "/": "除以", "×": "乘以", "÷": "除以", "=": "等于"}
             if language == "zh" else
             {"+": " plus ", "-": " minus ", "*": " times ", "/": " divided by ", "×": " times ", "÷": " divided by ", "=": " equals "})
    number = lambda value: (("负" if language == "zh" else "negative ") + value[1:]
                            if value.startswith("-") else value)
    first = re.match(_NUMBER, text)
    spoken = number(first.group())
    for operation in re.finditer(rf"([+\-*/×÷=])\s*({_NUMBER})", text[first.end():]):
        spoken += words[operation.group(1)] + number(operation.group(2))
    return spoken


def _table_cells(line: str) -> list[tuple[int, int]]:
    """Locate cell spans without splitting escaped or inline-code pipes."""
    boundaries, ticks, index = [], 0, 0
    while index < len(line):
        if line[index] == "\\":
            index += 2
            continue
        if line[index] == "`":
            end = index + 1
            while end < len(line) and line[end] == "`":
                end += 1
            count = end - index
            ticks = 0 if ticks == count else count if not ticks else ticks
            index = end
            continue
        if line[index] == "|" and not ticks:
            boundaries.append(index)
        index += 1
    if not boundaries:
        return []
    points = [-1, *boundaries, len(line)]
    spans = list(zip((point + 1 for point in points[:-1]), points[1:]))
    if not line[:boundaries[0]].strip():
        spans = spans[1:]
    if not line[boundaries[-1] + 1:].strip():
        spans = spans[:-1]
    return spans


@dataclass(frozen=True)
class SpeechText:
    text: str
    source_ends: tuple[int, ...]
    consumed: int
    previous_end: int


class MarkdownSpeech:
    """Project display Markdown into speech with bounded structured lookahead.

    Code blocks and complex delimited math become varied viewing cues. Short
    inline identifiers and ordinary arithmetic remain available to speech.
    Normal EOF flushes pending syntax; cancellation must discard the parser.
    """

    def __init__(self, language: str = "zh", *, cue_offset: int | None = None):
        self.language = "en" if language == "en" else "zh"
        self.cue_offset = random.randrange(5) if cue_offset is None else cue_offset
        self.cue_counts: dict[str, int] = {}
        self.prose_since_cue = 64
        self.pending = ""
        self.offset = 0
        self.previous = ""
        self.line_start = True
        self.fence = ""
        self.skip_line = False
        self.inline_ticks = ""
        self.inline_start = 0
        self.inline_skipped = False
        self.math_close = ""
        self.math_open = ""
        self.math_content = ""
        self.math_start = 0
        self.math_inline = False
        self.math_skipped = False
        self.emphasis: list[str] = []
        self.link_depth = 0
        self.table_rows = False
        self.heading = False

    def _cue(self, kind: str, *, inline: bool = False) -> str:
        """Rotate per-reply wording and avoid narrating every adjacent block."""
        count = self.cue_counts.get(kind, 0)
        if not inline and (count >= 3 or (count and self.prose_since_cue < 64)):
            return ""
        key = "inline_" + kind if inline else kind
        choices = _CUES[self.language][key]
        self.cue_counts[kind] = count + 1
        self.prose_since_cue = 0
        return choices[(self.cue_offset + count) % len(choices)] + ("\n" if not inline else "")

    def feed(self, delta: str, *, final: bool = False) -> SpeechText:
        self.pending += delta
        output, ends = [], []
        previous_end = self.offset

        def consume(count, text=None, positions=None, *, prose=True):
            nonlocal previous_end
            raw = self.pending[:count]
            if text is not None:
                if not output and text:
                    previous_end = self.offset
                output.extend(text)
                if prose:
                    self.prose_since_cue += sum(char.isalnum() for char in text)
                ends.extend(positions if positions is not None else
                            [self.offset + count] * len(text))
            elif ends:
                ends[-1] = self.offset + count
            self.pending = self.pending[count:]
            self.offset += count
            if raw:
                self.previous = raw[-1]
                self.line_start = raw.endswith("\n")
                if self.line_start:
                    self.heading = False

        def reference(count, kind, *, inline=False, start=None):
            spoken = self._cue(kind, inline=inline)
            begin = max(self.offset, start or 0)
            positions = [begin] * max(0, len(spoken) - 1) + ([self.offset + count] if spoken else [])
            consume(count, spoken, positions, prose=False)

        def math_finished(count):
            spoken = _simple_math(self.math_content, self.language)
            if spoken is not None:
                positions = [self.offset] * max(0, len(spoken) - 1) + ([self.offset + count] if spoken else [])
                consume(count, spoken, positions)
            elif self.math_open == "$" and not _MATH_SYNTAX.search(self.math_content):
                literal = self.math_open + self.math_content + (self.math_close if count > len(self.math_content) else "")
                consume(count, literal, [self.offset + count] * len(literal))
            else:
                reference(count, "math", inline=self.math_inline, start=self.math_start)
            self.math_close = ""
            self.math_content = ""

        def math_literal(count):
            spoken = self.math_open + self.math_content
            consume(count, spoken, [self.offset + count] * len(spoken))
            self.math_close = ""
            self.math_content = ""

        def table_row(line, count):
            spoken, positions = [], []
            spans = _table_cells(line)
            for index, (start, end) in enumerate(spans):
                cell = line[start:end]
                leading = len(cell) - len(cell.lstrip())
                parsed = MarkdownSpeech(self.language, cue_offset=self.cue_offset).feed(cell.strip(), final=True)
                spoken.extend(parsed.text)
                positions.extend(self.offset + start + leading + end for end in parsed.source_ends)
                if index < len(spans) - 1:
                    spoken.append("，")
                    positions.append(self.offset + end + 1)
            if count > len(line):
                spoken.append("\n")
                positions.append(self.offset + count)
            consume(count, "".join(spoken), positions)

        while self.pending:
            text = self.pending
            if self.math_close:
                if self.math_skipped:
                    if text.startswith(self.math_close):
                        consume(len(self.math_close))
                        self.math_close = ""
                    elif not final and self.math_close.startswith(text):
                        break
                    else:
                        consume(1)
                    continue
                end = text.find(self.math_close)
                newline = text.find("\n") if self.math_open == "$" else -1
                content = text[:end] if end >= 0 else text[:256]
                if _COMPLEX_MATH.search(content):
                    reference(0, "math", inline=self.math_inline, start=self.math_start)
                    self.math_skipped = True
                    continue
                if newline >= 0 and (end < 0 or newline < end):
                    self.math_content = text[:newline + 1]
                    math_literal(newline + 1)
                elif end >= 0 and end <= 256:
                    self.math_content = text[:end]
                    math_finished(end + len(self.math_close))
                elif len(text) > 256:
                    self.math_content = text[:256]
                    if self.math_open == "$" and not _MATH_SYNTAX.search(self.math_content):
                        math_literal(256)
                    else:
                        reference(256, "math", inline=self.math_inline, start=self.math_start)
                        self.math_skipped = True
                        self.math_content = ""
                elif final:
                    self.math_content = text
                    if self.math_open == "$" and not _MATH_SYNTAX.search(text):
                        math_literal(len(text))
                    else:
                        math_finished(len(text))
                else:
                    break
                continue
            if self.inline_ticks:
                marker = re.search(r"(?<!`)" + re.escape(self.inline_ticks) + r"(?!`)", text)
                if self.inline_skipped:
                    if marker and marker.start() == 0:
                        if marker.end() == len(text) and not final:
                            break
                        consume(marker.end())
                        self.inline_ticks = ""
                    elif not final and self.inline_ticks.startswith(text):
                        break
                    else:
                        consume(1)
                    continue
                if marker and marker.start() <= 48 and "\n" not in text[:marker.start()]:
                    if marker.end() == len(text) and not final:
                        break
                    content = text[:marker.start()]
                    if len(re.findall(r"[{}();=@/\\]", content)) >= 4:
                        reference(marker.end(), "code", inline=True, start=self.inline_start)
                    else:
                        spoken = _simple_math(content, self.language) or content
                        positions = (list(range(self.offset + 1, self.offset + len(content) + 1))
                                     if spoken == content else [self.offset] * len(spoken))
                        if positions:
                            positions[-1] = self.offset + marker.end()
                        consume(marker.end(), spoken, positions)
                    self.inline_ticks = ""
                elif len(text) > 48 or "\n" in text:
                    trailing = len(text) - len(text.rstrip("`"))
                    safe = marker.start() if marker else len(text) - trailing
                    reference(min(49, safe), "code", inline=True, start=self.inline_start)
                    self.inline_skipped = True
                elif final:
                    consume(len(text), text, range(self.offset + 1, self.offset + len(text) + 1))
                    self.inline_ticks = ""
                else:
                    break
                continue
            if self.skip_line:
                end = text.find("\n")
                if end < 0:
                    consume(len(text))
                    break
                consume(end + 1)
                self.skip_line = False
                continue
            if self.link_depth:
                char = text[0]
                if char == "(":
                    self.link_depth += 1
                elif char == ")":
                    self.link_depth -= 1
                consume(1)
                continue
            if self.line_start and not self.inline_ticks:
                table_prefix = text.startswith(("| ", "|\t")) or text == "|"
                if self.table_rows and not table_prefix:
                    self.table_rows = False
                if not self.fence and (self.table_rows or table_prefix):
                    end = text.find("\n")
                    if end < 0 and not final and len(text) < 512:
                        break
                    line = text[:end] if end >= 0 else text
                    if self.table_rows and _table_cells(line):
                        table_row(line, len(line) + int(end >= 0))
                        continue
                    self.table_rows = False
                    if end >= 0 and len(_table_cells(line)) >= 2:
                        next_line = text[end + 1:].split("\n", 1)[0]
                        if "\n" not in text[end + 1:] and not final and len(next_line) < 512:
                            break
                        if _TABLE_RULE.fullmatch(next_line):
                            count = end + 1 + len(next_line) + int(len(text) > end + 1 + len(next_line))
                            table_row(line, count)
                            self.table_rows = True
                            continue
                fence = _FENCE.match(text)
                if fence:
                    marker = fence.group(1)
                    if fence.end() == len(text) and not final:
                        break
                    if not self.fence:
                        header = text[fence.end():].split("\n", 1)[0]
                        if not final and "\n" not in text and len(header) < 128:
                            break
                        self.fence = marker
                        self.skip_line = True
                        kind = "math" if header.strip().lower() in {"math", "latex", "tex"} else "code"
                        reference(fence.end(), kind)
                        continue
                    if marker[0] == self.fence[0] and len(marker) >= len(self.fence):
                        tail = text[fence.end():]
                        if not final and "\n" not in tail and len(tail) < 128:
                            break
                        if not tail.split("\n", 1)[0].strip():
                            self.fence = ""
                            self.skip_line = True
                            consume(fence.end())
                            continue
                if not final and re.fullmatch(r" {0,3}[`~]{1,2}", text):
                    break
                if not self.fence:
                    rule = _RULE.match(text)
                    if rule and (final or "\n" in rule.group()):
                        consume(rule.end(), "\n" if rule.group().endswith("\n") else "")
                        continue
                    prefix = _PREFIX.match(text)
                    if prefix:
                        if prefix.end() == len(text) and not final:
                            break
                        heading = prefix.group().lstrip().startswith("#")
                        consume(prefix.end())
                        self.heading = self.heading or heading
                        self.line_start = True
                        continue
                    if not final and len(text) < 16 and (
                            re.fullmatch(r" {0,3}[#>*+\-]+", text) or
                            re.fullmatch(r" {0,3}\d{1,9}[.)]?", text)):
                        break
            if self.fence:
                consume(1)
                continue
            char = text[0]
            if text.startswith(("\\[", "\\(", "$$")) or char == "$":
                if char == "$" and len(text) == 1 and not final:
                    break
                opener = text[:2] if text.startswith(("\\[", "\\(", "$$")) else "$"
                self.math_close = {"\\[": "\\]", "\\(": "\\)", "$$": "$$", "$": "$"}[opener]
                self.math_open = opener
                self.math_start = self.offset + len(opener)
                self.math_inline = opener in ("$", "\\(")
                self.math_content = ""
                self.math_skipped = False
                consume(len(opener))
                continue
            if self.heading and char == "#" and self.previous.isspace():
                tail = text.split("\n", 1)[0]
                if not final and "\n" not in text and len(text) < 32:
                    break
                if re.fullmatch(r"#+[ \t]*", tail):
                    consume(len(tail))
                    continue
            if char == "`":
                size = len(text) - len(text.lstrip("`"))
                if size == len(text) and not final:
                    break
                marker = text[:size]
                self.inline_ticks = marker
                self.inline_start = self.offset + size
                self.inline_skipped = False
                consume(size)
                continue
            if char == "\\":
                if len(text) == 1 and not final:
                    break
                if len(text) > 1 and text[1] in string.punctuation:
                    consume(2, text[1])
                    continue
            if char == "[" or text.startswith("![") or text == "!":
                link = _LINK.match(text)
                if link:
                    label = link.group(1)
                    base = self.offset + link.start(1)
                    spoken = MarkdownSpeech(self.language, cue_offset=self.cue_offset).feed(label, final=True)
                    consume(link.end(), spoken.text, [base + end for end in spoken.source_ends])
                    self.link_depth = 1
                    continue
                closing = text.find("]")
                possible = ("\n" not in text and len(text) < 260 and
                            (closing < 0 or closing == len(text) - 1))
                if not final and possible:
                    break
            if char in "*_~":
                size = len(text) - len(text.lstrip(char))
                if size == len(text) and not final:
                    break
                marker, following = text[:size], text[size:size + 1]
                previous = self.previous
                if marker in self.emphasis and previous and not previous.isspace():
                    self.emphasis.remove(marker)
                    consume(size)
                    continue
                word = lambda c: bool(c and c.isascii() and c.isalnum())
                arithmetic = (char == "*" and word(previous) and word(following) and
                              (size == 1 or following.isdigit()))
                identifier = char == "_" and previous.isalnum() and following.isalnum()
                if (size <= 3 and (char != "~" or size == 2) and following and
                        not following.isspace() and not arithmetic and not identifier):
                    self.emphasis.append(marker)
                    consume(size)
                    continue
                consume(size, marker, range(self.offset + 1, self.offset + size + 1))
                continue
            consume(1, char)
        if final:
            self.math_close = ""
            self.inline_ticks = ""
        return SpeechText("".join(output), tuple(ends), self.offset,
                          previous_end if output else self.offset)
