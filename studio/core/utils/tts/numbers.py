"""Bounded numeric speech projection; display text and numeric values stay intact."""
from __future__ import annotations

import re


_TOKEN = re.compile(r"(?P<sign>[+-]?)(?P<integer>[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)"
                    r"(?:\.(?P<fraction>[0-9]+))?")
_FRAGMENT = re.compile(r"[+-]?[0-9][0-9,.]*")
_IDENTIFIER = re.compile(
    r"(?:编号|号码|电话|手机号|账号|账户|验证码|订单号|身份证号|邮编|序列号|"
    r"\b(?:id|code|phone|account|serial|order)\s*(?:number)?)\s*(?:为|是|[:：])?\s*$",
    re.IGNORECASE)
_ZH_DIGITS = "零一二三四五六七八九"
_EN_DIGITS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine")
_SMALL = (*_EN_DIGITS, "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
          "sixteen", "seventeen", "eighteen", "nineteen")
_TENS = ("", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety")


def _zh_integer(digits: str) -> str:
    groups = []
    while digits:
        groups.append(int(digits[-4:]))
        digits = digits[:-4]
    parts, gap = [], False
    for index in range(len(groups) - 1, -1, -1):
        value = groups[index]
        if not value:
            gap = bool(parts)
            continue
        if parts and (gap or value < 1000):
            parts.append("零")
        block, zero = [], False
        for place, digit in enumerate(f"{value:04d}"):
            if digit == "0":
                zero = bool(block)
            else:
                if zero:
                    block.append("零")
                block.extend((_ZH_DIGITS[int(digit)], ("千", "百", "十", "")[place]))
                zero = False
        parts.append("".join(block) + ("", "万", "亿", "万亿")[index])
        gap = False
    result = "".join(parts) or "零"
    return result[1:] if result.startswith("一十") else result


def _en_integer(digits: str) -> str:
    value = int(digits)
    if value < 20:
        return _SMALL[value]
    if value < 100:
        return _TENS[value // 10] + (" " + _SMALL[value % 10] if value % 10 else "")
    if value < 1000:
        return _SMALL[value // 100] + " hundred" + (
            " " + _en_integer(str(value % 100)) if value % 100 else "")
    for power, unit in ((15, "quadrillion"), (12, "trillion"), (9, "billion"),
                        (6, "million"), (3, "thousand")):
        scale = 10 ** power
        if value >= scale:
            return _en_integer(str(value // scale)) + " " + unit + (
                ", " + _en_integer(str(value % scale)) if value % scale else "")
    raise ValueError("Unsupported numeric scale")


def _digits(raw: str, language: str, count: int = 0):
    output, ends = [], []
    for index, char in enumerate(raw):
        if not char.isascii() or not char.isdecimal():
            continue
        separator = ("，" if language == "zh" else ", ") if count and count % 4 == 0 else (
            " " if count and language == "en" else "")
        word = _ZH_DIGITS[int(char)] if language == "zh" else _EN_DIGITS[int(char)]
        output.append(separator + word)
        ends.extend([index] * (len(separator) + len(word) - 1) + [index + 1])
        count += 1
    return "".join(output), tuple(ends), count


def number_text(raw: str, language: str, *, identifier: bool = False,
                force: bool = False) -> tuple[str, tuple[int, ...]]:
    """Render one complete number and relative source boundaries without rounding.

    Ordinary short numbers keep the provider's existing reading. Large values
    use cardinal words through 16 integer digits; longer values, explicit
    identifiers and leading-zero strings use grouped digits. Decimal tails are
    always lossless digit sequences when normalization is needed.
    """
    match = _TOKEN.fullmatch(raw)
    if match is None:
        return raw, tuple(range(1, len(raw) + 1))
    integer = match['integer'].replace(',', '')
    fraction = match['fraction'] or ''
    leading_zero = len(integer) > 1 and integer.startswith('0')
    if not (force or identifier or leading_zero or len(integer) >= 5 or len(fraction) >= 5):
        return raw, tuple(range(1, len(raw) + 1))
    sign = ({'-': '负', '+': '正'} if language == 'zh' else
            {'-': 'negative ', '+': 'positive '}).get(match['sign'], '')
    start = len(match['sign'])
    output = sign
    ends = [0] * max(0, len(sign) - 1) + ([start] if sign else [])
    if identifier or len(integer) > 16 or leading_zero:
        spoken, positions, _ = _digits(match['integer'], language)
        output += spoken
        ends.extend(start + end for end in positions)
    else:
        spoken = _zh_integer(integer) if language == 'zh' else _en_integer(integer)
        output += spoken
        ends.extend([start] * (len(spoken) - 1) + [match.end('integer')])
    if fraction:
        point = '点' if language == 'zh' else ' point '
        output += point
        ends.extend([match.end('integer')] * (len(point) - 1) + [match.start('fraction')])
        spoken, positions, _ = _digits(fraction, language)
        output += spoken
        ends.extend(match.start('fraction') + end for end in positions)
    return output, tuple(ends)


class NumberReader:
    """Hold only a bounded numeric prefix; stream long digit tails in fixed groups."""

    def __init__(self):
        self.mode = ''
        self.count = 0

    def read(self, text: str, language: str, *, previous: str = '', context: str = '',
             final: bool = False) -> tuple[int, str, tuple[int, ...]] | None:
        """Return a numeric projection, zero consumed to wait, or None for prose.

        The caller must retry the same prefix after a wait and discard this
        reader on cancellation. Source boundaries are relative to that prefix.
        """
        if self.mode:
            digits = re.match(r'[0-9]+', text[:16])
            if digits:
                if self.mode == 'literal':
                    return digits.end(), digits.group(), tuple(range(1, digits.end() + 1))
                size = min(16, digits.end())
                if size < 16 and digits.end() == len(text) and not final:
                    return 0, '', ()
                spoken, ends, self.count = _digits(text[:size], language, self.count)
                return size, spoken, ends
            if text.startswith('.') and self.mode == 'integer':
                if len(text) == 1 and not final:
                    return 0, '', ()
                if len(text) > 1 and text[1].isdigit():
                    self.mode, self.count = 'fraction', 0
                    point = '点' if language == 'zh' else ' point '
                    return 1, point, (0,) * (len(point) - 1) + (1,)
            if text.startswith(',') and self.mode == 'integer':
                if not final and len(text) < 4 and (not text[1:] or text[1:].isdigit()):
                    return 0, '', ()
                if re.match(r',[0-9]{3}(?![0-9])', text):
                    return 1, '', ()
            self.mode, self.count = '', 0
        if ((previous and previous.isascii() and (previous.isalpha() or previous == '_'))
                or re.search(r'https?://\S*$', context)):
            digits = re.match(r'[0-9]+', text)
            if digits:
                self.mode = 'literal'
                return digits.end(), digits.group(), tuple(range(1, digits.end() + 1))
            return None
        if text[:1] in ('+', '-') and re.search(r'[0-9)\]]\s*$', context):
            return None
        if text in ('+', '-') and not final:
            return 0, '', ()
        fragment = _FRAGMENT.match(text[:256])
        if fragment is None:
            return None
        match = _TOKEN.match(fragment.group())
        integer = match['integer'].replace(',', '')
        fraction = match['fraction'] or ''
        if len(integer) > 16:
            size, digits = len(match['sign']), 0
            while digits < 16:
                digits += text[size].isdigit()
                size += 1
            self.mode, self.count = 'integer', 16
            start = len(match['sign'])
            sign = ({'-': '负', '+': '正'} if language == 'zh' else
                    {'-': 'negative ', '+': 'positive '}).get(match['sign'], '')
            spoken, positions, _ = _digits(text[start:size], language)
            ends = ((0,) * max(0, len(sign) - 1) + ((start,) if sign else ())
                    + tuple(start + end for end in positions))
            spoken = sign + spoken
            return size, spoken, ends
        if len(fraction) > 16:
            size = match.start('fraction') + 16
            self.mode, self.count = 'fraction', 16
            spoken, ends = number_text(text[:size], language, force=True)
            return size, spoken, ends
        if not final and fragment.end() == len(text) and fragment.end() < 256:
            return 0, '', ()
        spoken, ends = number_text(match.group(), language,
                                   identifier=bool(_IDENTIFIER.search(context)))
        return match.end(), spoken, ends
