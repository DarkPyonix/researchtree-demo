"""Spell out numbers the way a reader would say them.

Applied in this order: money ("$5.50" -> "five dollars, fifty cents"), ordinals ("3rd" -> "third"),
years ("1963" -> "nineteen sixty-three", for 1100 to 2099 standing alone), decimals
("2.5" -> "two point five") and other whole numbers ("1,254" -> "one thousand two hundred fifty-four").
"""

from __future__ import annotations

import re

_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
         "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
_SCALES = [(10**9, "billion"), (10**6, "million"), (1000, "thousand"), (100, "hundred")]
_IRREGULAR_ORDINALS = {"one": "first", "two": "second", "three": "third", "five": "fifth",
                       "eight": "eighth", "nine": "ninth", "twelve": "twelfth"}

_MONEY = re.compile(r"\$(\d[\d,]*)(?:\.(\d\d))?")
_ORDINAL = re.compile(r"\b(\d+)(st|nd|rd|th)\b")
_YEAR = re.compile(r"\b(1[1-9]\d\d|20\d\d)\b")
_DECIMAL = re.compile(r"(\d+)\.(\d+)")
_NUMBER = re.compile(r"\d[\d,]*")


def cardinal(n: int) -> str:
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + (f"-{_ONES[n % 10]}" if n % 10 else "")
    for value, name in _SCALES:
        if n >= value:
            head, rest = divmod(n, value)
            return f"{cardinal(head)} {name}" + (f" {cardinal(rest)}" if rest else "")
    raise AssertionError(n)


def ordinal(n: int) -> str:
    """Only the last word changes: "twenty-one" -> "twenty-first", "forty" -> "fortieth"."""
    words = cardinal(n)
    cut = max(words.rfind(" "), words.rfind("-")) + 1
    head, last = words[:cut], words[cut:]
    if last in _IRREGULAR_ORDINALS:
        return head + _IRREGULAR_ORDINALS[last]
    if last.endswith("y"):
        return head + last[:-1] + "ieth"
    return head + last + "th"


def year(n: int) -> str:
    if 2000 <= n < 2010:
        return cardinal(n)
    high, low = divmod(n, 100)
    if low == 0:
        return f"{cardinal(high)} hundred"
    return f"{cardinal(high)} {'oh ' if low < 10 else ''}{cardinal(low)}"


def _money(m: re.Match) -> str:
    dollars = int(m[1].replace(",", ""))
    out = f"{cardinal(dollars)} dollar{'' if dollars == 1 else 's'}"
    cents = int(m[2]) if m[2] else 0
    if cents:
        out += f", {cardinal(cents)} cent{'' if cents == 1 else 's'}"
    return out


def expand_numbers(text: str) -> str:
    text = _MONEY.sub(_money, text)
    text = _ORDINAL.sub(lambda m: ordinal(int(m[1])), text)
    text = _YEAR.sub(lambda m: year(int(m[1])), text)
    text = _DECIMAL.sub(lambda m: f"{cardinal(int(m[1]))} point {' '.join(_ONES[int(d)] for d in m[2])}", text)
    return _NUMBER.sub(_whole, text)


def _whole(m: re.Match) -> str:
    digits = m[0].replace(",", "")
    if len(digits) > 1 and digits.startswith("0"):
        return " ".join(_ONES[int(d)] for d in digits)  # "007" is read "zero zero seven"
    return cardinal(int(digits))
