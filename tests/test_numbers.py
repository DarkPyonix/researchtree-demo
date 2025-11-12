import pytest

from tinyspeech.text.numbers import expand_numbers


@pytest.mark.parametrize(
    "text, spoken",
    [
        ("in 1963", "in nineteen sixty-three"),
        ("in 1900", "in nineteen hundred"),
        ("in 2005", "in two thousand five"),
        ("in 1805", "in eighteen oh five"),
        ("$5.50", "five dollars, fifty cents"),
        ("$1", "one dollar"),
        ("the 3rd floor", "the third floor"),
        ("the 21st day", "the twenty-first day"),
        ("the 40th time", "the fortieth time"),
        ("1,254 people", "one thousand two hundred fifty-four people"),
        ("2.5 miles", "two point five miles"),
        ("room 007", "room zero zero seven"),
    ],
)
def test_expand_numbers(text, spoken):
    assert expand_numbers(text) == spoken
