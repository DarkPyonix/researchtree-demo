from tinyspeech.text.abbreviations import expand_abbreviations


def test_titles():
    assert expand_abbreviations("mr. and mrs. smith met dr. jones") == "mister and missus smith met doctor jones"


def test_only_with_period():
    assert expand_abbreviations("the co-op") == "the co-op"
