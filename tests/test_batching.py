import pytest

from reviewlens.batching import MAX_CHARS_PER_DOC, chunked, truncate


def test_chunked_sizes():
    assert [len(b) for b in chunked(list(range(23)), 10)] == [10, 10, 3]


def test_chunked_empty():
    assert list(chunked([], 10)) == []


def test_chunked_rejects_zero():
    with pytest.raises(ValueError):
        list(chunked([1], 0))


def test_truncate_short_text_untouched():
    assert truncate("hello") == ("hello", False)


def test_truncate_long_text_within_limit():
    text, cut = truncate("word " * 3000)
    assert cut and len(text) <= MAX_CHARS_PER_DOC
