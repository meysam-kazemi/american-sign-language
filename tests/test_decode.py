from ml.decode import StreamDecoder


def feed(decoder, label, confidence, times):
    text = decoder.text
    for _ in range(times):
        text = decoder.update(label, confidence)
    return text


def test_commits_after_stability_reached():
    d = StreamDecoder(stability=3, min_confidence=0.6)
    assert feed(d, "a", 0.9, 2) == ""      # not yet stable
    assert feed(d, "a", 0.9, 1) == "A"     # third frame commits


def test_low_confidence_is_ignored():
    d = StreamDecoder(stability=3, min_confidence=0.6)
    assert feed(d, "a", 0.5, 5) == ""


def test_same_sign_not_committed_twice_in_a_row():
    d = StreamDecoder(stability=2, min_confidence=0.6)
    assert feed(d, "a", 0.9, 5) == "A"     # repeated frames still commit once


def test_drop_then_repeat_commits_again():
    d = StreamDecoder(stability=2, min_confidence=0.6)
    feed(d, "a", 0.9, 2)
    feed(d, None, 0.0, 1)                   # gap clears last-committed
    assert feed(d, "a", 0.9, 2) == "AA"


def test_space_and_del_labels():
    d = StreamDecoder(stability=2, min_confidence=0.6)
    feed(d, "a", 0.9, 2)
    feed(d, "space", 0.9, 2)
    feed(d, "b", 0.9, 2)
    assert d.text == "A B"
    feed(d, "del", 0.9, 2)
    assert d.text == "A "


def test_nothing_label_is_noop():
    d = StreamDecoder(stability=2, min_confidence=0.6)
    assert feed(d, "nothing", 0.9, 4) == ""


def test_reset_clears_state():
    d = StreamDecoder(stability=2, min_confidence=0.6)
    feed(d, "a", 0.9, 2)
    d.reset()
    assert d.text == ""
    assert feed(d, "a", 0.9, 2) == "A"
