from src.smoother import FENSmoother


def test_single_outlier_does_not_win():
    s = FENSmoother(window_size=5)
    values = ["A", "A", "B", "A", "A"]
    for v in values:
        s.add(v)
    assert s.stable() == "A"


def test_full_consensus():
    s = FENSmoother(window_size=4)
    for _ in range(4):
        s.add("same")
    assert s.stable() == "same"

