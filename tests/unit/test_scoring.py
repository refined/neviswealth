import pytest

from app.search.scoring import noisy_or


def test_noisy_or_is_bounded_and_monotonic() -> None:
    assert noisy_or([]) == 0
    assert noisy_or([0.5, 0.5]) == pytest.approx(0.75)
    assert noisy_or([1.5]) == 1.0
    assert noisy_or([0.6, 0.3]) > noisy_or([0.6])
